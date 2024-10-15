import Mathlib

namespace NUMINAMATH_CALUDE_milk_problem_l896_89667

theorem milk_problem (initial_milk : ℚ) (tim_fraction : ℚ) (kim_fraction : ℚ) : 
  initial_milk = 3/4 →
  tim_fraction = 1/3 →
  kim_fraction = 1/2 →
  kim_fraction * (initial_milk - tim_fraction * initial_milk) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_milk_problem_l896_89667


namespace NUMINAMATH_CALUDE_smallest_prime_20_less_than_square_l896_89627

theorem smallest_prime_20_less_than_square : ∃ (m : ℕ), 
  (∀ (n : ℕ), n > 0 ∧ Nat.Prime n ∧ (∃ (k : ℕ), n = k^2 - 20) → n ≥ 5) ∧
  5 > 0 ∧ Nat.Prime 5 ∧ 5 = m^2 - 20 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_20_less_than_square_l896_89627


namespace NUMINAMATH_CALUDE_bertha_family_women_without_daughters_l896_89622

/-- Represents a woman in Bertha's family tree -/
structure Woman where
  has_daughters : Bool

/-- Bertha's family tree -/
structure Family where
  daughters : Finset Woman
  granddaughters : Finset Woman

/-- The number of women who have no daughters in Bertha's family -/
def num_women_without_daughters (f : Family) : Nat :=
  (f.daughters.filter (fun w => !w.has_daughters)).card +
  (f.granddaughters.filter (fun w => !w.has_daughters)).card

theorem bertha_family_women_without_daughters :
  ∃ f : Family,
    f.daughters.card = 8 ∧
    (∀ d ∈ f.daughters, d.has_daughters) ∧
    (∀ d ∈ f.daughters, (f.granddaughters.filter (fun g => g.has_daughters.not)).card = 4) ∧
    (f.daughters.card + f.granddaughters.card = 40) ∧
    num_women_without_daughters f = 32 := by
  sorry

end NUMINAMATH_CALUDE_bertha_family_women_without_daughters_l896_89622


namespace NUMINAMATH_CALUDE_no_intersection_l896_89600

/-- Parabola C: y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Point M(x₀, y₀) is inside the parabola if y₀² < 4x₀ -/
def inside_parabola (x₀ y₀ : ℝ) : Prop := y₀^2 < 4*x₀

/-- Line l: y₀y = 2(x + x₀) -/
def line (x₀ y₀ x y : ℝ) : Prop := y₀*y = 2*(x + x₀)

theorem no_intersection (x₀ y₀ : ℝ) (h : inside_parabola x₀ y₀) :
  ¬∃ x y, parabola x y ∧ line x₀ y₀ x y :=
sorry

end NUMINAMATH_CALUDE_no_intersection_l896_89600


namespace NUMINAMATH_CALUDE_equation_solutions_l896_89687

theorem equation_solutions (x : ℝ) (y : ℝ) : 
  x^2 + 6 * (x / (x - 3))^2 = 81 →
  y = ((x - 3)^2 * (x + 4)) / (3*x - 4) →
  (y = -9 ∨ y = 225/176) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l896_89687


namespace NUMINAMATH_CALUDE_instructor_schedule_lcm_l896_89655

theorem instructor_schedule_lcm : Nat.lcm 9 (Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 10 12))) = 360 := by
  sorry

end NUMINAMATH_CALUDE_instructor_schedule_lcm_l896_89655


namespace NUMINAMATH_CALUDE_down_payment_proof_l896_89613

/-- Calculates the down payment for a car loan given the total price, monthly payment, and loan duration in years. -/
def calculate_down_payment (total_price : ℕ) (monthly_payment : ℕ) (loan_years : ℕ) : ℕ :=
  total_price - monthly_payment * loan_years * 12

/-- Proves that the down payment for a $20,000 car with a 5-year loan and $250 monthly payment is $5,000. -/
theorem down_payment_proof :
  calculate_down_payment 20000 250 5 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_down_payment_proof_l896_89613


namespace NUMINAMATH_CALUDE_same_parity_iff_square_sum_l896_89663

theorem same_parity_iff_square_sum (a b : ℤ) :
  (∃ k : ℤ, a - b = 2 * k) ↔ (∃ c d : ℤ, a^2 + b^2 + c^2 + 1 = d^2) := by sorry

end NUMINAMATH_CALUDE_same_parity_iff_square_sum_l896_89663


namespace NUMINAMATH_CALUDE_abs_fraction_inequality_l896_89642

theorem abs_fraction_inequality (x : ℝ) :
  x ≠ 0 → (|(x - 2) / x| > (x - 2) / x ↔ 0 < x ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_abs_fraction_inequality_l896_89642


namespace NUMINAMATH_CALUDE_mitchell_antonio_pencil_difference_l896_89639

theorem mitchell_antonio_pencil_difference :
  ∀ (mitchell_pencils antonio_pencils : ℕ),
    mitchell_pencils = 30 →
    mitchell_pencils + antonio_pencils = 54 →
    mitchell_pencils > antonio_pencils →
    mitchell_pencils - antonio_pencils = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_mitchell_antonio_pencil_difference_l896_89639


namespace NUMINAMATH_CALUDE_pentagonal_sum_theorem_l896_89676

def pentagonal_layer_sum (n : ℕ) : ℕ := 4 * (3^(n-1) - 1)

theorem pentagonal_sum_theorem (n : ℕ) :
  n ≥ 1 →
  (pentagonal_layer_sum 1 = 0) →
  (∀ k : ℕ, k ≥ 1 → pentagonal_layer_sum (k+1) = 3 * pentagonal_layer_sum k + 4) →
  pentagonal_layer_sum n = 4 * (3^(n-1) - 1) :=
by sorry

end NUMINAMATH_CALUDE_pentagonal_sum_theorem_l896_89676


namespace NUMINAMATH_CALUDE_test_questions_count_l896_89635

theorem test_questions_count (sections : Nat) (correct_answers : Nat) 
  (h1 : sections = 4)
  (h2 : correct_answers = 20)
  (h3 : ∀ x : Nat, x > 0 → (60 : Real) / 100 < (correct_answers : Real) / x → (correct_answers : Real) / x < (70 : Real) / 100 → x % sections = 0 → x = 32) :
  ∃ total_questions : Nat, 
    total_questions > 0 ∧ 
    (60 : Real) / 100 < (correct_answers : Real) / total_questions ∧ 
    (correct_answers : Real) / total_questions < (70 : Real) / 100 ∧ 
    total_questions % sections = 0 ∧
    total_questions = 32 :=
by sorry

end NUMINAMATH_CALUDE_test_questions_count_l896_89635


namespace NUMINAMATH_CALUDE_green_guards_with_shields_l896_89619

theorem green_guards_with_shields (total : ℝ) (green : ℝ) (yellow : ℝ) (special : ℝ) 
  (h1 : green = (3/8) * total)
  (h2 : yellow = (5/8) * total)
  (h3 : special = (1/5) * total)
  (h4 : ∃ (r s : ℝ), (green * (r/s) + yellow * (r/(3*s)) = special) ∧ (r/s > 0) ∧ (s ≠ 0)) :
  ∃ (r s : ℝ), (r/s = 12/35) ∧ (green * (r/s) = (3/5) * special) := by
  sorry

end NUMINAMATH_CALUDE_green_guards_with_shields_l896_89619


namespace NUMINAMATH_CALUDE_katie_game_difference_l896_89653

def katie_new_games : ℕ := 57
def katie_old_games : ℕ := 39
def friends_new_games : ℕ := 34

theorem katie_game_difference :
  (katie_new_games + katie_old_games) - friends_new_games = 62 := by
  sorry

end NUMINAMATH_CALUDE_katie_game_difference_l896_89653


namespace NUMINAMATH_CALUDE_probability_coprime_pairs_l896_89697

def S : Finset Nat := Finset.range 8

theorem probability_coprime_pairs (a b : Nat) (h : a ∈ S ∧ b ∈ S ∧ a ≠ b) :
  (Finset.filter (fun p : Nat × Nat => p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 ≠ p.2 ∧ Nat.gcd p.1 p.2 = 1) 
    (S.product S)).card / (Finset.filter (fun p : Nat × Nat => p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 ≠ p.2) 
    (S.product S)).card = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_coprime_pairs_l896_89697


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l896_89625

theorem arithmetic_calculations :
  ((5 : ℤ) - (-10) + (-32) - 7 = -24) ∧
  ((1/4 + 1/6 - 1/2 : ℚ) * 12 + (-2)^3 / (-4) = 1) ∧
  ((3^2 : ℚ) + (-2-5) / 7 - |-(1/4)| * (-2)^4 + (-1)^2023 = 3) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l896_89625


namespace NUMINAMATH_CALUDE_bryans_books_l896_89647

theorem bryans_books (num_shelves : ℕ) (books_per_shelf : ℕ) 
  (h1 : num_shelves = 9) 
  (h2 : books_per_shelf = 56) : 
  num_shelves * books_per_shelf = 504 := by
  sorry

end NUMINAMATH_CALUDE_bryans_books_l896_89647


namespace NUMINAMATH_CALUDE_sandy_clothes_cost_l896_89679

/-- The amount Sandy spent on shorts -/
def shorts_cost : ℚ := 13.99

/-- The amount Sandy spent on a shirt -/
def shirt_cost : ℚ := 12.14

/-- The amount Sandy spent on a jacket -/
def jacket_cost : ℚ := 7.43

/-- The total amount Sandy spent on clothes -/
def total_cost : ℚ := shorts_cost + shirt_cost + jacket_cost

theorem sandy_clothes_cost : total_cost = 33.56 := by sorry

end NUMINAMATH_CALUDE_sandy_clothes_cost_l896_89679


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l896_89680

theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l896_89680


namespace NUMINAMATH_CALUDE_no_root_greater_than_three_l896_89617

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the theorem
theorem no_root_greater_than_three (a b c : ℝ) :
  (quadratic a b c (-1) = -1) →
  (quadratic a b c 0 = 2) →
  (quadratic a b c 2 = 2) →
  (quadratic a b c 4 = -6) →
  ∀ x > 3, quadratic a b c x ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_no_root_greater_than_three_l896_89617


namespace NUMINAMATH_CALUDE_find_C_l896_89658

theorem find_C (A B C : ℕ) : A = 348 → B = A + 173 → C = B + 299 → C = 820 := by
  sorry

end NUMINAMATH_CALUDE_find_C_l896_89658


namespace NUMINAMATH_CALUDE_f_prime_at_two_l896_89645

-- Define f as a real-valued function
variable (f : ℝ → ℝ)

-- State the theorem
theorem f_prime_at_two
  (h1 : (1 - 0) / (2 - 0) = 1 / 2)  -- Slope of line through (0,0) and (2,1) is 1/2
  (h2 : f 0 = 0)                    -- f(0) = 0
  (h3 : f 2 = 2)                    -- f(2) = 2
  (h4 : (2 * (deriv f 2) - (f 2)) / (2^2) = 1 / 2)  -- Derivative of f(x)/x at x=2 equals slope
  : deriv f 2 = 2 := by
sorry

end NUMINAMATH_CALUDE_f_prime_at_two_l896_89645


namespace NUMINAMATH_CALUDE_puppies_sold_l896_89683

theorem puppies_sold (initial_puppies cages puppies_per_cage : ℕ) :
  initial_puppies = 78 →
  puppies_per_cage = 8 →
  cages = 6 →
  initial_puppies - (cages * puppies_per_cage) = 30 :=
by sorry

end NUMINAMATH_CALUDE_puppies_sold_l896_89683


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l896_89607

/-- The area of a square with adjacent points (2,1) and (3,4) on a Cartesian coordinate plane is 10. -/
theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (2, 1)
  let p2 : ℝ × ℝ := (3, 4)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 10 := by sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l896_89607


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l896_89685

theorem inequality_and_equality_condition (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a*b)) ∧ 
  ((1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a*b)) ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l896_89685


namespace NUMINAMATH_CALUDE_major_premise_false_correct_answer_is_major_premise_wrong_l896_89672

-- Define the properties of a rhombus
structure Rhombus where
  diagonals_perpendicular : Bool
  diagonals_bisect : Bool
  diagonals_equal : Bool

-- Define a square as a special case of rhombus
def Square : Rhombus where
  diagonals_perpendicular := true
  diagonals_bisect := true
  diagonals_equal := true

-- Define the syllogism
def syllogism : Prop :=
  ∀ (r : Rhombus), r.diagonals_equal = true

-- Theorem stating that the major premise of the syllogism is false
theorem major_premise_false : ¬syllogism := by
  sorry

-- Theorem stating that the correct answer is that the major premise is wrong
theorem correct_answer_is_major_premise_wrong : 
  (¬syllogism) ∧ (Square.diagonals_equal = true) := by
  sorry

end NUMINAMATH_CALUDE_major_premise_false_correct_answer_is_major_premise_wrong_l896_89672


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l896_89608

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = -1 ∧ 
  (x₁^2 - 2*x₁ - 3 = 0) ∧ (x₂^2 - 2*x₂ - 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l896_89608


namespace NUMINAMATH_CALUDE_line_intersection_theorem_l896_89657

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the properties and relations
variable (skew : Line → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (intersects : Line → Line → Prop)
variable (plane_intersection : Plane → Plane → Line)

-- State the theorem
theorem line_intersection_theorem 
  (m n l : Line) (α β : Plane) 
  (h1 : skew m n)
  (h2 : contains α m)
  (h3 : contains β n)
  (h4 : plane_intersection α β = l) :
  intersects l m ∨ intersects l n :=
sorry

end NUMINAMATH_CALUDE_line_intersection_theorem_l896_89657


namespace NUMINAMATH_CALUDE_x_twelve_equals_one_l896_89686

theorem x_twelve_equals_one (x : ℝ) (h : x + 1/x = 2) : x^12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_twelve_equals_one_l896_89686


namespace NUMINAMATH_CALUDE_fraction_equality_l896_89628

theorem fraction_equality (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l896_89628


namespace NUMINAMATH_CALUDE_central_octagon_area_l896_89603

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square tile -/
structure SquareTile where
  sideLength : ℝ
  center : Point

/-- Theorem: Area of the central octagon in a square tile -/
theorem central_octagon_area (tile : SquareTile) (X Y Z : Point) :
  tile.sideLength = 8 →
  (X.x - Y.x)^2 + (X.y - Y.y)^2 = 2^2 →
  (Y.x - Z.x)^2 + (Y.y - Z.y)^2 = 2^2 →
  (Z.y - Y.y) / (Z.x - Y.x) = 0 →
  let U : Point := { x := (X.x + Z.x) / 2, y := (X.y + Z.y) / 2 }
  let V : Point := { x := (Y.x + Z.x) / 2, y := (Y.y + Z.y) / 2 }
  let octagonArea := (U.x - V.x)^2 + (U.y - V.y)^2 + 4 * ((X.x - U.x)^2 + (X.y - U.y)^2)
  octagonArea = 10 := by
  sorry


end NUMINAMATH_CALUDE_central_octagon_area_l896_89603


namespace NUMINAMATH_CALUDE_sodium_thiosulfate_properties_l896_89699

/-- Represents the sodium thiosulfate anion -/
structure SodiumThiosulfateAnion where
  has_s_s_bond : Bool
  has_s_o_s_bond : Bool
  has_o_o_bond : Bool

/-- Represents the formation method of sodium thiosulfate -/
inductive FormationMethod
  | ThermalDecomposition
  | SulfiteWithSulfur
  | AnodicOxidation

/-- Properties of sodium thiosulfate -/
structure SodiumThiosulfate where
  anion : SodiumThiosulfateAnion
  formation : FormationMethod

/-- Theorem stating the correct structure and formation of sodium thiosulfate -/
theorem sodium_thiosulfate_properties :
  ∃ (st : SodiumThiosulfate),
    st.anion.has_s_s_bond = true ∧
    st.formation = FormationMethod.SulfiteWithSulfur :=
  sorry

end NUMINAMATH_CALUDE_sodium_thiosulfate_properties_l896_89699


namespace NUMINAMATH_CALUDE_arthurs_purchases_l896_89623

/-- The cost of Arthur's purchases on two days -/
theorem arthurs_purchases (hamburger_price : ℚ) :
  (3 * hamburger_price + 4 * 1 = 10) →
  (2 * hamburger_price + 3 * 1 = 7) :=
by sorry

end NUMINAMATH_CALUDE_arthurs_purchases_l896_89623


namespace NUMINAMATH_CALUDE_one_intersection_implies_a_range_l896_89654

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a^2*x + 1

-- State the theorem
theorem one_intersection_implies_a_range (a : ℝ) :
  (∃! x : ℝ, f a x = 3) → -1 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_one_intersection_implies_a_range_l896_89654


namespace NUMINAMATH_CALUDE_distance_on_number_line_l896_89624

theorem distance_on_number_line : 
  let point_a : ℤ := -2006
  let point_b : ℤ := 17
  abs (point_b - point_a) = 2023 := by sorry

end NUMINAMATH_CALUDE_distance_on_number_line_l896_89624


namespace NUMINAMATH_CALUDE_odd_function_property_l896_89620

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_prop : ∀ x, f (1 + x) = f (-x))
  (h_value : f (-1/3) = 1/3) :
  f (5/3) = 1/3 := by sorry

end NUMINAMATH_CALUDE_odd_function_property_l896_89620


namespace NUMINAMATH_CALUDE_h_properties_l896_89696

-- Define the functions
noncomputable def g (x : ℝ) : ℝ := 2^x

-- f is symmetric to g with respect to y = x
def f_symmetric_to_g (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- Define h in terms of f
def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f (1 - |x|)

-- Main theorem
theorem h_properties (f : ℝ → ℝ) (hf : f_symmetric_to_g f) :
  (∀ x, h f x = h f (-x)) ∧ 
  (∀ x, h f x ≤ 0 ∧ h f 0 = 0) :=
sorry

end NUMINAMATH_CALUDE_h_properties_l896_89696


namespace NUMINAMATH_CALUDE_sin_sum_product_l896_89677

theorem sin_sum_product (x : ℝ) : 
  Real.sin (7 * x) + Real.sin (9 * x) = 2 * Real.sin (8 * x) * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_product_l896_89677


namespace NUMINAMATH_CALUDE_mitya_travel_schedule_unique_l896_89649

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents the months of the year -/
inductive Month
  | February
  | March

/-- Represents a date within a month -/
structure Date where
  month : Month
  day : Nat

/-- Represents Mitya's travel schedule -/
structure TravelSchedule where
  smolensk : Date
  vologda : Date
  pskov : Date
  vladimir : Date

/-- Returns the day of the week for a given date -/
def dayOfWeek (date : Date) : DayOfWeek :=
  sorry

/-- Returns true if the given year is a leap year -/
def isLeapYear (year : Nat) : Bool :=
  sorry

/-- Returns the number of days in a given month -/
def daysInMonth (month : Month) (isLeap : Bool) : Nat :=
  sorry

/-- Theorem: Given the conditions of Mitya's travel and the calendar structure,
    there exists a unique travel schedule that satisfies all constraints -/
theorem mitya_travel_schedule_unique :
  ∃! (schedule : TravelSchedule),
    (dayOfWeek schedule.smolensk = DayOfWeek.Tuesday) ∧
    (dayOfWeek schedule.vologda = DayOfWeek.Tuesday) ∧
    (dayOfWeek schedule.pskov = DayOfWeek.Tuesday) ∧
    (dayOfWeek schedule.vladimir = DayOfWeek.Tuesday) ∧
    (schedule.smolensk.month = Month.February) ∧
    (schedule.vologda.month = Month.February) ∧
    (schedule.pskov.month = Month.March) ∧
    (schedule.vladimir.month = Month.March) ∧
    (schedule.smolensk.day = 1) ∧
    (schedule.vologda.day > schedule.smolensk.day) ∧
    (schedule.pskov.day = 1) ∧
    (schedule.vladimir.day > schedule.pskov.day) ∧
    (¬isLeapYear 0) ∧
    (daysInMonth Month.February false = 28) ∧
    (daysInMonth Month.March false = 31) :=
  sorry

end NUMINAMATH_CALUDE_mitya_travel_schedule_unique_l896_89649


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l896_89662

/-- Given a geometric sequence {a_n} where the sum of the first n terms S_n
    is defined as S_n = x · 3^n + 1, this theorem states that x = -1. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (x : ℝ) :
  (∀ n, S n = x * 3^n + 1) →
  (∀ n, a (n+1) / a n = a (n+2) / a (n+1)) →
  x = -1 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l896_89662


namespace NUMINAMATH_CALUDE_instrument_players_fraction_l896_89656

theorem instrument_players_fraction (total : ℕ) (two_or_more : ℕ) (prob_one : ℚ) :
  total = 800 →
  two_or_more = 32 →
  prob_one = 1/10 + 3/50 →
  (((prob_one * total) + two_or_more) : ℚ) / total = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_instrument_players_fraction_l896_89656


namespace NUMINAMATH_CALUDE_book_cost_in_cny_l896_89675

/-- Exchange rate from US dollar to Namibian dollar -/
def usd_to_nad : ℚ := 7

/-- Exchange rate from US dollar to Chinese yuan -/
def usd_to_cny : ℚ := 6

/-- Cost of the book in Namibian dollars -/
def book_cost_nad : ℚ := 168

/-- Calculate the cost of the book in Chinese yuan -/
def book_cost_cny : ℚ := book_cost_nad * (usd_to_cny / usd_to_nad)

/-- Theorem stating that the book costs 144 Chinese yuan -/
theorem book_cost_in_cny : book_cost_cny = 144 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_in_cny_l896_89675


namespace NUMINAMATH_CALUDE_arithmetic_progression_cube_sum_l896_89630

/-- 
Given integers x, y, z, u forming an arithmetic progression and satisfying x^3 + y^3 + z^3 = u^3,
prove that there exists an integer d such that x = 3d, y = 4d, z = 5d, and u = 6d.
-/
theorem arithmetic_progression_cube_sum (x y z u : ℤ) 
  (h_arith_prog : ∃ (d : ℤ), y = x + d ∧ z = y + d ∧ u = z + d)
  (h_cube_sum : x^3 + y^3 + z^3 = u^3) :
  ∃ (d : ℤ), x = 3*d ∧ y = 4*d ∧ z = 5*d ∧ u = 6*d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_cube_sum_l896_89630


namespace NUMINAMATH_CALUDE_gumball_multiple_proof_l896_89633

theorem gumball_multiple_proof :
  ∀ (joanna_initial jacques_initial total_final multiple : ℕ),
    joanna_initial = 40 →
    jacques_initial = 60 →
    total_final = 500 →
    (joanna_initial + joanna_initial * multiple) +
    (jacques_initial + jacques_initial * multiple) = total_final →
    multiple = 4 := by
  sorry

end NUMINAMATH_CALUDE_gumball_multiple_proof_l896_89633


namespace NUMINAMATH_CALUDE_least_positive_linear_combination_l896_89626

theorem least_positive_linear_combination : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x y z : ℤ), 72 * x + 54 * y + 36 * z > 0 → 72 * x + 54 * y + 36 * z ≥ n) ∧
  (∃ (x y z : ℤ), 72 * x + 54 * y + 36 * z = n) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_positive_linear_combination_l896_89626


namespace NUMINAMATH_CALUDE_conference_handshakes_l896_89671

theorem conference_handshakes (n : ℕ) (h : n = 7) : 
  (2 * n) * ((2 * n - 1) - 2) / 2 = 77 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l896_89671


namespace NUMINAMATH_CALUDE_problem_solution_l896_89646

theorem problem_solution (x k : ℕ) (h1 : (2^x) - (2^(x-2)) = k * (2^10)) (h2 : x = 12) : k = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l896_89646


namespace NUMINAMATH_CALUDE_smallest_k_for_inequality_l896_89615

theorem smallest_k_for_inequality : 
  ∃ (k : ℕ), k = 4 ∧ 
  (∀ (a : ℝ) (n : ℕ), 0 ≤ a ∧ a ≤ 1 → a^k * (1-a)^n < 1 / (n+1)^3) ∧
  (∀ (k' : ℕ), k' < k → 
    ∃ (a : ℝ) (n : ℕ), 0 ≤ a ∧ a ≤ 1 ∧ a^k' * (1-a)^n ≥ 1 / (n+1)^3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_inequality_l896_89615


namespace NUMINAMATH_CALUDE_min_product_xyz_l896_89650

theorem min_product_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (sum_eq_one : x + y + z = 1) (z_eq_2x : z = 2 * x) (y_eq_3x : y = 3 * x) :
  x * y * z ≥ 1 / 36 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
    x₀ + y₀ + z₀ = 1 ∧ z₀ = 2 * x₀ ∧ y₀ = 3 * x₀ ∧ x₀ * y₀ * z₀ = 1 / 36 :=
by sorry

end NUMINAMATH_CALUDE_min_product_xyz_l896_89650


namespace NUMINAMATH_CALUDE_french_students_count_l896_89641

theorem french_students_count (total : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 79)
  (h2 : german = 22)
  (h3 : both = 9)
  (h4 : neither = 25)
  : ∃ french : ℕ, french = 41 ∧ total = french + german - both + neither :=
by sorry

end NUMINAMATH_CALUDE_french_students_count_l896_89641


namespace NUMINAMATH_CALUDE_largest_minus_smallest_is_52_l896_89664

def digits : Finset Nat := {8, 3, 4, 6}

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

def valid_number (n : Nat) : Prop :=
  is_two_digit n ∧ ∃ (a b : Nat), a ∈ digits ∧ b ∈ digits ∧ a ≠ b ∧ n = 10 * a + b

theorem largest_minus_smallest_is_52 :
  ∃ (max min : Nat),
    valid_number max ∧
    valid_number min ∧
    (∀ n, valid_number n → n ≤ max) ∧
    (∀ n, valid_number n → min ≤ n) ∧
    max - min = 52 := by
  sorry

end NUMINAMATH_CALUDE_largest_minus_smallest_is_52_l896_89664


namespace NUMINAMATH_CALUDE_expected_value_r₃_l896_89611

/-- The expected value of a single fair six-sided die roll -/
def single_die_ev : ℝ := 3.5

/-- The number of dice rolled in the first round -/
def first_round_dice : ℕ := 8

/-- The expected value of r₁ (the sum of first_round_dice fair dice rolls) -/
def r₁_ev : ℝ := first_round_dice * single_die_ev

/-- The expected value of r₂ (the sum of r₁_ev fair dice rolls) -/
def r₂_ev : ℝ := r₁_ev * single_die_ev

/-- The expected value of r₃ (the sum of r₂_ev fair dice rolls) -/
def r₃_ev : ℝ := r₂_ev * single_die_ev

theorem expected_value_r₃ : r₃_ev = 343 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_r₃_l896_89611


namespace NUMINAMATH_CALUDE_num_arrangements_equals_5040_l896_89661

/-- The number of candidates --/
def n : ℕ := 8

/-- The number of volunteers to be selected --/
def k : ℕ := 5

/-- The number of days --/
def days : ℕ := 5

/-- Function to calculate the number of arrangements --/
def num_arrangements (n k : ℕ) : ℕ :=
  let only_one := 2 * (n - 2).choose (k - 1) * k.factorial
  let both := (n - 2).choose (k - 2) * (k - 2).factorial * 2 * (k - 1)
  only_one + both

/-- Theorem stating the number of arrangements --/
theorem num_arrangements_equals_5040 :
  num_arrangements n k = 5040 := by sorry

end NUMINAMATH_CALUDE_num_arrangements_equals_5040_l896_89661


namespace NUMINAMATH_CALUDE_rectangle_area_from_diagonal_l896_89604

/-- Theorem: Area of a rectangle with length thrice its width and diagonal x -/
theorem rectangle_area_from_diagonal (x : ℝ) (h : x > 0) : 
  ∃ w l : ℝ, w > 0 ∧ l = 3 * w ∧ w^2 + l^2 = x^2 ∧ w * l = (3/10) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_from_diagonal_l896_89604


namespace NUMINAMATH_CALUDE_smallest_cube_ending_888_l896_89643

theorem smallest_cube_ending_888 : 
  ∃ n : ℕ, (∀ m : ℕ, m < n → m^3 % 1000 ≠ 888) ∧ n^3 % 1000 = 888 :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_888_l896_89643


namespace NUMINAMATH_CALUDE_fraction_equality_and_sum_l896_89681

theorem fraction_equality_and_sum : ∃! (α β : ℝ),
  (∀ x : ℝ, x ≠ -β → x ≠ -110.36 →
    (x - α) / (x + β) = (x^2 - 64*x + 1007) / (x^2 + 81*x - 3240)) ∧
  α + β = 146.483 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_and_sum_l896_89681


namespace NUMINAMATH_CALUDE_point_A_in_first_quadrant_l896_89605

-- Define the Cartesian coordinate system
def CartesianCoordinate := ℝ × ℝ

-- Define the point A
def A : CartesianCoordinate := (1, 2)

-- Define the first quadrant
def FirstQuadrant (p : CartesianCoordinate) : Prop :=
  p.1 > 0 ∧ p.2 > 0

-- Theorem statement
theorem point_A_in_first_quadrant : FirstQuadrant A := by
  sorry

end NUMINAMATH_CALUDE_point_A_in_first_quadrant_l896_89605


namespace NUMINAMATH_CALUDE_range_reduction_after_five_trials_l896_89636

/-- The reduction factor for each trial using the 0.618 method -/
def reduction_factor : ℝ := 0.618

/-- The number of trials -/
def num_trials : ℕ := 5

/-- The range reduction after a given number of trials -/
def range_reduction (n : ℕ) : ℝ := reduction_factor ^ n

theorem range_reduction_after_five_trials :
  range_reduction (num_trials - 1) = reduction_factor ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_reduction_after_five_trials_l896_89636


namespace NUMINAMATH_CALUDE_francie_remaining_money_l896_89640

def initial_allowance : ℕ := 5
def initial_weeks : ℕ := 8
def raised_allowance : ℕ := 6
def raised_weeks : ℕ := 6
def cash_gift : ℕ := 20
def investment_amount : ℕ := 10
def investment_return_rate : ℚ := 5 / 100
def video_game_cost : ℕ := 35

def total_savings : ℚ :=
  (initial_allowance * initial_weeks +
   raised_allowance * raised_weeks +
   cash_gift : ℚ)

def total_with_investment : ℚ :=
  total_savings + investment_amount * investment_return_rate

def remaining_after_clothes : ℚ :=
  total_with_investment / 2

theorem francie_remaining_money :
  remaining_after_clothes - video_game_cost = 13.25 := by
  sorry

end NUMINAMATH_CALUDE_francie_remaining_money_l896_89640


namespace NUMINAMATH_CALUDE_ellipse_equation_l896_89612

/-- An ellipse with given properties has the equation x²/3 + y²/2 = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
  let e := Real.sqrt 3 / 3
  let c := a * e
  let perimeter := 4 * Real.sqrt 3
  (c^2 = a^2 - b^2) →
  (perimeter = 2 * a + 2 * a) →
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ x^2/3 + y^2/2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l896_89612


namespace NUMINAMATH_CALUDE_objects_meeting_probability_l896_89659

/-- The probability of two objects meeting on a coordinate plane -/
theorem objects_meeting_probability :
  let start_C : ℕ × ℕ := (0, 0)
  let start_D : ℕ × ℕ := (4, 6)
  let step_length : ℕ := 1
  let prob_C_right : ℚ := 1/2
  let prob_C_up : ℚ := 1/2
  let prob_D_left : ℚ := 1/2
  let prob_D_down : ℚ := 1/2
  ∃ (meeting_prob : ℚ), meeting_prob = 55/1024 :=
by sorry

end NUMINAMATH_CALUDE_objects_meeting_probability_l896_89659


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l896_89660

/-- Represents a sampling method -/
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling
  | SystematicSampling

/-- Represents a region -/
inductive Region
  | A
  | B
  | C
  | D

/-- Represents the company's sales point distribution -/
structure SalesPointDistribution where
  total_points : Nat
  region_points : Region → Nat
  large_points_in_C : Nat

/-- Represents an investigation -/
structure Investigation where
  sample_size : Nat
  population_size : Nat

/-- Determines the appropriate sampling method for an investigation -/
def appropriate_sampling_method (dist : SalesPointDistribution) (inv : Investigation) : SamplingMethod :=
  sorry

/-- The company's actual sales point distribution -/
def company_distribution : SalesPointDistribution :=
  { total_points := 600,
    region_points := fun r => match r with
      | Region.A => 150
      | Region.B => 120
      | Region.C => 180
      | Region.D => 150,
    large_points_in_C := 20 }

/-- Investigation ① -/
def investigation_1 : Investigation :=
  { sample_size := 100,
    population_size := 600 }

/-- Investigation ② -/
def investigation_2 : Investigation :=
  { sample_size := 7,
    population_size := 20 }

/-- Theorem stating the appropriate sampling methods for the given investigations -/
theorem appropriate_sampling_methods :
  appropriate_sampling_method company_distribution investigation_1 = SamplingMethod.StratifiedSampling ∧
  appropriate_sampling_method company_distribution investigation_2 = SamplingMethod.SimpleRandomSampling :=
sorry

end NUMINAMATH_CALUDE_appropriate_sampling_methods_l896_89660


namespace NUMINAMATH_CALUDE_sum_of_four_digit_numbers_l896_89609

/-- The set of digits used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- A four-digit number formed from the given digits -/
structure FourDigitNumber where
  d₁ : Nat
  d₂ : Nat
  d₃ : Nat
  d₄ : Nat
  h₁ : d₁ ∈ digits
  h₂ : d₂ ∈ digits
  h₃ : d₃ ∈ digits
  h₄ : d₄ ∈ digits
  distinct : d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄

/-- The value of a four-digit number -/
def value (n : FourDigitNumber) : Nat :=
  1000 * n.d₁ + 100 * n.d₂ + 10 * n.d₃ + n.d₄

/-- The set of all valid four-digit numbers -/
def allFourDigitNumbers : Finset FourDigitNumber :=
  sorry

/-- The theorem to be proved -/
theorem sum_of_four_digit_numbers :
  (allFourDigitNumbers.sum value) = 399960 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_digit_numbers_l896_89609


namespace NUMINAMATH_CALUDE_presidency_meeting_ways_l896_89648

def num_schools : Nat := 4
def members_per_school : Nat := 6
def host_representatives : Nat := 3
def non_host_representatives : Nat := 2
def seniors_per_school : Nat := 3

theorem presidency_meeting_ways :
  let choose_host := num_schools
  let host_rep_ways := Nat.choose members_per_school host_representatives
  let non_host_school_ways := Nat.choose seniors_per_school 1 * Nat.choose (members_per_school - seniors_per_school) 1
  let non_host_schools_ways := non_host_school_ways ^ (num_schools - 1)
  choose_host * host_rep_ways * non_host_schools_ways = 58320 := by
  sorry

end NUMINAMATH_CALUDE_presidency_meeting_ways_l896_89648


namespace NUMINAMATH_CALUDE_smallest_with_eight_factors_l896_89695

/-- The number of distinct positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Prove that 16 is the smallest positive integer with exactly eight distinct positive factors -/
theorem smallest_with_eight_factors : 
  (∀ m : ℕ+, m < 16 → num_factors m ≠ 8) ∧ num_factors 16 = 8 := by sorry

end NUMINAMATH_CALUDE_smallest_with_eight_factors_l896_89695


namespace NUMINAMATH_CALUDE_proportion_change_l896_89601

def is_proportion (a b c d : ℚ) : Prop := a * d = b * c

theorem proportion_change (x y : ℚ) :
  is_proportion 3 5 6 10 →
  is_proportion 12 y 6 10 →
  y = 20 := by sorry

end NUMINAMATH_CALUDE_proportion_change_l896_89601


namespace NUMINAMATH_CALUDE_triangle_area_calculation_l896_89651

theorem triangle_area_calculation (base : ℝ) (height_factor : ℝ) :
  base = 3.6 →
  height_factor = 2.5 →
  (base * (height_factor * base)) / 2 = 16.2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_calculation_l896_89651


namespace NUMINAMATH_CALUDE_last_segment_speed_prove_last_segment_speed_l896_89665

theorem last_segment_speed (total_distance : ℝ) (total_time : ℝ) 
  (speed1 : ℝ) (speed2 : ℝ) (speed3 : ℝ) : ℝ :=
  let total_segments : ℝ := 4
  let segment_time : ℝ := total_time / total_segments
  let overall_avg_speed : ℝ := total_distance / total_time
  let last_segment_speed : ℝ := 
    total_segments * overall_avg_speed - (speed1 + speed2 + speed3)
  last_segment_speed

theorem prove_last_segment_speed : 
  last_segment_speed 160 2 55 75 60 = 130 := by
  sorry

end NUMINAMATH_CALUDE_last_segment_speed_prove_last_segment_speed_l896_89665


namespace NUMINAMATH_CALUDE_sqrt_square_negative_l896_89684

theorem sqrt_square_negative : Real.sqrt ((-2023)^2) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_negative_l896_89684


namespace NUMINAMATH_CALUDE_schedule_theorem_l896_89652

-- Define the number of periods in a day
def periods : ℕ := 7

-- Define the number of courses to be scheduled
def courses : ℕ := 4

-- Define a function to calculate the number of ways to schedule courses
def schedule_ways (p : ℕ) (c : ℕ) : ℕ := sorry

-- Theorem statement
theorem schedule_theorem : 
  schedule_ways periods courses = 120 := by sorry

end NUMINAMATH_CALUDE_schedule_theorem_l896_89652


namespace NUMINAMATH_CALUDE_quadratic_minimum_l896_89632

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 10

/-- The point where the minimum occurs -/
def min_point : ℝ := -4

theorem quadratic_minimum :
  ∀ x : ℝ, f x ≥ f min_point :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l896_89632


namespace NUMINAMATH_CALUDE_square_area_ratio_l896_89610

theorem square_area_ratio (side_c side_d : ℝ) 
  (h1 : side_c = 45)
  (h2 : side_d = 60) : 
  (side_c^2) / (side_d^2) = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l896_89610


namespace NUMINAMATH_CALUDE_angle_of_inclination_slope_one_l896_89692

/-- The angle of inclination of a line with slope 1 is π/4 --/
theorem angle_of_inclination_slope_one :
  let line : ℝ → ℝ := λ x ↦ x + 1
  let slope : ℝ := 1
  let angle_of_inclination : ℝ := Real.arctan slope
  angle_of_inclination = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_of_inclination_slope_one_l896_89692


namespace NUMINAMATH_CALUDE_root_product_cubic_l896_89637

theorem root_product_cubic (a b c : ℂ) : 
  (3 * a^3 - 4 * a^2 + a - 2 = 0) →
  (3 * b^3 - 4 * b^2 + b - 2 = 0) →
  (3 * c^3 - 4 * c^2 + c - 2 = 0) →
  a * b * c = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_root_product_cubic_l896_89637


namespace NUMINAMATH_CALUDE_sin_function_smallest_c_l896_89629

/-- 
Given a sinusoidal function f(x) = a * sin(b * x + c) where a, b, and c are positive constants,
if f(x) reaches its maximum at x = 0, then the smallest possible value of c is π/2.
-/
theorem sin_function_smallest_c (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x, a * Real.sin (b * x + c) ≤ a * Real.sin c) → c = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_function_smallest_c_l896_89629


namespace NUMINAMATH_CALUDE_curve_inequality_l896_89621

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the curve equation
def curve_equation (a b c x y : ℝ) : Prop :=
  a * (lg x)^2 + 2 * b * (lg x) * (lg y) + c * (lg y)^2 = 1

-- Main theorem
theorem curve_inequality (a b c : ℝ) 
  (h1 : b^2 - a*c < 0) 
  (h2 : curve_equation a b c 10 (1/10)) :
  ∀ x y : ℝ, curve_equation a b c x y →
  -1 / Real.sqrt (a*c - b^2) ≤ lg (x*y) ∧ lg (x*y) ≤ 1 / Real.sqrt (a*c - b^2) := by
  sorry

end NUMINAMATH_CALUDE_curve_inequality_l896_89621


namespace NUMINAMATH_CALUDE_painted_square_ratio_exists_l896_89669

/-- Represents a square with a painted pattern -/
structure PaintedSquare where
  s : ℝ  -- side length of the square
  w : ℝ  -- width of the brush
  h_positive_s : 0 < s
  h_positive_w : 0 < w
  h_painted_area : w^2 + 2 * Real.sqrt 2 * ((s - w * Real.sqrt 2) / 2)^2 = s^2 / 3

/-- There exists a ratio between the side length and brush width for a painted square -/
theorem painted_square_ratio_exists (ps : PaintedSquare) : 
  ∃ r : ℝ, ps.s = r * ps.w :=
sorry

end NUMINAMATH_CALUDE_painted_square_ratio_exists_l896_89669


namespace NUMINAMATH_CALUDE_snake_head_fraction_l896_89688

theorem snake_head_fraction (total_length body_length : ℝ) 
  (h1 : total_length = 10)
  (h2 : body_length = 9)
  (h3 : body_length < total_length) :
  (total_length - body_length) / total_length = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_snake_head_fraction_l896_89688


namespace NUMINAMATH_CALUDE_train_length_l896_89670

/-- Given a train that crosses a platform in a certain time and a signal pole in another time,
    this theorem proves the length of the train. -/
theorem train_length
  (platform_crossing_time : ℝ)
  (pole_crossing_time : ℝ)
  (platform_length : ℝ)
  (h1 : platform_crossing_time = 39)
  (h2 : pole_crossing_time = 24)
  (h3 : platform_length = 187.5) :
  (platform_crossing_time * platform_length) / (platform_crossing_time - pole_crossing_time) = 300 :=
by sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l896_89670


namespace NUMINAMATH_CALUDE_circle_and_line_problem_l896_89631

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 3)^2 + (y + 2)^2 = 25

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the line m
def line_m (x y : ℝ) : Prop := x = 1 ∨ (5/12)*x - y + 43/12 = 0

-- Theorem statement
theorem circle_and_line_problem :
  -- Given conditions
  (circle_C 0 2) ∧ 
  (circle_C 2 (-2)) ∧ 
  (∃ (x y : ℝ), circle_C x y ∧ line_l x y) ∧
  (line_m 1 4) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    circle_C x₁ y₁ ∧ 
    circle_C x₂ y₂ ∧ 
    line_m x₁ y₁ ∧ 
    line_m x₂ y₂ ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 36) →
  -- Conclusion
  (∀ (x y : ℝ), circle_C x y ↔ (x + 3)^2 + (y + 2)^2 = 25) ∧
  (∀ (x y : ℝ), line_m x y ↔ (x = 1 ∨ (5/12)*x - y + 43/12 = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_and_line_problem_l896_89631


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l896_89666

/-- Proves that for a parabola y^2 = 2px with p > 0, if a point P(2,m) on the parabola
    is at a distance of 4 from its focus, then p = 4. -/
theorem parabola_focus_distance (p : ℝ) (m : ℝ) (h1 : p > 0) :
  m^2 = 2*p*2 →  -- Point P(2,m) is on the parabola y^2 = 2px
  (2 - p/2)^2 + m^2 = 4^2 →  -- Distance from P to focus F(p/2, 0) is 4
  p = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l896_89666


namespace NUMINAMATH_CALUDE_triangle_inequality_l896_89693

theorem triangle_inequality (a b c α β γ : ℝ) (n : ℕ) : 
  a > 0 → b > 0 → c > 0 → 
  α > 0 → β > 0 → γ > 0 → 
  α + β + γ = π → 
  (π/3)^n ≤ (a*α^n + b*β^n + c*γ^n) / (a + b + c) ∧ 
  (a*α^n + b*β^n + c*γ^n) / (a + b + c) < π^n/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l896_89693


namespace NUMINAMATH_CALUDE_sum_at_thirteenth_position_l896_89694

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℕ
  is_permutation : Function.Bijective vertices

/-- The sum of numbers in a specific position across all orientations of a regular polygon -/
def sum_at_position (p : RegularPolygon 100) (pos : ℕ) : ℕ :=
  sorry

/-- Theorem: The sum of numbers in the 13th position from the left across all orientations of a regular 100-gon is 10100 -/
theorem sum_at_thirteenth_position (p : RegularPolygon 100) :
  sum_at_position p 13 = 10100 := by
  sorry

end NUMINAMATH_CALUDE_sum_at_thirteenth_position_l896_89694


namespace NUMINAMATH_CALUDE_closest_perfect_square_to_350_l896_89668

def closest_perfect_square (n : ℕ) : ℕ :=
  let root := n.sqrt
  if (root + 1)^2 - n < n - root^2
  then (root + 1)^2
  else root^2

theorem closest_perfect_square_to_350 :
  closest_perfect_square 350 = 361 := by
  sorry

end NUMINAMATH_CALUDE_closest_perfect_square_to_350_l896_89668


namespace NUMINAMATH_CALUDE_quadratic_inequalities_solution_sets_l896_89618

theorem quadratic_inequalities_solution_sets :
  (∀ x : ℝ, -3 * x^2 + x + 1 > 0 ↔ x ∈ Set.Ioo ((1 - Real.sqrt 13) / 6) ((1 + Real.sqrt 13) / 6)) ∧
  (∀ x : ℝ, x^2 - 2*x + 1 ≤ 0 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_solution_sets_l896_89618


namespace NUMINAMATH_CALUDE_circle_line_intersection_l896_89634

/-- A circle C with center (a, 0) and radius r -/
structure Circle where
  a : ℝ
  r : ℝ
  r_pos : r > 0

/-- A line with slope k passing through (-1, 0) -/
structure Line where
  k : ℝ

/-- Theorem: Given a circle C and a line l satisfying certain conditions, 
    the dot product of OA and OB is -(26 + 9√2) / 5 -/
theorem circle_line_intersection 
  (C : Circle) 
  (l : Line) 
  (h1 : C.r = |C.a - 2 * Real.sqrt 2| / Real.sqrt 2)  -- C is tangent to x + y - 2√2 = 0
  (h2 : 4 * Real.sqrt 2 = 2 * Real.sqrt (C.r^2 - (|C.a| / Real.sqrt 2)^2))  -- chord length on y = x is 4√2
  (h3 : ∃ (m : ℝ), m / l.k^2 = -3 - Real.sqrt 2)  -- condition on slopes product
  : ∃ (A B : ℝ × ℝ), 
    (A.1 - C.a)^2 + A.2^2 = C.r^2 ∧   -- A is on circle C
    (B.1 - C.a)^2 + B.2^2 = C.r^2 ∧   -- B is on circle C
    A.2 = l.k * (A.1 + 1) ∧           -- A is on line l
    B.2 = l.k * (B.1 + 1) ∧           -- B is on line l
    (A.1 - C.a) * (B.1 - C.a) + A.2 * B.2 = -(26 + 9 * Real.sqrt 2) / 5  -- OA · OB
    := by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l896_89634


namespace NUMINAMATH_CALUDE_factorization_difference_of_squares_l896_89698

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_difference_of_squares_l896_89698


namespace NUMINAMATH_CALUDE_right_triangle_max_sum_l896_89602

theorem right_triangle_max_sum (a b c : ℝ) : 
  c = 5 →
  a ≤ 3 →
  b ≥ 3 →
  a^2 + b^2 = c^2 →
  a + b ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_max_sum_l896_89602


namespace NUMINAMATH_CALUDE_part_1_part_2_l896_89638

-- Define the circle C
def circle_C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

-- Define Line 1
def line_1 (x y : ℝ) : Prop := 3*x + 4*y - 6 = 0

-- Define Line 2
def line_2 (x y : ℝ) : Prop := x - y - 1 = 0

-- Part I
theorem part_1 (x y m : ℝ) (M N : ℝ × ℝ) :
  circle_C x y m →
  line_1 (M.1) (M.2) →
  line_1 (N.1) (N.2) →
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 12 →
  m = 1 :=
sorry

-- Part II
theorem part_2 :
  ∃ m : ℝ, m = -2 ∧
  ∃ A B : ℝ × ℝ,
    circle_C A.1 A.2 m ∧
    circle_C B.1 B.2 m ∧
    line_2 A.1 A.2 ∧
    line_2 B.1 B.2 ∧
    (A.1 * B.1 + A.2 * B.2 = 0) :=
sorry

end NUMINAMATH_CALUDE_part_1_part_2_l896_89638


namespace NUMINAMATH_CALUDE_axiom_1_l896_89606

-- Define the types for points, lines, and planes
variable {Point Line Plane : Type}

-- Define the relations for points being on lines and planes
variable (on_line : Point → Line → Prop)
variable (on_plane : Point → Plane → Prop)

-- Define the subset relation for lines and planes
variable (line_on_plane : Line → Plane → Prop)

-- State the theorem
theorem axiom_1 (l : Line) (α : Plane) :
  (∃ (A B : Point), on_line A l ∧ on_line B l ∧ on_plane A α ∧ on_plane B α) →
  line_on_plane l α :=
sorry

end NUMINAMATH_CALUDE_axiom_1_l896_89606


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_sum_l896_89614

theorem min_x_prime_factorization_sum (x y a b e : ℕ+) (c d f : ℕ) :
  (∀ x' y' : ℕ+, 7 * x'^5 = 13 * y'^11 → x ≤ x') →
  7 * x^5 = 13 * y^11 →
  x = a^c * b^d * e^f →
  a.val ≠ b.val ∧ b.val ≠ e.val ∧ a.val ≠ e.val →
  Nat.Prime a.val ∧ Nat.Prime b.val ∧ Nat.Prime e.val →
  a.val + b.val + c + d + e.val + f = 37 :=
by sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_sum_l896_89614


namespace NUMINAMATH_CALUDE_pyramid_height_theorem_l896_89689

/-- Properties of the Great Pyramid of Giza --/
structure Pyramid where
  h : ℝ  -- The certain height
  height : ℝ := h + 20  -- The actual height of the pyramid
  width : ℝ := height + 234  -- The width of the pyramid

/-- Theorem about the height of the Great Pyramid of Giza --/
theorem pyramid_height_theorem (p : Pyramid) 
    (sum_condition : p.height + p.width = 1274) : 
    p.h = 1000 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_height_theorem_l896_89689


namespace NUMINAMATH_CALUDE_bakers_friend_cakes_prove_bakers_friend_cakes_l896_89678

/-- Given that Baker initially made 169 cakes and has 32 cakes left,
    prove that the number of cakes bought by Baker's friend is 137. -/
theorem bakers_friend_cakes : ℕ → ℕ → ℕ → Prop :=
  fun initial_cakes remaining_cakes cakes_bought =>
    initial_cakes = 169 →
    remaining_cakes = 32 →
    cakes_bought = initial_cakes - remaining_cakes →
    cakes_bought = 137

/-- Proof of the theorem -/
theorem prove_bakers_friend_cakes :
  bakers_friend_cakes 169 32 137 := by
  sorry

end NUMINAMATH_CALUDE_bakers_friend_cakes_prove_bakers_friend_cakes_l896_89678


namespace NUMINAMATH_CALUDE_remainder_sum_l896_89616

theorem remainder_sum (x y : ℤ) (hx : x % 80 = 75) (hy : y % 120 = 115) :
  (x + y) % 40 = 30 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l896_89616


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l896_89674

def A : Set ℝ := {x | x < -1 ∨ (2 ≤ x ∧ x < 3)}
def B : Set ℝ := {x | -2 ≤ x ∧ x < 4}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | x < 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l896_89674


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l896_89690

/-- The asymptotes of the hyperbola x²/16 - y²/9 = -1 are y = ±(3/4)x -/
theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2/16 - y^2/9 = -1
  ∀ x y : ℝ, (∃ (ε : ℝ), ε > 0 ∧ ∀ δ : ℝ, δ > ε → h (δ * x) (δ * y)) →
    y = (3/4) * x ∨ y = -(3/4) * x :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l896_89690


namespace NUMINAMATH_CALUDE_regression_properties_l896_89691

/-- Regression line equation -/
def regression_line (x : ℝ) : ℝ := 6 * x + 8

/-- Data points -/
def data_points : List (ℝ × ℝ) := [(2, 19), (3, 25), (4, 0), (5, 38), (6, 44)]

/-- The value of the unclear data point -/
def unclear_data : ℝ := 34

/-- Theorem stating the properties of the regression line and data points -/
theorem regression_properties :
  let third_point := (4, unclear_data)
  let residual := (third_point.2 - regression_line third_point.1)
  (unclear_data = 34) ∧
  (residual = 2) ∧
  (regression_line 7 = 50) := by sorry

end NUMINAMATH_CALUDE_regression_properties_l896_89691


namespace NUMINAMATH_CALUDE_mary_crayons_left_l896_89682

/-- The number of crayons Mary has left after giving some away and breaking some -/
def crayons_left (initial_green initial_blue initial_yellow : ℚ)
  (given_green given_blue given_yellow broken_yellow : ℚ) : ℚ :=
  (initial_green - given_green) + (initial_blue - given_blue) + (initial_yellow - given_yellow - broken_yellow)

/-- Theorem stating that Mary has 12 crayons left -/
theorem mary_crayons_left :
  crayons_left 5 8 7 3.5 1.25 2.75 0.5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mary_crayons_left_l896_89682


namespace NUMINAMATH_CALUDE_parabola_equation_is_correct_coefficient_x2_positive_gcd_of_coefficients_is_one_l896_89673

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola in general form -/
structure Parabola where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- The focus of the parabola -/
def focus : Point := { x := 2, y := -1 }

/-- The directrix of the parabola -/
def directrix : Line := { a := 1, b := 2, c := -4 }

/-- The equation of the parabola -/
def parabola_equation : Parabola := { a := 4, b := -4, c := 1, d := -12, e := -6, f := 9 }

/-- Theorem stating that the given equation represents the parabola with the given focus and directrix -/
theorem parabola_equation_is_correct (p : Point) : 
  (parabola_equation.a * p.x^2 + parabola_equation.b * p.x * p.y + parabola_equation.c * p.y^2 + 
   parabola_equation.d * p.x + parabola_equation.e * p.y + parabola_equation.f = 0) ↔ 
  ((p.x - focus.x)^2 + (p.y - focus.y)^2 = 
   ((directrix.a * p.x + directrix.b * p.y + directrix.c)^2) / (directrix.a^2 + directrix.b^2)) :=
sorry

/-- Theorem stating that the coefficient of x^2 is positive -/
theorem coefficient_x2_positive : parabola_equation.a > 0 :=
sorry

/-- Theorem stating that the GCD of absolute values of coefficients is 1 -/
theorem gcd_of_coefficients_is_one : 
  Nat.gcd (Int.natAbs parabola_equation.a) 
    (Nat.gcd (Int.natAbs parabola_equation.b) 
      (Nat.gcd (Int.natAbs parabola_equation.c) 
        (Nat.gcd (Int.natAbs parabola_equation.d) 
          (Nat.gcd (Int.natAbs parabola_equation.e) 
            (Int.natAbs parabola_equation.f))))) = 1 :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_is_correct_coefficient_x2_positive_gcd_of_coefficients_is_one_l896_89673


namespace NUMINAMATH_CALUDE_higher_selling_price_is_360_l896_89644

/-- The higher selling price of an article, given its cost and profit conditions -/
def higherSellingPrice (cost : ℚ) (lowerPrice : ℚ) : ℚ :=
  let profitAtLowerPrice := lowerPrice - cost
  let additionalProfit := (5 / 100) * cost
  cost + profitAtLowerPrice + additionalProfit

/-- Theorem stating that the higher selling price is 360, given the conditions -/
theorem higher_selling_price_is_360 :
  higherSellingPrice 400 340 = 360 := by
  sorry

#eval higherSellingPrice 400 340

end NUMINAMATH_CALUDE_higher_selling_price_is_360_l896_89644
