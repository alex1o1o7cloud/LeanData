import Mathlib

namespace NUMINAMATH_CALUDE_watch_cost_price_l35_3538

/-- The cost price of a watch satisfying certain conditions -/
theorem watch_cost_price : ∃ (cp : ℝ), 
  (cp > 0) ∧ 
  (0.80 * cp + 520 = 1.06 * cp) ∧ 
  (cp = 2000) := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l35_3538


namespace NUMINAMATH_CALUDE_domain_of_g_l35_3549

-- Define the function f with domain [0,4]
def f : Set ℝ := Set.Icc 0 4

-- Define the function g
def g (f : Set ℝ) : Set ℝ := {x | x ∈ f ∧ x^2 ∈ f}

-- Theorem statement
theorem domain_of_g (f : Set ℝ) (hf : f = Set.Icc 0 4) : 
  g f = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_domain_of_g_l35_3549


namespace NUMINAMATH_CALUDE_adjacent_probability_l35_3584

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define a type for arrangements
def Arrangement := List Person

-- Define a function to check if A and B are adjacent in an arrangement
def areAdjacent (arr : Arrangement) : Prop :=
  ∃ i, (arr.get? i = some Person.A ∧ arr.get? (i+1) = some Person.B) ∨
       (arr.get? i = some Person.B ∧ arr.get? (i+1) = some Person.A)

-- Define the set of all possible arrangements
def allArrangements : Finset Arrangement :=
  sorry

-- Define the set of arrangements where A and B are adjacent
def adjacentArrangements : Finset Arrangement :=
  sorry

-- State the theorem
theorem adjacent_probability :
  (adjacentArrangements.card : ℚ) / (allArrangements.card : ℚ) = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_adjacent_probability_l35_3584


namespace NUMINAMATH_CALUDE_min_max_sum_l35_3530

theorem min_max_sum (a b c d e : ℕ+) (h_sum : a + b + c + d + e = 2018) :
  673 ≤ max (a + b) (max (b + c) (max (c + d) (d + e))) ∧
  ∃ (a' b' c' d' e' : ℕ+), a' + b' + c' + d' + e' = 2018 ∧
    max (a' + b') (max (b' + c') (max (c' + d') (d' + e'))) = 673 :=
by sorry

end NUMINAMATH_CALUDE_min_max_sum_l35_3530


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l35_3533

theorem regular_polygon_sides (central_angle : ℝ) : 
  central_angle = 20 → (360 : ℝ) / central_angle = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l35_3533


namespace NUMINAMATH_CALUDE_christine_money_l35_3552

theorem christine_money (total : ℕ) (difference : ℕ) : 
  total = 50 → difference = 30 → ∃ (christine siri : ℕ), 
    christine = siri + difference ∧ 
    christine + siri = total ∧ 
    christine = 40 := by sorry

end NUMINAMATH_CALUDE_christine_money_l35_3552


namespace NUMINAMATH_CALUDE_F_is_even_l35_3524

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property that f(-x) + f(x) = 0 for all x
axiom f_property : ∀ x, f (-x) + f x = 0

-- Define F(x) = |f(x)|
def F (x : ℝ) : ℝ := |f x|

-- Theorem statement
theorem F_is_even : ∀ x, F x = F (-x) := by sorry

end NUMINAMATH_CALUDE_F_is_even_l35_3524


namespace NUMINAMATH_CALUDE_sum_of_gcd_and_lcm_l35_3537

def A : ℕ := Nat.gcd 18 (Nat.gcd 24 36)
def B : ℕ := Nat.lcm 18 (Nat.lcm 24 36)

theorem sum_of_gcd_and_lcm : A + B = 78 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_gcd_and_lcm_l35_3537


namespace NUMINAMATH_CALUDE_equation_solution_l35_3555

theorem equation_solution : ∃ x : ℕ, 5 + x = 10 + 20 ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l35_3555


namespace NUMINAMATH_CALUDE_inequality_proof_l35_3572

theorem inequality_proof (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 5/9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l35_3572


namespace NUMINAMATH_CALUDE_function_composition_properties_l35_3510

theorem function_composition_properties :
  (¬ ∃ (f g : ℝ → ℝ), ∀ x, f (g x) = x^2 ∧ g (f x) = x^3) ∧
  (∃ (f g : ℝ → ℝ), ∀ x, f (g x) = x^2 ∧ g (f x) = x^4) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_properties_l35_3510


namespace NUMINAMATH_CALUDE_log_sum_equals_zero_l35_3516

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The main theorem -/
theorem log_sum_equals_zero
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_prod : a 3 * a 5 * a 7 = 1) :
  Real.log (a 1) + Real.log (a 9) = 0 :=
sorry

end NUMINAMATH_CALUDE_log_sum_equals_zero_l35_3516


namespace NUMINAMATH_CALUDE_cubic_roots_collinear_k_l35_3532

/-- A cubic polynomial with coefficient k -/
def cubic_polynomial (k : ℤ) (x : ℂ) : ℂ :=
  x^3 - 15*x^2 + k*x - 1105

/-- Predicate for three complex numbers being distinct and collinear -/
def distinct_collinear (z₁ z₂ z₃ : ℂ) : Prop :=
  z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₂ ≠ z₃ ∧
  ∃ (a b : ℝ), (z₁.im - a * z₁.re = b) ∧ 
               (z₂.im - a * z₂.re = b) ∧ 
               (z₃.im - a * z₃.re = b)

theorem cubic_roots_collinear_k (k : ℤ) :
  (∃ (z₁ z₂ z₃ : ℂ), 
    distinct_collinear z₁ z₂ z₃ ∧
    (cubic_polynomial k z₁ = 0) ∧
    (cubic_polynomial k z₂ = 0) ∧
    (cubic_polynomial k z₃ = 0)) →
  k = 271 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_collinear_k_l35_3532


namespace NUMINAMATH_CALUDE_truck_loading_time_l35_3590

theorem truck_loading_time 
  (worker1_rate : ℝ) 
  (worker2_rate : ℝ) 
  (h1 : worker1_rate = 1 / 6) 
  (h2 : worker2_rate = 1 / 4) : 
  1 / (worker1_rate + worker2_rate) = 12 / 5 := by
sorry

end NUMINAMATH_CALUDE_truck_loading_time_l35_3590


namespace NUMINAMATH_CALUDE_perfect_square_from_condition_l35_3529

theorem perfect_square_from_condition (a b c : ℤ) (h : a * b + b * c + c * a = 1) :
  ∃ n : ℤ, (1 + a^2) * (1 + b^2) * (1 + c^2) = n^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_from_condition_l35_3529


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l35_3518

/-- An increasing arithmetic sequence of integers -/
def ArithmeticSequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, d > 0 ∧ ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  ArithmeticSequence b → b 4 * b 5 = 21 → b 3 * b 6 = -779 ∨ b 3 * b 6 = -11 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_product_l35_3518


namespace NUMINAMATH_CALUDE_sum_of_divisors_119_l35_3597

/-- The sum of all positive integer divisors of 119 is 144. -/
theorem sum_of_divisors_119 : (Finset.filter (· ∣ 119) (Finset.range 120)).sum id = 144 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_119_l35_3597


namespace NUMINAMATH_CALUDE_unique_two_digit_number_with_remainder_one_l35_3531

theorem unique_two_digit_number_with_remainder_one : 
  ∃! n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n % 4 = 1 ∧ n % 17 = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_with_remainder_one_l35_3531


namespace NUMINAMATH_CALUDE_base_sum_theorem_l35_3570

theorem base_sum_theorem : ∃! (R_A R_B : ℕ), 
  (R_A > 0 ∧ R_B > 0) ∧
  ((4 * R_A + 5) * (R_B^2 - 1) = (3 * R_B + 6) * (R_A^2 - 1)) ∧
  ((5 * R_A + 4) * (R_B^2 - 1) = (6 * R_B + 3) * (R_A^2 - 1)) ∧
  (R_A + R_B = 19) := by
sorry

end NUMINAMATH_CALUDE_base_sum_theorem_l35_3570


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l35_3522

def f (x : ℝ) : ℝ := x^4 - 8*x^3 + 12*x^2 + 20*x - 18

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := sorry

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - 2) * q x + 22 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l35_3522


namespace NUMINAMATH_CALUDE_students_taking_german_prove_students_taking_german_l35_3596

theorem students_taking_german (total : ℕ) (french : ℕ) (both : ℕ) (neither : ℕ) : ℕ :=
  let students_taking_at_least_one := total - neither
  let students_taking_only_french := french - both
  let students_taking_german := students_taking_at_least_one - students_taking_only_french
  students_taking_german

/-- Given a class of 69 students, where 41 are taking French, 9 are taking both French and German,
    and 15 are not taking either course, prove that 22 students are taking German. -/
theorem prove_students_taking_german :
  students_taking_german 69 41 9 15 = 22 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_german_prove_students_taking_german_l35_3596


namespace NUMINAMATH_CALUDE_ironman_age_relation_l35_3543

/-- Represents the age relationship between superheroes -/
structure SuperheroAges where
  thor : ℝ
  captainAmerica : ℝ
  peterParker : ℝ
  ironman : ℝ

/-- The age relationships between the superheroes are valid -/
def validAgeRelationships (ages : SuperheroAges) : Prop :=
  ages.thor = 13 * ages.captainAmerica ∧
  ages.captainAmerica = 7 * ages.peterParker ∧
  ages.ironman = ages.peterParker + 32

/-- Theorem stating the relationship between Ironman's age and Thor's age -/
theorem ironman_age_relation (ages : SuperheroAges) 
  (h : validAgeRelationships ages) : 
  ages.ironman = ages.thor / 91 + 32 := by
  sorry

end NUMINAMATH_CALUDE_ironman_age_relation_l35_3543


namespace NUMINAMATH_CALUDE_symmetry_about_x_axis_and_origin_l35_3559

/-- Given point A (2, -3) and point B symmetrical to A about the x-axis,
    prove that the coordinates of point C, which is symmetrical to point B about the origin,
    are (-2, -3). -/
theorem symmetry_about_x_axis_and_origin :
  let A : ℝ × ℝ := (2, -3)
  let B : ℝ × ℝ := (A.1, -A.2)  -- B is symmetrical to A about the x-axis
  let C : ℝ × ℝ := (-B.1, -B.2) -- C is symmetrical to B about the origin
  C = (-2, -3) := by sorry

end NUMINAMATH_CALUDE_symmetry_about_x_axis_and_origin_l35_3559


namespace NUMINAMATH_CALUDE_min_surface_area_circumscribed_sphere_l35_3557

/-- The minimum surface area of a sphere circumscribed around a right rectangular prism --/
theorem min_surface_area_circumscribed_sphere (h : ℝ) (a : ℝ) :
  h = 3 →
  a * a = 7 / 2 →
  ∃ (S : ℝ), S = 16 * Real.pi ∧ ∀ (R : ℝ), R ≥ 2 → 4 * Real.pi * R^2 ≥ S :=
by sorry

end NUMINAMATH_CALUDE_min_surface_area_circumscribed_sphere_l35_3557


namespace NUMINAMATH_CALUDE_square_ratio_problem_l35_3501

theorem square_ratio_problem (area_ratio : ℚ) (a b c : ℕ) :
  area_ratio = 75 / 48 →
  (a : ℚ) * Real.sqrt b / c = Real.sqrt (area_ratio) →
  a + b + c = 10 :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_problem_l35_3501


namespace NUMINAMATH_CALUDE_victor_insect_stickers_l35_3564

/-- The number of insect stickers Victor has -/
def insect_stickers (flower_stickers : ℝ) (total_stickers : ℝ) : ℝ :=
  total_stickers - (2 * flower_stickers - 3.5) - (1.5 * flower_stickers + 5.5)

theorem victor_insect_stickers :
  insect_stickers 15 70 = 15.5 := by sorry

end NUMINAMATH_CALUDE_victor_insect_stickers_l35_3564


namespace NUMINAMATH_CALUDE_five_digit_sum_l35_3583

def is_valid_digit (d : ℕ) : Prop := d ≥ 1 ∧ d ≤ 9

theorem five_digit_sum (x : ℕ) (h1 : is_valid_digit x) 
  (h2 : x ≠ 1 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 6) 
  (h3 : 120 * (1 + 3 + 4 + 6 + x) = 2640) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_sum_l35_3583


namespace NUMINAMATH_CALUDE_work_completion_time_l35_3550

/-- Given a work that can be completed by person A in 60 days, and together with person B in 24 days,
    this theorem proves that B can complete the remaining work alone in 40 days after A works for 15 days. -/
theorem work_completion_time (total_work : ℝ) : 
  (∃ (rate_a rate_b : ℝ), 
    rate_a * 60 = total_work ∧ 
    (rate_a + rate_b) * 24 = total_work ∧ 
    rate_b * 40 = total_work - rate_a * 15) := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l35_3550


namespace NUMINAMATH_CALUDE_game_ends_after_63_rounds_l35_3525

/-- Represents a player in the game -/
inductive Player : Type
| A | B | C | D

/-- Represents the state of the game -/
structure GameState :=
  (tokens : Player → ℕ)
  (round : ℕ)

/-- Initial state of the game -/
def initialState : GameState :=
  { tokens := λ p => match p with
    | Player.A => 20
    | Player.B => 18
    | Player.C => 16
    | Player.D => 14
  , round := 0 }

/-- Updates the game state for one round -/
def updateState (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Theorem stating that the game ends after 63 rounds -/
theorem game_ends_after_63_rounds :
  ∃ (finalState : GameState),
    finalState.round = 63 ∧
    isGameOver finalState ∧
    (∀ (prevState : GameState),
      prevState.round < 63 →
      ¬isGameOver prevState) :=
  sorry

end NUMINAMATH_CALUDE_game_ends_after_63_rounds_l35_3525


namespace NUMINAMATH_CALUDE_weight_of_K2Cr2O7_l35_3534

/-- The atomic weight of potassium in g/mol -/
def atomic_weight_K : ℝ := 39.10

/-- The atomic weight of chromium in g/mol -/
def atomic_weight_Cr : ℝ := 52.00

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of potassium atoms in K2Cr2O7 -/
def K_count : ℕ := 2

/-- The number of chromium atoms in K2Cr2O7 -/
def Cr_count : ℕ := 2

/-- The number of oxygen atoms in K2Cr2O7 -/
def O_count : ℕ := 7

/-- The number of moles of K2Cr2O7 -/
def moles : ℝ := 4

/-- The molecular weight of K2Cr2O7 in g/mol -/
def molecular_weight_K2Cr2O7 : ℝ := 
  K_count * atomic_weight_K + Cr_count * atomic_weight_Cr + O_count * atomic_weight_O

/-- The total weight of 4 moles of K2Cr2O7 in grams -/
theorem weight_of_K2Cr2O7 : moles * molecular_weight_K2Cr2O7 = 1176.80 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_K2Cr2O7_l35_3534


namespace NUMINAMATH_CALUDE_solution_is_optimal_l35_3576

/-- Represents a feed mixture with its cost and nutrient composition -/
structure FeedMixture where
  cost : ℝ
  componentA : ℝ
  componentB : ℝ
  componentC : ℝ

/-- The available feed mixtures -/
def typeI : FeedMixture := ⟨30, 10, 10, 0⟩
def typeII : FeedMixture := ⟨50, 10, 20, 5⟩

/-- Daily nutritional requirements -/
def requiredA : ℝ := 45
def requiredB : ℝ := 60
def requiredC : ℝ := 5

/-- The proposed solution -/
def solution : ℝ × ℝ := (3, 1.5)

/-- Checks if the given quantities meet the nutritional requirements -/
def meetsRequirements (x y : ℝ) : Prop :=
  x * typeI.componentA + y * typeII.componentA ≥ requiredA ∧
  x * typeI.componentB + y * typeII.componentB ≥ requiredB ∧
  x * typeI.componentC + y * typeII.componentC ≥ requiredC

/-- Calculates the total cost for given quantities of feed mixtures -/
def totalCost (x y : ℝ) : ℝ :=
  x * typeI.cost + y * typeII.cost

/-- Theorem stating that the proposed solution minimizes cost while meeting requirements -/
theorem solution_is_optimal :
  let (x, y) := solution
  meetsRequirements x y ∧
  ∀ x' y', x' ≥ 0 → y' ≥ 0 → meetsRequirements x' y' →
    totalCost x y ≤ totalCost x' y' := by
  sorry

end NUMINAMATH_CALUDE_solution_is_optimal_l35_3576


namespace NUMINAMATH_CALUDE_special_sequence_second_term_l35_3571

/-- An arithmetic sequence with three terms -/
structure ArithmeticSequence3 where
  a : ℤ  -- First term
  b : ℤ  -- Second term
  c : ℤ  -- Third term
  is_arithmetic : b - a = c - b

/-- The second term of an arithmetic sequence with 3² as first term and 3⁴ as third term -/
def second_term_of_special_sequence : ℤ := 45

/-- Theorem stating that the second term of the special arithmetic sequence is 45 -/
theorem special_sequence_second_term :
  ∀ (seq : ArithmeticSequence3), 
  seq.a = 3^2 ∧ seq.c = 3^4 → seq.b = second_term_of_special_sequence :=
by sorry

end NUMINAMATH_CALUDE_special_sequence_second_term_l35_3571


namespace NUMINAMATH_CALUDE_wait_ratio_l35_3598

def total_time : ℕ := 180
def uber_to_house : ℕ := 10
def check_bag : ℕ := 15
def wait_for_boarding : ℕ := 20

def uber_to_airport : ℕ := 5 * uber_to_house
def security : ℕ := 3 * check_bag

def time_before_takeoff : ℕ := 
  uber_to_house + uber_to_airport + check_bag + security + wait_for_boarding

def wait_before_takeoff : ℕ := total_time - time_before_takeoff

theorem wait_ratio : 
  wait_before_takeoff = 2 * wait_for_boarding :=
sorry

end NUMINAMATH_CALUDE_wait_ratio_l35_3598


namespace NUMINAMATH_CALUDE_det_equation_solution_l35_3547

/-- Definition of 2nd order determinant -/
def det2 (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem: If |x+1 1-x; 1-x x+1| = 8, then x = 2 -/
theorem det_equation_solution (x : ℝ) : 
  det2 (x + 1) (1 - x) (1 - x) (x + 1) = 8 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_det_equation_solution_l35_3547


namespace NUMINAMATH_CALUDE_german_enrollment_l35_3506

theorem german_enrollment (total_students : ℕ) (both_subjects : ℕ) (only_english : ℕ) 
  (h1 : total_students = 45)
  (h2 : both_subjects = 12)
  (h3 : only_english = 23)
  (h4 : total_students = only_english + both_subjects + (total_students - (only_english + both_subjects))) :
  total_students - (only_english + both_subjects) + both_subjects = 22 := by
  sorry

end NUMINAMATH_CALUDE_german_enrollment_l35_3506


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_y_negative_l35_3551

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def in_fourth_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

theorem point_in_fourth_quadrant_y_negative (y : ℝ) :
  in_fourth_quadrant (Point2D.mk 5 y) → y < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_y_negative_l35_3551


namespace NUMINAMATH_CALUDE_f_f_zero_l35_3560

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 3 * x^2 - 4
  else if x = 0 then Real.pi
  else 0

theorem f_f_zero : f (f 0) = 3 * Real.pi^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_f_f_zero_l35_3560


namespace NUMINAMATH_CALUDE_max_rectangle_area_l35_3585

/-- The perimeter of the rectangle in feet -/
def perimeter : ℕ := 190

/-- The maximum area of the rectangle in square feet -/
def max_area : ℕ := 2256

/-- A function to calculate the area of a rectangle given one side length -/
def area (x : ℕ) : ℕ := x * (perimeter / 2 - x)

/-- Theorem stating that the maximum area of a rectangle with the given perimeter and integer side lengths is 2256 square feet -/
theorem max_rectangle_area :
  ∀ x : ℕ, x > 0 ∧ x < perimeter / 2 → area x ≤ max_area :=
sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l35_3585


namespace NUMINAMATH_CALUDE_total_earnings_l35_3587

def weekly_earnings : ℕ := 16
def harvest_duration : ℕ := 76

theorem total_earnings : weekly_earnings * harvest_duration = 1216 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_l35_3587


namespace NUMINAMATH_CALUDE_first_year_rate_is_12_percent_l35_3515

/-- Profit rate in the first year -/
def first_year_rate : ℝ := 0.12

/-- Initial investment in millions of yuan -/
def initial_investment : ℝ := 5

/-- Profit rate increase in the second year -/
def rate_increase : ℝ := 0.08

/-- Net profit in the second year in millions of yuan -/
def second_year_profit : ℝ := 1.12

theorem first_year_rate_is_12_percent :
  (initial_investment + initial_investment * first_year_rate) * 
  (first_year_rate + rate_increase) = second_year_profit := by sorry

end NUMINAMATH_CALUDE_first_year_rate_is_12_percent_l35_3515


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l35_3512

/-- 
Theorem: The largest value of n for which 2x^2 + nx + 50 can be factored 
as the product of two linear factors with integer coefficients is 101.
-/
theorem largest_n_for_factorization : 
  (∃ (n : ℤ), ∀ (m : ℤ), 
    (∃ (a b : ℤ), 2 * X^2 + n * X + 50 = (2 * X + a) * (X + b)) → 
    m ≤ n) ∧ 
  (∃ (a b : ℤ), 2 * X^2 + 101 * X + 50 = (2 * X + a) * (X + b)) :=
by sorry


end NUMINAMATH_CALUDE_largest_n_for_factorization_l35_3512


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l35_3581

-- Define the Cartesian coordinate system
def CartesianPoint := ℝ × ℝ

-- Define the fourth quadrant
def is_in_fourth_quadrant (p : CartesianPoint) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Define pi as a real number
noncomputable def π : ℝ := Real.pi

-- Theorem statement
theorem point_in_fourth_quadrant :
  is_in_fourth_quadrant (π, -1) := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l35_3581


namespace NUMINAMATH_CALUDE_meal_combinations_eq_sixty_l35_3517

/-- The number of menu items in the restaurant -/
def total_menu_items : ℕ := 12

/-- The number of vegetarian dishes available -/
def vegetarian_dishes : ℕ := 5

/-- The number of different meal combinations for Elena and Nasir -/
def meal_combinations : ℕ := total_menu_items * vegetarian_dishes

/-- Theorem stating that the number of meal combinations is 60 -/
theorem meal_combinations_eq_sixty :
  meal_combinations = 60 := by sorry

end NUMINAMATH_CALUDE_meal_combinations_eq_sixty_l35_3517


namespace NUMINAMATH_CALUDE_max_value_g_on_interval_l35_3519

def g (x : ℝ) : ℝ := x * (x^2 - 1)

theorem max_value_g_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧
  ∀ (y : ℝ), y ∈ Set.Icc 0 1 → g y ≤ g x ∧
  g x = 0 :=
sorry

end NUMINAMATH_CALUDE_max_value_g_on_interval_l35_3519


namespace NUMINAMATH_CALUDE_soda_cost_l35_3563

theorem soda_cost (bob_burgers bob_sodas bob_total carol_burgers carol_sodas carol_total : ℕ) 
  (h_bob : bob_burgers = 4 ∧ bob_sodas = 3 ∧ bob_total = 500)
  (h_carol : carol_burgers = 3 ∧ carol_sodas = 4 ∧ carol_total = 540) :
  ∃ (burger_cost soda_cost : ℕ), 
    burger_cost * bob_burgers + soda_cost * bob_sodas = bob_total ∧
    burger_cost * carol_burgers + soda_cost * carol_sodas = carol_total ∧
    soda_cost = 94 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_l35_3563


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l35_3591

theorem candidate_vote_percentage 
  (total_votes : ℕ) 
  (invalid_percentage : ℚ) 
  (candidate_valid_votes : ℕ) 
  (h1 : total_votes = 560000) 
  (h2 : invalid_percentage = 15 / 100) 
  (h3 : candidate_valid_votes = 357000) : 
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) = 75 / 100 := by
sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l35_3591


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l35_3505

theorem average_of_three_numbers (y : ℝ) : (15 + 25 + y) / 3 = 23 → y = 29 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l35_3505


namespace NUMINAMATH_CALUDE_sum_of_vertices_l35_3548

/-- A configuration of numbers on a triangle -/
structure TriangleConfig where
  vertices : Fin 3 → ℕ
  sides : Fin 3 → ℕ
  sum_property : ∀ i : Fin 3, vertices i + sides i + vertices (i + 1) = 17

/-- The set of numbers to be used in the triangle -/
def triangle_numbers : Finset ℕ := {1, 3, 5, 7, 9, 11}

/-- The theorem stating the sum of numbers at the vertices -/
theorem sum_of_vertices (config : TriangleConfig) 
  (h : ∀ n, n ∈ (Finset.image config.vertices Finset.univ ∪ Finset.image config.sides Finset.univ) → n ∈ triangle_numbers) :
  config.vertices 0 + config.vertices 1 + config.vertices 2 = 15 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_vertices_l35_3548


namespace NUMINAMATH_CALUDE_bitna_elementary_students_l35_3554

/-- The number of pencils purchased by Bitna Elementary School -/
def total_pencils : ℕ := 10395

/-- The number of pencils distributed to each student -/
def pencils_per_student : ℕ := 11

/-- The number of students in Bitna Elementary School -/
def number_of_students : ℕ := total_pencils / pencils_per_student

theorem bitna_elementary_students : number_of_students = 945 := by
  sorry

end NUMINAMATH_CALUDE_bitna_elementary_students_l35_3554


namespace NUMINAMATH_CALUDE_second_chord_length_l35_3527

/-- Represents a chord in a circle -/
structure Chord :=
  (length : ℝ)
  (segment1 : ℝ)
  (segment2 : ℝ)
  (valid : segment1 > 0 ∧ segment2 > 0 ∧ length = segment1 + segment2)

/-- Theorem: Length of the second chord given intersecting chords -/
theorem second_chord_length
  (chord1 : Chord)
  (chord2 : Chord)
  (h1 : chord1.segment1 = 12 ∧ chord1.segment2 = 18)
  (h2 : chord2.segment1 / chord2.segment2 = 3 / 8)
  (h3 : chord1.segment1 * chord1.segment2 = chord2.segment1 * chord2.segment2) :
  chord2.length = 33 :=
sorry

end NUMINAMATH_CALUDE_second_chord_length_l35_3527


namespace NUMINAMATH_CALUDE_farm_animal_ratio_l35_3542

theorem farm_animal_ratio (total animals goats cows pigs : ℕ) : 
  total = 56 ∧ 
  goats = 11 ∧ 
  cows = goats + 4 ∧ 
  total = pigs + cows + goats → 
  pigs * 1 = cows * 2 := by
  sorry

end NUMINAMATH_CALUDE_farm_animal_ratio_l35_3542


namespace NUMINAMATH_CALUDE_max_degree_polynomial_l35_3562

theorem max_degree_polynomial (p : ℕ) (hp : Nat.Prime p) :
  ∃ (d : ℕ), d = p - 2 ∧
  (∃ (T : Polynomial ℤ), (Polynomial.degree T = d) ∧
    (∀ (m n : ℤ), T.eval m ≡ T.eval n [ZMOD p] → m ≡ n [ZMOD p])) ∧
  (∀ (d' : ℕ), d' > d →
    ¬∃ (T : Polynomial ℤ), (Polynomial.degree T = d') ∧
      (∀ (m n : ℤ), T.eval m ≡ T.eval n [ZMOD p] → m ≡ n [ZMOD p])) := by
  sorry

end NUMINAMATH_CALUDE_max_degree_polynomial_l35_3562


namespace NUMINAMATH_CALUDE_smallest_n_divisible_sixty_satisfies_smallest_n_is_sixty_l35_3569

theorem smallest_n_divisible (n : ℕ) : n > 0 ∧ 24 ∣ n^2 ∧ 480 ∣ n^3 → n ≥ 60 :=
by sorry

theorem sixty_satisfies : 24 ∣ 60^2 ∧ 480 ∣ 60^3 :=
by sorry

theorem smallest_n_is_sixty : ∃ (n : ℕ), n > 0 ∧ 24 ∣ n^2 ∧ 480 ∣ n^3 ∧ ∀ (m : ℕ), (m > 0 ∧ 24 ∣ m^2 ∧ 480 ∣ m^3) → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_sixty_satisfies_smallest_n_is_sixty_l35_3569


namespace NUMINAMATH_CALUDE_root_in_interval_implies_a_range_l35_3561

-- Define the function f
def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 2

-- Define the theorem
theorem root_in_interval_implies_a_range :
  ∀ a : ℝ, (∃ x ∈ Set.Icc (-1) 1, f a x = 0) → a ∈ Set.Iic (-1) ∪ Set.Ici 1 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_implies_a_range_l35_3561


namespace NUMINAMATH_CALUDE_smallest_difference_is_six_l35_3521

/-- The set of available digits --/
def availableDigits : Finset Nat := {0, 1, 2, 6, 9}

/-- Function to create a three-digit number from three digits --/
def threeDigitNumber (x y z : Nat) : Nat := 100 * x + 10 * y + z

/-- Function to create a two-digit number from two digits --/
def twoDigitNumber (u v : Nat) : Nat := 10 * u + v

/-- Predicate to check if a list of digits uses each available digit exactly once --/
def validDigitUsage (digits : List Nat) : Prop :=
  digits.toFinset = availableDigits ∧ digits.length = 5

/-- The main theorem --/
theorem smallest_difference_is_six :
  ∃ (x y z u v : Nat),
    validDigitUsage [x, y, z, u, v] ∧
    x ≠ 0 ∧ u ≠ 0 ∧
    ∀ (a b c d e : Nat),
      validDigitUsage [a, b, c, d, e] →
      a ≠ 0 → d ≠ 0 →
      threeDigitNumber x y z - twoDigitNumber u v ≤ threeDigitNumber a b c - twoDigitNumber d e ∧
      threeDigitNumber x y z - twoDigitNumber u v = 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_difference_is_six_l35_3521


namespace NUMINAMATH_CALUDE_range_of_negative_values_l35_3540

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≥ f y

-- State the theorem
theorem range_of_negative_values
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_decreasing : decreasing_on_neg f)
  (h_f2 : f 2 = 0) :
  {x : ℝ | f x < 0} = Set.Ioo (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_negative_values_l35_3540


namespace NUMINAMATH_CALUDE_quadratic_with_real_roots_l35_3573

/-- 
Given a quadratic equation with complex coefficients that has real roots, 
prove that the value of the real parameter m is 1/12.
-/
theorem quadratic_with_real_roots (i : ℂ) :
  (∃ x : ℝ, x^2 - (2*i - 1)*x + 3*m - i = 0) → m = 1/12 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_with_real_roots_l35_3573


namespace NUMINAMATH_CALUDE_smallest_root_of_quadratic_l35_3567

theorem smallest_root_of_quadratic (x : ℝ) :
  (4 * x^2 - 20 * x + 24 = 0) → (x ≥ 2) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_root_of_quadratic_l35_3567


namespace NUMINAMATH_CALUDE_alexandra_magazines_l35_3528

/-- Alexandra's magazine problem -/
theorem alexandra_magazines :
  let friday_magazines : ℕ := 15
  let saturday_magazines : ℕ := 20
  let sunday_magazines : ℕ := 4 * friday_magazines
  let chewed_magazines : ℕ := 8
  let total_magazines : ℕ := friday_magazines + saturday_magazines + sunday_magazines - chewed_magazines
  total_magazines = 87 := by sorry

end NUMINAMATH_CALUDE_alexandra_magazines_l35_3528


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l35_3558

theorem logarithm_sum_simplification : 
  1 / (Real.log 3 / Real.log 20 + 1) + 
  1 / (Real.log 5 / Real.log 12 + 1) + 
  1 / (Real.log 7 / Real.log 8 + 1) = 2 := by sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l35_3558


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l35_3577

def i : ℂ := Complex.I

def z : ℂ := (1 - 3 * i) * (2 + i)

theorem z_in_fourth_quadrant : Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l35_3577


namespace NUMINAMATH_CALUDE_lunks_for_apples_l35_3507

/-- The number of lunks that can be traded for a given number of kunks -/
def lunks_per_kunks : ℚ := 4 / 2

/-- The number of kunks that can be traded for a given number of apples -/
def kunks_per_apples : ℚ := 3 / 5

/-- The number of apples we want to purchase -/
def target_apples : ℕ := 20

/-- Theorem: The number of lunks needed to purchase 20 apples is 24 -/
theorem lunks_for_apples : 
  (target_apples : ℚ) * kunks_per_apples * lunks_per_kunks = 24 := by sorry

end NUMINAMATH_CALUDE_lunks_for_apples_l35_3507


namespace NUMINAMATH_CALUDE_twelve_factorial_mod_thirteen_l35_3526

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem twelve_factorial_mod_thirteen : 
  factorial 12 % 13 = 12 := by sorry

end NUMINAMATH_CALUDE_twelve_factorial_mod_thirteen_l35_3526


namespace NUMINAMATH_CALUDE_complex_distance_range_l35_3503

theorem complex_distance_range (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min max : ℝ), min = 3 ∧ max = 5 ∧
  (∀ w : ℂ, Complex.abs (w + 2 - 2*I) = 1 →
    min ≤ Complex.abs (w - 2 - 2*I) ∧ Complex.abs (w - 2 - 2*I) ≤ max) :=
by sorry

end NUMINAMATH_CALUDE_complex_distance_range_l35_3503


namespace NUMINAMATH_CALUDE_age_difference_l35_3509

theorem age_difference (younger_age elder_age : ℕ) 
  (h1 : younger_age = 33)
  (h2 : elder_age = 53) : 
  elder_age - younger_age = 20 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l35_3509


namespace NUMINAMATH_CALUDE_not_all_squares_congruent_l35_3579

-- Define a square
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define congruence for squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

-- Theorem statement
theorem not_all_squares_congruent :
  ¬ ∀ (s1 s2 : Square), congruent s1 s2 := by
  sorry


end NUMINAMATH_CALUDE_not_all_squares_congruent_l35_3579


namespace NUMINAMATH_CALUDE_minimize_quadratic_l35_3545

theorem minimize_quadratic (c : ℝ) :
  ∃ (b : ℝ), ∀ (x : ℝ), 3 * b^2 + 2 * b + c ≤ 3 * x^2 + 2 * x + c :=
by
  use (-1/3)
  sorry

end NUMINAMATH_CALUDE_minimize_quadratic_l35_3545


namespace NUMINAMATH_CALUDE_production_consistency_gizmo_production_zero_l35_3568

/-- Represents the production rate of gadgets and gizmos -/
structure ProductionRate where
  workers : ℕ
  hours : ℕ
  gadgets : ℕ
  gizmos : ℕ

/-- Given production rates -/
def rate1 : ProductionRate := ⟨120, 1, 360, 240⟩
def rate2 : ProductionRate := ⟨40, 3, 240, 360⟩
def rate3 : ProductionRate := ⟨60, 4, 240, 0⟩

/-- Time to produce one gadget -/
def gadgetTime (r : ProductionRate) : ℚ :=
  (r.workers * r.hours : ℚ) / r.gadgets

/-- Time to produce one gizmo -/
def gizmoTime (r : ProductionRate) : ℚ :=
  (r.workers * r.hours : ℚ) / r.gizmos

theorem production_consistency (r1 r2 : ProductionRate) :
  gadgetTime r1 = gadgetTime r2 ∧ gizmoTime r1 = gizmoTime r2 := by sorry

theorem gizmo_production_zero :
  rate3.gizmos = 0 := by sorry

end NUMINAMATH_CALUDE_production_consistency_gizmo_production_zero_l35_3568


namespace NUMINAMATH_CALUDE_line_with_y_intercept_two_l35_3575

/-- A line in two-dimensional space. -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The equation of a line in slope-intercept form. -/
def line_equation (l : Line) (x : ℝ) : ℝ :=
  l.slope * x + l.y_intercept

/-- Theorem: The equation of a line with y-intercept 2 is y = kx + 2 -/
theorem line_with_y_intercept_two (k : ℝ) :
  ∃ (l : Line), l.y_intercept = 2 ∧ ∀ (x y : ℝ), y = line_equation l x ↔ y = k * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_line_with_y_intercept_two_l35_3575


namespace NUMINAMATH_CALUDE_seven_thousand_twenty_two_l35_3593

theorem seven_thousand_twenty_two : 7000 + 22 = 7022 := by
  sorry

end NUMINAMATH_CALUDE_seven_thousand_twenty_two_l35_3593


namespace NUMINAMATH_CALUDE_unique_integer_solution_l35_3574

theorem unique_integer_solution : ∃! x : ℤ, 3 * (x + 200000) = 10 * x + 2 :=
  by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l35_3574


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l35_3594

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_a1 : a 1 = -2)
  (h_a5 : a 5 = -8) :
  a 3 = -4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l35_3594


namespace NUMINAMATH_CALUDE_joyful_point_properties_l35_3592

-- Define a "joyful point"
def is_joyful_point (m n : ℝ) : Prop := 2 * m = 6 - n

-- Define the point P
def P (m n : ℝ) : ℝ × ℝ := (m, n + 2)

theorem joyful_point_properties :
  -- Part 1: (1, 6) is a joyful point
  is_joyful_point 1 4 ∧
  P 1 4 = (1, 6) ∧
  -- Part 2: If P(a, -a+3) is a joyful point, then a = 5 and P is in the fourth quadrant
  (∀ a : ℝ, is_joyful_point a (-a + 3) → a = 5 ∧ 5 > 0 ∧ -2 < 0) ∧
  -- Part 3: The midpoint of OP is (5/2, -1)
  (let O : ℝ × ℝ := (0, 0);
   let P : ℝ × ℝ := (5, -2);
   (O.1 + P.1) / 2 = 5 / 2 ∧ (O.2 + P.2) / 2 = -1) :=
by sorry

end NUMINAMATH_CALUDE_joyful_point_properties_l35_3592


namespace NUMINAMATH_CALUDE_power_division_equality_l35_3539

theorem power_division_equality (a : ℝ) (h : a ≠ 0) : a^10 / a^9 = a := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l35_3539


namespace NUMINAMATH_CALUDE_train_length_calculation_l35_3566

/-- Calculates the length of a train given its speed, tunnel length, and time to pass through the tunnel. -/
theorem train_length_calculation (train_speed : ℝ) (tunnel_length : ℝ) (passing_time : ℝ) :
  train_speed = 72 →
  tunnel_length = 1.7 →
  passing_time = 1.5 / 60 →
  (train_speed * passing_time) - tunnel_length = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l35_3566


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l35_3582

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 8*y - 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 4*y - 1 = 0

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (-2, -4)
def radius1 : ℝ := 5
def center2 : ℝ × ℝ := (-2, -2)
def radius2 : ℝ := 3

-- Theorem statement
theorem circles_internally_tangent :
  let d := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  d = abs (radius1 - radius2) ∧ d < radius1 + radius2 := by sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_l35_3582


namespace NUMINAMATH_CALUDE_clinton_wardrobe_problem_l35_3580

/-- Clinton's wardrobe problem -/
theorem clinton_wardrobe_problem (hats belts shoes : ℕ) :
  hats = 5 →
  belts = hats + 2 →
  shoes = 2 * belts →
  shoes = 14 := by sorry

end NUMINAMATH_CALUDE_clinton_wardrobe_problem_l35_3580


namespace NUMINAMATH_CALUDE_my_circle_center_l35_3520

/-- A circle in the 2D plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The center of a circle -/
def center (c : Circle) : ℝ × ℝ := sorry

/-- Our specific circle -/
def my_circle : Circle :=
  { equation := fun x y => (x + 2)^2 + y^2 = 5 }

/-- Theorem: The center of our specific circle is (-2, 0) -/
theorem my_circle_center :
  center my_circle = (-2, 0) := by sorry

end NUMINAMATH_CALUDE_my_circle_center_l35_3520


namespace NUMINAMATH_CALUDE_paint_problem_l35_3500

theorem paint_problem (initial_paint : ℚ) : 
  initial_paint = 1 →
  let first_day_used := initial_paint / 2
  let first_day_remaining := initial_paint - first_day_used
  let second_day_first_op := first_day_remaining / 4
  let second_day_mid_remaining := first_day_remaining - second_day_first_op
  let second_day_second_op := second_day_mid_remaining / 8
  let final_remaining := second_day_mid_remaining - second_day_second_op
  final_remaining = (21 : ℚ) / 64 * initial_paint :=
by sorry

end NUMINAMATH_CALUDE_paint_problem_l35_3500


namespace NUMINAMATH_CALUDE_complex_number_equidistant_l35_3546

theorem complex_number_equidistant (z : ℂ) :
  Complex.abs (z - Complex.I) = Complex.abs (z - 1) ∧
  Complex.abs (z - 1) = Complex.abs (z - 2015) →
  z = Complex.mk 1008 1008 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_equidistant_l35_3546


namespace NUMINAMATH_CALUDE_rain_duration_l35_3578

/-- Given a 9-hour period where it did not rain for 5 hours, prove that it rained for 4 hours. -/
theorem rain_duration (total_hours : ℕ) (no_rain_hours : ℕ) (h1 : total_hours = 9) (h2 : no_rain_hours = 5) :
  total_hours - no_rain_hours = 4 := by
  sorry

end NUMINAMATH_CALUDE_rain_duration_l35_3578


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l35_3589

theorem binomial_expansion_coefficient (x : ℝ) :
  let expansion := (1 + 2*x)^5
  ∃ a₀ a₁ a₂ a₃ a₄ a₅ : ℝ,
    expansion = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 ∧
    a₃ = 80 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l35_3589


namespace NUMINAMATH_CALUDE_f_difference_at_three_l35_3511

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + 3*x^3 + 2*x^2 + 7*x

-- Theorem statement
theorem f_difference_at_three : f 3 - f (-3) = 690 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_at_three_l35_3511


namespace NUMINAMATH_CALUDE_B_subset_A_iff_m_in_range_l35_3513

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | 2*m - 1 < x ∧ x < m + 1}

-- State the theorem
theorem B_subset_A_iff_m_in_range (m : ℝ) : 
  B m ⊆ A ↔ m ∈ Set.Ici (-1) := by sorry

end NUMINAMATH_CALUDE_B_subset_A_iff_m_in_range_l35_3513


namespace NUMINAMATH_CALUDE_correct_operation_l35_3556

theorem correct_operation (a : ℝ) : 3 * a - 2 * a = a := by sorry

end NUMINAMATH_CALUDE_correct_operation_l35_3556


namespace NUMINAMATH_CALUDE_sum_of_solutions_l35_3541

-- Define the equations
def equation1 (x : ℝ) : Prop := x + Real.log x = 3
def equation2 (x : ℝ) : Prop := x + (10 : ℝ) ^ x = 3

-- State the theorem
theorem sum_of_solutions (x₁ x₂ : ℝ) 
  (h1 : equation1 x₁) (h2 : equation2 x₂) : x₁ + x₂ = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l35_3541


namespace NUMINAMATH_CALUDE_gcd_divides_n_plus_two_l35_3523

theorem gcd_divides_n_plus_two (a b n : ℤ) 
  (h_coprime : Nat.Coprime a.natAbs b.natAbs) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  ∃ k : ℤ, k * Int.gcd (a^2 + b^2 - n*a*b) (a + b) = n + 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_divides_n_plus_two_l35_3523


namespace NUMINAMATH_CALUDE_rectangle_y_value_l35_3588

/-- Given a rectangle with vertices at (-2, y), (8, y), (-2, 2), and (8, 2),
    with an area of 100 square units and y > 2, prove that y = 12. -/
theorem rectangle_y_value (y : ℝ) 
    (h1 : (8 - (-2)) * (y - 2) = 100)  -- Area of rectangle is 100
    (h2 : y > 2) :                     -- y is greater than 2
  y = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l35_3588


namespace NUMINAMATH_CALUDE_seventh_term_value_l35_3535

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- The sum of the first five terms
  sum_first_five : ℚ
  -- The sixth term
  sixth_term : ℚ
  -- Property: The sum of the first five terms is 15
  sum_property : sum_first_five = 15
  -- Property: The sixth term is 7
  sixth_property : sixth_term = 7

/-- The seventh term of the arithmetic sequence -/
def seventh_term (seq : ArithmeticSequence) : ℚ := 25/3

/-- Theorem: The seventh term of the arithmetic sequence is 25/3 -/
theorem seventh_term_value (seq : ArithmeticSequence) :
  seventh_term seq = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_value_l35_3535


namespace NUMINAMATH_CALUDE_distribution_count_l35_3508

/-- Represents a distribution of tickets to people -/
structure TicketDistribution where
  /-- The number of tickets -/
  num_tickets : Nat
  /-- The number of people -/
  num_people : Nat
  /-- Condition that each person receives at least one ticket -/
  at_least_one_ticket : num_tickets ≥ num_people
  /-- Condition that the number of tickets is 5 -/
  five_tickets : num_tickets = 5
  /-- Condition that the number of people is 4 -/
  four_people : num_people = 4

/-- Counts the number of valid distributions -/
def count_distributions (d : TicketDistribution) : Nat :=
  -- The actual implementation is not provided here
  sorry

/-- Theorem stating that the number of valid distributions is 96 -/
theorem distribution_count (d : TicketDistribution) : count_distributions d = 96 := by
  sorry

end NUMINAMATH_CALUDE_distribution_count_l35_3508


namespace NUMINAMATH_CALUDE_opposite_of_2023_l35_3553

theorem opposite_of_2023 : -(2023 : ℤ) = -2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l35_3553


namespace NUMINAMATH_CALUDE_amoeba_reproduction_time_verify_16_amoebae_l35_3565

/-- Represents the number of amoebae after a certain number of divisions -/
def amoebae_count (divisions : ℕ) : ℕ := 2^divisions

/-- Represents the time taken for a given number of divisions -/
def time_for_divisions (divisions : ℕ) : ℕ := 8

/-- The number of divisions required to reach 16 amoebae from 1 -/
def divisions_to_16 : ℕ := 4

/-- Theorem stating that it takes 2 days for an amoeba to reproduce -/
theorem amoeba_reproduction_time : 
  (time_for_divisions divisions_to_16) / divisions_to_16 = 2 := by
  sorry

/-- Verifies that 16 amoebae are indeed reached after 4 divisions -/
theorem verify_16_amoebae : amoebae_count divisions_to_16 = 16 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_reproduction_time_verify_16_amoebae_l35_3565


namespace NUMINAMATH_CALUDE_two_numbers_sum_and_difference_l35_3536

theorem two_numbers_sum_and_difference (x y : ℝ) : 
  x + y = 18 ∧ x - y = 6 → x = 12 ∧ y = 6 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_sum_and_difference_l35_3536


namespace NUMINAMATH_CALUDE_smallest_perfect_square_sum_of_20_consecutive_integers_l35_3595

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

def sum_of_consecutive_integers (start : ℕ) (count : ℕ) : ℕ :=
  count * (2 * start + count - 1) / 2

theorem smallest_perfect_square_sum_of_20_consecutive_integers :
  ∀ n : ℕ, 
    (∃ start : ℕ, sum_of_consecutive_integers start 20 = n ∧ is_perfect_square n) →
    n ≥ 490 :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_sum_of_20_consecutive_integers_l35_3595


namespace NUMINAMATH_CALUDE_salary_solution_l35_3544

def salary_problem (s : ℕ) : Prop :=
  s - s / 3 - s / 4 - s / 5 = 1760

theorem salary_solution : ∃ (s : ℕ), salary_problem s ∧ s = 812 := by
  sorry

end NUMINAMATH_CALUDE_salary_solution_l35_3544


namespace NUMINAMATH_CALUDE_multiplication_of_powers_l35_3514

theorem multiplication_of_powers (m : ℝ) : 3 * m^2 * (2 * m^3) = 6 * m^5 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_of_powers_l35_3514


namespace NUMINAMATH_CALUDE_total_pencils_and_crayons_l35_3599

theorem total_pencils_and_crayons (rows : ℕ) (pencils_per_row : ℕ) (crayons_per_row : ℕ)
  (h_rows : rows = 11)
  (h_pencils : pencils_per_row = 31)
  (h_crayons : crayons_per_row = 27) :
  rows * pencils_per_row + rows * crayons_per_row = 638 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_and_crayons_l35_3599


namespace NUMINAMATH_CALUDE_used_car_clients_l35_3504

theorem used_car_clients (num_cars : ℕ) (selections_per_car : ℕ) (cars_per_client : ℕ) : 
  num_cars = 16 → 
  selections_per_car = 3 → 
  cars_per_client = 2 → 
  (num_cars * selections_per_car) / cars_per_client = 24 := by
sorry

end NUMINAMATH_CALUDE_used_car_clients_l35_3504


namespace NUMINAMATH_CALUDE_decimal_fraction_sum_equals_one_l35_3586

theorem decimal_fraction_sum_equals_one : ∃ (a b c d e f g h : Nat),
  (a = 2 ∨ a = 3) ∧ (b = 2 ∨ b = 3) ∧
  (c = 2 ∨ c = 3) ∧ (d = 2 ∨ d = 3) ∧
  (e = 2 ∨ e = 3) ∧ (f = 2 ∨ f = 3) ∧
  (g = 2 ∨ g = 3) ∧ (h = 2 ∨ h = 3) ∧
  (a * 10 + b) / 100 + (c * 10 + d) / 100 + (e * 10 + f) / 100 + (g * 10 + h) / 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_decimal_fraction_sum_equals_one_l35_3586


namespace NUMINAMATH_CALUDE_proposition_p_equivalence_l35_3502

theorem proposition_p_equivalence :
  (∃ x : ℝ, x > 0 ∧ Real.sqrt x ≤ x + 1) ↔ ¬(∀ x : ℝ, x > 0 → Real.sqrt x > x + 1) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_equivalence_l35_3502
