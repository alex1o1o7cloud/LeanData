import Mathlib

namespace NUMINAMATH_CALUDE_car_distance_l3726_372681

/-- The distance traveled by a car in 30 minutes, given that it travels at 2/3 the speed of a train moving at 90 miles per hour -/
theorem car_distance (train_speed : ℝ) (car_speed_ratio : ℝ) (time : ℝ) : 
  train_speed = 90 →
  car_speed_ratio = 2 / 3 →
  time = 1 / 2 →
  car_speed_ratio * train_speed * time = 30 := by
sorry

end NUMINAMATH_CALUDE_car_distance_l3726_372681


namespace NUMINAMATH_CALUDE_sum_P_2_neg_2_l3726_372656

/-- A cubic polynomial with specific properties -/
structure CubicPolynomial (k : ℝ) where
  P : ℝ → ℝ
  is_cubic : ∃ (a b c : ℝ), ∀ x, P x = a * x^3 + b * x^2 + c * x + k
  P_0 : P 0 = k
  P_1 : P 1 = 3 * k
  P_neg_1 : P (-1) = 4 * k

/-- The sum of P(2) and P(-2) for a cubic polynomial with specific properties -/
theorem sum_P_2_neg_2 (k : ℝ) (P : CubicPolynomial k) :
  P.P 2 + P.P (-2) = 24 * k := by sorry

end NUMINAMATH_CALUDE_sum_P_2_neg_2_l3726_372656


namespace NUMINAMATH_CALUDE_inequality_solution_l3726_372682

theorem inequality_solution (x : ℝ) :
  x ≠ 1 →
  (x^3 - 3*x^2 + 2*x + 1) / (x^2 - 2*x + 1) ≤ 2 ↔ 
  (2 - Real.sqrt 3 < x ∧ x < 1) ∨ (1 < x ∧ x < 2 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3726_372682


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3726_372663

theorem possible_values_of_a :
  ∀ (a b c : ℤ), (∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) → (a = 3 ∨ a = 7) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3726_372663


namespace NUMINAMATH_CALUDE_pony_price_calculation_l3726_372666

/-- The regular price of Fox jeans in dollars -/
def fox_price : ℝ := 15

/-- The discount rate for Pony jeans as a decimal -/
def pony_discount : ℝ := 0.10999999999999996

/-- The sum of discount rates for Fox and Pony jeans as a decimal -/
def total_discount : ℝ := 0.22

/-- The number of Fox jeans purchased -/
def fox_quantity : ℕ := 3

/-- The number of Pony jeans purchased -/
def pony_quantity : ℕ := 2

/-- The total savings from the purchase in dollars -/
def total_savings : ℝ := 8.91

/-- The regular price of Pony jeans in dollars -/
def pony_price : ℝ := 18

theorem pony_price_calculation :
  fox_price * fox_quantity * (total_discount - pony_discount) +
  pony_price * pony_quantity * pony_discount = total_savings :=
sorry

end NUMINAMATH_CALUDE_pony_price_calculation_l3726_372666


namespace NUMINAMATH_CALUDE_angle_alpha_properties_l3726_372692

def angle_alpha (α : Real) : Prop :=
  ∃ (x y : Real), x = 1 ∧ y = Real.sqrt 3 ∧ x = Real.cos α ∧ y = Real.sin α

theorem angle_alpha_properties (α : Real) (h : angle_alpha α) :
  (Real.sin (π - α) - Real.sin (π / 2 + α) = (Real.sqrt 3 - 1) / 2) ∧
  (∃ k : ℤ, α = 2 * π * (k : Real) + π / 3) :=
sorry

end NUMINAMATH_CALUDE_angle_alpha_properties_l3726_372692


namespace NUMINAMATH_CALUDE_height_of_pillar_D_l3726_372693

/-- Regular hexagon with pillars -/
structure HexagonWithPillars where
  -- Side length of the hexagon
  side_length : ℝ
  -- Heights of pillars at A, B, C
  height_A : ℝ
  height_B : ℝ
  height_C : ℝ

/-- Theorem: Height of pillar at D in a regular hexagon with given pillar heights -/
theorem height_of_pillar_D (h : HexagonWithPillars) 
  (h_side : h.side_length = 10)
  (h_A : h.height_A = 8)
  (h_B : h.height_B = 11)
  (h_C : h.height_C = 12) : 
  ∃ (z : ℝ), z = 5 ∧ 
  ((-15 * Real.sqrt 3) * (-10) + 20 * 0 + (50 * Real.sqrt 3) * z = 400 * Real.sqrt 3) := by
  sorry

#check height_of_pillar_D

end NUMINAMATH_CALUDE_height_of_pillar_D_l3726_372693


namespace NUMINAMATH_CALUDE_part1_part2_l3726_372622

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x - a| + a

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x, f a x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) → a = 1 := by sorry

-- Part 2
theorem part2 (m : ℝ) :
  (∃ n : ℝ, |2 * n - 1| + 1 ≤ m - (|2 * (-n) - 1| + 1)) → m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3726_372622


namespace NUMINAMATH_CALUDE_blue_socks_cost_l3726_372646

/-- The cost of blue socks given the total cost, number of red and blue socks, and cost of red socks -/
def cost_of_blue_socks (total_cost : ℚ) (num_red : ℕ) (num_blue : ℕ) (cost_red : ℚ) : ℚ :=
  (total_cost - num_red * cost_red) / num_blue

/-- Theorem stating the cost of each pair of blue socks -/
theorem blue_socks_cost :
  cost_of_blue_socks 42 4 6 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_blue_socks_cost_l3726_372646


namespace NUMINAMATH_CALUDE_large_monkey_doll_cost_l3726_372670

theorem large_monkey_doll_cost (total_spent : ℚ) (price_difference : ℚ) (extra_dolls : ℕ) 
  (h1 : total_spent = 320)
  (h2 : price_difference = 4)
  (h3 : extra_dolls = 40)
  (h4 : total_spent / (large_cost - price_difference) = total_spent / large_cost + extra_dolls) :
  large_cost = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_large_monkey_doll_cost_l3726_372670


namespace NUMINAMATH_CALUDE_max_attendance_days_l3726_372601

structure Day where
  name : String
  dan_available : Bool
  eve_available : Bool
  frank_available : Bool
  grace_available : Bool

def schedule : List Day := [
  { name := "Monday",    dan_available := false, eve_available := true,  frank_available := false, grace_available := true  },
  { name := "Tuesday",   dan_available := true,  eve_available := false, frank_available := false, grace_available := true  },
  { name := "Wednesday", dan_available := false, eve_available := true,  frank_available := true,  grace_available := false },
  { name := "Thursday",  dan_available := true,  eve_available := false, frank_available := true,  grace_available := false },
  { name := "Friday",    dan_available := false, eve_available := false, frank_available := false, grace_available := true  }
]

def count_available (day : Day) : Nat :=
  (if day.dan_available then 1 else 0) +
  (if day.eve_available then 1 else 0) +
  (if day.frank_available then 1 else 0) +
  (if day.grace_available then 1 else 0)

def max_available (schedule : List Day) : Nat :=
  schedule.map count_available |>.maximum?
    |>.getD 0

theorem max_attendance_days (schedule : List Day) :
  max_available schedule = 2 ∧
  schedule.filter (fun day => count_available day = 2) =
    schedule.filter (fun day => day.name ∈ ["Monday", "Tuesday", "Wednesday", "Thursday"]) :=
by sorry

end NUMINAMATH_CALUDE_max_attendance_days_l3726_372601


namespace NUMINAMATH_CALUDE_project_completion_time_l3726_372615

theorem project_completion_time (a_time b_time total_time : ℕ) 
  (h1 : a_time = 20)
  (h2 : b_time = 20)
  (h3 : total_time = 15) :
  ∃ (x : ℕ), 
    (1 : ℚ) / a_time + (1 : ℚ) / b_time = (1 : ℚ) / (total_time - x) + 
    ((1 : ℚ) / b_time) * (x : ℚ) / total_time ∧ 
    x = 10 := by
  sorry

#check project_completion_time

end NUMINAMATH_CALUDE_project_completion_time_l3726_372615


namespace NUMINAMATH_CALUDE_correct_product_l3726_372603

/-- Given two positive integers a and b, where a is a two-digit number,
    if reversing the digits of a and multiplying by b results in 172,
    then the correct product of a and b is 136. -/
theorem correct_product (a b : ℕ) : 
  (a ≥ 10 ∧ a ≤ 99) →  -- a is a two-digit number
  (b > 0) →  -- b is positive
  (((a % 10) * 10 + (a / 10)) * b = 172) →  -- reversing digits of a and multiplying by b gives 172
  (a * b = 136) :=
by sorry

end NUMINAMATH_CALUDE_correct_product_l3726_372603


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3726_372620

theorem inequality_solution_set (x : ℝ) :
  (x + 5) * (3 - 2*x) ≤ 6 ↔ x ≤ -9/2 ∨ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3726_372620


namespace NUMINAMATH_CALUDE_function_local_extrema_l3726_372602

/-- The function f(x) = (x^2 + ax + 2)e^x has both a local maximum and a local minimum
    if and only if a > 2 or a < -2 -/
theorem function_local_extrema (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    IsLocalMax (fun x => (x^2 + a*x + 2) * Real.exp x) x₁ ∧
    IsLocalMin (fun x => (x^2 + a*x + 2) * Real.exp x) x₂) ↔
  (a > 2 ∨ a < -2) :=
sorry

end NUMINAMATH_CALUDE_function_local_extrema_l3726_372602


namespace NUMINAMATH_CALUDE_q_sufficient_not_necessary_for_p_l3726_372660

-- Define the conditions
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := x^2 - 5*x + 6 < 0

-- Theorem statement
theorem q_sufficient_not_necessary_for_p :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) :=
sorry

end NUMINAMATH_CALUDE_q_sufficient_not_necessary_for_p_l3726_372660


namespace NUMINAMATH_CALUDE_range_of_a_l3726_372637

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3726_372637


namespace NUMINAMATH_CALUDE_largest_d_for_negative_five_in_range_l3726_372638

-- Define the function g
def g (x d : ℝ) : ℝ := x^2 + 5*x + d

-- State the theorem
theorem largest_d_for_negative_five_in_range :
  (∃ (d : ℝ), ∀ (d' : ℝ), 
    (∃ (x : ℝ), g x d = -5) → 
    (∃ (x : ℝ), g x d' = -5) → 
    d' ≤ d) ∧
  (∃ (x : ℝ), g x (5/4) = -5) :=
sorry

end NUMINAMATH_CALUDE_largest_d_for_negative_five_in_range_l3726_372638


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l3726_372699

/-- The solution set of the system of equations {x - 2y = 1, x^3 - 6xy - 8y^3 = 1} 
    is equivalent to the line y = (x-1)/2 -/
theorem solution_set_equivalence (x y : ℝ) : 
  (x - 2*y = 1 ∧ x^3 - 6*x*y - 8*y^3 = 1) ↔ y = (x - 1) / 2 := by
  sorry

#check solution_set_equivalence

end NUMINAMATH_CALUDE_solution_set_equivalence_l3726_372699


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l3726_372676

theorem negation_of_existential_proposition :
  (¬ ∃ n : ℕ, n > 1 ∧ n^2 > 2^n) ↔ (∀ n : ℕ, n > 1 → n^2 ≤ 2^n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l3726_372676


namespace NUMINAMATH_CALUDE_peanut_butter_servings_l3726_372600

/-- The amount of peanut butter in the jar in tablespoons -/
def jar_amount : ℚ := 45 + 2/3

/-- The size of one serving of peanut butter in tablespoons -/
def serving_size : ℚ := 1 + 1/3

/-- The number of servings in the jar -/
def servings : ℚ := jar_amount / serving_size

theorem peanut_butter_servings : servings = 34 + 1/4 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_servings_l3726_372600


namespace NUMINAMATH_CALUDE_inequality_integer_solutions_l3726_372608

theorem inequality_integer_solutions :
  {x : ℤ | 3 ≤ 5 - 2*x ∧ 5 - 2*x ≤ 9} = {-2, -1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_integer_solutions_l3726_372608


namespace NUMINAMATH_CALUDE_polynomial_expansion_equality_l3726_372665

theorem polynomial_expansion_equality (x : ℝ) : 
  (3*x^2 + 4*x + 8)*(x - 1) - (x - 1)*(x^2 + 5*x - 72) + (4*x - 15)*(x - 1)*(x + 6) = 
  6*x^3 + 2*x^2 - 18*x + 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_equality_l3726_372665


namespace NUMINAMATH_CALUDE_count_valid_license_plates_l3726_372648

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of possible digits -/
def digit_range : ℕ := 10

/-- The number of letter positions in a license plate -/
def letter_positions : ℕ := 3

/-- The number of digit positions in a license plate -/
def digit_positions : ℕ := 2

/-- Calculates the total number of valid license plates -/
def valid_license_plates : ℕ := alphabet_size ^ letter_positions * digit_range ^ digit_positions

theorem count_valid_license_plates : valid_license_plates = 1757600 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_license_plates_l3726_372648


namespace NUMINAMATH_CALUDE_at_least_one_boy_and_girl_l3726_372662

def probability_boy_or_girl : ℚ := 1 / 2

def number_of_children : ℕ := 4

theorem at_least_one_boy_and_girl :
  let p := probability_boy_or_girl
  let n := number_of_children
  (1 : ℚ) - (p^n + (1 - p)^n) = 7 / 8 := by sorry

end NUMINAMATH_CALUDE_at_least_one_boy_and_girl_l3726_372662


namespace NUMINAMATH_CALUDE_tickets_theorem_l3726_372634

/-- Calculates the total number of tickets Tate and Peyton have together -/
def totalTickets (tateInitial : ℕ) (tateAdditional : ℕ) : ℕ :=
  let tateFinal := tateInitial + tateAdditional
  let peyton := tateFinal / 2
  tateFinal + peyton

/-- Theorem stating that given the initial conditions, the total number of tickets is 51 -/
theorem tickets_theorem :
  totalTickets 32 2 = 51 := by
  sorry

end NUMINAMATH_CALUDE_tickets_theorem_l3726_372634


namespace NUMINAMATH_CALUDE_determinant_equation_solution_l3726_372621

-- Define the determinant operation
def determinant (a b c d : ℝ) : ℝ := a * d - b * c

-- State the theorem
theorem determinant_equation_solution :
  ∃ (x : ℝ), determinant (x + 1) x (2*x - 6) (2*(x - 1)) = 10 ∧ x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_determinant_equation_solution_l3726_372621


namespace NUMINAMATH_CALUDE_sugar_water_concentration_increases_l3726_372610

theorem sugar_water_concentration_increases 
  (a b m : ℝ) 
  (h1 : b > a) 
  (h2 : a > 0) 
  (h3 : m > 0) : 
  a / b < (a + m) / (b + m) := by
sorry

end NUMINAMATH_CALUDE_sugar_water_concentration_increases_l3726_372610


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l3726_372642

theorem empty_solution_set_implies_a_range :
  (∀ x : ℝ, |x - 1| - |x - 2| ≤ 1) →
  (∀ x : ℝ, |x - 1| - |x - 2| < a^2 + a + 1) →
  a ∈ Set.Iio (-1) ∪ Set.Ioi 0 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l3726_372642


namespace NUMINAMATH_CALUDE_funfair_visitors_l3726_372617

theorem funfair_visitors (a : ℕ) : 
  a > 0 ∧ 
  (50 * a - 40 : ℤ) > 0 ∧ 
  (90 - 20 * a : ℤ) > 0 ∧ 
  (50 * a - 40 : ℤ) > (90 - 20 * a : ℤ) →
  (50 * a - 40 : ℤ) = 60 ∨ (50 * a - 40 : ℤ) = 110 ∨ (50 * a - 40 : ℤ) = 160 :=
by sorry

end NUMINAMATH_CALUDE_funfair_visitors_l3726_372617


namespace NUMINAMATH_CALUDE_arthur_muffins_l3726_372649

theorem arthur_muffins (total : ℕ) (more : ℕ) (initial : ℕ) 
    (h1 : total = 83)
    (h2 : more = 48)
    (h3 : total = initial + more) :
  initial = 35 := by
  sorry

end NUMINAMATH_CALUDE_arthur_muffins_l3726_372649


namespace NUMINAMATH_CALUDE_tangent_line_at_P_l3726_372616

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the point P
def P : ℝ × ℝ := (1, 1)

-- Theorem statement
theorem tangent_line_at_P :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, a*x + b*y + c = 0 ↔ 
      (y - f P.1 = (deriv f P.1) * (x - P.1) ∧ 
       (x, y) ≠ P)) ∧
    a = 3 ∧ b = -1 ∧ c = -2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_P_l3726_372616


namespace NUMINAMATH_CALUDE_soda_discount_percentage_l3726_372684

/-- Given the regular price per can and the discounted price for 72 cans,
    calculate the discount percentage. -/
theorem soda_discount_percentage
  (regular_price : ℝ)
  (discounted_price : ℝ)
  (h_regular_price : regular_price = 0.60)
  (h_discounted_price : discounted_price = 34.56) :
  (1 - discounted_price / (72 * regular_price)) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_soda_discount_percentage_l3726_372684


namespace NUMINAMATH_CALUDE_large_painting_area_is_150_l3726_372690

/-- Represents Davonte's art collection --/
structure ArtCollection where
  square_paintings : Nat
  small_paintings : Nat
  large_painting : Nat
  square_side : Nat
  small_width : Nat
  small_height : Nat
  total_area : Nat

/-- Calculates the area of the large painting in Davonte's collection --/
def large_painting_area (collection : ArtCollection) : Nat :=
  collection.total_area -
  (collection.square_paintings * collection.square_side * collection.square_side +
   collection.small_paintings * collection.small_width * collection.small_height)

/-- Theorem stating that the area of the large painting is 150 square feet --/
theorem large_painting_area_is_150 (collection : ArtCollection)
  (h1 : collection.square_paintings = 3)
  (h2 : collection.small_paintings = 4)
  (h3 : collection.square_side = 6)
  (h4 : collection.small_width = 2)
  (h5 : collection.small_height = 3)
  (h6 : collection.total_area = 282) :
  large_painting_area collection = 150 := by
  sorry

#eval large_painting_area { square_paintings := 3, small_paintings := 4, large_painting := 1,
                            square_side := 6, small_width := 2, small_height := 3, total_area := 282 }

end NUMINAMATH_CALUDE_large_painting_area_is_150_l3726_372690


namespace NUMINAMATH_CALUDE_whipped_cream_cans_needed_l3726_372628

/-- The number of pies Billie bakes per day -/
def pies_per_day : ℕ := 3

/-- The number of days Billie bakes pies -/
def baking_days : ℕ := 11

/-- The number of cans of whipped cream needed to cover one pie -/
def cans_per_pie : ℕ := 2

/-- The number of pies Tiffany eats -/
def pies_eaten : ℕ := 4

/-- The total number of pies Billie bakes -/
def total_pies : ℕ := pies_per_day * baking_days

/-- The number of pies remaining after Tiffany eats -/
def remaining_pies : ℕ := total_pies - pies_eaten

/-- The number of cans of whipped cream needed to cover the remaining pies -/
def cans_needed : ℕ := remaining_pies * cans_per_pie

theorem whipped_cream_cans_needed : cans_needed = 58 := by
  sorry

end NUMINAMATH_CALUDE_whipped_cream_cans_needed_l3726_372628


namespace NUMINAMATH_CALUDE_problem1_problem2_l3726_372675

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the axioms
axiom parallel_trans_LP {l1 l2 : Line} {p : Plane} :
  parallel l1 l2 → parallelLP l2 p → (parallelLP l1 p ∨ subset l1 p)

axiom parallel_trans_PP {p1 p2 p3 : Plane} :
  parallelPP p1 p2 → parallelPP p2 p3 → parallelPP p1 p3

axiom perpendicular_parallel {l : Line} {p1 p2 : Plane} :
  perpendicular l p1 → parallelPP p1 p2 → perpendicular l p2

-- State the theorems
theorem problem1 {m n : Line} {α : Plane} :
  parallel m n → parallelLP n α → (parallelLP m α ∨ subset m α) :=
by sorry

theorem problem2 {m : Line} {α β γ : Plane} :
  parallelPP α β → parallelPP β γ → perpendicular m α → perpendicular m γ :=
by sorry

end NUMINAMATH_CALUDE_problem1_problem2_l3726_372675


namespace NUMINAMATH_CALUDE_existence_of_m_n_l3726_372697

theorem existence_of_m_n (d : ℤ) : ∃ m n : ℤ, d * (m^2 - n) = n - 2*m + 1 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_n_l3726_372697


namespace NUMINAMATH_CALUDE_divisibility_theorem_l3726_372674

theorem divisibility_theorem (a b c : ℕ) (h1 : a ∣ b * c) (h2 : Nat.gcd a b = 1) : a ∣ c := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l3726_372674


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l3726_372667

theorem modulus_of_complex_number : 
  let i : ℂ := Complex.I
  let z : ℂ := 2 * i + 2 / (1 + i)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l3726_372667


namespace NUMINAMATH_CALUDE_binomial_coefficient_19_10_l3726_372688

theorem binomial_coefficient_19_10 : 
  (Nat.choose 17 7 = 19448) → (Nat.choose 17 9 = 24310) → (Nat.choose 19 10 = 87516) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_19_10_l3726_372688


namespace NUMINAMATH_CALUDE_colors_drying_time_l3726_372623

/-- Represents the time in minutes for a laundry load -/
structure LaundryTime where
  washing : ℕ
  drying : ℕ

/-- The total time for all three loads of laundry -/
def total_time : ℕ := 344

/-- The laundry time for the whites -/
def whites : LaundryTime := { washing := 72, drying := 50 }

/-- The laundry time for the darks -/
def darks : LaundryTime := { washing := 58, drying := 65 }

/-- The washing time for the colors -/
def colors_washing : ℕ := 45

/-- The theorem stating that the drying time for colors is 54 minutes -/
theorem colors_drying_time : 
  total_time - (whites.washing + whites.drying + darks.washing + darks.drying + colors_washing) = 54 := by
  sorry

end NUMINAMATH_CALUDE_colors_drying_time_l3726_372623


namespace NUMINAMATH_CALUDE_no_real_solutions_l3726_372625

theorem no_real_solutions :
  (¬ ∃ x : ℝ, Real.sqrt (x + 1) - Real.sqrt (x - 1) = 0) ∧
  (¬ ∃ x : ℝ, Real.sqrt x - Real.sqrt (x - Real.sqrt (1 - x)) = 1) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3726_372625


namespace NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l3726_372611

theorem recurring_decimal_to_fraction : 
  (0.3 : ℚ) + (23 : ℚ) / 99 = 527 / 990 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l3726_372611


namespace NUMINAMATH_CALUDE_mary_overtime_pay_increase_l3726_372677

/-- Represents Mary's work schedule and pay structure -/
structure WorkSchedule where
  maxHours : Nat
  regularHours : Nat
  regularRate : ℚ
  totalEarnings : ℚ

/-- Calculates the percentage increase in overtime pay given a work schedule -/
def overtimePayIncrease (schedule : WorkSchedule) : ℚ :=
  let regularEarnings := schedule.regularHours * schedule.regularRate
  let overtimeEarnings := schedule.totalEarnings - regularEarnings
  let overtimeHours := schedule.maxHours - schedule.regularHours
  let overtimeRate := overtimeEarnings / overtimeHours
  ((overtimeRate - schedule.regularRate) / schedule.regularRate) * 100

/-- Theorem stating that Mary's overtime pay increase is 25% -/
theorem mary_overtime_pay_increase :
  let mary_schedule : WorkSchedule := {
    maxHours := 45,
    regularHours := 20,
    regularRate := 8,
    totalEarnings := 410
  }
  overtimePayIncrease mary_schedule = 25 := by
  sorry


end NUMINAMATH_CALUDE_mary_overtime_pay_increase_l3726_372677


namespace NUMINAMATH_CALUDE_empty_set_proof_l3726_372678

theorem empty_set_proof : {x : ℝ | x^2 + x + 1 = 0} = ∅ := by
  sorry

end NUMINAMATH_CALUDE_empty_set_proof_l3726_372678


namespace NUMINAMATH_CALUDE_exponent_fraction_equality_l3726_372626

theorem exponent_fraction_equality : (3^2015 + 3^2013) / (3^2015 - 3^2013) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_fraction_equality_l3726_372626


namespace NUMINAMATH_CALUDE_student_path_probability_l3726_372691

/-- Represents the number of paths between two points given the number of eastward and southward moves -/
def num_paths (east south : ℕ) : ℕ := Nat.choose (east + south) east

/-- Represents the total number of paths from A to B -/
def total_paths : ℕ := num_paths 6 5

/-- Represents the number of paths from A to B that pass through C and D -/
def paths_through_C_and_D : ℕ := num_paths 3 2 * num_paths 2 1 * num_paths 1 2

/-- The probability of choosing a specific path given the number of moves -/
def path_probability (moves : ℕ) : ℚ := (1 / 2) ^ moves

theorem student_path_probability : 
  (paths_through_C_and_D : ℚ) / total_paths = 15 / 77 := by sorry

end NUMINAMATH_CALUDE_student_path_probability_l3726_372691


namespace NUMINAMATH_CALUDE_b_55_mod_55_eq_zero_l3726_372679

/-- The integer obtained by writing all the integers from 1 to n from left to right -/
def b (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The remainder when b₅₅ is divided by 55 is 0 -/
theorem b_55_mod_55_eq_zero : b 55 % 55 = 0 := by
  sorry

end NUMINAMATH_CALUDE_b_55_mod_55_eq_zero_l3726_372679


namespace NUMINAMATH_CALUDE_connie_marbles_l3726_372613

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℝ := 183.0

/-- The number of marbles Connie has left -/
def marbles_left : ℝ := 593.0

/-- The initial number of marbles Connie had -/
def initial_marbles : ℝ := marbles_given + marbles_left

theorem connie_marbles : initial_marbles = 776.0 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_l3726_372613


namespace NUMINAMATH_CALUDE_parking_lot_cars_l3726_372659

theorem parking_lot_cars (red_cars : ℕ) (black_cars : ℕ) : 
  (red_cars : ℚ) / black_cars = 3 / 8 →
  red_cars = 28 →
  black_cars = 75 := by
sorry

end NUMINAMATH_CALUDE_parking_lot_cars_l3726_372659


namespace NUMINAMATH_CALUDE_money_distribution_l3726_372607

/-- Given three people A, B, and C with money, prove that B and C together have 350 rupees. -/
theorem money_distribution (a b c : ℕ) : 
  a + b + c = 450 →  -- Total money
  a + c = 200 →      -- Money A and C have together
  c = 100 →          -- Money C has
  b + c = 350 :=     -- Money B and C have together
by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l3726_372607


namespace NUMINAMATH_CALUDE_bumper_car_line_joiners_l3726_372698

/-- The number of new people who joined a line for bumper cars at a fair -/
theorem bumper_car_line_joiners (initial : ℕ) (left : ℕ) (final : ℕ) : 
  initial = 12 → left = 10 → final = 17 → final - (initial - left) = 15 := by
  sorry

end NUMINAMATH_CALUDE_bumper_car_line_joiners_l3726_372698


namespace NUMINAMATH_CALUDE_smallest_draw_for_red_apple_probability_l3726_372695

theorem smallest_draw_for_red_apple_probability (total_apples : Nat) (red_apples : Nat) 
  (h1 : total_apples = 15) (h2 : red_apples = 9) : 
  (∃ n : Nat, n = 5 ∧ 
    ∀ k : Nat, k < n → (red_apples - k : Rat) / (total_apples - k) ≥ 1/2 ∧
    (red_apples - n : Rat) / (total_apples - n) < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_draw_for_red_apple_probability_l3726_372695


namespace NUMINAMATH_CALUDE_car_distribution_l3726_372627

theorem car_distribution (total_cars : ℕ) (first_supplier : ℕ) (fourth_fifth_each : ℕ) :
  total_cars = 5650000 →
  first_supplier = 1000000 →
  fourth_fifth_each = 325000 →
  ∃ (second_supplier : ℕ),
    second_supplier + first_supplier + (second_supplier + first_supplier) + 2 * fourth_fifth_each = total_cars ∧
    second_supplier = first_supplier + 500000 := by
  sorry

end NUMINAMATH_CALUDE_car_distribution_l3726_372627


namespace NUMINAMATH_CALUDE_AQ_length_l3726_372657

/-- Square ABCD with side length 10 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 0) ∧ B = (10, 0) ∧ C = (10, 10) ∧ D = (0, 10))

/-- Points P, Q, R, X, Y -/
structure SpecialPoints (ABCD : Square) :=
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (R : ℝ × ℝ)
  (X : ℝ × ℝ)
  (Y : ℝ × ℝ)
  (P_on_CD : P.2 = 10)
  (Q_on_AD : Q.1 = 0)
  (R_on_CD : R.2 = 10)
  (BQ_perp_AP : (Q.2 / 10) * ((10 - Q.2) / P.1) = -1)
  (RQ_parallel_PA : (Q.2 - 10) / (-P.1) = (10 - Q.2) / P.1)
  (X_on_BC_AP : X.1 = 10 ∧ X.2 = (10 - Q.2) * (X.1 / P.1) + Q.2)
  (Y_on_circumcircle : ∃ (center : ℝ × ℝ) (radius : ℝ),
    (Y.1 - center.1)^2 + (Y.2 - center.2)^2 = radius^2 ∧
    (P.1 - center.1)^2 + (P.2 - center.2)^2 = radius^2 ∧
    (Q.1 - center.1)^2 + (Q.2 - center.2)^2 = radius^2 ∧
    (0 - center.1)^2 + (10 - center.2)^2 = radius^2)
  (angle_PYR : Real.cos (105 * π / 180) = 
    ((Y.1 - P.1) * (R.1 - Y.1) + (Y.2 - P.2) * (R.2 - Y.2)) /
    (Real.sqrt ((Y.1 - P.1)^2 + (Y.2 - P.2)^2) * Real.sqrt ((R.1 - Y.1)^2 + (R.2 - Y.2)^2)))

/-- The main theorem -/
theorem AQ_length (ABCD : Square) (points : SpecialPoints ABCD) :
  Real.sqrt ((points.Q.1 - ABCD.A.1)^2 + (points.Q.2 - ABCD.A.2)^2) = 10 * Real.sqrt 3 - 10 := by
  sorry

end NUMINAMATH_CALUDE_AQ_length_l3726_372657


namespace NUMINAMATH_CALUDE_range_of_m_l3726_372652

-- Define the equations
def equation1 (m x : ℝ) := x^2 + m*x + 1 = 0
def equation2 (m x : ℝ) := 4*x^2 + 4*(m-2)*x + 1 = 0

-- Define the conditions
def condition_p (m : ℝ) := ∃ x y, x < 0 ∧ y < 0 ∧ x ≠ y ∧ equation1 m x ∧ equation1 m y
def condition_q (m : ℝ) := ∀ x, ¬(equation2 m x)

-- Theorem statement
theorem range_of_m (m : ℝ) : 
  ((condition_p m ∨ condition_q m) ∧ ¬(condition_p m ∧ condition_q m)) →
  (m ∈ Set.Ioo 1 2 ∪ Set.Ici 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3726_372652


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l3726_372632

/-- Given that 2/3 of 15 bananas are worth 12 oranges,
    prove that 1/4 of 20 bananas are worth 6 oranges. -/
theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℚ),
  (2 / 3 : ℚ) * 15 * banana_value = 12 * orange_value →
  (1 / 4 : ℚ) * 20 * banana_value = 6 * orange_value :=
by sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l3726_372632


namespace NUMINAMATH_CALUDE_inequality_proof_l3726_372643

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1 / a + 1 / b + 1 / (a * b) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3726_372643


namespace NUMINAMATH_CALUDE_egg_groups_l3726_372680

theorem egg_groups (total_eggs : ℕ) (eggs_per_group : ℕ) (h1 : total_eggs = 35) (h2 : eggs_per_group = 7) :
  total_eggs / eggs_per_group = 5 := by
  sorry

end NUMINAMATH_CALUDE_egg_groups_l3726_372680


namespace NUMINAMATH_CALUDE_math_club_team_selection_l3726_372644

def math_club_selection (total_boys : ℕ) (total_girls : ℕ) (team_size : ℕ) (boys_in_team : ℕ) (girls_in_team : ℕ) : ℕ :=
  (total_boys.choose boys_in_team) * (total_girls.choose girls_in_team)

theorem math_club_team_selection :
  math_club_selection 10 12 8 4 4 = 103950 := by
sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l3726_372644


namespace NUMINAMATH_CALUDE_nested_radical_equality_l3726_372640

theorem nested_radical_equality : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_equality_l3726_372640


namespace NUMINAMATH_CALUDE_cake_area_l3726_372606

/-- Represents the size of a piece of cake in inches -/
def piece_size : ℝ := 2

/-- Represents the number of pieces that can be cut from the cake -/
def num_pieces : ℕ := 100

/-- Calculates the area of a single piece of cake -/
def piece_area : ℝ := piece_size * piece_size

/-- Theorem: The total area of the cake is 400 square inches -/
theorem cake_area : piece_area * num_pieces = 400 := by
  sorry

end NUMINAMATH_CALUDE_cake_area_l3726_372606


namespace NUMINAMATH_CALUDE_weight_of_b_l3726_372655

/-- Given three weights a, b, and c, prove that b equals 60 when:
    1. The average of a, b, and c is 60.
    2. The average of a and b is 70.
    3. The average of b and c is 50. -/
theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 60)
  (h2 : (a + b) / 2 = 70)
  (h3 : (b + c) / 2 = 50) : 
  b = 60 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l3726_372655


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3726_372641

theorem quadratic_inequality_solution_range (k : ℝ) :
  (k > 0) →
  (∃ x : ℝ, x^2 - 8*x + k < 0) ↔ (k < 16) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3726_372641


namespace NUMINAMATH_CALUDE_digit_difference_750_150_l3726_372696

/-- The number of digits in the base-2 representation of a positive integer -/
def numDigitsBase2 (n : ℕ+) : ℕ :=
  Nat.log2 n + 1

/-- The difference in the number of digits between 750 and 150 in base 2 -/
theorem digit_difference_750_150 : numDigitsBase2 750 - numDigitsBase2 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_750_150_l3726_372696


namespace NUMINAMATH_CALUDE_two_numbers_difference_l3726_372619

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 50)
  (triple_minus_quadruple : 3 * y - 4 * x = 10)
  (y_geq_x : y ≥ x) :
  |y - x| = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l3726_372619


namespace NUMINAMATH_CALUDE_minimum_packaging_cost_l3726_372669

/-- Calculates the minimum cost for packaging a collection given box dimensions, cost per box, and total volume to be packaged -/
theorem minimum_packaging_cost 
  (box_length : ℝ) 
  (box_width : ℝ) 
  (box_height : ℝ) 
  (cost_per_box : ℝ) 
  (total_volume : ℝ) 
  (h_box_length : box_length = 20)
  (h_box_width : box_width = 20)
  (h_box_height : box_height = 15)
  (h_cost_per_box : cost_per_box = 0.70)
  (h_total_volume : total_volume = 3060000) :
  ⌈total_volume / (box_length * box_width * box_height)⌉ * cost_per_box = 357 :=
by sorry

end NUMINAMATH_CALUDE_minimum_packaging_cost_l3726_372669


namespace NUMINAMATH_CALUDE_least_valid_integer_l3726_372636

def is_valid (n : ℕ) : Prop :=
  ∃ (d : ℕ) (m : ℕ), 
    n = 10 * d + m ∧ 
    d ≠ 0 ∧ 
    d < 10 ∧ 
    19 * m = n

theorem least_valid_integer : 
  (is_valid 95) ∧ (∀ n : ℕ, n < 95 → ¬(is_valid n)) :=
sorry

end NUMINAMATH_CALUDE_least_valid_integer_l3726_372636


namespace NUMINAMATH_CALUDE_greatest_multiple_of_six_remainder_l3726_372661

/-- A function that checks if a natural number has no repeated digits -/
def has_no_repeated_digits (n : ℕ) : Prop := sorry

/-- The greatest integer multiple of 6 with no repeated digits -/
def M : ℕ := sorry

theorem greatest_multiple_of_six_remainder :
  has_no_repeated_digits M ∧
  ∀ k : ℕ, has_no_repeated_digits k → k % 6 = 0 → k ≤ M →
  M % 100 = 78 := by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_six_remainder_l3726_372661


namespace NUMINAMATH_CALUDE_divisibility_condition_l3726_372651

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

theorem divisibility_condition (a b : ℕ) :
  (a^2 + b^2 + 1) % (a * b) = 0 ↔
  ((a = 1 ∧ b = 1) ∨ ∃ n : ℕ, n ≥ 1 ∧ a = fibonacci (2*n + 1) ∧ b = fibonacci (2*n - 1)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3726_372651


namespace NUMINAMATH_CALUDE_last_equal_sum_date_l3726_372683

def is_valid_date (year month day : ℕ) : Prop :=
  year = 2008 ∧ month ≥ 1 ∧ month ≤ 12 ∧ day ≥ 1 ∧ day ≤ 31

def sum_first_four (year month day : ℕ) : ℕ :=
  (day / 10) + (day % 10) + (month / 10) + (month % 10)

def sum_last_four (year : ℕ) : ℕ :=
  (year / 1000) + ((year / 100) % 10) + ((year / 10) % 10) + (year % 10)

def has_equal_sum (year month day : ℕ) : Prop :=
  sum_first_four year month day = sum_last_four year

def is_after (year1 month1 day1 year2 month2 day2 : ℕ) : Prop :=
  year1 > year2 ∨ (year1 = year2 ∧ (month1 > month2 ∨ (month1 = month2 ∧ day1 > day2)))

theorem last_equal_sum_date :
  ∀ (year month day : ℕ),
    is_valid_date year month day →
    has_equal_sum year month day →
    ¬(is_after year month day 2008 12 25) →
    year = 2008 ∧ month = 12 ∧ day = 25 :=
sorry

end NUMINAMATH_CALUDE_last_equal_sum_date_l3726_372683


namespace NUMINAMATH_CALUDE_smallest_non_odd_units_digit_l3726_372664

def isOddUnitsDigit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def isSingleDigit (d : ℕ) : Prop := d < 10

theorem smallest_non_odd_units_digit :
  ∀ d : ℕ, isSingleDigit d → (d < 0 ∨ isOddUnitsDigit d) → 0 ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_odd_units_digit_l3726_372664


namespace NUMINAMATH_CALUDE_ivan_commute_l3726_372653

theorem ivan_commute (T : ℝ) (D : ℝ) (h1 : T > 0) (h2 : D > 0) : 
  let v := D / T
  let new_time := T - 65
  (D / (1.6 * v) = new_time) → 
  (D / (1.3 * v) = T - 40) :=
by sorry

end NUMINAMATH_CALUDE_ivan_commute_l3726_372653


namespace NUMINAMATH_CALUDE_dani_pants_after_five_years_l3726_372689

/-- The number of pants Dani will have after a given number of years -/
def total_pants (initial_pants : ℕ) (pairs_per_year : ℕ) (years : ℕ) : ℕ :=
  initial_pants + pairs_per_year * 2 * years

/-- Theorem stating that Dani will have 90 pants after 5 years -/
theorem dani_pants_after_five_years :
  total_pants 50 4 5 = 90 := by
  sorry

end NUMINAMATH_CALUDE_dani_pants_after_five_years_l3726_372689


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3726_372658

theorem quadratic_equation_solution (m : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 - (m + 3) * x₁ + m + 2 = 0) →
  (x₂^2 - (m + 3) * x₂ + m + 2 = 0) →
  (x₁ / (x₁ + 1) + x₂ / (x₂ + 1) = 13 / 10) →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3726_372658


namespace NUMINAMATH_CALUDE_at_equals_rc_l3726_372687

-- Define the points
variable (A B C D M P R Q S T : Point)

-- Define the cyclic quadrilateral ABCD
def is_cyclic_quadrilateral (A B C D : Point) : Prop := sorry

-- Define M as midpoint of CD
def is_midpoint (M C D : Point) : Prop := sorry

-- Define P as intersection of diagonals AC and BD
def is_diagonal_intersection (P A B C D : Point) : Prop := sorry

-- Define circle through P touching CD at M and meeting AC at R and BD at Q
def circle_touches_and_meets (P M C D R Q : Point) : Prop := sorry

-- Define S on BD such that BS = DQ
def point_on_line_with_equal_distance (S B D Q : Point) : Prop := sorry

-- Define line through S parallel to AB meeting AC at T
def parallel_line_intersection (S T A B C : Point) : Prop := sorry

-- Theorem statement
theorem at_equals_rc 
  (h1 : is_cyclic_quadrilateral A B C D)
  (h2 : is_midpoint M C D)
  (h3 : is_diagonal_intersection P A B C D)
  (h4 : circle_touches_and_meets P M C D R Q)
  (h5 : point_on_line_with_equal_distance S B D Q)
  (h6 : parallel_line_intersection S T A B C) :
  AT = RC := by sorry

end NUMINAMATH_CALUDE_at_equals_rc_l3726_372687


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_l3726_372672

theorem half_abs_diff_squares : (1/2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_l3726_372672


namespace NUMINAMATH_CALUDE_mark_bench_press_value_l3726_372645

/-- Dave's weight in pounds -/
def dave_weight : ℝ := 175

/-- Dave's bench press multiplier -/
def dave_multiplier : ℝ := 3

/-- Craig's bench press percentage compared to Dave -/
def craig_percentage : ℝ := 0.2

/-- Difference between Craig's and Mark's bench press in pounds -/
def mark_difference : ℝ := 50

/-- Calculate Dave's bench press weight -/
def dave_bench_press : ℝ := dave_weight * dave_multiplier

/-- Calculate Craig's bench press weight -/
def craig_bench_press : ℝ := dave_bench_press * craig_percentage

/-- Calculate Mark's bench press weight -/
def mark_bench_press : ℝ := craig_bench_press - mark_difference

theorem mark_bench_press_value : mark_bench_press = 55 := by
  sorry

end NUMINAMATH_CALUDE_mark_bench_press_value_l3726_372645


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3726_372604

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_rate_calculation (principal : ℝ) (time : ℝ) (interest : ℝ)
  (h1 : principal = 1100)
  (h2 : time = 8)
  (h3 : interest = principal - 572) :
  ∃ (rate : ℝ), simple_interest principal rate time = interest ∧ rate = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3726_372604


namespace NUMINAMATH_CALUDE_fraction_sum_l3726_372614

theorem fraction_sum (w x y : ℝ) (h1 : (w + x) / 2 = 0.5) (h2 : w * x = y) : 
  5 / w + 5 / x = 20 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_l3726_372614


namespace NUMINAMATH_CALUDE_student_selection_probability_l3726_372647

theorem student_selection_probability 
  (total_students : ℕ) 
  (selected_students : ℕ) 
  (excluded_students : ℕ) 
  (h1 : total_students = 2008) 
  (h2 : selected_students = 50) 
  (h3 : excluded_students = 8) :
  (selected_students : ℚ) / total_students = 25 / 1004 :=
sorry

end NUMINAMATH_CALUDE_student_selection_probability_l3726_372647


namespace NUMINAMATH_CALUDE_find_divisor_l3726_372605

def nearest_number : ℕ := 3108
def original_number : ℕ := 3105

theorem find_divisor : 
  (nearest_number - original_number = 3) →
  (∃ d : ℕ, d > 1 ∧ nearest_number % d = 0 ∧ 
   ∀ n : ℕ, n > original_number ∧ n < nearest_number → n % d ≠ 0) →
  (∃ d : ℕ, d = 3 ∧ nearest_number % d = 0) :=
by sorry

end NUMINAMATH_CALUDE_find_divisor_l3726_372605


namespace NUMINAMATH_CALUDE_bicycle_trip_time_l3726_372618

/-- Proves that the time taken to go forth is 1 hour given the conditions of the bicycle problem -/
theorem bicycle_trip_time (speed_forth speed_back : ℝ) (time_diff : ℝ) 
  (h1 : speed_forth = 15)
  (h2 : speed_back = 10)
  (h3 : time_diff = 0.5)
  : ∃ (time_forth : ℝ), 
    speed_forth * time_forth = speed_back * (time_forth + time_diff) ∧ 
    time_forth = 1 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_trip_time_l3726_372618


namespace NUMINAMATH_CALUDE_initial_number_proof_l3726_372654

theorem initial_number_proof (x : ℕ) : x + 17 = 29 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l3726_372654


namespace NUMINAMATH_CALUDE_max_d_value_l3726_372612

def a (n : ℕ+) : ℕ := 100 + 2 * n ^ 2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  (∃ k : ℕ+, d k = 49) ∧ (∀ n : ℕ+, d n ≤ 49) :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l3726_372612


namespace NUMINAMATH_CALUDE_newspaper_delivery_difference_l3726_372631

/-- Calculates the difference in monthly newspaper deliveries between Miranda and Jake -/
def monthly_delivery_difference (jake_weekly : ℕ) (miranda_multiplier : ℕ) (weeks_per_month : ℕ) : ℕ :=
  (jake_weekly * miranda_multiplier - jake_weekly) * weeks_per_month

/-- Proves that the difference in monthly newspaper deliveries between Miranda and Jake is 936 -/
theorem newspaper_delivery_difference :
  monthly_delivery_difference 234 2 4 = 936 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_delivery_difference_l3726_372631


namespace NUMINAMATH_CALUDE_sequence_properties_l3726_372639

/-- Given a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ+) : ℚ := 3 * n.val^2 + 4 * n.val

/-- The nth term of the sequence -/
def a (n : ℕ+) : ℚ := S n - S (n - 1)

theorem sequence_properties :
  (∀ n : ℕ+, a n = 6 * n.val + 1) ∧
  (∀ n : ℕ+, n ≥ 2 → a n - a (n - 1) = 6) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l3726_372639


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l3726_372673

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 5 / 2)
  (hy : y / z = 3 / 2)
  (hz : z / x = 1 / 4) :
  w / y = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l3726_372673


namespace NUMINAMATH_CALUDE_parabola_translation_right_l3726_372686

/-- Represents a parabola in the form y = a(x - h)² + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Translates a parabola horizontally -/
def translate_horizontal (p : Parabola) (dx : ℝ) : Parabola :=
  { a := p.a, h := p.h + dx, k := p.k }

theorem parabola_translation_right :
  let original := Parabola.mk (-1) 0 0
  let translated := translate_horizontal original 1
  translated = Parabola.mk (-1) 1 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_right_l3726_372686


namespace NUMINAMATH_CALUDE_line_BM_equation_angle_ABM_equals_ABN_l3726_372694

-- Define the parabola C
def C (x y : ℝ) : Prop := y^2 = 2*x

-- Define points A and B
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (-2, 0)

-- Define a line l passing through A
def l (t : ℝ) (x y : ℝ) : Prop := x = t*y + 2

-- Define points M and N as intersections of l and C
def M (t : ℝ) : ℝ × ℝ := sorry
def N (t : ℝ) : ℝ × ℝ := sorry

-- Theorem 1: When l is perpendicular to x-axis, equation of BM
theorem line_BM_equation (t : ℝ) : 
  t = 0 → (
    let (x₁, y₁) := M t
    (x₁ - 2*y₁ + 2 = 0) ∨ (x₁ + 2*y₁ + 2 = 0)
  ) := by sorry

-- Theorem 2: ∠ABM = ∠ABN for any line l
theorem angle_ABM_equals_ABN (t : ℝ) :
  let (x₁, y₁) := M t
  let (x₂, y₂) := N t
  (y₁ / (x₁ + 2)) + (y₂ / (x₂ + 2)) = 0 := by sorry

end NUMINAMATH_CALUDE_line_BM_equation_angle_ABM_equals_ABN_l3726_372694


namespace NUMINAMATH_CALUDE_abs_sum_range_l3726_372609

theorem abs_sum_range : 
  (∀ x : ℝ, |x + 2| + |x + 3| ≥ 1) ∧ 
  (∃ y : ℝ, ∀ ε > 0, ∃ x : ℝ, |x + 2| + |x + 3| < y + ε) ∧ 
  y = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_range_l3726_372609


namespace NUMINAMATH_CALUDE_fraction_invariance_l3726_372650

theorem fraction_invariance (x y : ℝ) :
  (2 * x + y) / (3 * x + y) = (2 * (10 * x) + 10 * y) / (3 * (10 * x) + 10 * y) :=
by sorry

end NUMINAMATH_CALUDE_fraction_invariance_l3726_372650


namespace NUMINAMATH_CALUDE_color_assignment_l3726_372624

-- Define the colors
inductive Color
| White
| Red
| Blue

-- Define the friends
inductive Friend
| Tamara
| Valya
| Lida

-- Define a function to assign colors to dresses
def dress : Friend → Color := sorry

-- Define a function to assign colors to shoes
def shoes : Friend → Color := sorry

-- Define the theorem
theorem color_assignment :
  -- Tamara's dress and shoes match
  (dress Friend.Tamara = shoes Friend.Tamara) ∧
  -- Valya wore white shoes
  (shoes Friend.Valya = Color.White) ∧
  -- Neither Lida's dress nor her shoes were red
  (dress Friend.Lida ≠ Color.Red ∧ shoes Friend.Lida ≠ Color.Red) ∧
  -- All friends have different dress colors
  (dress Friend.Tamara ≠ dress Friend.Valya ∧
   dress Friend.Tamara ≠ dress Friend.Lida ∧
   dress Friend.Valya ≠ dress Friend.Lida) ∧
  -- All friends have different shoe colors
  (shoes Friend.Tamara ≠ shoes Friend.Valya ∧
   shoes Friend.Tamara ≠ shoes Friend.Lida ∧
   shoes Friend.Valya ≠ shoes Friend.Lida) →
  -- The only valid assignment is:
  (dress Friend.Tamara = Color.Red ∧ shoes Friend.Tamara = Color.Red) ∧
  (dress Friend.Valya = Color.Blue ∧ shoes Friend.Valya = Color.White) ∧
  (dress Friend.Lida = Color.White ∧ shoes Friend.Lida = Color.Blue) :=
by
  sorry

end NUMINAMATH_CALUDE_color_assignment_l3726_372624


namespace NUMINAMATH_CALUDE_positive_x_solution_l3726_372629

theorem positive_x_solution (x y z : ℝ) 
  (eq1 : x * y = 8 - 3 * x - 2 * y)
  (eq2 : y * z = 8 - 3 * y - 3 * z)
  (eq3 : x * z = 40 - 5 * x - 4 * z)
  (x_pos : x > 0) :
  x = Real.sqrt (14 * 17 * 60) / 17 - 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_x_solution_l3726_372629


namespace NUMINAMATH_CALUDE_score_mode_l3726_372668

/-- Represents a score in the stem-and-leaf plot -/
structure Score where
  stem : Nat
  leaf : Nat

/-- The list of all scores from the stem-and-leaf plot -/
def scores : List Score := [
  {stem := 5, leaf := 5}, {stem := 5, leaf := 5},
  {stem := 6, leaf := 4}, {stem := 6, leaf := 8},
  {stem := 7, leaf := 2}, {stem := 7, leaf := 6}, {stem := 7, leaf := 6}, {stem := 7, leaf := 9},
  {stem := 8, leaf := 1}, {stem := 8, leaf := 3}, {stem := 8, leaf := 3}, {stem := 8, leaf := 3}, {stem := 8, leaf := 9}, {stem := 8, leaf := 9},
  {stem := 9, leaf := 0}, {stem := 9, leaf := 5}, {stem := 9, leaf := 5}, {stem := 9, leaf := 5}, {stem := 9, leaf := 7}, {stem := 9, leaf := 8},
  {stem := 10, leaf := 2}, {stem := 10, leaf := 2}, {stem := 10, leaf := 2}, {stem := 10, leaf := 3}, {stem := 10, leaf := 3}, {stem := 10, leaf := 3}, {stem := 10, leaf := 4},
  {stem := 11, leaf := 0}, {stem := 11, leaf := 0}, {stem := 11, leaf := 1}
]

/-- Converts a Score to its numerical value -/
def scoreValue (s : Score) : Nat := s.stem * 10 + s.leaf

/-- Defines the mode of a list of scores -/
def mode (l : List Score) : Set Nat := sorry

/-- Theorem stating that the mode of the given scores is {83, 95, 102, 103} -/
theorem score_mode : mode scores = {83, 95, 102, 103} := by sorry

end NUMINAMATH_CALUDE_score_mode_l3726_372668


namespace NUMINAMATH_CALUDE_razorback_shop_revenue_l3726_372685

/-- Calculates the total revenue from selling discounted t-shirts and hats -/
theorem razorback_shop_revenue 
  (t_shirt_price : ℕ) 
  (hat_price : ℕ) 
  (t_shirt_discount : ℕ) 
  (hat_discount : ℕ) 
  (t_shirts_sold : ℕ) 
  (hats_sold : ℕ) 
  (h1 : t_shirt_price = 51)
  (h2 : hat_price = 28)
  (h3 : t_shirt_discount = 8)
  (h4 : hat_discount = 5)
  (h5 : t_shirts_sold = 130)
  (h6 : hats_sold = 85) :
  (t_shirts_sold * (t_shirt_price - t_shirt_discount) + 
   hats_sold * (hat_price - hat_discount)) = 7545 := by
  sorry

end NUMINAMATH_CALUDE_razorback_shop_revenue_l3726_372685


namespace NUMINAMATH_CALUDE_age_equality_l3726_372671

theorem age_equality (joe_current_age : ℕ) (james_current_age : ℕ) (years_until_equality : ℕ) : 
  joe_current_age = 22 →
  james_current_age = joe_current_age - 10 →
  2 * (joe_current_age + years_until_equality) = 3 * (james_current_age + years_until_equality) →
  years_until_equality = 8 := by
sorry

end NUMINAMATH_CALUDE_age_equality_l3726_372671


namespace NUMINAMATH_CALUDE_multiplication_of_powers_l3726_372630

theorem multiplication_of_powers (a : ℝ) : 2 * a^2 * a = 2 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_of_powers_l3726_372630


namespace NUMINAMATH_CALUDE_value_of_a_l3726_372633

-- Define the conversion rate from paise to rupees
def paiseToRupees (paise : ℚ) : ℚ := paise / 100

-- Define the problem statement
theorem value_of_a (a : ℚ) (h : (0.5 / 100) * a = paiseToRupees 70) : a = 140 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3726_372633


namespace NUMINAMATH_CALUDE_exists_increasing_omega_sequence_l3726_372635

/-- The number of distinct prime factors of a natural number -/
def omega (n : ℕ) : ℕ := sorry

/-- For any k, there exists an n > k satisfying the omega inequality -/
theorem exists_increasing_omega_sequence (k : ℕ) :
  ∃ n : ℕ, n > k ∧ omega n < omega (n + 1) ∧ omega (n + 1) < omega (n + 2) :=
sorry

end NUMINAMATH_CALUDE_exists_increasing_omega_sequence_l3726_372635
