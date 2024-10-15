import Mathlib

namespace NUMINAMATH_CALUDE_cubic_expression_evaluation_l2974_297441

theorem cubic_expression_evaluation : 7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_evaluation_l2974_297441


namespace NUMINAMATH_CALUDE_root_product_theorem_l2974_297424

-- Define the polynomial h(y)
def h (y : ℝ) : ℝ := y^5 - y^3 + 2

-- Define the function k(y)
def k (y : ℝ) : ℝ := y^2 - 3

-- State the theorem
theorem root_product_theorem (y₁ y₂ y₃ y₄ y₅ : ℝ) :
  h y₁ = 0 → h y₂ = 0 → h y₃ = 0 → h y₄ = 0 → h y₅ = 0 →
  k y₁ * k y₂ * k y₃ * k y₄ * k y₅ = 104 := by
  sorry

end NUMINAMATH_CALUDE_root_product_theorem_l2974_297424


namespace NUMINAMATH_CALUDE_parabola_vertex_sum_max_l2974_297420

/-- Given a parabola y = ax^2 + bx + c passing through (0,0), (2T,0), and (2T+1,15),
    where a and T are integers and T ≠ 0, prove that the largest possible value of N is -10,
    where N is the sum of the coordinates of the vertex point. -/
theorem parabola_vertex_sum_max (a T : ℤ) (hT : T ≠ 0) : 
  ∃ (b c : ℤ),
    (0 = c) ∧
    (0 = 4*a*T^2 + 2*b*T + c) ∧
    (15 = a*(2*T+1)^2 + b*(2*T+1) + c) →
    (∀ N : ℤ, N = T - a*T^2 → N ≤ -10) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_sum_max_l2974_297420


namespace NUMINAMATH_CALUDE_some_magical_beings_are_mystical_creatures_l2974_297411

-- Define the universe
variable (U : Type)

-- Define the predicates
variable (Dragon : U → Prop)
variable (MagicalBeing : U → Prop)
variable (MysticalCreature : U → Prop)

-- State the theorem
theorem some_magical_beings_are_mystical_creatures :
  (∀ x, Dragon x → MagicalBeing x) →  -- All dragons are magical beings
  (∃ x, MysticalCreature x ∧ Dragon x) →  -- Some mystical creatures are dragons
  (∃ x, MagicalBeing x ∧ MysticalCreature x)  -- Some magical beings are mystical creatures
:= by sorry

end NUMINAMATH_CALUDE_some_magical_beings_are_mystical_creatures_l2974_297411


namespace NUMINAMATH_CALUDE_lcm_12_18_l2974_297476

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_18_l2974_297476


namespace NUMINAMATH_CALUDE_problem_solution_l2974_297401

theorem problem_solution (s t : ℝ) 
  (eq1 : 12 * s + 8 * t = 160)
  (eq2 : s = t^2 + 2) : 
  t = (Real.sqrt 103 - 1) / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2974_297401


namespace NUMINAMATH_CALUDE_birthday_dinner_cost_l2974_297465

theorem birthday_dinner_cost (num_people : ℕ) (meal_cost drink_cost dessert_cost : ℚ) :
  num_people = 5 →
  meal_cost = 12 →
  drink_cost = 3 →
  dessert_cost = 5 →
  (num_people : ℚ) * (meal_cost + drink_cost + dessert_cost) = 100 := by
  sorry

end NUMINAMATH_CALUDE_birthday_dinner_cost_l2974_297465


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l2974_297461

/-- A line y = 3x + c is tangent to the parabola y^2 = 12x if and only if c = 3 -/
theorem line_tangent_to_parabola (c : ℝ) : 
  (∃ x y : ℝ, y = 3*x + c ∧ y^2 = 12*x ∧ 
   ∀ x' y' : ℝ, y' = 3*x' + c → y'^2 = 12*x' → (x', y') = (x, y)) ↔ 
  c = 3 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l2974_297461


namespace NUMINAMATH_CALUDE_necessary_implies_sufficient_l2974_297477

theorem necessary_implies_sufficient (A B : Prop) :
  (A → B) → (A → B) :=
by
  sorry

end NUMINAMATH_CALUDE_necessary_implies_sufficient_l2974_297477


namespace NUMINAMATH_CALUDE_fraction_simplification_l2974_297474

theorem fraction_simplification : 
  (4 : ℝ) / (Real.sqrt 108 + 2 * Real.sqrt 12 + 2 * Real.sqrt 27) = Real.sqrt 3 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2974_297474


namespace NUMINAMATH_CALUDE_max_value_constraint_max_value_attained_l2974_297484

theorem max_value_constraint (x y : ℝ) (h : 3 * x^2 + 2 * y^2 ≤ 6) : 
  2 * x + y ≤ Real.sqrt 11 := by
sorry

theorem max_value_attained : ∃ (x y : ℝ), 3 * x^2 + 2 * y^2 ≤ 6 ∧ 2 * x + y = Real.sqrt 11 := by
sorry

end NUMINAMATH_CALUDE_max_value_constraint_max_value_attained_l2974_297484


namespace NUMINAMATH_CALUDE_appliance_final_cost_l2974_297439

theorem appliance_final_cost (initial_price : ℝ) : 
  initial_price * 1.4 = 1680 →
  (1680 * 0.8) * 0.9 = 1209.6 :=
by sorry

end NUMINAMATH_CALUDE_appliance_final_cost_l2974_297439


namespace NUMINAMATH_CALUDE_equation_solution_l2974_297491

theorem equation_solution (x y : ℝ) (h : 17 * x + 51 * y = 102) : 9 * x + 27 * y = 54 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2974_297491


namespace NUMINAMATH_CALUDE_rent_spending_percentage_l2974_297496

theorem rent_spending_percentage (x : ℝ) : 
  x > 0 ∧ x < 100 ∧ 
  x + (x - 0.2 * x) + 28 = 100 → 
  x = 40 := by
sorry

end NUMINAMATH_CALUDE_rent_spending_percentage_l2974_297496


namespace NUMINAMATH_CALUDE_quadratic_roots_l2974_297453

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), x₁ = 0 ∧ x₂ = 2 ∧ 
  (∀ x : ℝ, x^2 - 2*x = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2974_297453


namespace NUMINAMATH_CALUDE_random_phenomena_l2974_297486

-- Define a type for phenomena
inductive Phenomenon
| TrafficCount
| IntegerSuccessor
| ShellFiring
| ProductInspection

-- Define a predicate for random phenomena
def isRandom (p : Phenomenon) : Prop :=
  match p with
  | Phenomenon.TrafficCount => true
  | Phenomenon.IntegerSuccessor => false
  | Phenomenon.ShellFiring => true
  | Phenomenon.ProductInspection => true

-- Theorem statement
theorem random_phenomena :
  (isRandom Phenomenon.TrafficCount) ∧
  (¬isRandom Phenomenon.IntegerSuccessor) ∧
  (isRandom Phenomenon.ShellFiring) ∧
  (isRandom Phenomenon.ProductInspection) :=
by sorry

end NUMINAMATH_CALUDE_random_phenomena_l2974_297486


namespace NUMINAMATH_CALUDE_foreign_trade_analysis_l2974_297427

-- Define the data points
def x : List ℝ := [1.8, 2.2, 2.6, 3.0]
def y : List ℝ := [2.0, 2.8, 3.2, 4.0]

-- Define the linear correlation function
def linear_correlation (b : ℝ) (x : ℝ) : ℝ := b * x - 0.84

-- Theorem statement
theorem foreign_trade_analysis :
  let x_mean := (List.sum x) / (List.length x : ℝ)
  let y_mean := (List.sum y) / (List.length y : ℝ)
  let b_hat := (y_mean + 0.84) / x_mean
  ∀ (ε : ℝ), ε > 0 →
    (abs (b_hat - 1.6) < ε) ∧
    (abs ((linear_correlation b_hat⁻¹ 6 + 0.84) / b_hat - 4.275) < ε) :=
by sorry

end NUMINAMATH_CALUDE_foreign_trade_analysis_l2974_297427


namespace NUMINAMATH_CALUDE_cubic_equation_root_l2974_297434

theorem cubic_equation_root (c d : ℚ) : 
  (3 + 2 * Real.sqrt 5)^3 + c * (3 + 2 * Real.sqrt 5)^2 + d * (3 + 2 * Real.sqrt 5) + 45 = 0 →
  c = -10 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l2974_297434


namespace NUMINAMATH_CALUDE_polygon_angle_sums_l2974_297436

/-- For an n-sided polygon, the sum of exterior angles is 360° and the sum of interior angles is (n-2) × 180° -/
theorem polygon_angle_sums (n : ℕ) (h : n ≥ 3) :
  ∃ (exterior_sum interior_sum : ℝ),
    exterior_sum = 360 ∧
    interior_sum = (n - 2) * 180 :=
by sorry

end NUMINAMATH_CALUDE_polygon_angle_sums_l2974_297436


namespace NUMINAMATH_CALUDE_converse_abs_inequality_l2974_297428

theorem converse_abs_inequality (x y : ℝ) : x > |y| → x > y := by sorry

end NUMINAMATH_CALUDE_converse_abs_inequality_l2974_297428


namespace NUMINAMATH_CALUDE_binary_10011_equals_19_l2974_297431

/-- Convert a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 19 -/
def binary_19 : List Bool := [true, true, false, false, true]

/-- Theorem stating that the binary number 10011 is equal to the decimal number 19 -/
theorem binary_10011_equals_19 : binary_to_decimal binary_19 = 19 := by
  sorry

end NUMINAMATH_CALUDE_binary_10011_equals_19_l2974_297431


namespace NUMINAMATH_CALUDE_square_area_error_l2974_297470

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 4.04 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l2974_297470


namespace NUMINAMATH_CALUDE_max_side_length_triangle_l2974_297492

theorem max_side_length_triangle (a b c : ℕ) : 
  a < b ∧ b < c ∧                -- Three different side lengths
  a + b + c = 24 ∧               -- Perimeter is 24
  a ≥ 4 ∧                        -- Shortest side is at least 4
  c < a + b →                    -- Triangle inequality
  c ≤ 11 :=                      -- Maximum side length is 11
by sorry

end NUMINAMATH_CALUDE_max_side_length_triangle_l2974_297492


namespace NUMINAMATH_CALUDE_g_behavior_at_infinity_l2974_297419

def g (x : ℝ) := -3 * x^3 + 5 * x^2 + 4

theorem g_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x > M) :=
by sorry

end NUMINAMATH_CALUDE_g_behavior_at_infinity_l2974_297419


namespace NUMINAMATH_CALUDE_special_function_zero_location_l2974_297475

/-- A function f satisfying the given conditions -/
structure SpecialFunction (f : ℝ → ℝ) : Prop :=
  (decreasing : ∀ x₁ x₂ : ℝ, (x₁ - x₂) * (f x₁ - f x₂) < 0)

/-- The theorem statement -/
theorem special_function_zero_location
  (f : ℝ → ℝ) (hf : SpecialFunction f) (a b c d : ℝ)
  (h_order : c < b ∧ b < a)
  (h_product : f a * f b * f c < 0)
  (h_zero : f d = 0) :
  (d < c) ∨ (b < d ∧ d < a) :=
sorry

end NUMINAMATH_CALUDE_special_function_zero_location_l2974_297475


namespace NUMINAMATH_CALUDE_weaving_woman_problem_l2974_297437

/-- Represents the amount of cloth woven on a given day -/
def cloth_woven (day : ℕ) (initial_amount : ℚ) : ℚ :=
  initial_amount * 2^(day - 1)

/-- The problem of the weaving woman -/
theorem weaving_woman_problem :
  ∃ (initial_amount : ℚ),
    (∀ (day : ℕ), day > 0 → cloth_woven day initial_amount = initial_amount * 2^(day - 1)) ∧
    cloth_woven 5 initial_amount = 5 ∧
    initial_amount = 5/31 := by
  sorry

end NUMINAMATH_CALUDE_weaving_woman_problem_l2974_297437


namespace NUMINAMATH_CALUDE_cube_root_function_l2974_297400

theorem cube_root_function (k : ℝ) : 
  (∀ x : ℝ, x > 0 → (k * x^(1/3) = 4 * Real.sqrt 3 ↔ x = 64)) →
  k * 8^(1/3) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_function_l2974_297400


namespace NUMINAMATH_CALUDE_constant_k_value_l2974_297458

theorem constant_k_value : ∃ k : ℝ, ∀ x : ℝ, 
  -x^2 - (k + 7)*x - 8 = -(x - 2)*(x - 4) ↔ k = -13 := by
  sorry

end NUMINAMATH_CALUDE_constant_k_value_l2974_297458


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l2974_297425

/-- Proves that the cost of each adult ticket is $31.50 given the problem conditions --/
theorem adult_ticket_cost : 
  let child_ticket_cost : ℚ := 15/2
  let total_bill : ℚ := 138
  let total_tickets : ℕ := 12
  ∀ (adult_tickets : ℕ) (child_tickets : ℕ) (adult_ticket_cost : ℚ),
    child_tickets = adult_tickets + 8 →
    adult_tickets + child_tickets = total_tickets →
    adult_tickets * adult_ticket_cost + child_tickets * child_ticket_cost = total_bill →
    adult_ticket_cost = 63/2 :=
by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l2974_297425


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_one_range_of_a_for_existence_condition_l2974_297464

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |2*x + a|

-- Part I
theorem solution_set_for_a_equals_one :
  let a := 1
  {x : ℝ | f a x ≥ 5} = Set.Ici 2 ∪ Set.Iic (-4/3) := by sorry

-- Part II
theorem range_of_a_for_existence_condition :
  (∃ x₀ : ℝ, f a x₀ + |x₀ - 2| < 3) ↔ -7 < a ∧ a < -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_one_range_of_a_for_existence_condition_l2974_297464


namespace NUMINAMATH_CALUDE_road_completion_proof_l2974_297445

def road_paving (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n+1 => road_paving n + 1 / road_paving n

theorem road_completion_proof :
  ∃ n : ℕ, n ≤ 5001 ∧ road_paving n ≥ 100 := by
  sorry

end NUMINAMATH_CALUDE_road_completion_proof_l2974_297445


namespace NUMINAMATH_CALUDE_price_difference_l2974_297446

def original_price : ℚ := 150
def tax_rate : ℚ := 0.07
def discount_rate : ℚ := 0.25
def service_charge_rate : ℚ := 0.05

def ann_price : ℚ :=
  original_price * (1 + tax_rate) * (1 - discount_rate) * (1 + service_charge_rate)

def ben_price : ℚ :=
  original_price * (1 - discount_rate) * (1 + tax_rate)

theorem price_difference :
  ann_price - ben_price = 6.01875 := by sorry

end NUMINAMATH_CALUDE_price_difference_l2974_297446


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2974_297493

theorem absolute_value_inequality (a b c : ℝ) (h : |a + b| < -c) :
  (∃! n : ℕ, n = 2 ∧
    (a < -b - c) ∧
    (a + b > c) ∧
    ¬(a + c < b) ∧
    ¬(|a| + c < b)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2974_297493


namespace NUMINAMATH_CALUDE_largest_whole_number_nine_times_less_than_150_l2974_297448

theorem largest_whole_number_nine_times_less_than_150 : 
  ∃ (x : ℕ), x = 16 ∧ (∀ y : ℕ, 9 * y < 150 → y ≤ x) := by
  sorry

end NUMINAMATH_CALUDE_largest_whole_number_nine_times_less_than_150_l2974_297448


namespace NUMINAMATH_CALUDE_XY₂_atomic_numbers_l2974_297417

/-- Represents an element in the periodic table -/
structure Element where
  atomic_number : ℕ
  charge : ℤ
  group : ℕ

/-- Represents an ionic compound -/
structure IonicCompound where
  metal : Element
  nonmetal : Element
  metal_count : ℕ
  nonmetal_count : ℕ

/-- The XY₂ compound -/
def XY₂ : IonicCompound :=
  { metal := { atomic_number := 12, charge := 2, group := 2 },
    nonmetal := { atomic_number := 9, charge := -1, group := 17 },
    metal_count := 1,
    nonmetal_count := 2 }

theorem XY₂_atomic_numbers :
  XY₂.metal.atomic_number = 12 ∧ XY₂.nonmetal.atomic_number = 9 :=
by sorry

end NUMINAMATH_CALUDE_XY₂_atomic_numbers_l2974_297417


namespace NUMINAMATH_CALUDE_circle_bisection_l2974_297415

/-- Circle represented by its equation -/
structure Circle where
  center : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- A circle bisects another circle if the line through their intersection points passes through the center of the bisected circle -/
def bisects (c1 c2 : Circle) : Prop :=
  ∃ (l : ℝ → ℝ → Prop), (∀ x y, l x y ↔ c1.equation x y ∧ c2.equation x y) ∧
                         l c2.center.1 c2.center.2

theorem circle_bisection (a b : ℝ) :
  let c1 : Circle := ⟨(a, b), λ x y => (x - a)^2 + (y - b)^2 = b^2 + 1⟩
  let c2 : Circle := ⟨(-1, -1), λ x y => (x + 1)^2 + (y + 1)^2 = 4⟩
  bisects c1 c2 → a^2 + 2*a + 2*b + 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_bisection_l2974_297415


namespace NUMINAMATH_CALUDE_students_using_red_l2974_297460

/-- Given a group of students painting a picture, calculate the number using red color. -/
theorem students_using_red (total green both : ℕ) (h1 : total = 70) (h2 : green = 52) (h3 : both = 38) :
  total = green + (green + both - total) - both → green + both - total = 56 := by
  sorry

end NUMINAMATH_CALUDE_students_using_red_l2974_297460


namespace NUMINAMATH_CALUDE_largest_divisor_n4_minus_n_l2974_297438

/-- A positive integer greater than 1 is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ k, 1 < k ∧ k < n ∧ n % k = 0

theorem largest_divisor_n4_minus_n (n : ℕ) (h : IsComposite n) :
  (∀ d : ℕ, d > 6 → ¬(d ∣ (n^4 - n))) ∧
  (6 ∣ (n^4 - n)) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_n4_minus_n_l2974_297438


namespace NUMINAMATH_CALUDE_total_missed_pitches_l2974_297497

-- Define the constants from the problem
def pitches_per_token : ℕ := 15
def macy_tokens : ℕ := 11
def piper_tokens : ℕ := 17
def macy_hits : ℕ := 50
def piper_hits : ℕ := 55

-- Theorem statement
theorem total_missed_pitches :
  (macy_tokens * pitches_per_token + piper_tokens * pitches_per_token) - (macy_hits + piper_hits) = 315 := by
  sorry


end NUMINAMATH_CALUDE_total_missed_pitches_l2974_297497


namespace NUMINAMATH_CALUDE_employee_savings_l2974_297402

/-- Calculates the combined savings of three employees over a given period. -/
def combinedSavings (hourlyWage : ℚ) (hoursPerDay : ℚ) (daysPerWeek : ℚ) (weeks : ℚ)
  (savingsRate1 savingsRate2 savingsRate3 : ℚ) : ℚ :=
  let weeklyWage := hourlyWage * hoursPerDay * daysPerWeek
  let totalPeriod := weeklyWage * weeks
  totalPeriod * (savingsRate1 + savingsRate2 + savingsRate3)

/-- The combined savings of three employees with given work conditions and savings rates
    over four weeks is $3000. -/
theorem employee_savings : 
  combinedSavings 10 10 5 4 (2/5) (3/5) (1/2) = 3000 := by
  sorry

end NUMINAMATH_CALUDE_employee_savings_l2974_297402


namespace NUMINAMATH_CALUDE_twin_pairs_probability_l2974_297450

/-- Represents the gender composition of a pair of twins -/
inductive TwinPair
  | BothBoys
  | BothGirls
  | Mixed

/-- The probability of each outcome for a pair of twins -/
def pairProbability : TwinPair → ℚ
  | TwinPair.BothBoys => 1/3
  | TwinPair.BothGirls => 1/3
  | TwinPair.Mixed => 1/3

/-- The probability of two pairs of twins having a specific composition -/
def twoTwinPairsProbability (pair1 pair2 : TwinPair) : ℚ :=
  pairProbability pair1 * pairProbability pair2

theorem twin_pairs_probability :
  (twoTwinPairsProbability TwinPair.BothBoys TwinPair.BothBoys +
   twoTwinPairsProbability TwinPair.BothGirls TwinPair.BothGirls) =
  (twoTwinPairsProbability TwinPair.Mixed TwinPair.BothBoys +
   twoTwinPairsProbability TwinPair.Mixed TwinPair.BothGirls) ∧
  (twoTwinPairsProbability TwinPair.BothBoys TwinPair.BothBoys +
   twoTwinPairsProbability TwinPair.BothGirls TwinPair.BothGirls) = 2/9 :=
by sorry

#check twin_pairs_probability

end NUMINAMATH_CALUDE_twin_pairs_probability_l2974_297450


namespace NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l2974_297467

theorem no_solution_absolute_value_equation : ¬∃ (x : ℝ), |(-4 * x)| + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l2974_297467


namespace NUMINAMATH_CALUDE_basketball_spectators_l2974_297498

/-- Proves the number of children at a basketball match -/
theorem basketball_spectators (total : ℕ) (men : ℕ) (women : ℕ) (children : ℕ) : 
  total = 10000 →
  men = 7000 →
  children = 5 * women →
  total = men + women + children →
  children = 2500 := by
sorry

end NUMINAMATH_CALUDE_basketball_spectators_l2974_297498


namespace NUMINAMATH_CALUDE_sum_of_slopes_constant_l2974_297482

/-- An ellipse with eccentricity 1/2 passing through (2,0) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_a_gt_b : a > b
  h_ecc : a^2 - b^2 = (a/2)^2
  h_thru_point : 4/a^2 + 0/b^2 = 1

/-- A line passing through (1,0) intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  h_intersect : ∃ x y, x^2/E.a^2 + y^2/E.b^2 = 1 ∧ y = k*(x-1)

/-- The point P -/
def P : ℝ × ℝ := (4, 3)

/-- Slopes of PA and PB -/
def slopes (E : Ellipse) (L : IntersectingLine E) : ℝ × ℝ :=
  sorry

/-- The theorem to be proved -/
theorem sum_of_slopes_constant (E : Ellipse) (L : IntersectingLine E) :
  let (k₁, k₂) := slopes E L
  k₁ + k₂ = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_slopes_constant_l2974_297482


namespace NUMINAMATH_CALUDE_expression_evaluation_l2974_297429

theorem expression_evaluation (x : ℝ) (h : x = 6) :
  (x^9 - 27*x^6 + 216*x^3 - 512) / (x^3 - 8) = 43264 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2974_297429


namespace NUMINAMATH_CALUDE_puzzle_solution_l2974_297473

def addition_puzzle (F I V N E : Nat) : Prop :=
  F ≠ I ∧ F ≠ V ∧ F ≠ N ∧ F ≠ E ∧
  I ≠ V ∧ I ≠ N ∧ I ≠ E ∧
  V ≠ N ∧ V ≠ E ∧
  N ≠ E ∧
  F = 8 ∧
  I % 2 = 0 ∧
  1000 * N + 100 * I + 10 * N + E = 100 * F + 10 * I + V + 100 * F + 10 * I + V

theorem puzzle_solution :
  ∀ F I V N E, addition_puzzle F I V N E → V = 5 :=
by sorry

end NUMINAMATH_CALUDE_puzzle_solution_l2974_297473


namespace NUMINAMATH_CALUDE_cara_catches_47_l2974_297412

/-- The number of animals Martha's cat catches -/
def martha_animals : ℕ := 10

/-- The number of animals Cara's cat catches -/
def cara_animals : ℕ := 5 * martha_animals - 3

/-- Theorem stating that Cara's cat catches 47 animals -/
theorem cara_catches_47 : cara_animals = 47 := by
  sorry

end NUMINAMATH_CALUDE_cara_catches_47_l2974_297412


namespace NUMINAMATH_CALUDE_school_population_l2974_297405

theorem school_population (b g t : ℕ) : 
  b = 4 * g → g = 5 * t → b + g + t = 26 * t := by
  sorry

end NUMINAMATH_CALUDE_school_population_l2974_297405


namespace NUMINAMATH_CALUDE_complex_exponential_to_rectangular_l2974_297455

theorem complex_exponential_to_rectangular : 2 * Complex.exp (15 * π * I / 4) = Complex.mk (Real.sqrt 2) (- Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_to_rectangular_l2974_297455


namespace NUMINAMATH_CALUDE_rational_sqrt_fraction_l2974_297404

theorem rational_sqrt_fraction (r q n : ℚ) 
  (h : 1 / (r + q * n) + 1 / (q + r * n) = 1 / (r + q)) :
  ∃ (m : ℚ), (n - 3) / (n + 1) = m^2 := by
sorry

end NUMINAMATH_CALUDE_rational_sqrt_fraction_l2974_297404


namespace NUMINAMATH_CALUDE_salad_ratio_l2974_297442

/-- Given a salad with cucumbers and tomatoes, prove the ratio of tomatoes to cucumbers -/
theorem salad_ratio (total : ℕ) (cucumbers : ℕ) (h1 : total = 280) (h2 : cucumbers = 70) :
  (total - cucumbers) / cucumbers = 3 := by
  sorry

end NUMINAMATH_CALUDE_salad_ratio_l2974_297442


namespace NUMINAMATH_CALUDE_company_daily_production_l2974_297472

/-- Proves that a company producing enough bottles to fill 2000 cases, 
    where each case holds 25 bottles, produces 50000 bottles daily. -/
theorem company_daily_production 
  (bottles_per_case : ℕ) 
  (cases_per_day : ℕ) 
  (h1 : bottles_per_case = 25)
  (h2 : cases_per_day = 2000) :
  bottles_per_case * cases_per_day = 50000 := by
  sorry

#check company_daily_production

end NUMINAMATH_CALUDE_company_daily_production_l2974_297472


namespace NUMINAMATH_CALUDE_no_solution_implies_a_range_l2974_297489

def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem no_solution_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ¬(otimes (x - a) (x + 1) ≥ 1)) → -2 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_range_l2974_297489


namespace NUMINAMATH_CALUDE_smallest_AAB_l2974_297435

def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def AB (a b : ℕ) : ℕ := 10 * a + b

def AAB (a b : ℕ) : ℕ := 100 * a + 10 * a + b

theorem smallest_AAB :
  ∃ (a b : ℕ),
    is_digit a ∧
    is_digit b ∧
    two_digit (AB a b) ∧
    three_digit (AAB a b) ∧
    AB a b = (AAB a b) / 7 ∧
    AAB a b = 996 ∧
    (∀ (x y : ℕ),
      is_digit x ∧
      is_digit y ∧
      two_digit (AB x y) ∧
      three_digit (AAB x y) ∧
      AB x y = (AAB x y) / 7 →
      AAB x y ≥ 996) :=
by sorry

end NUMINAMATH_CALUDE_smallest_AAB_l2974_297435


namespace NUMINAMATH_CALUDE_road_trip_distance_ratio_l2974_297481

theorem road_trip_distance_ratio : 
  ∀ (tracy michelle katie : ℕ),
  tracy + michelle + katie = 1000 →
  tracy = 2 * michelle + 20 →
  michelle = 294 →
  (michelle : ℚ) / (katie : ℚ) = 3 / 1 := by
sorry

end NUMINAMATH_CALUDE_road_trip_distance_ratio_l2974_297481


namespace NUMINAMATH_CALUDE_distance_between_points_l2974_297466

def point1 : ℝ × ℝ := (2, -3)
def point2 : ℝ × ℝ := (8, 9)

theorem distance_between_points :
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2974_297466


namespace NUMINAMATH_CALUDE_sum_equals_five_l2974_297421

/-- The mapping f that transforms (x, y) to (x, x+y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.1 + p.2)

/-- Theorem stating that a + b = 5 given the conditions -/
theorem sum_equals_five (a b : ℝ) (h : f (a, b) = (1, 3)) : a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_five_l2974_297421


namespace NUMINAMATH_CALUDE_jane_bagels_l2974_297414

theorem jane_bagels (b m : ℕ) : 
  b + m = 5 →
  (75 * b + 50 * m) % 100 = 0 →
  b = 2 :=
by sorry

end NUMINAMATH_CALUDE_jane_bagels_l2974_297414


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l2974_297449

theorem no_prime_roots_for_quadratic : ¬∃ k : ℤ, ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ p ≠ q ∧ 
  (p : ℤ) + q = 57 ∧ 
  (p : ℤ) * q = k ∧
  (p : ℤ) * q = 57 * (p + q) - (p^2 + q^2) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l2974_297449


namespace NUMINAMATH_CALUDE_cars_combined_efficiency_l2974_297463

/-- Calculates the combined fuel efficiency of three cars given their individual efficiencies -/
def combinedFuelEfficiency (e1 e2 e3 : ℚ) : ℚ :=
  3 / (1 / e1 + 1 / e2 + 1 / e3)

/-- Theorem: The combined fuel efficiency of cars with 30, 15, and 20 mpg is 20 mpg -/
theorem cars_combined_efficiency :
  combinedFuelEfficiency 30 15 20 = 20 := by
  sorry

#eval combinedFuelEfficiency 30 15 20

end NUMINAMATH_CALUDE_cars_combined_efficiency_l2974_297463


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l2974_297443

theorem chocolate_box_problem (N : ℕ) (rows columns : ℕ) :
  -- Initial conditions
  N > 0 ∧ rows > 0 ∧ columns > 0 ∧
  -- After operations, one-third remains
  N / 3 > 0 ∧
  -- Three rows minus one can be filled at one point
  (3 * columns - 1 ≤ N ∧ 3 * columns > N / 3) ∧
  -- Five columns minus one can be filled at another point
  (5 * rows - 1 ≤ N ∧ 5 * rows > N / 3) →
  -- Conclusions
  N = 60 ∧ N - (3 * columns - 1) = 25 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l2974_297443


namespace NUMINAMATH_CALUDE_election_votes_calculation_l2974_297494

theorem election_votes_calculation (total_votes : ℕ) : 
  (75 : ℚ) / 100 * ((100 : ℚ) - 15) / 100 * total_votes = 357000 → 
  total_votes = 560000 :=
by
  sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l2974_297494


namespace NUMINAMATH_CALUDE_subtraction_result_l2974_297426

theorem subtraction_result : 
  let total : ℚ := 8000
  let fraction1 : ℚ := 1 / 10
  let fraction2 : ℚ := 1 / 20 * (1 / 100)
  (total * fraction1) - (total * fraction2) = 796 :=
by sorry

end NUMINAMATH_CALUDE_subtraction_result_l2974_297426


namespace NUMINAMATH_CALUDE_positive_integer_problem_l2974_297454

theorem positive_integer_problem (n p : ℕ) (h_p_prime : Nat.Prime p) 
  (h_division : n / (12 * p) = 2) (h_n_ge_48 : n ≥ 48) : n = 48 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_problem_l2974_297454


namespace NUMINAMATH_CALUDE_pen_profit_calculation_l2974_297430

/-- Calculates the profit from selling pens given the purchase quantity, cost rate, and selling rate. -/
def calculate_profit (purchase_quantity : ℕ) (cost_rate : ℚ × ℚ) (selling_rate : ℚ × ℚ) : ℚ :=
  let cost_per_pen := cost_rate.2 / cost_rate.1
  let total_cost := cost_per_pen * purchase_quantity
  let selling_price_per_pen := selling_rate.2 / selling_rate.1
  let total_revenue := selling_price_per_pen * purchase_quantity
  total_revenue - total_cost

/-- The profit from selling 1200 pens, bought at 4 for $3 and sold at 3 for $2, is -$96. -/
theorem pen_profit_calculation :
  calculate_profit 1200 (4, 3) (3, 2) = -96 := by
  sorry

end NUMINAMATH_CALUDE_pen_profit_calculation_l2974_297430


namespace NUMINAMATH_CALUDE_one_negative_number_l2974_297413

theorem one_negative_number (numbers : List ℝ := [-2, 1/2, 0, 3]) : 
  (numbers.filter (λ x => x < 0)).length = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_negative_number_l2974_297413


namespace NUMINAMATH_CALUDE_diagonal_of_square_l2974_297407

theorem diagonal_of_square (side_length : ℝ) (h : side_length = 10) :
  let diagonal := Real.sqrt (2 * side_length ^ 2)
  diagonal = 10 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_diagonal_of_square_l2974_297407


namespace NUMINAMATH_CALUDE_average_increase_is_four_l2974_297479

/-- Represents a cricket player's performance --/
structure CricketPerformance where
  innings : ℕ
  totalRuns : ℕ
  newInningsRuns : ℕ

/-- Calculates the average runs per innings --/
def average (cp : CricketPerformance) : ℚ :=
  cp.totalRuns / cp.innings

/-- Calculates the new average after playing an additional innings --/
def newAverage (cp : CricketPerformance) : ℚ :=
  (cp.totalRuns + cp.newInningsRuns) / (cp.innings + 1)

/-- Theorem: The increase in average is 4 runs --/
theorem average_increase_is_four (cp : CricketPerformance) 
  (h1 : cp.innings = 10)
  (h2 : average cp = 18)
  (h3 : cp.newInningsRuns = 62) : 
  newAverage cp - average cp = 4 := by
  sorry


end NUMINAMATH_CALUDE_average_increase_is_four_l2974_297479


namespace NUMINAMATH_CALUDE_milford_lake_algae_increase_l2974_297487

theorem milford_lake_algae_increase (original_algae current_algae : ℕ) 
  (h1 : original_algae = 809)
  (h2 : current_algae = 3263) : 
  current_algae - original_algae = 2454 := by
  sorry

end NUMINAMATH_CALUDE_milford_lake_algae_increase_l2974_297487


namespace NUMINAMATH_CALUDE_farmer_land_ownership_l2974_297469

theorem farmer_land_ownership (T : ℝ) 
  (h1 : T > 0)
  (h2 : 0.8 * T + 0.2 * T = T)
  (h3 : 0.05 * (0.8 * T) + 0.3 * (0.2 * T) = 720) :
  0.8 * T = 5760 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_ownership_l2974_297469


namespace NUMINAMATH_CALUDE_basketball_weight_proof_l2974_297433

/-- The weight of a skateboard in pounds -/
def skateboard_weight : ℝ := 32

/-- The number of skateboards that balance with the basketballs -/
def num_skateboards : ℕ := 4

/-- The number of basketballs that balance with the skateboards -/
def num_basketballs : ℕ := 8

/-- The weight of a single basketball in pounds -/
def basketball_weight : ℝ := 16

theorem basketball_weight_proof :
  num_basketballs * basketball_weight = num_skateboards * skateboard_weight :=
by sorry

end NUMINAMATH_CALUDE_basketball_weight_proof_l2974_297433


namespace NUMINAMATH_CALUDE_no_real_solutions_l2974_297490

theorem no_real_solutions : ¬∃ x : ℝ, (2*x^2 - 6*x + 5)^2 + 1 = -|x| := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2974_297490


namespace NUMINAMATH_CALUDE_james_sheets_used_l2974_297432

/-- The number of books James prints -/
def num_books : ℕ := 2

/-- The number of pages in each book -/
def pages_per_book : ℕ := 600

/-- The number of pages printed on one side of a sheet -/
def pages_per_side : ℕ := 4

/-- Whether the printing is double-sided -/
def is_double_sided : Bool := true

/-- Calculate the total number of pages to be printed -/
def total_pages : ℕ := num_books * pages_per_book

/-- Calculate the number of pages that can be printed on a single sheet -/
def pages_per_sheet : ℕ := if is_double_sided then 2 * pages_per_side else pages_per_side

/-- The number of sheets of paper James uses -/
def sheets_used : ℕ := total_pages / pages_per_sheet

theorem james_sheets_used : sheets_used = 150 := by
  sorry

end NUMINAMATH_CALUDE_james_sheets_used_l2974_297432


namespace NUMINAMATH_CALUDE_inequality_sqrt_ratios_l2974_297444

theorem inequality_sqrt_ratios (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : 
  Real.sqrt (a / d) > Real.sqrt (b / c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_sqrt_ratios_l2974_297444


namespace NUMINAMATH_CALUDE_simplify_expression_l2974_297462

theorem simplify_expression (a : ℝ) (h : a < (1/4 : ℝ)) :
  4 * (4*a - 1)^2 = Real.sqrt (1 - 4*a) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2974_297462


namespace NUMINAMATH_CALUDE_justine_colored_sheets_l2974_297468

/-- Given 2450 sheets of paper evenly split into 5 binders, 
    prove that Justine colors 245 sheets when she colors 
    half the sheets in one binder. -/
theorem justine_colored_sheets : 
  let total_sheets : ℕ := 2450
  let num_binders : ℕ := 5
  let sheets_per_binder : ℕ := total_sheets / num_binders
  let justine_colored : ℕ := sheets_per_binder / 2
  justine_colored = 245 := by
  sorry

#check justine_colored_sheets

end NUMINAMATH_CALUDE_justine_colored_sheets_l2974_297468


namespace NUMINAMATH_CALUDE_power_multiplication_equality_l2974_297488

theorem power_multiplication_equality : (-0.25)^2023 * 4^2024 = -4 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_equality_l2974_297488


namespace NUMINAMATH_CALUDE_gary_egg_collection_l2974_297495

/-- Calculates the number of eggs collected per week given the initial number of chickens,
    the multiplication factor after two years, eggs laid per chicken per day, and days in a week. -/
def eggs_per_week (initial_chickens : ℕ) (multiplication_factor : ℕ) (eggs_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  initial_chickens * multiplication_factor * eggs_per_day * days_in_week

/-- Proves that Gary collects 1344 eggs per week given the initial conditions. -/
theorem gary_egg_collection :
  eggs_per_week 4 8 6 7 = 1344 :=
by sorry

end NUMINAMATH_CALUDE_gary_egg_collection_l2974_297495


namespace NUMINAMATH_CALUDE_inequality_proof_l2974_297409

theorem inequality_proof (x y : ℝ) (n k : ℕ) 
  (h1 : x > y) (h2 : y > 0) (h3 : n > k) : 
  (x^k - y^k)^n < (x^n - y^n)^k := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2974_297409


namespace NUMINAMATH_CALUDE_largest_base7_five_digit_to_base10_l2974_297403

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ (digits.length - 1 - i))) 0

/-- The largest five-digit number in base-7 --/
def largestBase7FiveDigit : List Nat := [6, 6, 6, 6, 6]

theorem largest_base7_five_digit_to_base10 :
  base7ToBase10 largestBase7FiveDigit = 16806 := by
  sorry

end NUMINAMATH_CALUDE_largest_base7_five_digit_to_base10_l2974_297403


namespace NUMINAMATH_CALUDE_x_squared_less_than_abs_x_l2974_297471

theorem x_squared_less_than_abs_x (x : ℝ) :
  x^2 < |x| ↔ (-1 < x ∧ x < 0) ∨ (0 < x ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_less_than_abs_x_l2974_297471


namespace NUMINAMATH_CALUDE_smallest_absolute_value_rational_l2974_297478

theorem smallest_absolute_value_rational : 
  ∀ q : ℚ, |0| ≤ |q| := by
  sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_rational_l2974_297478


namespace NUMINAMATH_CALUDE_train_average_speed_l2974_297416

-- Define the points and distances
def x : ℝ := 0
def y : ℝ := sorry
def z : ℝ := sorry

-- Define the speeds
def speed_xy : ℝ := 300
def speed_yz : ℝ := 100

-- State the theorem
theorem train_average_speed :
  -- Conditions
  (y - x = 2 * (z - y)) →  -- Distance from x to y is twice the distance from y to z
  -- Conclusion
  (z - x) / ((y - x) / speed_xy + (z - y) / speed_yz) = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_train_average_speed_l2974_297416


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l2974_297485

theorem polar_to_rectangular_conversion :
  let r : ℝ := 10
  let θ : ℝ := 5 * Real.pi / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 5 ∧ y = -5 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l2974_297485


namespace NUMINAMATH_CALUDE_triangle_area_proof_l2974_297423

/-- The line equation defining the hypotenuse of the triangle -/
def line_equation (x y : ℝ) : Prop := 3 * x + y = 9

/-- The x-intercept of the line -/
def x_intercept : ℝ := 3

/-- The y-intercept of the line -/
def y_intercept : ℝ := 9

/-- The area of the triangle -/
def triangle_area : ℝ := 13.5

theorem triangle_area_proof :
  triangle_area = (1/2) * x_intercept * y_intercept :=
sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l2974_297423


namespace NUMINAMATH_CALUDE_minks_set_free_ratio_l2974_297447

/-- Represents the mink coat problem -/
structure MinkCoatProblem where
  skins_per_coat : ℕ
  initial_minks : ℕ
  babies_per_mink : ℕ
  coats_made : ℕ

/-- Calculates the total number of minks -/
def total_minks (p : MinkCoatProblem) : ℕ :=
  p.initial_minks * (1 + p.babies_per_mink)

/-- Calculates the number of minks used for coats -/
def minks_used_for_coats (p : MinkCoatProblem) : ℕ :=
  p.skins_per_coat * p.coats_made

/-- Calculates the number of minks set free -/
def minks_set_free (p : MinkCoatProblem) : ℕ :=
  total_minks p - minks_used_for_coats p

/-- The main theorem stating the ratio of minks set free to total minks -/
theorem minks_set_free_ratio (p : MinkCoatProblem) 
  (h1 : p.skins_per_coat = 15)
  (h2 : p.initial_minks = 30)
  (h3 : p.babies_per_mink = 6)
  (h4 : p.coats_made = 7) :
  minks_set_free p * 2 = total_minks p :=
sorry

end NUMINAMATH_CALUDE_minks_set_free_ratio_l2974_297447


namespace NUMINAMATH_CALUDE_divisor_problem_l2974_297483

theorem divisor_problem (initial_number : ℕ) (added_number : ℝ) (divisor : ℕ) : 
  initial_number = 1782452 →
  added_number = 48.00000000010186 →
  divisor = 500 →
  divisor = (Int.toNat (round (initial_number + added_number))).gcd (Int.toNat (round (initial_number + added_number))) :=
by sorry

end NUMINAMATH_CALUDE_divisor_problem_l2974_297483


namespace NUMINAMATH_CALUDE_telescope_visual_range_l2974_297452

theorem telescope_visual_range 
  (original_range : ℝ) 
  (percentage_increase : ℝ) 
  (new_range : ℝ) : 
  original_range = 50 → 
  percentage_increase = 200 → 
  new_range = original_range + (percentage_increase / 100) * original_range → 
  new_range = 150 := by
sorry

end NUMINAMATH_CALUDE_telescope_visual_range_l2974_297452


namespace NUMINAMATH_CALUDE_ball_distribution_problem_l2974_297440

/-- Represents the number of ways to distribute balls among boxes -/
def distribute_balls (total_balls : ℕ) (num_boxes : ℕ) (min_balls : ℕ → ℕ) : ℕ :=
  sorry

/-- The specific problem setup -/
def problem_setup : ℕ × ℕ × (ℕ → ℕ) :=
  (9, 3, fun i => i)

theorem ball_distribution_problem :
  let (total_balls, num_boxes, min_balls) := problem_setup
  distribute_balls total_balls num_boxes min_balls = 10 := by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_problem_l2974_297440


namespace NUMINAMATH_CALUDE_ticket_distribution_l2974_297499

/-- The number of ways to distribute identical objects among people --/
def distribution_methods (n : ℕ) (m : ℕ) : ℕ :=
  if n + 1 = m then m else 0

/-- Theorem: There are 5 ways to distribute 4 identical tickets among 5 people --/
theorem ticket_distribution : distribution_methods 4 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ticket_distribution_l2974_297499


namespace NUMINAMATH_CALUDE_infinitely_many_powers_of_two_l2974_297408

def lastDigit (n : ℕ) : ℕ := n % 10

def sequenceA : ℕ → ℕ
  | 0 => 0  -- This is a placeholder, as a₁ is actually the first term
  | n + 1 => sequenceA n + lastDigit (sequenceA n)

theorem infinitely_many_powers_of_two 
  (h₁ : sequenceA 1 % 5 ≠ 0)  -- a₁ is not divisible by 5
  (h₂ : ∀ n, sequenceA (n + 1) = sequenceA n + lastDigit (sequenceA n)) :
  ∀ k, ∃ n, ∃ m, sequenceA n = 2^m ∧ m ≥ k :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_powers_of_two_l2974_297408


namespace NUMINAMATH_CALUDE_solution_interval_l2974_297456

theorem solution_interval (x : ℝ) : (x / 4 ≤ 3 + x ∧ 3 + x < -3 * (1 + x)) ↔ x ∈ Set.Ici (-4) ∩ Set.Iio (-1) := by
  sorry

end NUMINAMATH_CALUDE_solution_interval_l2974_297456


namespace NUMINAMATH_CALUDE_tan_double_angle_l2974_297410

theorem tan_double_angle (α : Real) (h : Real.tan α = 1/3) : Real.tan (2 * α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l2974_297410


namespace NUMINAMATH_CALUDE_red_tint_percentage_after_modification_l2974_297418

/-- Calculates the percentage of red tint in a modified paint mixture -/
theorem red_tint_percentage_after_modification
  (initial_volume : ℝ)
  (initial_red_tint_percentage : ℝ)
  (added_red_tint : ℝ)
  (h_initial_volume : initial_volume = 40)
  (h_initial_red_tint_percentage : initial_red_tint_percentage = 35)
  (h_added_red_tint : added_red_tint = 10) :
  let initial_red_tint := initial_volume * initial_red_tint_percentage / 100
  let final_red_tint := initial_red_tint + added_red_tint
  let final_volume := initial_volume + added_red_tint
  final_red_tint / final_volume * 100 = 48 := by
  sorry

end NUMINAMATH_CALUDE_red_tint_percentage_after_modification_l2974_297418


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l2974_297480

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : ℕ
  num_groups : ℕ
  students_per_group : ℕ
  selected_number : ℕ
  selected_group : ℕ

/-- Theorem: In a systematic sampling of 50 students into 5 groups of 10 each,
    if the student with number 22 is selected from the third group,
    then the student with number 42 will be selected from the fifth group. -/
theorem systematic_sampling_theorem (s : SystematicSampling)
    (h1 : s.total_students = 50)
    (h2 : s.num_groups = 5)
    (h3 : s.students_per_group = 10)
    (h4 : s.selected_number = 22)
    (h5 : s.selected_group = 3) :
    s.selected_number + (s.num_groups - s.selected_group) * s.students_per_group = 42 :=
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_theorem_l2974_297480


namespace NUMINAMATH_CALUDE_marys_cake_flour_l2974_297451

/-- Given a cake recipe that requires a certain amount of flour and an amount already added,
    calculate the remaining amount to be added. -/
def remaining_flour (required : ℕ) (added : ℕ) : ℕ :=
  required - added

/-- Prove that for Mary's cake, which requires 8 cups of flour and has 2 cups already added,
    the remaining amount to be added is 6 cups. -/
theorem marys_cake_flour : remaining_flour 8 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_marys_cake_flour_l2974_297451


namespace NUMINAMATH_CALUDE_cos_difference_from_sum_l2974_297422

theorem cos_difference_from_sum (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_from_sum_l2974_297422


namespace NUMINAMATH_CALUDE_max_product_l2974_297406

theorem max_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 40) :
  x^4 * y^3 ≤ (160/7)^4 * (120/7)^3 ∧
  x^4 * y^3 = (160/7)^4 * (120/7)^3 ↔ x = 160/7 ∧ y = 120/7 := by
  sorry

end NUMINAMATH_CALUDE_max_product_l2974_297406


namespace NUMINAMATH_CALUDE_count_divisors_not_divisible_by_three_of_180_l2974_297457

def divisors_not_divisible_by_three (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (λ x => x ∣ n ∧ ¬(3 ∣ x))

theorem count_divisors_not_divisible_by_three_of_180 :
  (divisors_not_divisible_by_three 180).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_divisors_not_divisible_by_three_of_180_l2974_297457


namespace NUMINAMATH_CALUDE_bennys_work_hours_l2974_297459

/-- Given that Benny worked for 6 days and a total of 18 hours, 
    prove that he worked 3 hours each day. -/
theorem bennys_work_hours (days : ℕ) (total_hours : ℕ) 
    (h1 : days = 6) (h2 : total_hours = 18) : 
    total_hours / days = 3 := by
  sorry

end NUMINAMATH_CALUDE_bennys_work_hours_l2974_297459
