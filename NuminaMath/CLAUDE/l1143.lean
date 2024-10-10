import Mathlib

namespace gcd_1554_2405_l1143_114332

theorem gcd_1554_2405 : Nat.gcd 1554 2405 = 37 := by
  sorry

end gcd_1554_2405_l1143_114332


namespace circle_ratio_l1143_114393

theorem circle_ratio (r R : ℝ) (h : r > 0) (H : R > 0) 
  (area_condition : π * R^2 - π * r^2 = 4 * (π * r^2)) : 
  r / R = 1 / Real.sqrt 5 := by
sorry

end circle_ratio_l1143_114393


namespace total_red_stripes_l1143_114388

/-- Calculates the number of red stripes in Flag A -/
def red_stripes_a (total_stripes : ℕ) : ℕ :=
  1 + (total_stripes - 1) / 2

/-- Calculates the number of red stripes in Flag B -/
def red_stripes_b (total_stripes : ℕ) : ℕ :=
  total_stripes / 3

/-- Calculates the number of red stripes in Flag C -/
def red_stripes_c (total_stripes : ℕ) : ℕ :=
  let full_patterns := total_stripes / 9
  let remaining_stripes := total_stripes % 9
  2 * full_patterns + min remaining_stripes 2

/-- The main theorem stating the total number of red stripes -/
theorem total_red_stripes :
  let flag_a_count := 20
  let flag_b_count := 30
  let flag_c_count := 40
  let flag_a_stripes := 30
  let flag_b_stripes := 45
  let flag_c_stripes := 60
  flag_a_count * red_stripes_a flag_a_stripes +
  flag_b_count * red_stripes_b flag_b_stripes +
  flag_c_count * red_stripes_c flag_c_stripes = 1310 := by
  sorry

end total_red_stripes_l1143_114388


namespace meadow_grazing_l1143_114378

/-- Represents the amount of grass one cow eats per day -/
def daily_cow_consumption : ℝ := sorry

/-- Represents the amount of grass that grows on the meadow per day -/
def daily_grass_growth : ℝ := sorry

/-- Represents the initial amount of grass in the meadow -/
def initial_grass : ℝ := sorry

/-- Condition: 9 cows will graze the meadow empty in 4 days -/
axiom condition1 : initial_grass + 4 * daily_grass_growth = 9 * 4 * daily_cow_consumption

/-- Condition: 8 cows will graze the meadow empty in 6 days -/
axiom condition2 : initial_grass + 6 * daily_grass_growth = 8 * 6 * daily_cow_consumption

/-- The number of cows that can graze continuously in the meadow -/
def continuous_grazing_cows : ℕ := 6

theorem meadow_grazing :
  daily_grass_growth = continuous_grazing_cows * daily_cow_consumption :=
sorry

end meadow_grazing_l1143_114378


namespace equal_roots_condition_l1143_114310

theorem equal_roots_condition (m : ℝ) : 
  (∀ x, x ≠ -3 ∧ m ≠ -1 ∧ m ≠ 0 → 
    (x * (x + 3) - (m - 3)) / ((x + 3) * (m + 1)) = x / m) →
  (∃! r, ∀ x, x ≠ -3 ∧ m ≠ -1 ∧ m ≠ 0 → 
    (x * (x + 3) - (m - 3)) / ((x + 3) * (m + 1)) = x / m → x = r) ↔ 
  m = 3/2 :=
sorry

end equal_roots_condition_l1143_114310


namespace actual_time_greater_than_planned_l1143_114327

/-- Represents the running competition scenario -/
structure RunningCompetition where
  V : ℝ  -- Planned constant speed
  D : ℝ  -- Total distance
  V1 : ℝ := 1.25 * V  -- Increased speed for first half
  V2 : ℝ := 0.80 * V  -- Decreased speed for second half

/-- Theorem stating that the actual time is greater than the planned time -/
theorem actual_time_greater_than_planned (rc : RunningCompetition) 
  (h_positive_speed : rc.V > 0) (h_positive_distance : rc.D > 0) : 
  (rc.D / (2 * rc.V1) + rc.D / (2 * rc.V2)) > (rc.D / rc.V) :=
by sorry

end actual_time_greater_than_planned_l1143_114327


namespace max_min_f_on_interval_l1143_114377

def f (x : ℝ) := x^3 - 12*x + 8

theorem max_min_f_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-3) 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-3) 3, f x = max) ∧
    (∀ x ∈ Set.Icc (-3) 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-3) 3, f x = min) ∧
    max = 24 ∧ min = -6 := by
  sorry

end max_min_f_on_interval_l1143_114377


namespace electric_bus_pricing_and_optimal_plan_l1143_114315

/-- Represents the unit price of a type A electric bus in million yuan -/
def type_a_price : ℝ := 36

/-- Represents the unit price of a type B electric bus in million yuan -/
def type_b_price : ℝ := 40

/-- Represents the number of type A buses in the optimal plan -/
def optimal_type_a : ℕ := 20

/-- Represents the number of type B buses in the optimal plan -/
def optimal_type_b : ℕ := 10

/-- Represents the total cost of the optimal plan in million yuan -/
def optimal_total_cost : ℝ := 1120

theorem electric_bus_pricing_and_optimal_plan :
  (type_b_price = type_a_price + 4) ∧
  (720 / type_a_price = 800 / type_b_price) ∧
  (optimal_type_a + optimal_type_b = 30) ∧
  (optimal_type_a ≥ 10) ∧
  (optimal_type_a ≤ 2 * optimal_type_b) ∧
  (∀ m n : ℕ, m + n = 30 → m ≥ 10 → m ≤ 2 * n →
    type_a_price * m + type_b_price * n ≥ optimal_total_cost) ∧
  (optimal_total_cost = type_a_price * optimal_type_a + type_b_price * optimal_type_b) :=
by sorry


end electric_bus_pricing_and_optimal_plan_l1143_114315


namespace condition_for_equation_l1143_114325

theorem condition_for_equation (x y z : ℤ) : x = y ∧ y = z → x * (x - y) + y * (y - z) + z * (z - x) = 0 := by
  sorry

end condition_for_equation_l1143_114325


namespace max_t_value_l1143_114318

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * |x - a| - x

-- State the theorem
theorem max_t_value (a : ℝ) (h : a ≤ 1) :
  (∃ t : ℝ, t = 1 + Real.sqrt 7 ∧
   (∀ x : ℝ, x ∈ Set.Icc 0 t → -1 ≤ f a x ∧ f a x ≤ 6) ∧
   (∀ t' : ℝ, t' > t →
     ∃ x : ℝ, x ∈ Set.Icc 0 t' ∧ (f a x < -1 ∨ f a x > 6))) ∧
  (∀ a' : ℝ, a' ≤ 1 →
    ∀ t : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 t → -1 ≤ f a' x ∧ f a' x ≤ 6) →
      t ≤ 1 + Real.sqrt 7) :=
by sorry

end max_t_value_l1143_114318


namespace race_symmetry_l1143_114317

/-- Represents a car in the race -/
structure Car where
  speed : ℝ
  direction : Bool -- true for clockwise, false for counterclockwise

/-- Represents the race scenario -/
structure RaceScenario where
  A : Car
  B : Car
  C : Car
  D : Car
  track_length : ℝ
  first_AC_meet_time : ℝ
  first_BD_meet_time : ℝ
  first_AB_meet_time : ℝ

/-- The main theorem statement -/
theorem race_symmetry (race : RaceScenario) :
  race.A.direction = true ∧
  race.B.direction = true ∧
  race.C.direction = false ∧
  race.D.direction = false ∧
  race.A.speed ≠ race.B.speed ∧
  race.A.speed ≠ race.C.speed ∧
  race.A.speed ≠ race.D.speed ∧
  race.B.speed ≠ race.C.speed ∧
  race.B.speed ≠ race.D.speed ∧
  race.C.speed ≠ race.D.speed ∧
  race.first_AC_meet_time = 7 ∧
  race.first_BD_meet_time = 7 ∧
  race.first_AB_meet_time = 53 →
  ∃ (first_CD_meet_time : ℝ), first_CD_meet_time = race.first_AB_meet_time :=
by
  sorry

end race_symmetry_l1143_114317


namespace three_zeros_l1143_114347

noncomputable def f (a b x : ℝ) : ℝ := (1/2) * a * x^2 - (a^2 + a + 2) * x + (2*a + 2) * Real.log x + b

theorem three_zeros (a b : ℝ) (ha : a > 3) (hb : a^2 + a + 1 < b) (hb' : b < 2*a^2 - 2*a + 2) :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, f a b x = 0 :=
sorry

end three_zeros_l1143_114347


namespace greatest_x_with_lcm_l1143_114399

/-- Given that the least common multiple of x, 15, and 21 is 105, 
    the greatest possible value of x is 105. -/
theorem greatest_x_with_lcm (x : ℕ) : 
  Nat.lcm x (Nat.lcm 15 21) = 105 → x ≤ 105 ∧ ∃ y : ℕ, y > 105 → Nat.lcm y (Nat.lcm 15 21) > 105 :=
by sorry

end greatest_x_with_lcm_l1143_114399


namespace intersection_point_solution_l1143_114314

/-- Given two lines y = x + 1 and y = mx + n that intersect at point (1,b),
    prove that the solution to the system of equations { x + 1 = y, y - mx = n }
    is x = 1 and y = 2. -/
theorem intersection_point_solution (m n b : ℝ) :
  (∃ x y : ℝ, x + 1 = y ∧ y - m*x = n) →
  (1 + 1 = b) →
  (∀ x y : ℝ, x + 1 = y ∧ y - m*x = n → x = 1 ∧ y = 2) :=
by sorry

end intersection_point_solution_l1143_114314


namespace correct_statements_l1143_114338

theorem correct_statements (a b c d : ℝ) :
  (ab > 0 ∧ bc - ad > 0 → c / a - d / b > 0) ∧
  (a > b ∧ c > d → a - d > b - c) :=
by sorry

end correct_statements_l1143_114338


namespace solution_set_f_geq_1_range_of_a_l1143_114330

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 2

-- Theorem for the solution set of f(x) ≥ 1
theorem solution_set_f_geq_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

-- Theorem for the range of a where f(x) ≥ a^2 - a - 2 for all x in ℝ
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ a^2 - a - 2) ↔ -1 ≤ a ∧ a ≤ 2 := by sorry

end solution_set_f_geq_1_range_of_a_l1143_114330


namespace circle_plus_two_four_l1143_114311

-- Define the operation ⊕
def circle_plus (a b : ℝ) : ℝ := 5 * a + 2 * b

-- Theorem statement
theorem circle_plus_two_four : circle_plus 2 4 = 18 := by
  sorry

end circle_plus_two_four_l1143_114311


namespace marble_problem_l1143_114361

theorem marble_problem : 
  ∀ (x : ℚ), x > 0 →
  let bag1 := x
  let bag2 := 2 * x
  let bag3 := 3 * x
  let green1 := (1 / 2) * bag1
  let green2 := (1 / 3) * bag2
  let green3 := (1 / 4) * bag3
  let total_green := green1 + green2 + green3
  let total_marbles := bag1 + bag2 + bag3
  (total_green / total_marbles) = 23 / 72 :=
by
  sorry

end marble_problem_l1143_114361


namespace sum_of_roots_l1143_114380

def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

theorem sum_of_roots (a b : ℝ) (ha : f a = 1) (hb : f b = 19) : a + b = -2 := by
  sorry

end sum_of_roots_l1143_114380


namespace andrew_remaining_vacation_days_l1143_114390

/-- Calculates the remaining vacation days for an employee given their work days and vacation days taken. -/
def remaining_vacation_days (work_days : ℕ) (march_vacation : ℕ) : ℕ :=
  let earned_days := work_days / 10
  let taken_days := march_vacation + 2 * march_vacation
  earned_days - taken_days

/-- Theorem stating that Andrew has 15 remaining vacation days. -/
theorem andrew_remaining_vacation_days :
  remaining_vacation_days 300 5 = 15 := by
  sorry

#eval remaining_vacation_days 300 5

end andrew_remaining_vacation_days_l1143_114390


namespace count_D_eq_3_is_33_l1143_114305

/-- D(n) is the number of pairs of different adjacent digits in the binary representation of n -/
def D (n : ℕ) : ℕ := sorry

/-- Count of positive integers n ≤ 200 for which D(n) = 3 -/
def count_D_eq_3 : ℕ := sorry

theorem count_D_eq_3_is_33 : count_D_eq_3 = 33 := by sorry

end count_D_eq_3_is_33_l1143_114305


namespace july_birth_percentage_l1143_114304

def total_people : ℕ := 100
def born_in_july : ℕ := 13

theorem july_birth_percentage :
  (born_in_july : ℚ) / total_people * 100 = 13 := by
  sorry

end july_birth_percentage_l1143_114304


namespace tomatoes_calculation_l1143_114357

/-- The number of tomato plants -/
def num_plants : ℕ := 50

/-- The number of tomatoes produced by each plant -/
def tomatoes_per_plant : ℕ := 15

/-- The fraction of tomatoes that are dried -/
def dried_fraction : ℚ := 2 / 3

/-- The fraction of remaining tomatoes used for marinara sauce -/
def marinara_fraction : ℚ := 1 / 2

/-- The number of tomatoes left after drying and making marinara sauce -/
def tomatoes_left : ℕ := 125

theorem tomatoes_calculation :
  (num_plants * tomatoes_per_plant : ℚ) * (1 - dried_fraction) * (1 - marinara_fraction) = tomatoes_left := by
  sorry

end tomatoes_calculation_l1143_114357


namespace museum_entrance_cost_l1143_114344

theorem museum_entrance_cost (group_size : ℕ) (ticket_price : ℚ) (tax_rate : ℚ) : 
  group_size = 25 →
  ticket_price = 35.91 →
  tax_rate = 0.05 →
  (group_size : ℚ) * ticket_price * (1 + tax_rate) = 942.64 := by
sorry

end museum_entrance_cost_l1143_114344


namespace nested_fraction_equality_l1143_114349

theorem nested_fraction_equality : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = 21 / 55 := by
  sorry

end nested_fraction_equality_l1143_114349


namespace isosceles_triangle_sides_isosceles_triangle_4_exists_l1143_114309

/-- An isosceles triangle with perimeter 18 and legs twice the base length -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  perimeter_eq : base + 2 * leg = 18
  leg_eq : leg = 2 * base

/-- An isosceles triangle with one side 4 -/
structure IsoscelesTriangle4 where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  perimeter_eq : side1 + side2 + side3 = 18
  isosceles_eq : side2 = side3
  one_side_4 : side1 = 4 ∨ side2 = 4

theorem isosceles_triangle_sides (t : IsoscelesTriangle) :
  t.base = 18 / 5 ∧ t.leg = 36 / 5 := by sorry

theorem isosceles_triangle_4_exists :
  ∃ (t : IsoscelesTriangle4), t.side2 = 7 ∧ t.side3 = 7 := by sorry

end isosceles_triangle_sides_isosceles_triangle_4_exists_l1143_114309


namespace ab_over_c_equals_two_l1143_114336

theorem ab_over_c_equals_two 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_eq1 : a * b - c = 3) 
  (h_eq2 : a * b * c = 18) : 
  a * b / c = 2 := by
sorry

end ab_over_c_equals_two_l1143_114336


namespace tangent_line_at_one_max_value_of_f_l1143_114395

/-- The function f(x) defined as 2a ln x - x^2 --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.log x - x^2

/-- Theorem stating the equation of the tangent line when a = 2 --/
theorem tangent_line_at_one (a : ℝ) (h : a = 2) :
  ∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ 2 * x - y - 3 = 0 :=
sorry

/-- Theorem stating the maximum value of f(x) when a > 0 --/
theorem max_value_of_f (a : ℝ) (h : a > 0) :
  ∃ x_max : ℝ, x_max = Real.sqrt 2 ∧
    ∀ x : ℝ, x > 0 → f a x ≤ f a x_max ∧ f a x_max = Real.log 2 - 2 :=
sorry

end tangent_line_at_one_max_value_of_f_l1143_114395


namespace quadratic_roots_sum_l1143_114350

theorem quadratic_roots_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 1 = 0 → x₂^2 - 3*x₂ + 1 = 0 → x₁^2 + 3*x₁*x₂ + x₂^2 = 10 := by
  sorry

end quadratic_roots_sum_l1143_114350


namespace exists_integer_divisible_by_15_with_sqrt_between_30_and_30_5_l1143_114307

theorem exists_integer_divisible_by_15_with_sqrt_between_30_and_30_5 :
  ∃ n : ℕ+, 15 ∣ n ∧ 30 ≤ (n : ℝ).sqrt ∧ (n : ℝ).sqrt < 30.5 := by
  sorry

end exists_integer_divisible_by_15_with_sqrt_between_30_and_30_5_l1143_114307


namespace leo_current_weight_l1143_114301

/-- Leo's current weight in pounds -/
def leo_weight : ℝ := 80

/-- Kendra's current weight in pounds -/
def kendra_weight : ℝ := 140 - leo_weight

/-- The combined weight of Leo and Kendra in pounds -/
def combined_weight : ℝ := 140

theorem leo_current_weight :
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = combined_weight) →
  leo_weight = 80 := by
sorry

end leo_current_weight_l1143_114301


namespace blue_cap_cost_l1143_114367

/-- The cost of items before applying a discount --/
structure PreDiscountCost where
  tshirt : ℕ
  backpack : ℕ
  bluecap : ℕ

/-- The total cost after applying a discount --/
def total_after_discount (cost : PreDiscountCost) (discount : ℕ) : ℕ :=
  cost.tshirt + cost.backpack + cost.bluecap - discount

/-- The theorem stating the cost of the blue cap --/
theorem blue_cap_cost (cost : PreDiscountCost) (discount : ℕ) :
  cost.tshirt = 30 →
  cost.backpack = 10 →
  discount = 2 →
  total_after_discount cost discount = 43 →
  cost.bluecap = 5 := by
  sorry

#check blue_cap_cost

end blue_cap_cost_l1143_114367


namespace arithmetic_sequence_ratio_l1143_114374

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  S : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, 2 * (S (n + 1) - S n) = S (n + 2) - S n

/-- Theorem: If S_5 : S_10 = 2 : 3, then S_15 : S_5 = 3 : 2 -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequence) 
  (h : seq.S 5 / seq.S 10 = 2 / 3) : 
  seq.S 15 / seq.S 5 = 3 / 2 := by
  sorry

end arithmetic_sequence_ratio_l1143_114374


namespace a_range_l1143_114321

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

/-- The condition for the function -/
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0

/-- The theorem stating the range of a -/
theorem a_range (a : ℝ) :
  (strictly_increasing (f a)) ↔ (3/2 ≤ a ∧ a < 2) :=
sorry

end a_range_l1143_114321


namespace high_school_student_distribution_l1143_114363

theorem high_school_student_distribution :
  ∀ (freshmen sophomores juniors seniors : ℕ),
    freshmen + sophomores + juniors + seniors = 800 →
    juniors = 216 →
    sophomores = 200 →
    seniors = 160 →
    freshmen - sophomores = 24 := by
  sorry

end high_school_student_distribution_l1143_114363


namespace min_ratio_folded_to_total_area_ratio_two_thirds_achievable_min_ratio_is_two_thirds_l1143_114306

/-- Represents a point on the square tablecloth -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the square tablecloth with dark spots -/
structure Tablecloth where
  side_length : ℝ
  spots : Set Point
  total_area : ℝ
  folded_area : ℝ

/-- The ratio of folded visible area to total area is at least 2/3 -/
theorem min_ratio_folded_to_total_area (t : Tablecloth) : 
  t.folded_area / t.total_area ≥ 2/3 := by
  sorry

/-- The ratio of 2/3 is achievable -/
theorem ratio_two_thirds_achievable : 
  ∃ t : Tablecloth, t.folded_area / t.total_area = 2/3 := by
  sorry

/-- The minimum ratio of folded visible area to total area is exactly 2/3 -/
theorem min_ratio_is_two_thirds : 
  (∀ t : Tablecloth, t.folded_area / t.total_area ≥ 2/3) ∧
  (∃ t : Tablecloth, t.folded_area / t.total_area = 2/3) := by
  sorry

end min_ratio_folded_to_total_area_ratio_two_thirds_achievable_min_ratio_is_two_thirds_l1143_114306


namespace pure_imaginary_product_l1143_114300

theorem pure_imaginary_product (x : ℝ) : 
  (∃ b : ℝ, (x + 2*I)*((x + 2) + 2*I)*((x + 4) + 2*I) = b*I) ↔ (x = -4 ∨ x = 1) :=
by sorry

end pure_imaginary_product_l1143_114300


namespace carpet_length_proof_l1143_114339

theorem carpet_length_proof (length width diagonal : ℝ) : 
  length > 0 ∧ width > 0 ∧
  length * width = 60 ∧
  diagonal + length = 5 * width ∧
  diagonal^2 = length^2 + width^2 →
  length = 2 * Real.sqrt 30 :=
by sorry

end carpet_length_proof_l1143_114339


namespace flame_time_calculation_l1143_114368

/-- Represents the duration of one minute in seconds -/
def minute_duration : ℕ := 60

/-- Represents the interval between weapon fires in seconds -/
def fire_interval : ℕ := 15

/-- Represents the duration of each flame shot in seconds -/
def flame_duration : ℕ := 5

/-- Calculates the total time spent shooting flames in one minute -/
def flame_time_per_minute : ℕ := (minute_duration / fire_interval) * flame_duration

theorem flame_time_calculation :
  flame_time_per_minute = 20 := by sorry

end flame_time_calculation_l1143_114368


namespace professor_count_l1143_114352

theorem professor_count (p : ℕ) 
  (h1 : 6480 % p = 0)  -- 6480 is divisible by p
  (h2 : 11200 % (p + 3) = 0)  -- 11200 is divisible by (p + 3)
  (h3 : (6480 : ℚ) / p < (11200 : ℚ) / (p + 3))  -- grades per professor increased
  : p = 5 := by
  sorry

end professor_count_l1143_114352


namespace soda_bottle_difference_l1143_114385

theorem soda_bottle_difference :
  let diet_soda : ℕ := 4
  let regular_soda : ℕ := 83
  regular_soda - diet_soda = 79 :=
by sorry

end soda_bottle_difference_l1143_114385


namespace mobile_payment_probability_l1143_114355

def group_size : ℕ := 10

def mobile_payment_prob (p : ℝ) : ℝ := p

def is_independent (p : ℝ) : Prop := true

def num_mobile_users (X : ℕ) : ℕ := X

def variance (X : ℕ) (p : ℝ) : ℝ := group_size * p * (1 - p)

def prob_X_eq (k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose group_size k : ℝ) * p^k * (1 - p)^(group_size - k)

theorem mobile_payment_probability :
  ∀ p : ℝ,
    0 ≤ p ∧ p ≤ 1 →
    is_independent p →
    variance (num_mobile_users X) p = 2.4 →
    prob_X_eq 4 p < prob_X_eq 6 p →
    p = 0.6 := by
  sorry

end mobile_payment_probability_l1143_114355


namespace caroline_lassi_production_l1143_114346

/-- Given that Caroline can make 15 lassis out of 3 mangoes, 
    prove that she can make 75 lassis out of 15 mangoes. -/
theorem caroline_lassi_production :
  (∃ (lassis_per_3_mangoes : ℕ), lassis_per_3_mangoes = 15) →
  (∃ (lassis_per_15_mangoes : ℕ), lassis_per_15_mangoes = 75) :=
by
  sorry

end caroline_lassi_production_l1143_114346


namespace books_in_pile_A_l1143_114394

/-- Given three piles of books with the following properties:
  - The total number of books is 240
  - Pile A has 30 more than three times the books in pile B
  - Pile C has 15 fewer books than pile B
  Prove that pile A contains 165 books. -/
theorem books_in_pile_A (total : ℕ) (books_B : ℕ) (books_A : ℕ) (books_C : ℕ) : 
  total = 240 →
  books_A = 3 * books_B + 30 →
  books_C = books_B - 15 →
  books_A + books_B + books_C = total →
  books_A = 165 := by
sorry

end books_in_pile_A_l1143_114394


namespace integer_roots_imply_n_values_l1143_114373

theorem integer_roots_imply_n_values (n : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ x^2 - 6*x - 4*n^2 - 32*n = 0 ∧ y^2 - 6*y - 4*n^2 - 32*n = 0) →
  (n = 10 ∨ n = 0 ∨ n = -8 ∨ n = -18) :=
by sorry

end integer_roots_imply_n_values_l1143_114373


namespace investment_interest_rate_l1143_114331

/-- Proves that given the specified conditions, the interest rate for the second part of an investment is 5% -/
theorem investment_interest_rate 
  (total_investment : ℕ)
  (first_part : ℕ)
  (first_rate : ℚ)
  (total_interest : ℕ)
  (h1 : total_investment = 3400)
  (h2 : first_part = 1300)
  (h3 : first_rate = 3 / 100)
  (h4 : total_interest = 144) :
  let second_part := total_investment - first_part
  let first_interest := (first_part : ℚ) * first_rate
  let second_interest := (total_interest : ℚ) - first_interest
  let second_rate := second_interest / (second_part : ℚ)
  second_rate = 5 / 100 := by
sorry

end investment_interest_rate_l1143_114331


namespace total_blocks_is_55_l1143_114329

/-- Calculates the total number of blocks in Thomas's stacks --/
def total_blocks : ℕ :=
  let first_stack := 7
  let second_stack := first_stack + 3
  let third_stack := second_stack - 6
  let fourth_stack := third_stack + 10
  let fifth_stack := 2 * second_stack
  first_stack + second_stack + third_stack + fourth_stack + fifth_stack

/-- Theorem stating that the total number of blocks is 55 --/
theorem total_blocks_is_55 : total_blocks = 55 := by
  sorry

end total_blocks_is_55_l1143_114329


namespace intersection_A_complement_B_l1143_114387

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x ≥ 1}

-- Define set B
def B : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (Set.compl B) = {x | 1 ≤ x ∧ x < 2} := by sorry

end intersection_A_complement_B_l1143_114387


namespace max_at_2_implies_c_6_l1143_114362

/-- The function f(x) = x(x-c)² has a maximum value at x = 2 -/
def has_max_at_2 (c : ℝ) : Prop :=
  let f := fun x => x * (x - c)^2
  ∀ x, f x ≤ f 2

/-- Theorem: If f(x) = x(x-c)² has a maximum value at x = 2, then c = 6 -/
theorem max_at_2_implies_c_6 : 
  ∀ c : ℝ, has_max_at_2 c → c = 6 := by
  sorry

end max_at_2_implies_c_6_l1143_114362


namespace shirt_price_calculation_l1143_114335

theorem shirt_price_calculation (total_cost sweater_price shirt_price : ℝ) :
  total_cost = 80.34 →
  sweater_price - shirt_price = 7.43 →
  total_cost = sweater_price + shirt_price →
  shirt_price = 36.455 := by
sorry

end shirt_price_calculation_l1143_114335


namespace early_arrival_speed_l1143_114383

/-- Represents the travel scenario for Mrs. Early --/
structure TravelScenario where
  speed : ℝ
  timeDifference : ℝ  -- in hours, positive for early, negative for late

/-- Calculates the required speed to arrive exactly on time --/
def exactTimeSpeed (scenario1 scenario2 : TravelScenario) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the correct speed for Mrs. Early to arrive on time --/
theorem early_arrival_speed : 
  let scenario1 : TravelScenario := { speed := 50, timeDifference := -1/15 }
  let scenario2 : TravelScenario := { speed := 70, timeDifference := 1/12 }
  let requiredSpeed := exactTimeSpeed scenario1 scenario2
  57 < requiredSpeed ∧ requiredSpeed < 58 := by
  sorry

end early_arrival_speed_l1143_114383


namespace completing_square_equivalence_l1143_114323

theorem completing_square_equivalence (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) := by
  sorry

end completing_square_equivalence_l1143_114323


namespace sin_cos_pi_12_l1143_114397

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end sin_cos_pi_12_l1143_114397


namespace p_sufficient_not_necessary_for_q_l1143_114386

-- Define condition p
def condition_p (x y : ℝ) : Prop := x > 2 ∧ y > 3

-- Define condition q
def condition_q (x y : ℝ) : Prop := x + y > 5 ∧ x * y > 6

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x y : ℝ, condition_p x y → condition_q x y) ∧
  ¬(∀ x y : ℝ, condition_q x y → condition_p x y) :=
by sorry

end p_sufficient_not_necessary_for_q_l1143_114386


namespace broken_crayons_percentage_l1143_114382

theorem broken_crayons_percentage (total : ℕ) (slightly_used : ℕ) :
  total = 120 →
  slightly_used = 56 →
  (total / 3 : ℚ) + slightly_used + (total / 5 : ℚ) = total →
  (total / 5 : ℚ) / total * 100 = 20 := by
  sorry

end broken_crayons_percentage_l1143_114382


namespace point_in_quadrant_iv_l1143_114343

/-- Given a system of equations x - y = a and 6x + 5y = -1, where x = 1,
    prove that the point (a, y) is in Quadrant IV -/
theorem point_in_quadrant_iv (a : ℚ) : 
  let x : ℚ := 1
  let y : ℚ := -7/5
  (x - y = a) → (6 * x + 5 * y = -1) → (a > 0 ∧ y < 0) := by
  sorry

#check point_in_quadrant_iv

end point_in_quadrant_iv_l1143_114343


namespace billy_ice_trays_l1143_114372

theorem billy_ice_trays (ice_cubes_per_tray : ℕ) (total_ice_cubes : ℕ) 
  (h1 : ice_cubes_per_tray = 9)
  (h2 : total_ice_cubes = 72) :
  total_ice_cubes / ice_cubes_per_tray = 8 := by
  sorry

end billy_ice_trays_l1143_114372


namespace total_money_proof_l1143_114348

/-- The total amount of money p, q, and r have among themselves -/
def total_amount (r_amount : ℚ) (r_fraction : ℚ) : ℚ :=
  r_amount / r_fraction

theorem total_money_proof (r_amount : ℚ) (h1 : r_amount = 2000) 
  (r_fraction : ℚ) (h2 : r_fraction = 2/3) : 
  total_amount r_amount r_fraction = 5000 := by
  sorry

#check total_money_proof

end total_money_proof_l1143_114348


namespace three_digit_number_divisible_by_seven_l1143_114370

theorem three_digit_number_divisible_by_seven (a b : ℕ) 
  (h1 : a ≥ 1 ∧ a ≤ 9) 
  (h2 : b ≥ 0 ∧ b ≤ 9) 
  (h3 : (a + b + b) % 7 = 0) : 
  ∃ k : ℕ, (100 * a + 10 * b + b) = 7 * k :=
sorry

end three_digit_number_divisible_by_seven_l1143_114370


namespace line_slope_proof_l1143_114322

theorem line_slope_proof (x y : ℝ) : 
  (((Real.sqrt 3) / 3) * x + y - 7 = 0) → 
  (∃ m : ℝ, m = -(Real.sqrt 3) / 3 ∧ y = m * x + 7) := by
  sorry

end line_slope_proof_l1143_114322


namespace jewelry_sweater_difference_l1143_114369

theorem jewelry_sweater_difference (sweater_cost initial_fraction remaining : ℚ) :
  sweater_cost = 40 →
  initial_fraction = 1/4 →
  remaining = 20 →
  let initial_money := sweater_cost / initial_fraction
  let jewelry_cost := initial_money - sweater_cost - remaining
  jewelry_cost - sweater_cost = 60 := by
  sorry

end jewelry_sweater_difference_l1143_114369


namespace solve_for_k_l1143_114376

-- Define the polynomials
def p (x y k : ℝ) : ℝ := x^3 - 2*k*x*y
def q (x y : ℝ) : ℝ := y^2 + 4*x*y

-- Define the condition that the difference doesn't contain xy term
def no_xy_term (k : ℝ) : Prop :=
  ∀ x y, ∃ a b c, p x y k - q x y = a*x^3 + b*y^2 + c

-- State the theorem
theorem solve_for_k :
  ∃ k : ℝ, no_xy_term k ∧ k = -2 :=
sorry

end solve_for_k_l1143_114376


namespace expression_evaluation_l1143_114337

theorem expression_evaluation : (-1)^3 + 4 * (-2) - 3 / (-3) = -8 := by
  sorry

end expression_evaluation_l1143_114337


namespace inverse_difference_simplification_l1143_114308

theorem inverse_difference_simplification (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : 3 * x - y / 3 ≠ 0) :
  (3 * x - y / 3)⁻¹ * ((3 * x)⁻¹ - (y / 3)⁻¹) = -(x * y)⁻¹ :=
by sorry

end inverse_difference_simplification_l1143_114308


namespace arithmetic_sequence_problem_l1143_114333

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) (m : ℕ) 
  (h_m : m > 1)
  (h_condition : seq.a (m - 1) + seq.a (m + 1) - (seq.a m)^2 = 0)
  (h_sum : seq.S (2 * m - 1) = 38) :
  m = 10 := by
  sorry

end arithmetic_sequence_problem_l1143_114333


namespace work_ratio_l1143_114342

theorem work_ratio (a b : ℝ) (ha : a = 8) (hab : 1/a + 1/b = 0.375) : b/a = 1/2 := by
  sorry

end work_ratio_l1143_114342


namespace daniel_video_game_collection_l1143_114353

/-- The number of video games Daniel bought for $12 each -/
def games_at_12 : ℕ := 80

/-- The price of the first group of games -/
def price_1 : ℕ := 12

/-- The price of the second group of games -/
def price_2 : ℕ := 7

/-- The price of the third group of games -/
def price_3 : ℕ := 3

/-- The total amount Daniel spent on all games -/
def total_spent : ℕ := 2290

/-- Theorem stating the total number of video games in Daniel's collection -/
theorem daniel_video_game_collection :
  ∃ (games_at_7 games_at_3 : ℕ),
    games_at_7 = games_at_3 ∧
    games_at_12 * price_1 + games_at_7 * price_2 + games_at_3 * price_3 = total_spent ∧
    games_at_12 + games_at_7 + games_at_3 = 346 :=
by sorry

end daniel_video_game_collection_l1143_114353


namespace special_polynomial_sum_l1143_114384

theorem special_polynomial_sum (d₁ d₂ d₃ d₄ e₁ e₂ e₃ e₄ : ℝ) 
  (h : ∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + d₁*x + e₁)*(x^2 + d₂*x + e₂)*(x^2 + d₃*x + e₃)*(x^2 + d₄*x + e₄)) : 
  d₁*e₁ + d₂*e₂ + d₃*e₃ + d₄*e₄ = -1 := by
  sorry

end special_polynomial_sum_l1143_114384


namespace semicircle_inscriptions_l1143_114326

theorem semicircle_inscriptions (D : ℝ) (N : ℕ) (h : N > 0) : 
  let r := D / (2 * N)
  let R := N * r
  let A := N * (π * r^2 / 2)
  let B := π * R^2 / 2 - A
  A / B = 2 / 25 → N = 14 := by
sorry

end semicircle_inscriptions_l1143_114326


namespace fraction_equality_l1143_114360

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 5 / 6) 
  (h2 : s / u = 7 / 18) : 
  (5 * p * s - 6 * q * u) / (7 * q * u - 10 * p * s) = -473 / 406 := by
  sorry

end fraction_equality_l1143_114360


namespace polynomial_negative_l1143_114354

theorem polynomial_negative (a : ℝ) (x : ℝ) (h : 0 < x ∧ x < a) : 
  (a - x)^6 - 3*a*(a - x)^5 + (5/2)*a^2*(a - x)^4 - (1/2)*a^4*(a - x)^2 < 0 := by
  sorry

end polynomial_negative_l1143_114354


namespace trapezoid_area_l1143_114396

/-- A trapezoid bounded by y = 2x, y = 12, y = 8, and the y-axis -/
structure Trapezoid where
  /-- The line y = 2x -/
  line : ℝ → ℝ
  /-- The upper bound y = 12 -/
  upper_bound : ℝ
  /-- The lower bound y = 8 -/
  lower_bound : ℝ
  /-- The line is y = 2x -/
  line_eq : ∀ x, line x = 2 * x
  /-- The upper bound is 12 -/
  upper_eq : upper_bound = 12
  /-- The lower bound is 8 -/
  lower_eq : lower_bound = 8

/-- The area of the trapezoid -/
def area (t : Trapezoid) : ℝ := sorry

/-- Theorem: The area of the specified trapezoid is 20 square units -/
theorem trapezoid_area : ∀ t : Trapezoid, area t = 20 := by sorry

end trapezoid_area_l1143_114396


namespace scientific_notation_equivalence_l1143_114381

theorem scientific_notation_equivalence :
  ∃ (a : ℝ) (n : ℤ), 
    27017800000000 = a * (10 : ℝ) ^ n ∧ 
    1 ≤ a ∧ a < 10 ∧
    n = 13 ∧
    a = 2.70178 := by
  sorry

end scientific_notation_equivalence_l1143_114381


namespace student_money_proof_l1143_114340

/-- The amount of money (in rubles) the student has after buying 11 pens -/
def remaining_after_11 : ℝ := 8

/-- The additional amount (in rubles) needed to buy 15 pens -/
def additional_for_15 : ℝ := 12.24

/-- The cost of one pen in rubles -/
noncomputable def pen_cost : ℝ :=
  (additional_for_15 + remaining_after_11) / (15 - 11)

/-- The initial amount of money the student had in rubles -/
noncomputable def initial_amount : ℝ :=
  11 * pen_cost + remaining_after_11

theorem student_money_proof :
  initial_amount = 63.66 := by sorry

end student_money_proof_l1143_114340


namespace exists_divisible_figure_l1143_114334

/-- Represents a geometric shape --/
structure Shape :=
  (area : ℝ)

/-- Represents a T-shaped piece --/
def T_shape : Shape :=
  { area := 3 }

/-- Represents the set of five specific pieces --/
def five_pieces : Finset Shape :=
  sorry

/-- A figure that can be divided into different sets of pieces --/
structure DivisibleFigure :=
  (total_area : ℝ)
  (can_divide_into_four_T : Prop)
  (can_divide_into_five_pieces : Prop)

/-- The existence of a figure that satisfies both division conditions --/
theorem exists_divisible_figure : 
  ∃ (fig : DivisibleFigure), 
    fig.can_divide_into_four_T ∧ 
    fig.can_divide_into_five_pieces :=
sorry

end exists_divisible_figure_l1143_114334


namespace remainder_3_2015_mod_13_l1143_114328

theorem remainder_3_2015_mod_13 : ∃ k : ℤ, 3^2015 = 13 * k + 9 :=
by
  sorry

end remainder_3_2015_mod_13_l1143_114328


namespace circle_c_properties_l1143_114366

-- Define the circle C
structure CircleC where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line l
structure LineL where
  b : ℝ

-- Define point N
def pointN : ℝ × ℝ := (0, 3)

-- Define the theorem
theorem circle_c_properties (c : CircleC) (l : LineL) :
  -- Condition 1: Circle C's center is on the line x - 2y = 0
  c.center.1 = 2 * c.center.2 →
  -- Condition 2: Circle C is tangent to the positive half of the y-axis
  c.center.2 > 0 →
  -- Condition 3: The chord obtained by intersecting the x-axis is 2√3 long
  2 * Real.sqrt 3 = 2 * Real.sqrt (c.radius^2 - c.center.2^2) →
  -- Condition 4: Line l intersects circle C at two points
  ∃ (A B : ℝ × ℝ), A ≠ B ∧ 
    (A.1 - c.center.1)^2 + (A.2 - c.center.2)^2 = c.radius^2 ∧
    (B.1 - c.center.1)^2 + (B.2 - c.center.2)^2 = c.radius^2 ∧
    A.2 = -2 * A.1 + l.b ∧ B.2 = -2 * B.1 + l.b →
  -- Condition 5: The circle with AB as its diameter passes through the origin
  ∃ (A B : ℝ × ℝ), A ≠ B ∧ 
    (A.1 - c.center.1)^2 + (A.2 - c.center.2)^2 = c.radius^2 ∧
    (B.1 - c.center.1)^2 + (B.2 - c.center.2)^2 = c.radius^2 ∧
    A.2 = -2 * A.1 + l.b ∧ B.2 = -2 * B.1 + l.b ∧
    A.1 * B.1 + A.2 * B.2 = 0 →
  -- Condition 6-9 are implicitly included in the structure of CircleC
  -- Prove:
  -- 1. The standard equation of circle C is (x - 2)² + (y - 1)² = 4
  ((c.center = (2, 1) ∧ c.radius = 2) ∨
  -- 2. The value of b in the equation y = -2x + b is (5 ± √15) / 2
   (l.b = (5 + Real.sqrt 15) / 2 ∨ l.b = (5 - Real.sqrt 15) / 2)) ∧
  -- 3. The y-coordinate of the center of circle C is in the range (0, 2]
   (0 < c.center.2 ∧ c.center.2 ≤ 2) :=
by sorry

end circle_c_properties_l1143_114366


namespace point_A_in_transformed_plane_l1143_114398

/-- The similarity transformation coefficient -/
def k : ℚ := 1/2

/-- The original plane equation: 4x - 3y + 5z - 10 = 0 -/
def plane_a (x y z : ℚ) : Prop := 4*x - 3*y + 5*z - 10 = 0

/-- The transformed plane equation: 4x - 3y + 5z - 5 = 0 -/
def plane_a' (x y z : ℚ) : Prop := 4*x - 3*y + 5*z - 5 = 0

/-- Point A -/
def point_A : ℚ × ℚ × ℚ := (1/4, 1/3, 1)

/-- Theorem: Point A belongs to the image of plane a under the similarity transformation -/
theorem point_A_in_transformed_plane :
  plane_a' point_A.1 point_A.2.1 point_A.2.2 :=
by sorry

end point_A_in_transformed_plane_l1143_114398


namespace subtraction_value_l1143_114359

theorem subtraction_value (x y : ℝ) : 
  (x - 5) / 7 = 7 → (x - y) / 8 = 6 → y = 6 := by
sorry

end subtraction_value_l1143_114359


namespace function_value_at_negative_pi_fourth_l1143_114345

theorem function_value_at_negative_pi_fourth 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, f x = a * Real.tan x - b * Real.sin x + 1) 
  (h2 : f (π/4) = 7) : 
  f (-π/4) = -5 := by
sorry

end function_value_at_negative_pi_fourth_l1143_114345


namespace revolution_volume_formula_l1143_114371

/-- Region P in the coordinate plane -/
def P : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |6 - p.1| + p.2 ≤ 8 ∧ 4 * p.2 - p.1 ≥ 20}

/-- The line around which P is revolved -/
def revolveLine (x y : ℝ) : Prop := 4 * y - x = 20

/-- The volume of the solid formed by revolving P around the line -/
noncomputable def revolutionVolume : ℝ := sorry

theorem revolution_volume_formula :
  revolutionVolume = 24 * Real.pi / (85 * Real.sqrt 3741) := by sorry

end revolution_volume_formula_l1143_114371


namespace inequality_solution_set_l1143_114391

theorem inequality_solution_set (x : ℝ) :
  (3 * x - 1) / (x - 2) > 0 ↔ x < 1/3 ∨ x > 2 :=
sorry

end inequality_solution_set_l1143_114391


namespace locus_of_vertex_A_l1143_114316

def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  B = (-6, 0) ∧ C = (6, 0)

def angle_condition (A B C : ℝ) : Prop :=
  Real.sin B - Real.sin C = (1/2) * Real.sin A

def locus_equation (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 27 = 1 ∧ x < -3

theorem locus_of_vertex_A (A B C : ℝ × ℝ) (angleA angleB angleC : ℝ) :
  triangle_ABC A B C →
  angle_condition angleA angleB angleC →
  locus_equation A.1 A.2 :=
sorry

end locus_of_vertex_A_l1143_114316


namespace gcd_problem_l1143_114302

theorem gcd_problem : Nat.gcd 7260 540 - 12 + 5 = 53 := by
  sorry

end gcd_problem_l1143_114302


namespace mn_value_l1143_114364

theorem mn_value (m n : ℤ) (h : |3*m - 6| + (n + 4)^2 = 0) : m * n = -8 := by
  sorry

end mn_value_l1143_114364


namespace number_problem_l1143_114375

theorem number_problem (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 25) : 
  0.40 * N = 300 := by
  sorry

end number_problem_l1143_114375


namespace triangle_area_from_perimeter_and_inradius_l1143_114389

/-- The area of a triangle with given perimeter and inradius -/
theorem triangle_area_from_perimeter_and_inradius 
  (perimeter : ℝ) (inradius : ℝ) (area : ℝ) 
  (h_perimeter : perimeter = 20) 
  (h_inradius : inradius = 3) : 
  area = 30 := by
  sorry

end triangle_area_from_perimeter_and_inradius_l1143_114389


namespace smallest_number_l1143_114351

/-- Given three numbers A, B, and C with the following properties:
  - A is 38 greater than 18
  - B is 26 less than A
  - C is the quotient of B divided by 3
  Prove that C is the smallest among A, B, and C. -/
theorem smallest_number (A B C : ℤ) 
  (h1 : A = 18 + 38)
  (h2 : B = A - 26)
  (h3 : C = B / 3) :
  C ≤ A ∧ C ≤ B := by
  sorry

end smallest_number_l1143_114351


namespace hostel_problem_l1143_114392

/-- Calculates the number of men who left a hostel given the initial conditions and the new duration of provisions. -/
def men_who_left (initial_men : ℕ) (initial_days : ℕ) (new_days : ℕ) : ℕ :=
  initial_men - (initial_men * initial_days) / new_days

/-- Proves that 50 men left the hostel under the given conditions. -/
theorem hostel_problem : men_who_left 250 48 60 = 50 := by
  sorry

end hostel_problem_l1143_114392


namespace dhoni_rent_percentage_l1143_114303

theorem dhoni_rent_percentage (rent_percentage : ℝ) 
  (h1 : rent_percentage > 0)
  (h2 : rent_percentage < 100)
  (h3 : rent_percentage + (rent_percentage - 10) + 52.5 = 100) :
  rent_percentage = 28.75 := by
sorry

end dhoni_rent_percentage_l1143_114303


namespace one_third_percent_of_180_l1143_114379

theorem one_third_percent_of_180 : (1 / 3 : ℚ) / 100 * 180 = 0.6 := by sorry

end one_third_percent_of_180_l1143_114379


namespace inequalities_problem_l1143_114341

theorem inequalities_problem (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) :
  (a + b < a * b) ∧
  (b/a + a/b > 2) ∧
  (a > b) ∧
  (abs a < abs b) := by
  sorry

end inequalities_problem_l1143_114341


namespace min_value_of_function_l1143_114319

theorem min_value_of_function (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + y = 2) :
  (2 / x + 1 / y) ≥ 3 / 2 + Real.sqrt 2 :=
sorry

end min_value_of_function_l1143_114319


namespace inequality_range_l1143_114324

theorem inequality_range (x : ℝ) : 
  (∀ p : ℝ, 0 ≤ p ∧ p ≤ 4 → x^2 + p*x > 4*x + p - 3) ↔ (x > 3 ∨ x < -1) :=
by sorry

end inequality_range_l1143_114324


namespace mikes_pears_l1143_114358

/-- Given that Jason picked 7 pears and the total number of pears picked was 15,
    prove that Mike picked 8 pears. -/
theorem mikes_pears (jason_pears total_pears : ℕ) 
    (h1 : jason_pears = 7)
    (h2 : total_pears = 15) :
    total_pears - jason_pears = 8 := by
  sorry

end mikes_pears_l1143_114358


namespace divisibility_implies_sum_ten_l1143_114320

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number (C F : ℕ) : ℕ := C * 1000000 + 854000 + F * 100 + 72

theorem divisibility_implies_sum_ten (C F : ℕ) 
  (h_C : is_digit C) (h_F : is_digit F) 
  (h_div_8 : number C F % 8 = 0) 
  (h_div_9 : number C F % 9 = 0) : 
  C + F = 10 := by
sorry

end divisibility_implies_sum_ten_l1143_114320


namespace coefficient_x_squared_in_expansion_l1143_114365

theorem coefficient_x_squared_in_expansion :
  (Finset.range 6).sum (fun k => (Nat.choose 5 k) * 2^k * (if k = 2 then 1 else 0)) = 40 :=
sorry

end coefficient_x_squared_in_expansion_l1143_114365


namespace expression_value_l1143_114313

theorem expression_value :
  let x : ℕ := 3
  5^3 - 2^x * 3 + 4^2 = 117 := by
  sorry

end expression_value_l1143_114313


namespace total_pictures_calculation_l1143_114356

/-- The number of pictures that can be contained in one album -/
def pictures_per_album : ℕ := 20

/-- The number of albums needed -/
def albums_needed : ℕ := 24

/-- The total number of pictures -/
def total_pictures : ℕ := pictures_per_album * albums_needed

theorem total_pictures_calculation :
  total_pictures = 480 :=
by sorry

end total_pictures_calculation_l1143_114356


namespace prime_condition_theorem_l1143_114312

def satisfies_condition (p : ℕ) : Prop :=
  Nat.Prime p ∧
  ∀ q : ℕ, Nat.Prime q → q < p →
    ∀ k r : ℕ, p = k * q + r → 0 ≤ r → r < q →
      ∀ a : ℕ, a > 1 → ¬(a^2 ∣ r)

theorem prime_condition_theorem :
  {p : ℕ | satisfies_condition p} = {2, 3, 5, 7, 13} :=
sorry

end prime_condition_theorem_l1143_114312
