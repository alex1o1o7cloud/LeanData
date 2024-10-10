import Mathlib

namespace fruit_cost_proof_l516_51694

/-- Given the cost of fruits, prove the cost of a different combination -/
theorem fruit_cost_proof (cost_six_apples_three_oranges : ℝ) 
                         (cost_one_apple : ℝ) : 
  cost_six_apples_three_oranges = 1.77 →
  cost_one_apple = 0.21 →
  2 * cost_one_apple + 5 * ((cost_six_apples_three_oranges - 6 * cost_one_apple) / 3) = 1.27 := by
  sorry

end fruit_cost_proof_l516_51694


namespace function_extrema_implies_a_range_l516_51625

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

-- State the theorem
theorem function_extrema_implies_a_range (a : ℝ) :
  (∃ (x_max x_min : ℝ), ∀ (x : ℝ), f a x ≤ f a x_max ∧ f a x_min ≤ f a x) →
  a < -3 ∨ a > 6 := by
  sorry

end function_extrema_implies_a_range_l516_51625


namespace smallest_number_l516_51620

theorem smallest_number (a b c d : ℝ) (ha : a = 0) (hb : b = -1/2) (hc : c = -1) (hd : d = Real.sqrt 2) :
  c ≤ a ∧ c ≤ b ∧ c ≤ d :=
sorry

end smallest_number_l516_51620


namespace at_least_one_non_negative_l516_51621

def f (x : ℝ) : ℝ := x^2 - x

theorem at_least_one_non_negative (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m * n > 1) :
  f m ≥ 0 ∨ f n ≥ 0 := by
  sorry

end at_least_one_non_negative_l516_51621


namespace inequality_equivalence_l516_51633

theorem inequality_equivalence (x y : ℝ) : 
  (y + x > |x/2|) ↔ ((x ≥ 0 ∧ y > -x/2) ∨ (x < 0 ∧ y > -3*x/2)) := by
  sorry

end inequality_equivalence_l516_51633


namespace two_digit_perfect_squares_divisible_by_four_l516_51604

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem two_digit_perfect_squares_divisible_by_four :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, is_two_digit n ∧ is_perfect_square n ∧ n % 4 = 0) ∧
    s.card = 3 := by
  sorry

end two_digit_perfect_squares_divisible_by_four_l516_51604


namespace program1_output_program2_output_l516_51679

-- Define a type to represent the state of the program
structure ProgramState where
  a : Int
  b : Int
  c : Int

-- Function to simulate the first program
def program1 (initial : ProgramState) : ProgramState :=
  { a := initial.b
  , b := initial.c
  , c := initial.c }

-- Function to simulate the second program
def program2 (initial : ProgramState) : ProgramState :=
  { a := initial.b
  , b := initial.c
  , c := initial.b }

-- Theorem for the first program
theorem program1_output :
  let initial := ProgramState.mk 3 (-5) 8
  let final := program1 initial
  final.a = -5 ∧ final.b = 8 ∧ final.c = 8 := by sorry

-- Theorem for the second program
theorem program2_output :
  let initial := ProgramState.mk 3 (-5) 8
  let final := program2 initial
  final.a = -5 ∧ final.b = 8 ∧ final.c = -5 := by sorry

end program1_output_program2_output_l516_51679


namespace total_notes_count_l516_51693

def total_amount : ℕ := 10350
def note_50_value : ℕ := 50
def note_500_value : ℕ := 500
def note_50_count : ℕ := 37

theorem total_notes_count : 
  ∃ (note_500_count : ℕ), 
    note_50_count * note_50_value + note_500_count * note_500_value = total_amount ∧
    note_50_count + note_500_count = 54 :=
by sorry

end total_notes_count_l516_51693


namespace inscribed_cube_surface_area_l516_51675

/-- The surface area of a cube inscribed in a sphere, which is itself inscribed in another cube --/
theorem inscribed_cube_surface_area (outer_cube_area : ℝ) : 
  outer_cube_area = 150 →
  ∃ (inner_cube_area : ℝ), inner_cube_area = 50 := by
  sorry

end inscribed_cube_surface_area_l516_51675


namespace min_value_theorem_min_value_achievable_l516_51629

theorem min_value_theorem (x : ℝ) (h : x > -3) :
  2 * x + 1 / (x + 3) ≥ 2 * Real.sqrt 2 - 6 :=
by sorry

theorem min_value_achievable :
  ∃ x > -3, 2 * x + 1 / (x + 3) = 2 * Real.sqrt 2 - 6 :=
by sorry

end min_value_theorem_min_value_achievable_l516_51629


namespace monotonicity_intervals_l516_51686

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (3/2) * a * x^2 + (2*a^2 + a - 1) * x + 3

theorem monotonicity_intervals (a : ℝ) :
  (a = 2 → ∀ x y, x < y → f a x < f a y) ∧
  (a < 2 → (∀ x y, x < y ∧ y < 2*a - 1 → f a x < f a y) ∧
           (∀ x y, 2*a - 1 < x ∧ x < y ∧ y < a + 1 → f a x > f a y) ∧
           (∀ x y, a + 1 < x ∧ x < y → f a x < f a y)) ∧
  (a > 2 → (∀ x y, x < y ∧ y < a + 1 → f a x < f a y) ∧
           (∀ x y, a + 1 < x ∧ x < y ∧ y < 2*a - 1 → f a x > f a y) ∧
           (∀ x y, 2*a - 1 < x ∧ x < y → f a x < f a y)) :=
by sorry

end monotonicity_intervals_l516_51686


namespace inscribed_triangle_ratio_l516_51661

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Represents a triangle -/
structure Triangle where
  p : Point
  q : Point
  r : Point

theorem inscribed_triangle_ratio (a : ℝ) (b : ℝ) (c : ℝ) (e : Ellipse) (t : Triangle) :
  e.a = a ∧ e.b = b ∧
  c = (3/5) * a ∧
  t.q = Point.mk 0 b ∧
  t.p.y = t.r.y ∧
  t.p.x = -c ∧ t.r.x = c ∧
  (t.p.x^2 / a^2) + (t.p.y^2 / b^2) = 1 ∧
  (t.q.x^2 / a^2) + (t.q.y^2 / b^2) = 1 ∧
  (t.r.x^2 / a^2) + (t.r.y^2 / b^2) = 1 ∧
  2 * c = 0.6 * a →
  (Real.sqrt ((t.p.x - t.q.x)^2 + (t.p.y - t.q.y)^2)) / (t.r.x - t.p.x) = 5/3 := by
  sorry

end inscribed_triangle_ratio_l516_51661


namespace exists_appropriate_ratio_in_small_interval_l516_51646

-- Define a type for the cutting ratio
def CuttingRatio := {a : ℝ // 0 < a ∧ a < 1}

-- Define a predicate for appropriate cutting ratios
def isAppropriate (a : CuttingRatio) : Prop :=
  ∃ (n : ℕ), ∀ (w : ℝ), w > 0 → ∃ (w1 w2 : ℝ), w1 = w2 ∧ w1 + w2 = w ∧
  ∃ (cuts : List ℝ), cuts.length ≤ n ∧ 
    (∀ c ∈ cuts, c = a.val * w ∨ c = (1 - a.val) * w)

-- State the theorem
theorem exists_appropriate_ratio_in_small_interval :
  ∀ x : ℝ, 0 < x → x < 0.999 →
  ∃ a : CuttingRatio, x < a.val ∧ a.val < x + 0.001 ∧ isAppropriate a :=
sorry

end exists_appropriate_ratio_in_small_interval_l516_51646


namespace forest_growth_l516_51631

/-- The number of trees in a forest follows a specific growth pattern --/
theorem forest_growth (trees : ℕ → ℕ) (k : ℚ) : 
  (∀ n, trees (n + 2) - trees n = k * trees (n + 1)) →
  trees 1993 = 50 →
  trees 1994 = 75 →
  trees 1996 = 140 →
  trees 1995 = 99 := by
sorry

end forest_growth_l516_51631


namespace max_S_value_l516_51634

theorem max_S_value (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let S := min x (min (y + 1/x) (1/y))
  ∃ (max_S : ℝ), max_S = Real.sqrt 2 ∧
    (∀ x' y' : ℝ, x' > 0 → y' > 0 → 
      min x' (min (y' + 1/x') (1/y')) ≤ max_S) ∧
    S = max_S ↔ x = Real.sqrt 2 ∧ y = Real.sqrt 2 / 2 :=
by sorry

end max_S_value_l516_51634


namespace perpendicular_parallel_implies_perpendicular_l516_51606

/-- A line in 3D space -/
structure Line3D where
  -- Define the line structure (omitted for brevity)

/-- A plane in 3D space -/
structure Plane3D where
  -- Define the plane structure (omitted for brevity)

/-- Two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- A line is parallel to a plane -/
def lineParallelToPlane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line is perpendicular to a plane -/
def linePerpendicularToPlane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line is perpendicular to another line -/
def linePerpendicular (l1 l2 : Line3D) : Prop :=
  sorry

/-- Theorem: If a line is perpendicular to a plane and another line is parallel to the plane,
    then the two lines are perpendicular to each other -/
theorem perpendicular_parallel_implies_perpendicular
  (a b : Line3D) (α : Plane3D)
  (h1 : linePerpendicularToPlane a α)
  (h2 : lineParallelToPlane b α) :
  linePerpendicular a b :=
sorry

end perpendicular_parallel_implies_perpendicular_l516_51606


namespace condition1_condition2_f_satisfies_conditions_l516_51691

/-- A function satisfying the given conditions -/
def f (x : ℝ) := -3 * x

/-- The first condition: f(x) + f(-x) = 0 for all x ∈ ℝ -/
theorem condition1 : ∀ x : ℝ, f x + f (-x) = 0 := by
  sorry

/-- The second condition: f(x + t) - f(x) < 0 for all x ∈ ℝ and t > 0 -/
theorem condition2 : ∀ x t : ℝ, t > 0 → f (x + t) - f x < 0 := by
  sorry

/-- The main theorem: f satisfies both conditions -/
theorem f_satisfies_conditions : 
  (∀ x : ℝ, f x + f (-x) = 0) ∧ 
  (∀ x t : ℝ, t > 0 → f (x + t) - f x < 0) := by
  sorry

end condition1_condition2_f_satisfies_conditions_l516_51691


namespace pirate_loot_sum_l516_51630

def base7_to_base10 (n : Nat) : Nat :=
  let digits := n.digits 7
  (List.range digits.length).foldl (fun acc i => acc + digits[i]! * (7 ^ i)) 0

def pirate_loot : Nat :=
  base7_to_base10 4516 + base7_to_base10 3216 + base7_to_base10 654 + base7_to_base10 301

theorem pirate_loot_sum :
  pirate_loot = 3251 := by sorry

end pirate_loot_sum_l516_51630


namespace tan_sum_15_30_l516_51614

theorem tan_sum_15_30 : 
  ∀ (tan : Real → Real),
  (∀ α β, tan (α + β) = (tan α + tan β) / (1 - tan α * tan β)) →
  tan (45 * π / 180) = 1 →
  tan (15 * π / 180) + tan (30 * π / 180) + tan (15 * π / 180) * tan (30 * π / 180) = 1 :=
by sorry

end tan_sum_15_30_l516_51614


namespace prob_roll_less_than_4_l516_51684

/-- A fair 8-sided die -/
def fair_8_sided_die : Finset (Fin 8) := Finset.univ

/-- The event of rolling a number less than 4 -/
def roll_less_than_4 : Finset (Fin 8) := Finset.filter (λ x => x.val < 4) fair_8_sided_die

/-- The probability of an event occurring when rolling a fair 8-sided die -/
def prob (event : Finset (Fin 8)) : ℚ :=
  (event.card : ℚ) / (fair_8_sided_die.card : ℚ)

theorem prob_roll_less_than_4 : 
  prob roll_less_than_4 = 3 / 8 := by
  sorry

end prob_roll_less_than_4_l516_51684


namespace original_paint_intensity_l516_51636

theorem original_paint_intensity 
  (f : ℝ) 
  (h1 : f = 2/3)
  (h2 : (1 - f) * I + f * 0.3 = 0.4) : 
  I = 0.6 :=
by sorry

end original_paint_intensity_l516_51636


namespace unique_number_property_l516_51698

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end unique_number_property_l516_51698


namespace intersection_implies_m_value_subset_implies_m_range_l516_51655

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 6 / (x + 1) ≥ 1}
def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x + 2*m < 0}

-- Theorem 1
theorem intersection_implies_m_value :
  ∀ m : ℝ, (A ∩ B m = {x : ℝ | -1 < x ∧ x < 4}) → m = -4 := by sorry

-- Theorem 2
theorem subset_implies_m_range :
  ∀ m : ℝ, (B m ⊆ A) → m ≥ -3/2 := by sorry

end intersection_implies_m_value_subset_implies_m_range_l516_51655


namespace zero_in_M_l516_51609

theorem zero_in_M : 0 ∈ ({-1, 0, 1} : Set ℤ) := by
  sorry

end zero_in_M_l516_51609


namespace distinct_prime_factors_of_420_l516_51605

theorem distinct_prime_factors_of_420 : Nat.card (Nat.factors 420).toFinset = 4 := by
  sorry

end distinct_prime_factors_of_420_l516_51605


namespace pollution_data_median_mode_l516_51624

def pollution_data : List ℕ := [31, 35, 31, 34, 30, 32, 31]

def median (l : List ℕ) : ℕ := sorry

def mode (l : List ℕ) : ℕ := sorry

theorem pollution_data_median_mode : 
  median pollution_data = 31 ∧ mode pollution_data = 31 := by sorry

end pollution_data_median_mode_l516_51624


namespace unique_sequence_existence_l516_51683

theorem unique_sequence_existence : ∃! a : ℕ → ℤ,
  (a 1 = 1) ∧
  (a 2 = 2) ∧
  (∀ n : ℕ, n ≥ 1 → (a (n + 1))^3 + 1 = (a n) * (a (n + 2))) :=
by sorry

end unique_sequence_existence_l516_51683


namespace triangle_area_l516_51677

/-- Given a triangle ABC with side lengths b and c, and angle C, prove that its area is √3/4 -/
theorem triangle_area (b c : ℝ) (C : ℝ) (h1 : b = 1) (h2 : c = Real.sqrt 3) (h3 : C = 2 * Real.pi / 3) :
  (1 / 2) * b * c * Real.sin (Real.pi / 6) = Real.sqrt 3 / 4 := by
  sorry

end triangle_area_l516_51677


namespace y_intercept_of_line_l516_51681

-- Define the line equation
def line_equation (x y a b : ℝ) : Prop := x / (a^2) - y / (b^2) = 1

-- Define y-intercept
def y_intercept (f : ℝ → ℝ) : ℝ := f 0

-- Theorem statement
theorem y_intercept_of_line (a b : ℝ) (h : b ≠ 0) :
  ∃ f : ℝ → ℝ, (∀ x, line_equation x (f x) a b) ∧ y_intercept f = -b^2 := by
  sorry

end y_intercept_of_line_l516_51681


namespace infinite_binary_decimal_divisible_by_2017_l516_51688

/-- A number composed only of digits 0 and 1 in decimal representation -/
def is_binary_decimal (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The set of numbers composed only of digits 0 and 1 in decimal representation -/
def binary_decimal_set : Set ℕ :=
  {n : ℕ | is_binary_decimal n}

/-- The theorem statement -/
theorem infinite_binary_decimal_divisible_by_2017 :
  ∃ S : Set ℕ, (∀ n ∈ S, is_binary_decimal n ∧ 2017 ∣ n) ∧ Set.Infinite S :=
sorry

end infinite_binary_decimal_divisible_by_2017_l516_51688


namespace inequality_solution_l516_51635

theorem inequality_solution (x : ℝ) :
  x ≠ 0 →
  ((2 * x - 7) * (x - 3)) / x ≥ 0 ↔ (0 < x ∧ x ≤ 3) ∨ (7/2 ≤ x) := by
  sorry

end inequality_solution_l516_51635


namespace veggies_per_day_l516_51645

/-- The number of servings of veggies eaten in a week -/
def weekly_servings : ℕ := 21

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of servings of veggies eaten per day -/
def daily_servings : ℕ := weekly_servings / days_in_week

theorem veggies_per_day : daily_servings = 3 := by
  sorry

end veggies_per_day_l516_51645


namespace pink_cookies_l516_51665

theorem pink_cookies (total : ℕ) (red : ℕ) (h1 : total = 86) (h2 : red = 36) :
  total - red = 50 := by
  sorry

end pink_cookies_l516_51665


namespace fish_tank_leak_ratio_l516_51641

/-- The ratio of a bucket's capacity to the amount of leaked fluid over a given time -/
def leakRatio (bucketCapacity leakRate hours : ℚ) : ℚ :=
  bucketCapacity / (leakRate * hours)

/-- Theorem stating that the ratio of a 36-ounce bucket's capacity to the amount of fluid
    leaking at 1.5 ounces per hour over 12 hours is 2:1 -/
theorem fish_tank_leak_ratio :
  leakRatio 36 (3/2) 12 = 2 := by
  sorry

end fish_tank_leak_ratio_l516_51641


namespace cube_surface_area_l516_51610

/-- The surface area of a cube with edge length 3 cm is 54 square centimeters. -/
theorem cube_surface_area : 
  let edge_length : ℝ := 3
  let face_area : ℝ := edge_length ^ 2
  let surface_area : ℝ := 6 * face_area
  surface_area = 54 := by sorry

end cube_surface_area_l516_51610


namespace prob_at_least_one_black_without_replacement_prob_exactly_one_black_with_replacement_l516_51692

/- Define the number of white and black balls -/
def num_white : ℕ := 4
def num_black : ℕ := 2
def total_balls : ℕ := num_white + num_black

/- Define the number of draws -/
def num_draws : ℕ := 3

/- Theorem for drawing without replacement -/
theorem prob_at_least_one_black_without_replacement :
  let total_ways := Nat.choose total_balls num_draws
  let all_white_ways := Nat.choose num_white num_draws
  (1 : ℚ) - (all_white_ways : ℚ) / (total_ways : ℚ) = 4/5 := by sorry

/- Theorem for drawing with replacement -/
theorem prob_exactly_one_black_with_replacement :
  let total_ways := total_balls ^ num_draws
  let one_black_ways := num_draws * num_black * (num_white ^ (num_draws - 1))
  (one_black_ways : ℚ) / (total_ways : ℚ) = 4/9 := by sorry

end prob_at_least_one_black_without_replacement_prob_exactly_one_black_with_replacement_l516_51692


namespace multiply_658217_and_99999_l516_51603

theorem multiply_658217_and_99999 : 658217 * 99999 = 65821034183 := by
  sorry

end multiply_658217_and_99999_l516_51603


namespace max_length_sum_xy_l516_51666

/-- The length of an integer is the number of positive prime factors (not necessarily distinct) whose product equals the integer. -/
def length (n : ℕ) : ℕ := sorry

/-- Given the constraints, the maximum sum of lengths of x and y is 15. -/
theorem max_length_sum_xy : 
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ x + 3*y < 920 ∧ 
  ∀ (a b : ℕ), a > 1 → b > 1 → a + 3*b < 920 → 
  length x + length y ≥ length a + length b ∧
  length x + length y = 15 := by
sorry

end max_length_sum_xy_l516_51666


namespace sin_cos_equivalence_l516_51676

theorem sin_cos_equivalence (x : ℝ) : 
  Real.sin (2 * x + π / 3) = Real.cos (2 * (x - π / 12)) := by sorry

end sin_cos_equivalence_l516_51676


namespace tank_filling_proof_l516_51652

/-- The number of buckets required to fill a tank with the original bucket size -/
def original_buckets : ℕ := 10

/-- The number of buckets required to fill the tank with reduced bucket capacity -/
def reduced_buckets : ℕ := 25

/-- The ratio of reduced bucket capacity to original bucket capacity -/
def capacity_ratio : ℚ := 2 / 5

theorem tank_filling_proof :
  original_buckets * 1 = reduced_buckets * capacity_ratio :=
by sorry

end tank_filling_proof_l516_51652


namespace abs_equation_unique_solution_l516_51668

theorem abs_equation_unique_solution :
  ∃! x : ℝ, |x - 9| = |x + 3| := by
sorry

end abs_equation_unique_solution_l516_51668


namespace simplify_fraction_l516_51611

theorem simplify_fraction : (5^4 + 5^2) / (5^3 - 5) = 65 / 12 := by
  sorry

end simplify_fraction_l516_51611


namespace joshua_toy_cars_l516_51617

theorem joshua_toy_cars (box1 box2 box3 : ℕ) 
  (h1 : box1 = 21) 
  (h2 : box2 = 31) 
  (h3 : box3 = 19) : 
  box1 + box2 + box3 = 71 := by
sorry

end joshua_toy_cars_l516_51617


namespace compare_expressions_l516_51650

theorem compare_expressions (x y : ℝ) (h1 : x * y > 0) (h2 : x ≠ y) :
  x^4 + 6*x^2*y^2 + y^4 > 4*x*y*(x^2 + y^2) := by
  sorry

end compare_expressions_l516_51650


namespace min_value_theorem_l516_51602

theorem min_value_theorem (x : ℝ) (h : x > 1) :
  x + 4 / (x - 1) ≥ 5 ∧ ∃ y > 1, y + 4 / (y - 1) = 5 := by
  sorry

end min_value_theorem_l516_51602


namespace oats_per_meal_is_four_l516_51607

/-- The amount of oats each horse eats per meal, in pounds -/
def oats_per_meal (num_horses : ℕ) (grain_per_horse : ℕ) (total_food : ℕ) (num_days : ℕ) : ℚ :=
  let total_food_per_day := total_food / num_days
  let grain_per_day := num_horses * grain_per_horse
  let oats_per_day := total_food_per_day - grain_per_day
  oats_per_day / (2 * num_horses)

theorem oats_per_meal_is_four :
  oats_per_meal 4 3 132 3 = 4 := by
  sorry

end oats_per_meal_is_four_l516_51607


namespace hill_climb_speed_l516_51671

/-- Proves that given a journey with an uphill climb taking 4 hours and a downhill descent
    taking 2 hours, if the average speed for the entire journey is 1.5 km/h,
    then the average speed for the uphill climb is 1.125 km/h. -/
theorem hill_climb_speed (distance : ℝ) (climb_time : ℝ) (descent_time : ℝ) 
    (average_speed : ℝ) (h1 : climb_time = 4) (h2 : descent_time = 2) 
    (h3 : average_speed = 1.5) :
  distance / climb_time = 1.125 := by
  sorry

end hill_climb_speed_l516_51671


namespace binomial_probability_two_successes_l516_51667

/-- The probability mass function for a binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- Theorem: For a random variable ξ following a binomial distribution B(6, 1/3),
    the probability P(ξ = 2) is equal to 80/243 -/
theorem binomial_probability_two_successes :
  binomial_pmf 6 (1/3) 2 = 80/243 :=
sorry

end binomial_probability_two_successes_l516_51667


namespace circle_area_irrational_when_radius_rational_l516_51615

/-- The area of a circle is irrational when its radius is rational -/
theorem circle_area_irrational_when_radius_rational :
  ∀ r : ℚ, ∃ A : ℝ, A = π * r^2 ∧ Irrational A :=
sorry

end circle_area_irrational_when_radius_rational_l516_51615


namespace arithmetic_sequence_common_difference_l516_51637

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  sum : ℕ → ℝ
  seq_def : ∀ n, a (n + 1) = a n + d
  sum_def : ∀ n, sum n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- The common difference of an arithmetic sequence is 2 if 2S₃ = 3S₂ + 6 -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h : 2 * seq.sum 3 = 3 * seq.sum 2 + 6) :
  seq.d = 2 := by
  sorry

end arithmetic_sequence_common_difference_l516_51637


namespace geometric_sequence_S5_l516_51673

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_S5 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 1 →
  (a 3 + a 4) / (a 1 + a 2) = 4 →
  ∃ S5 : ℝ, (S5 = 31 ∨ S5 = 11) ∧ S5 = (a 1 + a 2 + a 3 + a 4 + a 5) :=
by sorry

end geometric_sequence_S5_l516_51673


namespace quadratic_roots_transformation_l516_51619

variable {a b c x1 x2 : ℝ}

theorem quadratic_roots_transformation (ha : a ≠ 0)
  (hroots : a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) :
  ∃ k, k * ((x1 + 2*x2) - x)* ((x2 + 2*x1) - x) = a^2 * x^2 + 3*a*b * x + 2*b^2 + a*c :=
sorry

end quadratic_roots_transformation_l516_51619


namespace smallest_angle_theorem_l516_51647

/-- The smallest positive angle θ, in degrees, that satisfies the given equation is 50°. -/
theorem smallest_angle_theorem : 
  ∃ θ : ℝ, θ > 0 ∧ θ < 360 ∧ 
  Real.cos (θ * π / 180) = Real.sin (70 * π / 180) + Real.cos (50 * π / 180) - 
                           Real.sin (20 * π / 180) - Real.cos (10 * π / 180) ∧
  θ = 50 ∧ 
  ∀ φ : ℝ, 0 < φ ∧ φ < θ → 
    Real.cos (φ * π / 180) ≠ Real.sin (70 * π / 180) + Real.cos (50 * π / 180) - 
                             Real.sin (20 * π / 180) - Real.cos (10 * π / 180) :=
by sorry


end smallest_angle_theorem_l516_51647


namespace distinct_prime_factors_count_l516_51648

theorem distinct_prime_factors_count : 
  let n : ℕ := 101 * 103 * 107 * 109
  ∀ (is_prime_101 : Nat.Prime 101) 
    (is_prime_103 : Nat.Prime 103) 
    (is_prime_107 : Nat.Prime 107) 
    (is_prime_109 : Nat.Prime 109),
  Finset.card (Nat.factors n).toFinset = 4 := by
sorry

end distinct_prime_factors_count_l516_51648


namespace triangle_inequality_theorem_l516_51690

/-- Checks if three lengths can form a triangle according to the triangle inequality theorem -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality_theorem :
  can_form_triangle 3 4 5 ∧
  ¬can_form_triangle 2 4 7 ∧
  ¬can_form_triangle 3 6 9 ∧
  ¬can_form_triangle 4 4 9 :=
by sorry

end triangle_inequality_theorem_l516_51690


namespace divisibility_by_five_l516_51618

theorem divisibility_by_five (a b : ℕ) (h : 5 ∣ (a * b)) :
  ¬(¬(5 ∣ a) ∧ ¬(5 ∣ b)) := by
  sorry

end divisibility_by_five_l516_51618


namespace abs_three_minus_a_l516_51659

theorem abs_three_minus_a (a : ℝ) (h : |1 - a| = 1 + |a|) : |3 - a| = 3 - a := by
  sorry

end abs_three_minus_a_l516_51659


namespace simplify_fraction_l516_51601

theorem simplify_fraction : (72 : ℚ) / 108 = 2 / 3 := by
  sorry

end simplify_fraction_l516_51601


namespace sqrt_two_minus_one_to_zero_l516_51680

theorem sqrt_two_minus_one_to_zero : (Real.sqrt 2 - 1) ^ (0 : ℕ) = 1 := by
  sorry

end sqrt_two_minus_one_to_zero_l516_51680


namespace hall_volume_l516_51638

/-- A rectangular hall with specific dimensions and area properties -/
structure RectangularHall where
  length : ℝ
  width : ℝ
  height : ℝ
  area_equality : 2 * (length * width) = 2 * (length * height) + 2 * (width * height)

/-- The volume of a rectangular hall with the given properties is 972 cubic meters -/
theorem hall_volume (hall : RectangularHall) 
  (h_length : hall.length = 18)
  (h_width : hall.width = 9) : 
  hall.length * hall.width * hall.height = 972 := by
  sorry

end hall_volume_l516_51638


namespace order_of_constants_l516_51600

-- Define the constants
noncomputable def a : ℝ := Real.log 3 / Real.log (1/2)
noncomputable def b : ℝ := Real.exp (1/2)
noncomputable def c : ℝ := Real.log 2 / Real.log 10

-- Theorem statement
theorem order_of_constants : a < c ∧ c < b := by sorry

end order_of_constants_l516_51600


namespace zoo_ticket_cost_l516_51626

def adult_price : ℝ := 10

def grandparent_discount : ℝ := 0.2
def child_discount : ℝ := 0.6

def grandparent_price : ℝ := adult_price * (1 - grandparent_discount)
def child_price : ℝ := adult_price * (1 - child_discount)

def total_cost : ℝ := 2 * grandparent_price + adult_price + child_price

theorem zoo_ticket_cost : total_cost = 30 := by
  sorry

end zoo_ticket_cost_l516_51626


namespace tailor_buttons_l516_51663

/-- The number of buttons purchased by a tailor -/
def total_buttons (green : ℕ) : ℕ :=
  let yellow := green + 10
  let blue := green - 5
  let red := 2 * (yellow + blue)
  let white := red + green
  let black := red - green
  green + yellow + blue + red + white + black

/-- Theorem: The tailor purchased 1385 buttons -/
theorem tailor_buttons : total_buttons 90 = 1385 := by
  sorry

end tailor_buttons_l516_51663


namespace cheryl_same_color_probability_l516_51632

def total_marbles : ℕ := 6
def red_marbles : ℕ := 3
def green_marbles : ℕ := 2
def yellow_marbles : ℕ := 1

def carol_draw : ℕ := 2
def claudia_draw : ℕ := 2
def cheryl_draw : ℕ := 2

theorem cheryl_same_color_probability :
  let total_outcomes := (total_marbles.choose carol_draw) * ((total_marbles - carol_draw).choose claudia_draw) * ((total_marbles - carol_draw - claudia_draw).choose cheryl_draw)
  let favorable_outcomes := red_marbles.choose cheryl_draw * ((total_marbles - cheryl_draw).choose carol_draw) * ((total_marbles - cheryl_draw - carol_draw).choose claudia_draw)
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 5 := by
  sorry

end cheryl_same_color_probability_l516_51632


namespace trigonometric_expression_equality_l516_51658

theorem trigonometric_expression_equality : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 2 / Real.sin (70 * π / 180) = 4 := by
  sorry

end trigonometric_expression_equality_l516_51658


namespace sum_of_squares_101_to_200_l516_51697

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_101_to_200 :
  sum_of_squares 200 - sum_of_squares 100 = 2348350 := by
  sorry

end sum_of_squares_101_to_200_l516_51697


namespace trigonometric_product_transformation_l516_51669

theorem trigonometric_product_transformation (α : ℝ) :
  4.66 * Real.sin (5 * π / 2 + 4 * α) - Real.sin (5 * π / 2 + 2 * α) ^ 6 + Real.cos (7 * π / 2 - 2 * α) ^ 6 = 
  (1 / 8) * Real.sin (4 * α) * Real.sin (8 * α) := by
  sorry

end trigonometric_product_transformation_l516_51669


namespace no_divisors_between_2_and_100_l516_51628

theorem no_divisors_between_2_and_100 (n : ℕ+) 
  (h : ∀ k ∈ Finset.range 99, (Finset.sum (Finset.range n) (fun i => (i + 1) ^ (k + 1))) % n = 0) :
  ∀ d ∈ Finset.range 99, d > 1 → ¬(d ∣ n) := by
  sorry

end no_divisors_between_2_and_100_l516_51628


namespace Q_subset_complement_P_l516_51664

-- Define the sets P and Q
def P : Set ℝ := {x | x < 1}
def Q : Set ℝ := {x | x > 1}

-- Define the complement of P in the real numbers
def CₘP : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem Q_subset_complement_P : Q ⊆ CₘP := by sorry

end Q_subset_complement_P_l516_51664


namespace unique_phone_number_l516_51674

/-- A six-digit number -/
def SixDigitNumber := { n : ℕ // 100000 ≤ n ∧ n < 1000000 }

/-- The set of divisors we're interested in -/
def Divisors : Finset ℕ := {3, 4, 7, 9, 11, 13}

/-- The property that a number gives the same remainder when divided by all numbers in Divisors -/
def SameRemainder (n : ℕ) : Prop :=
  ∃ r, ∀ d ∈ Divisors, n % d = r

/-- The main theorem -/
theorem unique_phone_number :
  ∃! (T : SixDigitNumber),
    Odd T.val ∧
    (T.val / 100000 = 7) ∧
    ((T.val / 100) % 10 = 2) ∧
    SameRemainder T.val ∧
    T.val = 720721 := by
  sorry

#check unique_phone_number

end unique_phone_number_l516_51674


namespace cubic_function_values_l516_51662

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 - 6 * a * x^2 + b

-- State the theorem
theorem cubic_function_values (a b : ℝ) (ha : a ≠ 0) :
  (∀ x ∈ Set.Icc (-1) 2, f a b x ≤ 3) ∧
  (∃ x ∈ Set.Icc (-1) 2, f a b x = 3) ∧
  (∀ x ∈ Set.Icc (-1) 2, f a b x ≥ -29) ∧
  (∃ x ∈ Set.Icc (-1) 2, f a b x = -29) →
  ((a = 2 ∧ b = 3) ∨ (a = -2 ∧ b = -29)) :=
by sorry

end cubic_function_values_l516_51662


namespace quadratic_inequality_solution_range_l516_51699

theorem quadratic_inequality_solution_range (c : ℝ) : 
  (c > 0 ∧ ∃ x : ℝ, x^2 - 8*x + c < 0) ↔ (c > 0 ∧ c < 16) := by
sorry

end quadratic_inequality_solution_range_l516_51699


namespace harry_pencils_left_l516_51689

/-- Calculates the number of pencils left with Harry given Anna's pencils and Harry's lost pencils. -/
def pencils_left_with_harry (anna_pencils : ℕ) (harry_lost_pencils : ℕ) : ℕ :=
  2 * anna_pencils - harry_lost_pencils

/-- Proves that Harry has 81 pencils left given the problem conditions. -/
theorem harry_pencils_left : pencils_left_with_harry 50 19 = 81 := by
  sorry

end harry_pencils_left_l516_51689


namespace elliptic_curve_solutions_l516_51608

theorem elliptic_curve_solutions (p : ℕ) (hp : Nat.Prime p) :
  (∃ (S : Finset (Fin p × Fin p)),
    S.card = p ∧
    ∀ (x y : Fin p), (x, y) ∈ S ↔ y^2 ≡ x^3 + 4*x [ZMOD p]) ↔
  p = 2 ∨ p ≡ 3 [MOD 4] :=
sorry

end elliptic_curve_solutions_l516_51608


namespace probability_at_least_one_fuse_blows_l516_51627

/-- The probability that at least one fuse blows in a circuit with two independent fuses -/
theorem probability_at_least_one_fuse_blows 
  (prob_A : ℝ) 
  (prob_B : ℝ) 
  (h_prob_A : prob_A = 0.85) 
  (h_prob_B : prob_B = 0.74) 
  (h_independent : True) -- We don't need to express independence explicitly in this theorem
  : 1 - (1 - prob_A) * (1 - prob_B) = 0.961 := by
  sorry

end probability_at_least_one_fuse_blows_l516_51627


namespace colinear_vector_problem_l516_51616

/-- Given vector a and b in ℝ², prove that if a = (1, -2), b is colinear with a, and |b| = 4|a|, then b = (4, -8) or b = (-4, 8) -/
theorem colinear_vector_problem (a b : ℝ × ℝ) : 
  a = (1, -2) → 
  (∃ (k : ℝ), b = k • a) → 
  Real.sqrt ((b.1)^2 + (b.2)^2) = 4 * Real.sqrt ((a.1)^2 + (a.2)^2) → 
  b = (4, -8) ∨ b = (-4, 8) := by
sorry

end colinear_vector_problem_l516_51616


namespace dianes_honey_harvest_l516_51623

/-- Diane's honey harvest calculation -/
theorem dianes_honey_harvest (last_year_harvest : ℕ) (increase : ℕ) : 
  last_year_harvest = 2479 → increase = 6085 → last_year_harvest + increase = 8564 := by
  sorry

end dianes_honey_harvest_l516_51623


namespace largest_sum_proof_l516_51657

theorem largest_sum_proof : 
  let sums : List ℚ := [1/4 + 1/9, 1/4 + 1/10, 1/4 + 1/11, 1/4 + 1/12, 1/4 + 1/13]
  (∀ x ∈ sums, x ≤ (1/4 + 1/9)) ∧ (1/4 + 1/9 = 13/36) :=
by sorry

end largest_sum_proof_l516_51657


namespace odd_prime_divisibility_l516_51695

theorem odd_prime_divisibility (p a b c : ℤ) : 
  Prime p → 
  Odd p → 
  (p ∣ a^2023 + b^2023) → 
  (p ∣ b^2024 + c^2024) → 
  (p ∣ a^2025 + c^2025) → 
  (p ∣ a) ∧ (p ∣ b) ∧ (p ∣ c) := by
sorry

end odd_prime_divisibility_l516_51695


namespace F_odd_and_increasing_l516_51653

-- Define f(x) implicitly using the given condition
noncomputable def f : ℝ → ℝ := fun x => Real.exp (x * Real.log 2)

-- Define F(x) using f(x)
noncomputable def F : ℝ → ℝ := fun x => f x - 1 / f x

-- Theorem stating that F is odd and increasing
theorem F_odd_and_increasing :
  (∀ x : ℝ, F (-x) = -F x) ∧
  (∀ x y : ℝ, x < y → F x < F y) :=
by sorry

end F_odd_and_increasing_l516_51653


namespace exam_time_proof_l516_51612

/-- Proves that the examination time is 3 hours given the specified conditions -/
theorem exam_time_proof (total_questions : ℕ) (type_a_problems : ℕ) (type_a_time : ℚ) :
  total_questions = 200 →
  type_a_problems = 15 →
  type_a_time = 25116279069767444 / 1000000000000000 →
  ∃ (type_b_time : ℚ),
    type_b_time > 0 ∧
    type_a_time = 2 * type_b_time * type_a_problems ∧
    (type_b_time * (total_questions - type_a_problems) + type_a_time) / 60 = 3 :=
by sorry

end exam_time_proof_l516_51612


namespace line_perpendicular_to_parallel_planes_l516_51642

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_parallel_planes
  (l : Line) (α β : Plane)
  (h1 : perpendicular l β)
  (h2 : parallel α β) :
  perpendicular l α :=
sorry

end line_perpendicular_to_parallel_planes_l516_51642


namespace fuse_probability_l516_51682

/-- The probability of the union of two events -/
def prob_union (prob_A prob_B prob_A_and_B : ℝ) : ℝ :=
  prob_A + prob_B - prob_A_and_B

theorem fuse_probability (prob_A prob_B prob_A_and_B : ℝ) 
  (h1 : prob_A = 0.085)
  (h2 : prob_B = 0.074)
  (h3 : prob_A_and_B = 0.063) :
  prob_union prob_A prob_B prob_A_and_B = 0.096 := by
sorry

end fuse_probability_l516_51682


namespace units_digit_of_7_pow_5_l516_51643

theorem units_digit_of_7_pow_5 : (7^5) % 10 = 7 := by
  sorry

end units_digit_of_7_pow_5_l516_51643


namespace existence_of_many_prime_factors_l516_51672

theorem existence_of_many_prime_factors (N : ℕ+) :
  ∃ n : ℕ+, ∃ p : Finset ℕ,
    (∀ q ∈ p, Nat.Prime q) ∧
    (Finset.card p ≥ N) ∧
    (∀ q ∈ p, q ∣ (n^2013 - n^20 + n^13 - 2013)) :=
by sorry

end existence_of_many_prime_factors_l516_51672


namespace sticker_probability_l516_51613

def total_stickers : ℕ := 18
def selected_stickers : ℕ := 10
def missing_stickers : ℕ := 6

theorem sticker_probability :
  (Nat.choose missing_stickers missing_stickers * Nat.choose (total_stickers - missing_stickers) (selected_stickers - missing_stickers)) / 
  Nat.choose total_stickers selected_stickers = 5 / 442 := by
  sorry

end sticker_probability_l516_51613


namespace special_triangle_cosine_l516_51687

/-- A triangle with consecutive integer side lengths where the middle angle is 1.5 times the smallest angle -/
structure SpecialTriangle where
  n : ℕ
  side1 : ℕ := n
  side2 : ℕ := n + 1
  side3 : ℕ := n + 2
  smallest_angle : ℝ
  middle_angle : ℝ
  largest_angle : ℝ
  angle_sum : middle_angle = 1.5 * smallest_angle
  angle_total : smallest_angle + middle_angle + largest_angle = Real.pi

/-- The cosine of the smallest angle in a SpecialTriangle is 53/60 -/
theorem special_triangle_cosine (t : SpecialTriangle) : 
  Real.cos t.smallest_angle = 53 / 60 := by sorry

end special_triangle_cosine_l516_51687


namespace odd_functions_sufficient_not_necessary_l516_51685

-- Define the real-valued functions
variable (f g h : ℝ → ℝ)

-- Define odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the relationship between f, g, and h
def FunctionsRelated (f g h : ℝ → ℝ) : Prop := ∀ x, h x = f x * g x

-- Theorem statement
theorem odd_functions_sufficient_not_necessary :
  (∀ f g h, FunctionsRelated f g h → (IsOdd f ∧ IsOdd g → IsEven h)) ∧
  (∃ f g h, FunctionsRelated f g h ∧ IsEven h ∧ ¬(IsOdd f ∧ IsOdd g)) :=
sorry

end odd_functions_sufficient_not_necessary_l516_51685


namespace max_product_853_l516_51640

def digits : List Nat := [3, 5, 6, 8, 9]

def is_valid_combination (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def three_digit_number (a b c : Nat) : Nat := 100 * a + 10 * b + c
def two_digit_number (d e : Nat) : Nat := 10 * d + e

def product (a b c d e : Nat) : Nat :=
  (three_digit_number a b c) * (two_digit_number d e)

theorem max_product_853 :
  ∀ a b c d e,
    is_valid_combination a b c d e →
    product 8 5 3 9 6 ≥ product a b c d e :=
by sorry

end max_product_853_l516_51640


namespace cricket_bat_price_l516_51660

theorem cricket_bat_price (cost_price_A : ℝ) (profit_A_percent : ℝ) (profit_B_percent : ℝ) : 
  cost_price_A = 156 →
  profit_A_percent = 20 →
  profit_B_percent = 25 →
  let selling_price_B := cost_price_A * (1 + profit_A_percent / 100)
  let selling_price_C := selling_price_B * (1 + profit_B_percent / 100)
  selling_price_C = 234 :=
by sorry

end cricket_bat_price_l516_51660


namespace arithmetic_comparisons_l516_51644

theorem arithmetic_comparisons :
  (80 / 4 > 80 / 5) ∧
  (16 * 21 > 14 * 22) ∧
  (32 * 25 = 16 * 50) ∧
  (320 / 8 < 320 / 4) := by
  sorry

end arithmetic_comparisons_l516_51644


namespace A_initial_investment_l516_51651

/-- Represents the initial investment of A in dollars -/
def A_investment : ℝ := sorry

/-- Represents B's investment in dollars -/
def B_investment : ℝ := 9000

/-- Represents the number of months A invested -/
def A_months : ℕ := 12

/-- Represents the number of months B invested -/
def B_months : ℕ := 7

/-- Represents A's share in the profit ratio -/
def A_ratio : ℕ := 2

/-- Represents B's share in the profit ratio -/
def B_ratio : ℕ := 3

theorem A_initial_investment :
  A_investment * A_months * B_ratio = B_investment * B_months * A_ratio :=
sorry

end A_initial_investment_l516_51651


namespace difference_not_necessarily_periodic_l516_51696

-- Define a periodic function
def Periodic (f : ℝ → ℝ) : Prop :=
  (∃ x y, f x ≠ f y) ∧ 
  (∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x)

-- Define functions g and h with their respective periods
def g_periodic (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 6) = g x

def h_periodic (h : ℝ → ℝ) : Prop :=
  ∀ x, h (x + 2 * Real.pi) = h x

-- Theorem statement
theorem difference_not_necessarily_periodic 
  (g h : ℝ → ℝ) 
  (hg : g_periodic g) 
  (hh : h_periodic h) :
  ¬ (∀ f : ℝ → ℝ, f = g - h → Periodic f) :=
sorry

end difference_not_necessarily_periodic_l516_51696


namespace a_is_four_l516_51649

def rounds_to_9430 (a b : ℕ) : Prop :=
  9000 + 100 * a + 30 + b ≥ 9425 ∧ 9000 + 100 * a + 30 + b < 9435

theorem a_is_four (a b : ℕ) (h : rounds_to_9430 a b) : a = 4 :=
sorry

end a_is_four_l516_51649


namespace circle_center_and_radius_l516_51670

theorem circle_center_and_radius :
  ∀ (x y : ℝ), 4*x^2 - 8*x + 4*y^2 + 24*y + 28 = 0 ↔ 
  (x - 1)^2 + (y + 3)^2 = 3 :=
by sorry

end circle_center_and_radius_l516_51670


namespace detergent_in_altered_solution_l516_51678

/-- Represents the ratio of bleach : detergent : water in a solution -/
structure SolutionRatio :=
  (bleach : ℕ)
  (detergent : ℕ)
  (water : ℕ)

/-- Calculates the new ratio after tripling bleach to detergent and halving detergent to water -/
def alter_ratio (r : SolutionRatio) : SolutionRatio :=
  { bleach := 3 * r.bleach,
    detergent := 2 * r.detergent,
    water := 4 * r.water }

/-- Calculates the amount of detergent in the altered solution -/
def detergent_amount (r : SolutionRatio) (water_amount : ℕ) : ℕ :=
  (r.detergent * water_amount) / r.water

theorem detergent_in_altered_solution :
  let original_ratio : SolutionRatio := { bleach := 4, detergent := 40, water := 100 }
  let altered_ratio := alter_ratio original_ratio
  detergent_amount altered_ratio 300 = 60 := by
  sorry

end detergent_in_altered_solution_l516_51678


namespace at_most_two_match_count_l516_51656

/-- The number of ways to arrange 5 balls in 5 boxes -/
def total_arrangements : ℕ := 120

/-- The number of ways to arrange 5 balls in 5 boxes where exactly 3 balls match their box number -/
def three_match_arrangements : ℕ := 10

/-- The number of ways to arrange 5 balls in 5 boxes where all 5 balls match their box number -/
def all_match_arrangement : ℕ := 1

/-- The number of ways to arrange 5 balls in 5 boxes such that at most two balls have the same number as their respective boxes -/
def at_most_two_match : ℕ := total_arrangements - three_match_arrangements - all_match_arrangement

theorem at_most_two_match_count : at_most_two_match = 109 := by
  sorry

end at_most_two_match_count_l516_51656


namespace max_sum_of_arithmetic_progression_l516_51622

def arithmetic_progression (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem max_sum_of_arithmetic_progression (a : ℕ → ℕ) (d : ℕ) :
  arithmetic_progression a d →
  (∀ n, a n > 0) →
  a 3 = 13 →
  (∀ n, a (n + 1) > a n) →
  (a (a 1) + a (a 2) + a (a 3) + a (a 4) + a (a 5) ≤ 365) ∧
  (∃ a d, arithmetic_progression a d ∧
          (∀ n, a n > 0) ∧
          a 3 = 13 ∧
          (∀ n, a (n + 1) > a n) ∧
          a (a 1) + a (a 2) + a (a 3) + a (a 4) + a (a 5) = 365) :=
by sorry

end max_sum_of_arithmetic_progression_l516_51622


namespace sqrt_two_sqrt_two_power_l516_51654

theorem sqrt_two_sqrt_two_power : (((2 * Real.sqrt 2) ^ 4).sqrt) ^ 3 = 512 := by
  sorry

end sqrt_two_sqrt_two_power_l516_51654


namespace smallest_n_satisfying_conditions_l516_51639

theorem smallest_n_satisfying_conditions (n : ℕ) : 
  n > 10 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 2 → 
  n ≥ 27 ∧ 
  (∀ m : ℕ, m > 10 ∧ m % 4 = 3 ∧ m % 5 = 2 → m ≥ n) :=
by sorry

end smallest_n_satisfying_conditions_l516_51639
