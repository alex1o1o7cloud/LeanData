import Mathlib

namespace cat_speed_l2300_230005

/-- Proves that given a rabbit running at 25 miles per hour, a cat with a 15-minute head start,
    and the rabbit taking 1 hour to catch up, the cat's speed is 20 miles per hour. -/
theorem cat_speed (rabbit_speed : ℝ) (head_start : ℝ) (catch_up_time : ℝ) :
  rabbit_speed = 25 →
  head_start = 0.25 →
  catch_up_time = 1 →
  ∃ cat_speed : ℝ,
    cat_speed * (head_start + catch_up_time) = rabbit_speed * catch_up_time ∧
    cat_speed = 20 :=
by sorry

end cat_speed_l2300_230005


namespace seventh_term_is_seven_l2300_230079

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  -- The first term of the sequence
  a : ℝ
  -- The common difference of the sequence
  d : ℝ
  -- The sum of the first five terms is 15
  sum_first_five : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 15
  -- The sixth term is 6
  sixth_term : a + 5*d = 6

/-- The seventh term of the arithmetic sequence is 7 -/
theorem seventh_term_is_seven (seq : ArithmeticSequence) : seq.a + 6*seq.d = 7 := by
  sorry

end seventh_term_is_seven_l2300_230079


namespace prob_at_least_two_females_l2300_230057

/-- The probability of selecting at least two females when choosing three finalists
    from a group of eight contestants consisting of five females and three males. -/
theorem prob_at_least_two_females (total : ℕ) (females : ℕ) (males : ℕ) (finalists : ℕ) :
  total = 8 →
  females = 5 →
  males = 3 →
  finalists = 3 →
  (Nat.choose females 2 * Nat.choose males 1 + Nat.choose females 3) / Nat.choose total finalists = 5 / 7 := by
  sorry

end prob_at_least_two_females_l2300_230057


namespace sqrt_difference_equality_l2300_230088

theorem sqrt_difference_equality : Real.sqrt (49 + 121) - Real.sqrt (36 - 9) = Real.sqrt 170 - Real.sqrt 27 := by
  sorry

end sqrt_difference_equality_l2300_230088


namespace system_solution_l2300_230028

theorem system_solution :
  ∀ x y : ℝ, x > 0 ∧ y > 0 →
  (y - 2 * Real.sqrt (x * y) - Real.sqrt (y / x) + 2 = 0) ∧
  (3 * x^2 * y^2 + y^4 = 84) →
  ((x = 1/3 ∧ y = 3) ∨ (x = (21/76)^(1/4) ∧ y = 2 * (84/19)^(1/4))) :=
by sorry

end system_solution_l2300_230028


namespace smallest_x_cos_equality_l2300_230090

theorem smallest_x_cos_equality : ∃ x : ℝ, 
  x > 30 ∧ 
  Real.cos (x * Real.pi / 180) = Real.cos ((2 * x + 10) * Real.pi / 180) ∧
  x < 117 ∧
  ∀ y : ℝ, y > 30 ∧ 
    Real.cos (y * Real.pi / 180) = Real.cos ((2 * y + 10) * Real.pi / 180) → 
    y ≥ x ∧
  ⌈x⌉ = 117 :=
sorry

end smallest_x_cos_equality_l2300_230090


namespace triangle_problem_l2300_230012

theorem triangle_problem (a b c A B C : Real) 
  (h1 : 2 * Real.sqrt 3 * a * b * Real.sin C = a^2 + b^2 - c^2)
  (h2 : a * Real.sin B = b * Real.cos A)
  (h3 : a = 2) :
  C = π/6 ∧ (1/2 * a * c * Real.sin B = (Real.sqrt 3 + 1) / 2) := by
  sorry

end triangle_problem_l2300_230012


namespace exponential_function_properties_l2300_230003

theorem exponential_function_properties (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  let f : ℝ → ℝ := fun x ↦ 2^x
  (f (x₁ + x₂) = f x₁ * f x₂) ∧
  (f (-x₁) = 1 / f x₁) ∧
  ((f x₁ - f x₂) / (x₁ - x₂) > 0) :=
by sorry

end exponential_function_properties_l2300_230003


namespace thursday_to_tuesday_ratio_l2300_230001

/-- Represents the number of crates sold on each day --/
structure DailySales where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- Theorem stating the ratio of Thursday to Tuesday sales --/
theorem thursday_to_tuesday_ratio
  (sales : DailySales)
  (h1 : sales.monday = 5)
  (h2 : sales.tuesday = 2 * sales.monday)
  (h3 : sales.wednesday = sales.tuesday - 2)
  (h4 : sales.monday + sales.tuesday + sales.wednesday + sales.thursday = 28) :
  sales.thursday / sales.tuesday = 1 / 2 := by
  sorry

end thursday_to_tuesday_ratio_l2300_230001


namespace fishing_problem_l2300_230058

theorem fishing_problem (jason ryan jeffery : ℕ) : 
  ryan = 3 * jason →
  jeffery = 2 * ryan →
  jeffery = 60 →
  jason + ryan + jeffery = 100 := by
sorry

end fishing_problem_l2300_230058


namespace class_size_proof_l2300_230074

/-- Given a student's rank from top and bottom in a class, 
    calculate the total number of students -/
def total_students (rank_from_top rank_from_bottom : ℕ) : ℕ :=
  rank_from_top + rank_from_bottom - 1

/-- Theorem stating that a class with a student ranking 24th from top 
    and 34th from bottom has 57 students in total -/
theorem class_size_proof :
  total_students 24 34 = 57 := by
  sorry

end class_size_proof_l2300_230074


namespace quadratic_inequality_range_l2300_230080

-- Define the quadratic function
def f (m x : ℝ) : ℝ := (m + 1) * x^2 + (m^2 - 2*m - 3) * x - m + 3

-- State the theorem
theorem quadratic_inequality_range (m : ℝ) :
  (∀ x, f m x > 0) ↔ (m ∈ Set.Icc (-1) 1 ∪ Set.Ioo 1 3) :=
sorry

end quadratic_inequality_range_l2300_230080


namespace ghee_mixture_problem_l2300_230016

theorem ghee_mixture_problem (x : ℝ) : 
  (0.6 * x = x - 0.4 * x) →  -- 60% is pure ghee, 40% is vanaspati
  (0.4 * x = 0.2 * (x + 10)) →  -- After adding 10 kg, vanaspati becomes 20%
  (x = 10) :=  -- The original quantity was 10 kg
by sorry

end ghee_mixture_problem_l2300_230016


namespace rulers_in_drawer_l2300_230097

theorem rulers_in_drawer (initial_rulers : ℕ) (added_rulers : ℕ) : 
  initial_rulers = 46 → added_rulers = 25 → initial_rulers + added_rulers = 71 := by
  sorry

end rulers_in_drawer_l2300_230097


namespace binomial_10_0_l2300_230092

theorem binomial_10_0 : (10 : ℕ).choose 0 = 1 := by
  sorry

end binomial_10_0_l2300_230092


namespace f_inverse_a_eq_28_l2300_230060

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x^(1/3)
  else if x ≥ 1 then 4*(x-1)
  else 0  -- undefined for x ≤ 0

theorem f_inverse_a_eq_28 (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : f a = f (a + 1)) :
  f (1 / a) = 28 := by
  sorry

end f_inverse_a_eq_28_l2300_230060


namespace right_triangle_hypotenuse_l2300_230020

/-- A right triangle with specific median lengths has a hypotenuse of 4√14 -/
theorem right_triangle_hypotenuse (a b : ℝ) (h_right : a^2 + b^2 = (a + b)^2 / 4) 
  (h_median1 : b^2 + (a/2)^2 = 34) (h_median2 : a^2 + (b/2)^2 = 36) : 
  (a + b) = 4 * Real.sqrt 14 := by
  sorry

end right_triangle_hypotenuse_l2300_230020


namespace range_of_t_l2300_230002

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def cubic_for_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x ≥ 0, f x = x^3

theorem range_of_t (f : ℝ → ℝ) (t : ℝ) :
  is_even_function f →
  cubic_for_nonneg f →
  (∀ x ∈ Set.Icc (2*t - 1) (2*t + 3), f (3*x - t) ≥ 8 * f x) →
  t ∈ Set.Iic (-3) ∪ {0} ∪ Set.Ici 1 :=
sorry

end range_of_t_l2300_230002


namespace least_N_for_P_condition_l2300_230063

def P (N k : ℕ) : ℚ :=
  (N + 1 - 2 * ⌈(2 / 5 : ℚ) * N⌉) / (N + 1 : ℚ)

theorem least_N_for_P_condition :
  ∀ N : ℕ, N > 0 ∧ N % 10 = 0 →
    (P N 2 < 8 / 10 ↔ N ≥ 10) ∧
    (∀ M : ℕ, M > 0 ∧ M % 10 = 0 ∧ M < 10 → P M 2 ≥ 8 / 10) :=
by sorry

end least_N_for_P_condition_l2300_230063


namespace calculation_proof_l2300_230033

theorem calculation_proof : 65 + 5 * 12 / (180 / 3) = 66 := by
  sorry

end calculation_proof_l2300_230033


namespace two_number_difference_l2300_230072

theorem two_number_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 20) : 
  |y - x| = 80 / 7 := by
  sorry

end two_number_difference_l2300_230072


namespace tomatoes_left_after_yesterday_l2300_230054

/-- The number of tomatoes left after yesterday's picking -/
def tomatoes_left (initial : ℕ) (picked_yesterday : ℕ) : ℕ :=
  initial - picked_yesterday

/-- Theorem: Given 160 initial tomatoes and 56 tomatoes picked yesterday,
    the number of tomatoes left after yesterday's picking is 104. -/
theorem tomatoes_left_after_yesterday :
  tomatoes_left 160 56 = 104 := by
  sorry

end tomatoes_left_after_yesterday_l2300_230054


namespace twenty_four_game_l2300_230045

theorem twenty_four_game (a b c d : ℤ) (e f g h : ℕ) : 
  (a = 3 ∧ b = 4 ∧ c = -6 ∧ d = 10) →
  (e = 3 ∧ f = 2 ∧ g = 6 ∧ h = 7) →
  ∃ (expr1 expr2 : ℤ → ℤ → ℤ → ℤ → ℤ),
    expr1 a b c d = 24 ∧
    expr2 e f g h = 24 :=
by sorry

end twenty_four_game_l2300_230045


namespace model1_best_fit_l2300_230041

-- Define the coefficient of determination for each model
def R2_model1 : ℝ := 0.976
def R2_model2 : ℝ := 0.776
def R2_model3 : ℝ := 0.076
def R2_model4 : ℝ := 0.351

-- Define a function to determine if a model has the best fitting effect
def has_best_fit (model_R2 : ℝ) : Prop :=
  model_R2 > R2_model2 ∧ model_R2 > R2_model3 ∧ model_R2 > R2_model4

-- Theorem stating that Model 1 has the best fitting effect
theorem model1_best_fit : has_best_fit R2_model1 := by
  sorry

end model1_best_fit_l2300_230041


namespace g_max_value_f_upper_bound_l2300_230095

noncomputable def f (x : ℝ) := Real.log (x + 1)

noncomputable def g (x : ℝ) := f x - x / 4 - 1

theorem g_max_value :
  ∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 2 * Real.log 2 - 7 / 4 := by sorry

theorem f_upper_bound (x : ℝ) (hx : x > 0) :
  f x < (Real.exp x - 1) / x^2 := by sorry

end g_max_value_f_upper_bound_l2300_230095


namespace equilateral_triangle_side_length_l2300_230073

/-- An equilateral triangle with a point inside -/
structure EquilateralTriangleWithPoint where
  -- The side length of the equilateral triangle
  side_length : ℝ
  -- The perpendicular distances from the point to the sides
  dist_to_AB : ℝ
  dist_to_BC : ℝ
  dist_to_CA : ℝ
  -- Ensure the triangle is equilateral and the point is inside
  side_length_pos : 0 < side_length
  dist_pos : 0 < dist_to_AB ∧ 0 < dist_to_BC ∧ 0 < dist_to_CA
  point_inside : dist_to_AB + dist_to_BC + dist_to_CA < side_length * Real.sqrt 3

/-- The theorem statement -/
theorem equilateral_triangle_side_length 
  (triangle : EquilateralTriangleWithPoint) 
  (h1 : triangle.dist_to_AB = 2)
  (h2 : triangle.dist_to_BC = 3)
  (h3 : triangle.dist_to_CA = 4) : 
  triangle.side_length = 6 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_side_length_l2300_230073


namespace linear_function_composition_l2300_230007

-- Define a linear function
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

-- State the theorem
theorem linear_function_composition (f : ℝ → ℝ) :
  LinearFunction f → (∀ x, f (f x) = 9 * x + 8) →
  (∀ x, f x = 3 * x + 2) ∨ (∀ x, f x = -3 * x - 4) :=
by sorry

end linear_function_composition_l2300_230007


namespace diagonal_sum_is_384_l2300_230013

/-- A cyclic hexagon with five sides of length 81 and one side of length 31 -/
structure CyclicHexagon where
  -- Five sides have length 81
  side_length : ℝ
  side_length_eq : side_length = 81
  -- One side (AB) has length 31
  AB_length : ℝ
  AB_length_eq : AB_length = 31

/-- The sum of the lengths of the three diagonals drawn from one vertex in the hexagon -/
def diagonal_sum (h : CyclicHexagon) : ℝ := sorry

/-- Theorem: The sum of the lengths of the three diagonals drawn from one vertex is 384 -/
theorem diagonal_sum_is_384 (h : CyclicHexagon) : diagonal_sum h = 384 := by sorry

end diagonal_sum_is_384_l2300_230013


namespace squirrel_count_l2300_230062

theorem squirrel_count :
  ∀ (first_count second_count : ℕ),
  second_count = first_count + (first_count / 3) →
  first_count + second_count = 28 →
  first_count = 21 :=
by
  sorry

end squirrel_count_l2300_230062


namespace smallest_perimeter_l2300_230029

/-- Represents a triangle with two fixed sides and one variable side --/
structure Triangle where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ
  h1 : side1 = 7
  h2 : side2 = 10
  h3 : Even side3
  h4 : side1 + side2 > side3
  h5 : side1 + side3 > side2
  h6 : side2 + side3 > side1

/-- The perimeter of a triangle --/
def perimeter (t : Triangle) : ℕ := t.side1 + t.side2 + t.side3

/-- Theorem: The smallest possible perimeter of the given triangle is 21 --/
theorem smallest_perimeter :
  ∃ (t : Triangle), ∀ (t' : Triangle), perimeter t ≤ perimeter t' ∧ perimeter t = 21 :=
sorry

end smallest_perimeter_l2300_230029


namespace order_of_even_increasing_function_l2300_230059

-- Define an even function f on ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define an increasing function on [0, +∞)
def increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

-- Main theorem
theorem order_of_even_increasing_function (f : ℝ → ℝ) 
  (h_even : even_function f) (h_incr : increasing_on_nonneg f) :
  f (-2) < f 3 ∧ f 3 < f (-π) :=
by
  sorry


end order_of_even_increasing_function_l2300_230059


namespace burger_cost_is_12_l2300_230066

/-- The cost of each burger Owen bought in June -/
def burger_cost (burgers_per_day : ℕ) (total_spent : ℕ) (days_in_june : ℕ) : ℚ :=
  total_spent / (burgers_per_day * days_in_june)

/-- Theorem stating that each burger costs 12 dollars -/
theorem burger_cost_is_12 :
  burger_cost 2 720 30 = 12 := by
  sorry

end burger_cost_is_12_l2300_230066


namespace light_nanosecond_distance_l2300_230026

/-- The speed of light in meters per second -/
def speed_of_light : ℝ := 3e8

/-- One billionth of a second in seconds -/
def one_billionth : ℝ := 1e-9

/-- The distance traveled by light in one billionth of a second in meters -/
def light_nanosecond : ℝ := speed_of_light * one_billionth

/-- Conversion factor from meters to centimeters -/
def meters_to_cm : ℝ := 100

theorem light_nanosecond_distance :
  light_nanosecond * meters_to_cm = 30 := by sorry

end light_nanosecond_distance_l2300_230026


namespace second_day_sales_l2300_230043

def ice_cream_sales (x : ℕ) : List ℕ := [100, x, 109, 96, 103, 96, 105]

def mean (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

theorem second_day_sales :
  ∃ (x : ℕ), mean (ice_cream_sales x) = 100.1 ∧ x = 92 := by sorry

end second_day_sales_l2300_230043


namespace g_odd_g_strictly_increasing_l2300_230036

/-- The function g(x) = lg(x + √(x^2 + 1)) -/
noncomputable def g (x : ℝ) : ℝ := Real.log (x + Real.sqrt (x^2 + 1))

/-- g is an odd function -/
theorem g_odd : ∀ x, g (-x) = -g x := by sorry

/-- g is strictly increasing on ℝ -/
theorem g_strictly_increasing : StrictMono g := by sorry

end g_odd_g_strictly_increasing_l2300_230036


namespace divisible_by_19_l2300_230064

theorem divisible_by_19 (n : ℕ) : 
  19 ∣ (12000 + 3 * 10^n + 8) := by
  sorry

end divisible_by_19_l2300_230064


namespace function_with_two_zeros_l2300_230056

theorem function_with_two_zeros 
  (f : ℝ → ℝ) 
  (hcont : ContinuousOn f (Set.Icc 1 3))
  (h1 : f 1 * f 2 < 0)
  (h2 : f 2 * f 3 < 0) :
  ∃ (x y : ℝ), x ∈ Set.Ioo 1 3 ∧ y ∈ Set.Ioo 1 3 ∧ x ≠ y ∧ f x = 0 ∧ f y = 0 :=
sorry

end function_with_two_zeros_l2300_230056


namespace perpendicular_tangents_intersection_l2300_230046

/-- Given two curves y = x^2 - 1 and y = 1 + x^3 with perpendicular tangents at x = x_0,
    prove that x_0 = -1 / ∛6 -/
theorem perpendicular_tangents_intersection (x_0 : ℝ) :
  (2 * x_0) * (3 * x_0^2) = -1 →
  x_0 = -1 / Real.rpow 6 (1/3) := by
sorry

end perpendicular_tangents_intersection_l2300_230046


namespace sqrt_two_times_sqrt_eight_equals_four_l2300_230004

theorem sqrt_two_times_sqrt_eight_equals_four :
  Real.sqrt 2 * Real.sqrt 8 = 4 := by
  sorry

end sqrt_two_times_sqrt_eight_equals_four_l2300_230004


namespace sweatshirt_cost_is_15_l2300_230042

def hannah_shopping (sweatshirt_cost : ℝ) : Prop :=
  let num_sweatshirts : ℕ := 3
  let num_tshirts : ℕ := 2
  let tshirt_cost : ℝ := 10
  let total_spent : ℝ := 65
  (num_sweatshirts * sweatshirt_cost) + (num_tshirts * tshirt_cost) = total_spent

theorem sweatshirt_cost_is_15 : 
  ∃ (sweatshirt_cost : ℝ), hannah_shopping sweatshirt_cost ∧ sweatshirt_cost = 15 :=
by
  sorry

end sweatshirt_cost_is_15_l2300_230042


namespace sqrt_a_sqrt_a_sqrt_a_l2300_230034

theorem sqrt_a_sqrt_a_sqrt_a (a : ℝ) (ha : a ≥ 0) : 
  Real.sqrt (a * Real.sqrt a * Real.sqrt a) = a := by
sorry

end sqrt_a_sqrt_a_sqrt_a_l2300_230034


namespace probability_of_black_ball_l2300_230047

theorem probability_of_black_ball (prob_red prob_white : ℝ) 
  (h1 : prob_red = 0.42)
  (h2 : prob_white = 0.28)
  (h3 : prob_red + prob_white + prob_black = 1) :
  prob_black = 0.3 := by
  sorry

end probability_of_black_ball_l2300_230047


namespace marianne_age_always_12_more_than_bella_l2300_230018

/-- Represents the age difference between Marianne and Bella -/
def age_difference : ℕ := 12

/-- Marianne's age when Bella is 8 years old -/
def marianne_age_when_bella_8 : ℕ := 20

/-- Bella's age when Marianne is 30 years old -/
def bella_age_when_marianne_30 : ℕ := 18

/-- Marianne's age as a function of Bella's age -/
def marianne_age (bella_age : ℕ) : ℕ := bella_age + age_difference

theorem marianne_age_always_12_more_than_bella :
  ∀ (bella_age : ℕ),
    marianne_age bella_age = bella_age + age_difference :=
by
  sorry

#check marianne_age_always_12_more_than_bella

end marianne_age_always_12_more_than_bella_l2300_230018


namespace correct_calculation_l2300_230081

theorem correct_calculation : (-0.5)^2010 * 2^2011 = 2 := by
  sorry

end correct_calculation_l2300_230081


namespace preceding_binary_l2300_230039

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : Nat) : List Bool :=
  let rec aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc
    else aux (m / 2) ((m % 2 = 1) :: acc)
  aux n []

def M : List Bool := [true, false, true, false, true, false]

theorem preceding_binary (M : List Bool) : 
  M = [true, false, true, false, true, false] → 
  decimal_to_binary (binary_to_decimal M - 1) = [true, false, true, false, false, true] := by
  sorry

end preceding_binary_l2300_230039


namespace solution_set_l2300_230015

theorem solution_set (x : ℝ) : 2 < x / (3 * x - 7) ∧ x / (3 * x - 7) ≤ 7 ↔ 49 / 20 < x ∧ x ≤ 14 / 5 := by
  sorry

end solution_set_l2300_230015


namespace cycle_selling_price_l2300_230000

theorem cycle_selling_price (cost_price : ℝ) (loss_percentage : ℝ) (selling_price : ℝ) : 
  cost_price = 1400 → 
  loss_percentage = 5 → 
  selling_price = cost_price * (1 - loss_percentage / 100) → 
  selling_price = 1330 := by
sorry

end cycle_selling_price_l2300_230000


namespace nancy_bills_denomination_l2300_230082

/-- Given Nancy has 9 bills of equal denomination and a total of 45 dollars, 
    the denomination of each bill is $5. -/
theorem nancy_bills_denomination (num_bills : ℕ) (total_amount : ℕ) (denomination : ℕ) :
  num_bills = 9 →
  total_amount = 45 →
  num_bills * denomination = total_amount →
  denomination = 5 := by
sorry

end nancy_bills_denomination_l2300_230082


namespace lori_marble_sharing_l2300_230037

def total_marbles : ℕ := 30
def marbles_per_friend : ℕ := 6

theorem lori_marble_sharing :
  total_marbles / marbles_per_friend = 5 := by
  sorry

end lori_marble_sharing_l2300_230037


namespace max_intersection_points_l2300_230053

-- Define the geometric objects
def Circle : Type := Unit
def Ellipse : Type := Unit
def Line : Type := Unit

-- Define the intersection function
def intersection_points (c : Circle) (e : Ellipse) (l : Line) : ℕ := sorry

-- Theorem statement
theorem max_intersection_points :
  ∃ (c : Circle) (e : Ellipse) (l : Line),
    ∀ (c' : Circle) (e' : Ellipse) (l' : Line),
      intersection_points c e l ≥ intersection_points c' e' l' ∧
      intersection_points c e l = 8 :=
sorry

end max_intersection_points_l2300_230053


namespace parallel_perpendicular_line_coefficient_l2300_230049

/-- Given two lines in the plane, if there exists a third line parallel to one and perpendicular to the other, prove that the coefficient k in the equations must be zero. -/
theorem parallel_perpendicular_line_coefficient (k : ℝ) : 
  (∃ (c : ℝ), ∀ (x y : ℝ), (3 * x - k * y + c = 0) ∧ 
    ((3 * x - k * y + c = 0) ↔ (3 * x - k * y + 6 = 0)) ∧
    ((3 * k + (-k) * 1 = 0) ↔ (k * x + y + 1 = 0))) → 
  k = 0 := by
sorry

end parallel_perpendicular_line_coefficient_l2300_230049


namespace tv_price_change_l2300_230065

theorem tv_price_change (original_price : ℝ) (h : original_price > 0) :
  let price_after_decrease := original_price * (1 - 0.2)
  let final_price := price_after_decrease * (1 + 0.4)
  let net_change := (final_price - original_price) / original_price
  net_change = 0.12 := by
sorry

end tv_price_change_l2300_230065


namespace find_number_l2300_230089

theorem find_number : ∃! x : ℝ, ((x / 12 - 32) * 3 - 45) = 159 := by sorry

end find_number_l2300_230089


namespace lotto_ticket_cost_l2300_230098

/-- Proves that the cost per ticket is $2 given the lottery conditions --/
theorem lotto_ticket_cost (total_tickets : ℕ) (winning_percentage : ℚ)
  (five_dollar_winners_percentage : ℚ) (grand_prize_tickets : ℕ)
  (grand_prize_amount : ℕ) (other_winners_average : ℕ) (total_profit : ℕ) :
  total_tickets = 200 →
  winning_percentage = 1/5 →
  five_dollar_winners_percentage = 4/5 →
  grand_prize_tickets = 1 →
  grand_prize_amount = 5000 →
  other_winners_average = 10 →
  total_profit = 4830 →
  ∃ (cost_per_ticket : ℚ), cost_per_ticket = 2 :=
by
  sorry

end lotto_ticket_cost_l2300_230098


namespace total_participants_grandmasters_top_positions_l2300_230067

/-- A round-robin chess tournament with grandmasters and masters -/
structure ChessTournament where
  num_grandmasters : ℕ
  num_masters : ℕ
  total_points_grandmasters : ℕ
  total_points_masters : ℕ

/-- The conditions of the tournament -/
def tournament_conditions (t : ChessTournament) : Prop :=
  t.num_masters = 3 * t.num_grandmasters ∧
  t.total_points_masters = (12 * t.total_points_grandmasters) / 10 ∧
  t.total_points_grandmasters + t.total_points_masters = (t.num_grandmasters + t.num_masters) * (t.num_grandmasters + t.num_masters - 1)

/-- The theorem stating the total number of participants -/
theorem total_participants (t : ChessTournament) (h : tournament_conditions t) : 
  t.num_grandmasters + t.num_masters = 12 := by
  sorry

/-- The theorem stating that grandmasters took the top positions -/
theorem grandmasters_top_positions (t : ChessTournament) (h : tournament_conditions t) : 
  t.num_grandmasters ≤ 3 ∧ t.num_grandmasters > 0 := by
  sorry

end total_participants_grandmasters_top_positions_l2300_230067


namespace lava_lamp_probability_l2300_230008

/-- The number of green lava lamps -/
def green_lamps : ℕ := 4

/-- The number of purple lava lamps -/
def purple_lamps : ℕ := 4

/-- The total number of lamps -/
def total_lamps : ℕ := green_lamps + purple_lamps

/-- The number of lamps in each row -/
def lamps_per_row : ℕ := 4

/-- The number of rows -/
def num_rows : ℕ := 2

/-- The number of lamps turned on -/
def lamps_on : ℕ := 4

/-- The probability of the specific arrangement -/
def specific_arrangement_probability : ℚ := 1 / 7

theorem lava_lamp_probability :
  (green_lamps = 4) →
  (purple_lamps = 4) →
  (total_lamps = green_lamps + purple_lamps) →
  (lamps_per_row = 4) →
  (num_rows = 2) →
  (lamps_on = 4) →
  (specific_arrangement_probability = 1 / 7) := by
  sorry

end lava_lamp_probability_l2300_230008


namespace units_digit_17_1987_l2300_230069

theorem units_digit_17_1987 : (17^1987) % 10 = 3 := by
  sorry

end units_digit_17_1987_l2300_230069


namespace total_matches_l2300_230040

def dozen : ℕ := 12

def boxes : ℕ := 5 * dozen

def matches_per_box : ℕ := 20

theorem total_matches : boxes * matches_per_box = 1200 := by
  sorry

end total_matches_l2300_230040


namespace apple_price_difference_l2300_230019

/-- The price difference between Shimla apples and Fuji apples -/
def price_difference (shimla_price fuji_price : ℝ) : ℝ :=
  shimla_price - fuji_price

/-- The condition that the sum of Shimla and Red Delicious prices is 250 more than Red Delicious and Fuji -/
def price_condition (shimla_price red_delicious_price fuji_price : ℝ) : Prop :=
  shimla_price + red_delicious_price = red_delicious_price + fuji_price + 250

theorem apple_price_difference 
  (shimla_price red_delicious_price fuji_price : ℝ) 
  (h : price_condition shimla_price red_delicious_price fuji_price) : 
  price_difference shimla_price fuji_price = 250 := by
  sorry

end apple_price_difference_l2300_230019


namespace crayon_division_l2300_230077

theorem crayon_division (total_crayons : ℕ) (num_people : ℕ) (crayons_per_person : ℕ) :
  total_crayons = 24 →
  num_people = 3 →
  crayons_per_person = total_crayons / num_people →
  crayons_per_person = 8 :=
by
  sorry

end crayon_division_l2300_230077


namespace cone_base_radius_l2300_230024

theorem cone_base_radius (r : ℝ) (θ : ℝ) (base_radius : ℝ) : 
  r = 9 → θ = 240 * π / 180 → base_radius = r * θ / (2 * π) → base_radius = 6 := by
  sorry

end cone_base_radius_l2300_230024


namespace amanda_hourly_rate_l2300_230030

/-- Amanda's work scenario --/
structure AmandaWork where
  hours_per_day : ℕ
  pay_percentage : ℚ
  reduced_pay : ℚ

/-- Calculate Amanda's hourly rate --/
def hourly_rate (w : AmandaWork) : ℚ :=
  (w.reduced_pay / w.pay_percentage) / w.hours_per_day

/-- Theorem: Amanda's hourly rate is $50 --/
theorem amanda_hourly_rate (w : AmandaWork) 
  (h1 : w.hours_per_day = 10)
  (h2 : w.pay_percentage = 4/5)
  (h3 : w.reduced_pay = 400) :
  hourly_rate w = 50 := by
  sorry

#eval hourly_rate { hours_per_day := 10, pay_percentage := 4/5, reduced_pay := 400 }

end amanda_hourly_rate_l2300_230030


namespace min_reciprocal_sum_l2300_230011

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → 1/x + 1/y ≤ 1/a + 1/b) ∧ (1/x + 1/y = 4 → x = 1/2 ∧ y = 1/2) :=
by sorry

end min_reciprocal_sum_l2300_230011


namespace vegetable_garden_theorem_l2300_230025

def vegetable_garden_total (potatoes cucumbers tomatoes peppers carrots : ℕ) : Prop :=
  potatoes = 1200 ∧
  cucumbers = potatoes - 160 ∧
  tomatoes = 4 * cucumbers ∧
  peppers * peppers = cucumbers * tomatoes ∧
  carrots = (cucumbers + tomatoes) + (cucumbers + tomatoes) / 5 ∧
  potatoes + cucumbers + tomatoes + peppers + carrots = 14720

theorem vegetable_garden_theorem :
  ∃ (potatoes cucumbers tomatoes peppers carrots : ℕ),
    vegetable_garden_total potatoes cucumbers tomatoes peppers carrots :=
by
  sorry

end vegetable_garden_theorem_l2300_230025


namespace optimal_candy_purchase_l2300_230083

/-- Represents the number of candies in a purchase strategy -/
structure CandyPurchase where
  singles : ℕ
  packs : ℕ
  bulks : ℕ

/-- Calculates the total cost of a purchase strategy -/
def totalCost (p : CandyPurchase) : ℕ :=
  p.singles + 3 * p.packs + 4 * p.bulks

/-- Calculates the total number of candies in a purchase strategy -/
def totalCandies (p : CandyPurchase) : ℕ :=
  p.singles + 4 * p.packs + 7 * p.bulks

/-- Represents a valid purchase strategy within the $10 budget -/
def ValidPurchase (p : CandyPurchase) : Prop :=
  totalCost p ≤ 10

/-- The maximum number of candies that can be purchased with $10 -/
def maxCandies : ℕ := 16

theorem optimal_candy_purchase :
  ∀ p : CandyPurchase, ValidPurchase p → totalCandies p ≤ maxCandies ∧
  ∃ q : CandyPurchase, ValidPurchase q ∧ totalCandies q = maxCandies :=
by sorry

end optimal_candy_purchase_l2300_230083


namespace remainder_b_107_mod_64_l2300_230094

theorem remainder_b_107_mod_64 : (5^107 + 9^107) % 64 = 8 := by
  sorry

end remainder_b_107_mod_64_l2300_230094


namespace english_test_average_l2300_230068

theorem english_test_average (avg_two_months : ℝ) (third_month_score : ℝ) :
  avg_two_months = 86 →
  third_month_score = 98 →
  (2 * avg_two_months + third_month_score) / 3 = 90 := by
  sorry

end english_test_average_l2300_230068


namespace number_of_operations_indicates_quality_l2300_230014

-- Define a type for algorithms
structure Algorithm : Type where
  name : String

-- Define a measure for the number of operations
def numberOfOperations (a : Algorithm) : ℕ := sorry

-- Define a measure for algorithm quality
def algorithmQuality (a : Algorithm) : ℝ := sorry

-- Define a measure for computer speed
def computerSpeed : ℝ := sorry

-- State the theorem
theorem number_of_operations_indicates_quality (a : Algorithm) :
  computerSpeed > 0 →
  algorithmQuality a = (1 / numberOfOperations a) * computerSpeed :=
sorry

end number_of_operations_indicates_quality_l2300_230014


namespace surface_area_circumscribed_sphere_l2300_230038

/-- The surface area of a sphere circumscribed about a rectangular solid -/
theorem surface_area_circumscribed_sphere
  (length width height : ℝ)
  (h_length : length = 2)
  (h_width : width = 1)
  (h_height : height = 2) :
  4 * Real.pi * ((length^2 + width^2 + height^2) / 4) = 9 * Real.pi :=
by sorry

end surface_area_circumscribed_sphere_l2300_230038


namespace football_team_progress_l2300_230009

def team_progress (loss : Int) (gain : Int) : Int :=
  gain - loss

theorem football_team_progress :
  team_progress 5 8 = 3 := by sorry

end football_team_progress_l2300_230009


namespace opposite_face_is_t_l2300_230091

-- Define the faces of the cube
inductive Face : Type
  | p | q | r | s | t | u

-- Define the cube structure
structure Cube where
  top : Face
  right : Face
  left : Face
  bottom : Face
  front : Face
  back : Face

-- Define the conditions of the problem
def problem_cube : Cube :=
  { top := Face.p
  , right := Face.q
  , left := Face.r
  , bottom := Face.t  -- We'll prove this is correct
  , front := Face.s   -- Arbitrary assignment for remaining faces
  , back := Face.u }  -- Arbitrary assignment for remaining faces

-- Theorem statement
theorem opposite_face_is_t (c : Cube) :
  c.top = Face.p → c.right = Face.q → c.left = Face.r → c.bottom = Face.t :=
by
  sorry

end opposite_face_is_t_l2300_230091


namespace orange_harvest_theorem_l2300_230093

/-- The number of sacks of oranges harvested per day -/
def sacks_per_day : ℕ := 38

/-- The number of days of harvest -/
def harvest_days : ℕ := 49

/-- The total number of sacks of oranges harvested after the harvest period -/
def total_sacks : ℕ := sacks_per_day * harvest_days

theorem orange_harvest_theorem : total_sacks = 1862 := by
  sorry

end orange_harvest_theorem_l2300_230093


namespace decimal_division_l2300_230078

theorem decimal_division (x y : ℚ) (hx : x = 45/100) (hy : y = 5/1000) : x / y = 90 := by
  sorry

end decimal_division_l2300_230078


namespace fixed_point_theorem_l2300_230076

/-- The fixed point through which all lines of the form kx-y+1=3k pass -/
def fixed_point : ℝ × ℝ := (3, 1)

/-- The equation of the line parameterized by k -/
def line_equation (k x y : ℝ) : Prop := k*x - y + 1 = 3*k

/-- Theorem stating that the fixed_point is the unique point through which all lines pass -/
theorem fixed_point_theorem :
  ∀ (k : ℝ), line_equation k (fixed_point.1) (fixed_point.2) ∧
  ∀ (x y : ℝ), (∀ (k : ℝ), line_equation k x y) → (x, y) = fixed_point :=
by sorry

end fixed_point_theorem_l2300_230076


namespace remaining_pictures_l2300_230044

/-- The number of pictures Haley took at the zoo -/
def zoo_pictures : ℕ := 50

/-- The number of pictures Haley took at the museum -/
def museum_pictures : ℕ := 8

/-- The number of pictures Haley deleted -/
def deleted_pictures : ℕ := 38

/-- Theorem: The number of pictures Haley still has from her vacation is 20 -/
theorem remaining_pictures : 
  zoo_pictures + museum_pictures - deleted_pictures = 20 := by
  sorry

end remaining_pictures_l2300_230044


namespace smallest_sum_reciprocals_l2300_230032

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 ∧ (a : ℕ) + (b : ℕ) = 64 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 15 → (c : ℕ) + (d : ℕ) ≥ 64 :=
by sorry

end smallest_sum_reciprocals_l2300_230032


namespace sequence_sum_values_l2300_230070

def is_valid_sequence (a b : ℕ → ℕ) : Prop :=
  (∀ n, a (n + 1) > a n) ∧ 
  (∀ n, b (n + 1) > b n) ∧
  (a 10 = b 10) ∧ 
  (a 10 < 2017) ∧
  (∀ n, a (n + 2) = a (n + 1) + a n) ∧
  (∀ n, b (n + 1) = 2 * b n)

theorem sequence_sum_values (a b : ℕ → ℕ) :
  is_valid_sequence a b → (a 1 + b 1 = 13 ∨ a 1 + b 1 = 20) :=
by
  sorry

end sequence_sum_values_l2300_230070


namespace doll_cost_is_15_l2300_230051

/-- Represents the cost of gifts for each sister -/
def gift_cost : ℕ := 60

/-- Represents the number of dolls bought for the younger sister -/
def num_dolls : ℕ := 4

/-- Represents the number of Lego sets bought for the older sister -/
def num_lego_sets : ℕ := 3

/-- Represents the cost of each Lego set -/
def lego_set_cost : ℕ := 20

/-- Theorem stating that the cost of each doll is $15 -/
theorem doll_cost_is_15 : 
  gift_cost = num_lego_sets * lego_set_cost ∧ 
  gift_cost = num_dolls * 15 := by
  sorry

end doll_cost_is_15_l2300_230051


namespace y_exceeds_x_by_100_percent_l2300_230031

theorem y_exceeds_x_by_100_percent (x y : ℝ) (h : x = 0.5 * y) : 
  (y - x) / x = 1 := by sorry

end y_exceeds_x_by_100_percent_l2300_230031


namespace polynomial_factorization_l2300_230010

theorem polynomial_factorization (x : ℝ) : 
  ∃ (q : ℝ → ℝ), (x^2 - x - 1) * q x = (-987 * x^18 + 2584 * x^17 + 1) := by
  sorry

end polynomial_factorization_l2300_230010


namespace total_cats_sum_l2300_230048

/-- The number of cats owned by Mr. Thompson -/
def thompson_cats : ℝ := 15.5

/-- The number of cats owned by Mrs. Sheridan -/
def sheridan_cats : ℝ := 11.6

/-- The number of cats owned by Mrs. Garrett -/
def garrett_cats : ℝ := 24.2

/-- The number of cats owned by Mr. Ravi -/
def ravi_cats : ℝ := 18.3

/-- The total number of cats owned by all four people -/
def total_cats : ℝ := thompson_cats + sheridan_cats + garrett_cats + ravi_cats

theorem total_cats_sum :
  total_cats = 69.6 := by sorry

end total_cats_sum_l2300_230048


namespace quadratic_equation_from_means_l2300_230052

theorem quadratic_equation_from_means (η ζ : ℝ) 
  (h_arithmetic_mean : (η + ζ) / 2 = 7)
  (h_geometric_mean : Real.sqrt (η * ζ) = 8) :
  ∀ x : ℝ, x^2 - 14*x + 64 = 0 ↔ (x = η ∨ x = ζ) :=
by sorry

end quadratic_equation_from_means_l2300_230052


namespace sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l2300_230085

/-- The function f(x) = |x - a| is increasing on [-3, +∞) -/
def is_increasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x ∈ Set.Ici (-3) → y ∈ Set.Ici (-3) → x ≤ y → |x - a| ≤ |y - a|

/-- "a = -3" is a sufficient condition -/
theorem sufficient_condition (a : ℝ) (h : a = -3) : is_increasing_on_interval a :=
sorry

/-- "a = -3" is not a necessary condition -/
theorem not_necessary_condition : ∃ a : ℝ, a ≠ -3 ∧ is_increasing_on_interval a :=
sorry

/-- "a = -3" is a sufficient but not necessary condition -/
theorem sufficient_but_not_necessary :
  (∀ a : ℝ, a = -3 → is_increasing_on_interval a) ∧
  (∃ a : ℝ, a ≠ -3 ∧ is_increasing_on_interval a) :=
sorry

end sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l2300_230085


namespace village_cats_l2300_230086

theorem village_cats (total_cats : ℕ) 
  (striped_ratio : ℚ) (spotted_ratio : ℚ) 
  (fluffy_striped_ratio : ℚ) (fluffy_spotted_ratio : ℚ)
  (h_total : total_cats = 180)
  (h_striped : striped_ratio = 1/2)
  (h_spotted : spotted_ratio = 1/3)
  (h_fluffy_striped : fluffy_striped_ratio = 1/8)
  (h_fluffy_spotted : fluffy_spotted_ratio = 3/7) :
  ⌊striped_ratio * total_cats * fluffy_striped_ratio⌋ + 
  ⌊spotted_ratio * total_cats * fluffy_spotted_ratio⌋ = 36 := by
sorry

end village_cats_l2300_230086


namespace rectangular_solid_surface_area_bounds_l2300_230035

/-- Represents the dimensions of a rectangular solid --/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the surface area of a rectangular solid --/
def surfaceArea (d : Dimensions) : ℕ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- Checks if the given dimensions use exactly 12 unit cubes --/
def usestwelveCubes (d : Dimensions) : Prop :=
  d.length * d.width * d.height = 12

theorem rectangular_solid_surface_area_bounds :
  ∃ (min max : ℕ),
    (∀ d : Dimensions, usestwelveCubes d → min ≤ surfaceArea d) ∧
    (∃ d : Dimensions, usestwelveCubes d ∧ surfaceArea d = min) ∧
    (∀ d : Dimensions, usestwelveCubes d → surfaceArea d ≤ max) ∧
    (∃ d : Dimensions, usestwelveCubes d ∧ surfaceArea d = max) ∧
    min = 32 ∧ max = 50 := by
  sorry

end rectangular_solid_surface_area_bounds_l2300_230035


namespace graces_coins_worth_l2300_230023

/-- The total worth of Grace's coins in pennies -/
def total_worth (pennies nickels dimes quarters : ℕ) : ℕ :=
  pennies + 5 * nickels + 10 * dimes + 25 * quarters

/-- Theorem stating that Grace's coins are worth 550 pennies -/
theorem graces_coins_worth : total_worth 25 15 20 10 = 550 := by
  sorry

end graces_coins_worth_l2300_230023


namespace largest_in_set_l2300_230075

theorem largest_in_set (a : ℝ) (h : a = -3) :
  let S : Set ℝ := {-2*a, 3*a, 18/a, a^3, 2}
  ∀ x ∈ S, -2*a ≥ x :=
by sorry

end largest_in_set_l2300_230075


namespace find_number_to_multiply_l2300_230022

theorem find_number_to_multiply : ∃ x : ℤ, 43 * x - 34 * x = 1233 :=
by sorry

end find_number_to_multiply_l2300_230022


namespace initial_alcohol_percentage_l2300_230006

theorem initial_alcohol_percentage
  (initial_volume : Real)
  (added_alcohol : Real)
  (final_percentage : Real)
  (h1 : initial_volume = 6)
  (h2 : added_alcohol = 1.2)
  (h3 : final_percentage = 50)
  (h4 : (initial_percentage / 100) * initial_volume + added_alcohol = 
        (final_percentage / 100) * (initial_volume + added_alcohol)) :
  initial_percentage = 40 := by
  sorry

#check initial_alcohol_percentage

end initial_alcohol_percentage_l2300_230006


namespace percentage_difference_l2300_230087

theorem percentage_difference : (0.6 * 50) - (0.42 * 30) = 17.4 := by
  sorry

end percentage_difference_l2300_230087


namespace walnut_trees_increase_l2300_230050

/-- Calculates the total number of walnut trees after planting given the initial number and percentage increase -/
def total_trees_after_planting (initial_trees : ℕ) (percent_increase : ℕ) : ℕ :=
  initial_trees + (initial_trees * percent_increase) / 100

/-- Theorem stating that with 22 initial trees and 150% increase, the total after planting is 55 -/
theorem walnut_trees_increase :
  total_trees_after_planting 22 150 = 55 := by
  sorry

#eval total_trees_after_planting 22 150

end walnut_trees_increase_l2300_230050


namespace smaller_box_length_l2300_230071

/-- Given a larger box and smaller boxes with specified dimensions, 
    proves that the length of the smaller box is 60 cm when 1000 boxes fit. -/
theorem smaller_box_length 
  (large_box_length : ℕ) 
  (large_box_width : ℕ) 
  (large_box_height : ℕ)
  (small_box_width : ℕ) 
  (small_box_height : ℕ)
  (max_small_boxes : ℕ)
  (h1 : large_box_length = 600)
  (h2 : large_box_width = 500)
  (h3 : large_box_height = 400)
  (h4 : small_box_width = 50)
  (h5 : small_box_height = 40)
  (h6 : max_small_boxes = 1000) :
  ∃ (small_box_length : ℕ), 
    small_box_length = 60 ∧ 
    (small_box_length * small_box_width * small_box_height) * max_small_boxes ≤ 
      large_box_length * large_box_width * large_box_height :=
by sorry

end smaller_box_length_l2300_230071


namespace square_of_85_l2300_230021

theorem square_of_85 : (85 : ℕ)^2 = 7225 := by
  sorry

end square_of_85_l2300_230021


namespace seconds_in_minutes_l2300_230027

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes we are converting to seconds -/
def minutes : ℚ := 12.5

/-- Theorem: The number of seconds in 12.5 minutes is 750 -/
theorem seconds_in_minutes : (minutes * seconds_per_minute : ℚ) = 750 := by
  sorry

end seconds_in_minutes_l2300_230027


namespace nells_baseball_cards_l2300_230017

/-- Nell's card collection problem -/
theorem nells_baseball_cards
  (initial_ace : ℕ)
  (final_ace final_baseball : ℕ)
  (ace_difference baseball_difference : ℕ)
  (h1 : initial_ace = 18)
  (h2 : final_ace = 55)
  (h3 : final_baseball = 178)
  (h4 : baseball_difference = 123)
  (h5 : final_baseball = final_ace + baseball_difference)
  : final_baseball + baseball_difference = 301 := by
  sorry

#check nells_baseball_cards

end nells_baseball_cards_l2300_230017


namespace boys_passed_exam_l2300_230055

theorem boys_passed_exam (total : ℕ) (overall_avg passed_avg failed_avg : ℚ) : 
  total = 120 →
  overall_avg = 36 →
  passed_avg = 39 →
  failed_avg = 15 →
  ∃ (passed : ℕ), passed = 105 ∧ 
    (passed : ℚ) * passed_avg + (total - passed : ℚ) * failed_avg = (total : ℚ) * overall_avg :=
by sorry

end boys_passed_exam_l2300_230055


namespace problem_statement_l2300_230061

theorem problem_statement (x y : ℝ) :
  |x - 8*y| + (4*y - 1)^2 = 0 → (x + 2*y)^3 = 125/8 := by
sorry

end problem_statement_l2300_230061


namespace inequality_proof_l2300_230096

theorem inequality_proof (b c : ℝ) (hb : b > 0) (hc : c > 0) :
  (b - c)^2011 * (b + c)^2011 * (c - b)^2011 ≥ (b^2011 - c^2011) * (b^2011 + c^2011) * (c^2011 - b^2011) := by
  sorry

end inequality_proof_l2300_230096


namespace car_clock_accuracy_l2300_230084

def actual_time (start_time : ℕ) (elapsed_time : ℕ) (gain_rate : ℚ) : ℚ :=
  start_time + elapsed_time / gain_rate

theorem car_clock_accuracy (start_time : ℕ) (elapsed_time : ℕ) : 
  start_time = 8 * 60 →  -- 8:00 AM in minutes
  elapsed_time = 14 * 60 →  -- 14 hours from 8:00 AM to 10:00 PM in minutes
  actual_time start_time elapsed_time (37/36) = 21 * 60 + 47  -- 9:47 PM in minutes
  := by sorry

end car_clock_accuracy_l2300_230084


namespace games_that_didnt_work_l2300_230099

/-- The number of games that didn't work, given Edward's game purchases and good games. -/
theorem games_that_didnt_work (friend_games garage_games good_games : ℕ) : 
  friend_games = 41 → garage_games = 14 → good_games = 24 → 
  friend_games + garage_games - good_games = 31 := by
  sorry

end games_that_didnt_work_l2300_230099
