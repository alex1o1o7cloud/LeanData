import Mathlib

namespace NUMINAMATH_CALUDE_ice_cream_consumption_l1035_103505

theorem ice_cream_consumption (friday_amount saturday_amount : Real) 
  (h1 : friday_amount = 3.25) 
  (h2 : saturday_amount = 0.25) : 
  friday_amount + saturday_amount = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_consumption_l1035_103505


namespace NUMINAMATH_CALUDE_hapok_guarantee_l1035_103518

/-- Represents the coin division game between Hapok and Glazok -/
structure CoinGame where
  totalCoins : Nat
  maxHandfuls : Nat

/-- Represents a strategy for Hapok -/
structure Strategy where
  coinsPerHandful : Nat

/-- Calculates the minimum number of coins Hapok can guarantee with a given strategy -/
def guaranteedCoins (game : CoinGame) (strategy : Strategy) : Nat :=
  let fullHandfuls := game.totalCoins / strategy.coinsPerHandful
  let remainingCoins := game.totalCoins % strategy.coinsPerHandful
  if fullHandfuls ≥ 2 * game.maxHandfuls - 1 then
    (game.maxHandfuls - 1) * strategy.coinsPerHandful + remainingCoins
  else
    (fullHandfuls - game.maxHandfuls) * strategy.coinsPerHandful

/-- Theorem stating that Hapok can guarantee at least 46 coins -/
theorem hapok_guarantee (game : CoinGame) (strategy : Strategy) :
  game.totalCoins = 100 →
  game.maxHandfuls = 9 →
  strategy.coinsPerHandful = 6 →
  guaranteedCoins game strategy ≥ 46 := by
  sorry

#eval guaranteedCoins { totalCoins := 100, maxHandfuls := 9 } { coinsPerHandful := 6 }

end NUMINAMATH_CALUDE_hapok_guarantee_l1035_103518


namespace NUMINAMATH_CALUDE_factorization_of_x2y_plus_xy2_l1035_103547

theorem factorization_of_x2y_plus_xy2 (x y : ℝ) : x^2*y + x*y^2 = x*y*(x + y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x2y_plus_xy2_l1035_103547


namespace NUMINAMATH_CALUDE_num_workers_is_500_l1035_103536

/-- The number of workers who raised money by equal contribution -/
def num_workers : ℕ := sorry

/-- The original contribution amount per worker in rupees -/
def contribution_per_worker : ℕ := sorry

/-- The total contribution is 300,000 rupees -/
axiom total_contribution : num_workers * contribution_per_worker = 300000

/-- If each worker contributed 50 rupees extra, the total would be 325,000 rupees -/
axiom total_with_extra : num_workers * (contribution_per_worker + 50) = 325000

/-- Theorem: The number of workers is 500 -/
theorem num_workers_is_500 : num_workers = 500 := by sorry

end NUMINAMATH_CALUDE_num_workers_is_500_l1035_103536


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_integers_l1035_103516

theorem product_of_five_consecutive_integers (n : ℤ) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = n^5 - n^4 - 5*n^3 + 4*n^2 + 4*n :=
by sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_integers_l1035_103516


namespace NUMINAMATH_CALUDE_min_value_cos_sin_min_value_cos_sin_achieved_l1035_103542

theorem min_value_cos_sin (x : ℝ) : 2 * (Real.cos x)^2 - Real.sin (2 * x) ≥ 1 - Real.sqrt 2 := by
  sorry

theorem min_value_cos_sin_achieved : ∃ x : ℝ, 2 * (Real.cos x)^2 - Real.sin (2 * x) = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_cos_sin_min_value_cos_sin_achieved_l1035_103542


namespace NUMINAMATH_CALUDE_slope_of_CD_is_one_l1035_103529

/-- Given a line l passing through the origin O and intersecting y = e^(x-1) at two different points A and B,
    and lines parallel to y-axis drawn through A and B intersecting y = ln x at C and D respectively,
    prove that the slope of line CD is 1. -/
theorem slope_of_CD_is_one (k : ℝ) (hk : k > 0) : ∃ x₁ x₂ : ℝ, 
  x₁ ≠ x₂ ∧ 
  k * x₁ = Real.exp (x₁ - 1) ∧ 
  k * x₂ = Real.exp (x₂ - 1) ∧ 
  (Real.log (k * x₁) - Real.log (k * x₂)) / (k * x₁ - k * x₂) = 1 := by
  sorry


end NUMINAMATH_CALUDE_slope_of_CD_is_one_l1035_103529


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1035_103525

def f (x : ℝ) := x^3 + x

theorem solution_set_of_inequality (x : ℝ) :
  x ∈ Set.Ioo (1/3 : ℝ) 3 ↔ 
  (x ∈ Set.Icc (-5 : ℝ) 5 ∧ f (2*x - 1) + f x > 0) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1035_103525


namespace NUMINAMATH_CALUDE_diagonals_150_sided_polygon_l1035_103543

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a polygon with 150 sides is 11025 -/
theorem diagonals_150_sided_polygon : num_diagonals 150 = 11025 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_150_sided_polygon_l1035_103543


namespace NUMINAMATH_CALUDE_product_equals_eight_l1035_103540

theorem product_equals_eight : 
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_eight_l1035_103540


namespace NUMINAMATH_CALUDE_negative_nine_less_than_negative_two_l1035_103560

theorem negative_nine_less_than_negative_two : -9 < -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_nine_less_than_negative_two_l1035_103560


namespace NUMINAMATH_CALUDE_john_quiz_goal_l1035_103553

theorem john_quiz_goal (total_quizzes : ℕ) (goal_percentage : ℚ) (completed_quizzes : ℕ) (current_as : ℕ) :
  total_quizzes = 60 →
  goal_percentage = 70 / 100 →
  completed_quizzes = 40 →
  current_as = 25 →
  ∃ (max_non_as : ℕ),
    max_non_as = 3 ∧
    (total_quizzes - completed_quizzes - max_non_as) + current_as ≥ ⌈(goal_percentage * total_quizzes : ℚ)⌉ ∧
    ∀ (n : ℕ), n > max_non_as →
      (total_quizzes - completed_quizzes - n) + current_as < ⌈(goal_percentage * total_quizzes : ℚ)⌉ :=
by sorry

end NUMINAMATH_CALUDE_john_quiz_goal_l1035_103553


namespace NUMINAMATH_CALUDE_repeating_decimal_simplest_form_sum_of_numerator_and_denominator_l1035_103592

def repeating_decimal : ℚ := 24/99

theorem repeating_decimal_simplest_form : 
  repeating_decimal = 8/33 := by sorry

theorem sum_of_numerator_and_denominator : 
  (Nat.gcd 8 33 = 1) ∧ (8 + 33 = 41) := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_simplest_form_sum_of_numerator_and_denominator_l1035_103592


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l1035_103567

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^4

-- State the theorem
theorem f_derivative_at_zero : 
  (deriv f) 0 = 4 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l1035_103567


namespace NUMINAMATH_CALUDE_number_pairs_sum_800_or_400_l1035_103517

theorem number_pairs_sum_800_or_400 (x y : ℤ) (A : ℤ) (h1 : x ≥ y) (h2 : A = 800 ∨ A = 400) 
  (h3 : (x + y) + (x - y) + x * y + x / y = A) :
  (A = 800 ∧ ((x = 38 ∧ y = 19) ∨ (x = -42 ∧ y = -21) ∨ (x = 36 ∧ y = 9) ∨ 
              (x = -44 ∧ y = -11) ∨ (x = 40 ∧ y = 4) ∨ (x = -60 ∧ y = -6) ∨ 
              (x = 20 ∧ y = 1) ∨ (x = -60 ∧ y = -3))) ∨
  (A = 400 ∧ ((x = 19 ∧ y = 19) ∨ (x = -21 ∧ y = -21) ∨ (x = 36 ∧ y = 9) ∨ 
              (x = -44 ∧ y = -11) ∨ (x = 64 ∧ y = 4) ∨ (x = -96 ∧ y = -6) ∨ 
              (x = 75 ∧ y = 3) ∨ (x = -125 ∧ y = -5) ∨ (x = 100 ∧ y = 1) ∨ 
              (x = -300 ∧ y = -3))) :=
by sorry

end NUMINAMATH_CALUDE_number_pairs_sum_800_or_400_l1035_103517


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_3_pow_4_l1035_103502

theorem units_digit_of_7_pow_3_pow_4 : ∃ n : ℕ, 7^(3^4) ≡ 7 [ZMOD 10] ∧ n < 10 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_3_pow_4_l1035_103502


namespace NUMINAMATH_CALUDE_vector_sum_l1035_103577

/-- Given two plane vectors a and b, prove their sum is (0, 1) -/
theorem vector_sum (a b : ℝ × ℝ) (h1 : a = (1, -1)) (h2 : b = (-1, 2)) :
  a + b = (0, 1) := by sorry

end NUMINAMATH_CALUDE_vector_sum_l1035_103577


namespace NUMINAMATH_CALUDE_vehicle_distance_time_l1035_103562

/-- Proves that two vehicles traveling in opposite directions for 4 hours
    will be 384 miles apart, given their respective speeds -/
theorem vehicle_distance_time (slower_speed faster_speed : ℝ) 
    (h1 : slower_speed = 44)
    (h2 : faster_speed = slower_speed + 8)
    (distance : ℝ) (h3 : distance = 384) : 
    (slower_speed + faster_speed) * 4 = distance := by
  sorry

end NUMINAMATH_CALUDE_vehicle_distance_time_l1035_103562


namespace NUMINAMATH_CALUDE_jello_bathtub_cost_is_270_l1035_103521

/-- Represents the cost calculation for filling a bathtub with jello. -/
def jello_bathtub_cost (
  jello_mix_per_pound : Real
) (
  bathtub_capacity : Real
) (
  cubic_foot_to_gallon : Real
) (
  gallon_weight : Real
) (
  jello_mix_cost : Real
) : Real :=
  bathtub_capacity * cubic_foot_to_gallon * gallon_weight * jello_mix_per_pound * jello_mix_cost

/-- Theorem stating that the cost to fill the bathtub with jello is $270. -/
theorem jello_bathtub_cost_is_270 :
  jello_bathtub_cost 1.5 6 7.5 8 0.5 = 270 := by
  sorry

#check jello_bathtub_cost_is_270

end NUMINAMATH_CALUDE_jello_bathtub_cost_is_270_l1035_103521


namespace NUMINAMATH_CALUDE_spring_compression_l1035_103591

/-- The force-distance relationship for a spring -/
def spring_force (s : ℝ) : ℝ := 16 * s^2

/-- Theorem: When a force of 4 newtons is applied, the spring compresses by 0.5 meters -/
theorem spring_compression :
  spring_force 0.5 = 4 := by sorry

end NUMINAMATH_CALUDE_spring_compression_l1035_103591


namespace NUMINAMATH_CALUDE_tetrahedron_division_l1035_103572

/-- A regular tetrahedron with unit edge length -/
structure RegularTetrahedron where
  edge_length : ℝ
  is_unit : edge_length = 1

/-- Perpendicular bisector plane of a tetrahedron -/
structure PerpendicularBisectorPlane (t : RegularTetrahedron) where

/-- The number of parts the perpendicular bisector planes divide the tetrahedron into -/
def num_parts (t : RegularTetrahedron) : ℕ := sorry

/-- The volume of each part after division -/
def part_volume (t : RegularTetrahedron) : ℝ := sorry

/-- Theorem stating the number of parts and their volumes -/
theorem tetrahedron_division (t : RegularTetrahedron) :
  num_parts t = 24 ∧ part_volume t = Real.sqrt 2 / 288 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_division_l1035_103572


namespace NUMINAMATH_CALUDE_arrangements_theorem_l1035_103503

def number_of_arrangements (n : ℕ) (a_not_first : Bool) (b_not_last : Bool) : ℕ :=
  if n = 5 ∧ a_not_first ∧ b_not_last then
    78
  else
    0

theorem arrangements_theorem :
  ∀ (n : ℕ) (a_not_first b_not_last : Bool),
    n = 5 → a_not_first → b_not_last →
    number_of_arrangements n a_not_first b_not_last = 78 :=
by
  sorry

end NUMINAMATH_CALUDE_arrangements_theorem_l1035_103503


namespace NUMINAMATH_CALUDE_f_two_zeros_implies_a_range_l1035_103514

/-- The function f(x) defined in terms of the parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * a * x + 3 * a - 5

/-- The derivative of f(x) with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * (x^2 - 1)

/-- Theorem stating that if f has at least two zeros, then 1 ≤ a ≤ 5 -/
theorem f_two_zeros_implies_a_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) →
  1 ≤ a ∧ a ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_f_two_zeros_implies_a_range_l1035_103514


namespace NUMINAMATH_CALUDE_road_trip_ratio_l1035_103508

/-- Road trip distance calculation -/
theorem road_trip_ratio : 
  ∀ (D R : ℝ),
  D > 0 →
  R > 0 →
  D / 2 = 40 →
  2 * (D + R * D + 40) = 560 - (D + R * D + 40) →
  R = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_road_trip_ratio_l1035_103508


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l1035_103545

theorem min_value_sum_of_squares (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_9 : a + b + c = 9) : 
  (a^2 + b^2)/(a + b) + (a^2 + c^2)/(a + c) + (b^2 + c^2)/(b + c) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l1035_103545


namespace NUMINAMATH_CALUDE_dinner_meals_count_l1035_103559

/-- Represents the number of meals in a restaurant scenario -/
structure RestaurantMeals where
  lunch_prepared : ℕ
  lunch_sold : ℕ
  dinner_prepared : ℕ

/-- Calculates the total number of meals available for dinner -/
def meals_for_dinner (r : RestaurantMeals) : ℕ :=
  (r.lunch_prepared - r.lunch_sold) + r.dinner_prepared

/-- Theorem stating the number of meals available for dinner in the given scenario -/
theorem dinner_meals_count (r : RestaurantMeals) 
  (h1 : r.lunch_prepared = 17) 
  (h2 : r.lunch_sold = 12) 
  (h3 : r.dinner_prepared = 5) : 
  meals_for_dinner r = 10 := by
  sorry

end NUMINAMATH_CALUDE_dinner_meals_count_l1035_103559


namespace NUMINAMATH_CALUDE_tangency_point_satisfies_equations_tangency_point_is_unique_l1035_103500

/-- The point of tangency for two parabolas -/
def point_of_tangency : ℝ × ℝ := (-7, -25)

/-- First parabola equation -/
def parabola1 (x y : ℝ) : Prop := y = x^2 + 17*x + 40

/-- Second parabola equation -/
def parabola2 (x y : ℝ) : Prop := x = y^2 + 51*y + 650

/-- Theorem stating that the point_of_tangency satisfies both parabola equations -/
theorem tangency_point_satisfies_equations :
  parabola1 point_of_tangency.1 point_of_tangency.2 ∧
  parabola2 point_of_tangency.1 point_of_tangency.2 :=
by sorry

/-- Theorem stating that the point_of_tangency is the unique point satisfying both equations -/
theorem tangency_point_is_unique :
  ∀ (x y : ℝ), parabola1 x y ∧ parabola2 x y → (x, y) = point_of_tangency :=
by sorry

end NUMINAMATH_CALUDE_tangency_point_satisfies_equations_tangency_point_is_unique_l1035_103500


namespace NUMINAMATH_CALUDE_gain_percentage_proof_l1035_103519

/-- Proves that the gain percentage is 20% when an article is sold for 168 Rs,
    given that it incurs a 15% loss when sold for 119 Rs. -/
theorem gain_percentage_proof (cost_price : ℝ) : 
  (cost_price * 0.85 = 119) →  -- 15% loss when sold for 119
  ((168 - cost_price) / cost_price * 100 = 20) := by
sorry

end NUMINAMATH_CALUDE_gain_percentage_proof_l1035_103519


namespace NUMINAMATH_CALUDE_f_properties_l1035_103599

-- Define the function f(x) = x ln|x|
noncomputable def f (x : ℝ) : ℝ := x * Real.log (abs x)

-- Define the function g(x) = f(x) - m
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f x - m

theorem f_properties :
  (∀ x y, x < y ∧ x < -1/Real.exp 1 ∧ y < -1/Real.exp 1 → f x < f y) ∧
  (∀ m : ℝ, ∃ n : ℕ, n ≤ 3 ∧ (∃ s : Finset ℝ, s.card = n ∧ ∀ x ∈ s, g m x = 0) ∧
    ∀ s : Finset ℝ, (∀ x ∈ s, g m x = 0) → s.card ≤ n) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1035_103599


namespace NUMINAMATH_CALUDE_misery_ratio_bound_l1035_103546

/-- Represents a room with its total load -/
structure Room where
  load : ℝ
  load_positive : load > 0

/-- Represents a student with their download request -/
structure Student where
  bits : ℝ
  bits_positive : bits > 0

/-- Calculates the displeasure of a student in a given room -/
def displeasure (s : Student) (r : Room) : ℝ := s.bits * r.load

/-- Calculates the total misery for a given configuration -/
def misery (students : List Student) (rooms : List Room) (assignment : Student → Room) : ℝ :=
  (students.map (fun s => displeasure s (assignment s))).sum

/-- Defines a balanced configuration -/
def is_balanced (students : List Student) (rooms : List Room) (assignment : Student → Room) : Prop :=
  ∀ s : Student, ∀ r : Room, displeasure s (assignment s) ≤ displeasure s r

theorem misery_ratio_bound 
  (students : List Student) 
  (rooms : List Room) 
  (balanced_assignment : Student → Room)
  (other_assignment : Student → Room)
  (h_balanced : is_balanced students rooms balanced_assignment) :
  let M1 := misery students rooms balanced_assignment
  let M2 := misery students rooms other_assignment
  M1 / M2 ≤ 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_misery_ratio_bound_l1035_103546


namespace NUMINAMATH_CALUDE_school_dance_relationship_l1035_103565

theorem school_dance_relationship (b g : ℕ) : 
  (b > 0) →  -- There is at least one boy
  (g ≥ 7) →  -- There are at least 7 girls (for the first boy)
  (∀ i : ℕ, i > 0 ∧ i ≤ b → (7 + i - 1) ≤ g) →  -- Each boy can dance with his required number of girls
  (7 + b - 1 = g) →  -- The last boy dances with all girls
  b = g - 6 := by
sorry

end NUMINAMATH_CALUDE_school_dance_relationship_l1035_103565


namespace NUMINAMATH_CALUDE_remaining_budget_l1035_103580

/-- Proves that given a weekly food budget of $80, after purchasing a $12 bucket of fried chicken
    and 5 pounds of beef at $3 per pound, the remaining budget is $53. -/
theorem remaining_budget (weekly_budget : ℕ) (chicken_cost : ℕ) (beef_price : ℕ) (beef_amount : ℕ) :
  weekly_budget = 80 →
  chicken_cost = 12 →
  beef_price = 3 →
  beef_amount = 5 →
  weekly_budget - (chicken_cost + beef_price * beef_amount) = 53 := by
  sorry

end NUMINAMATH_CALUDE_remaining_budget_l1035_103580


namespace NUMINAMATH_CALUDE_f_zero_at_three_l1035_103548

-- Define the function f
def f (x r : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 5 * x + r

-- State the theorem
theorem f_zero_at_three (r : ℝ) : f 3 r = 0 ↔ r = -273 := by sorry

end NUMINAMATH_CALUDE_f_zero_at_three_l1035_103548


namespace NUMINAMATH_CALUDE_manny_cookie_slices_left_l1035_103598

/-- Calculates the number of cookie slices left after distribution --/
def cookie_slices_left (num_pies : ℕ) (slices_per_pie : ℕ) (total_people : ℕ) (half_slice_people : ℕ) : ℕ :=
  let total_slices := num_pies * slices_per_pie
  let full_slice_people := total_people - half_slice_people
  let distributed_slices := full_slice_people + (half_slice_people / 2)
  total_slices - distributed_slices

/-- Theorem stating the number of cookie slices left in Manny's scenario --/
theorem manny_cookie_slices_left : cookie_slices_left 6 12 39 3 = 33 := by
  sorry

#eval cookie_slices_left 6 12 39 3

end NUMINAMATH_CALUDE_manny_cookie_slices_left_l1035_103598


namespace NUMINAMATH_CALUDE_marble_probability_value_l1035_103594

/-- The probability of having one white and one blue marble left when drawing
    marbles randomly from a bag containing 3 blue and 5 white marbles until 2 are left -/
def marble_probability : ℚ :=
  let total_marbles : ℕ := 8
  let blue_marbles : ℕ := 3
  let white_marbles : ℕ := 5
  let marbles_drawn : ℕ := 6
  let favorable_outcomes : ℕ := Nat.choose white_marbles white_marbles * Nat.choose blue_marbles (blue_marbles - 1)
  let total_outcomes : ℕ := Nat.choose total_marbles marbles_drawn
  (favorable_outcomes : ℚ) / total_outcomes

/-- Theorem stating that the probability of having one white and one blue marble left
    is equal to 3/28 -/
theorem marble_probability_value : marble_probability = 3 / 28 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_value_l1035_103594


namespace NUMINAMATH_CALUDE_fourth_test_score_for_average_l1035_103569

def test1 : ℕ := 80
def test2 : ℕ := 70
def test3 : ℕ := 90
def test4 : ℕ := 100
def targetAverage : ℕ := 85

theorem fourth_test_score_for_average :
  (test1 + test2 + test3 + test4) / 4 = targetAverage :=
sorry

end NUMINAMATH_CALUDE_fourth_test_score_for_average_l1035_103569


namespace NUMINAMATH_CALUDE_x_squared_mod_20_l1035_103570

theorem x_squared_mod_20 (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 20]) 
  (h2 : 4 * x ≡ 12 [ZMOD 20]) : 
  x^2 ≡ 4 [ZMOD 20] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_20_l1035_103570


namespace NUMINAMATH_CALUDE_line_segment_param_sum_l1035_103523

/-- Given a line segment connecting points (-3,10) and (4,16) represented by
    parametric equations x = at + b and y = ct + d where 0 ≤ t ≤ 1,
    and t = 0 corresponds to (-3,10), prove that a² + b² + c² + d² = 194 -/
theorem line_segment_param_sum (a b c d : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (b = -3 ∧ d = 10) →
  (a + b = 4 ∧ c + d = 16) →
  a^2 + b^2 + c^2 + d^2 = 194 := by
sorry

end NUMINAMATH_CALUDE_line_segment_param_sum_l1035_103523


namespace NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l1035_103561

/-- Given three points on an inverse proportion function, prove their y-coordinates' relationship -/
theorem inverse_proportion_y_relationship
  (k : ℝ) (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ)
  (h_k : k < 0)
  (h_x : x₁ < x₂ ∧ x₂ < 0 ∧ 0 < x₃)
  (h_y₁ : y₁ = k / x₁)
  (h_y₂ : y₂ = k / x₂)
  (h_y₃ : y₃ = k / x₃) :
  y₂ > y₁ ∧ y₁ > y₃ := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l1035_103561


namespace NUMINAMATH_CALUDE_complement_intersection_equals_specific_set_l1035_103586

def U : Set Nat := {1,2,3,4,5,6,7,8}
def S : Set Nat := {1,3,5}
def T : Set Nat := {3,6}

theorem complement_intersection_equals_specific_set :
  (U \ S) ∩ (U \ T) = {2,4,7,8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_specific_set_l1035_103586


namespace NUMINAMATH_CALUDE_expression_equality_l1035_103556

theorem expression_equality : 12 + 5*(4-9)^2 - 3 = 134 := by sorry

end NUMINAMATH_CALUDE_expression_equality_l1035_103556


namespace NUMINAMATH_CALUDE_pentagon_area_l1035_103539

/-- The area of a pentagon with specific dimensions -/
theorem pentagon_area : 
  ∀ (right_triangle_base right_triangle_height trapezoid_base1 trapezoid_base2 trapezoid_height : ℝ),
  right_triangle_base = 28 →
  right_triangle_height = 30 →
  trapezoid_base1 = 25 →
  trapezoid_base2 = 18 →
  trapezoid_height = 39 →
  (1/2 * right_triangle_base * right_triangle_height) + 
  (1/2 * (trapezoid_base1 + trapezoid_base2) * trapezoid_height) = 1257 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_area_l1035_103539


namespace NUMINAMATH_CALUDE_bus_ride_difference_l1035_103537

theorem bus_ride_difference (oscar_ride : ℝ) (charlie_ride : ℝ) 
  (h1 : oscar_ride = 0.75) (h2 : charlie_ride = 0.25) :
  oscar_ride - charlie_ride = 0.50 := by
sorry

end NUMINAMATH_CALUDE_bus_ride_difference_l1035_103537


namespace NUMINAMATH_CALUDE_chord_length_of_perpendicular_bisector_l1035_103535

/-- 
Given a circle with radius 15 units and a chord that is the perpendicular bisector of a radius,
prove that the length of this chord is 26 units.
-/
theorem chord_length_of_perpendicular_bisector (r : ℝ) (chord_length : ℝ) : 
  r = 15 → 
  chord_length = 2 * Real.sqrt (r^2 - (r/2)^2) → 
  chord_length = 26 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_of_perpendicular_bisector_l1035_103535


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1035_103584

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_first_fifth : a 1 + a 5 = 10
  fourth_term : a 4 = 7

/-- The common difference of the arithmetic sequence is 2 -/
theorem arithmetic_sequence_common_difference (seq : ArithmeticSequence) :
  ∃ d, (∀ n, seq.a (n + 1) - seq.a n = d) ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1035_103584


namespace NUMINAMATH_CALUDE_circular_path_meeting_time_l1035_103530

theorem circular_path_meeting_time (c : ℝ) : 
  c > 0 ∧ 
  (6⁻¹ : ℝ) > 0 ∧ 
  c⁻¹ > 0 ∧
  (((6 * c) / (c + 6) + 1) * c⁻¹ = 1) →
  c = 3 :=
by sorry

end NUMINAMATH_CALUDE_circular_path_meeting_time_l1035_103530


namespace NUMINAMATH_CALUDE_preimage_of_three_one_l1035_103564

/-- A mapping from A to B -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, p.1 - p.2)

/-- The theorem stating that (2, 1) is the preimage of (3, 1) under f -/
theorem preimage_of_three_one :
  f (2, 1) = (3, 1) ∧ ∀ p : ℝ × ℝ, f p = (3, 1) → p = (2, 1) :=
by sorry

end NUMINAMATH_CALUDE_preimage_of_three_one_l1035_103564


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1035_103566

theorem min_value_of_expression (x y : ℝ) (h : x^2 + y^2 - 4*x + 6*y + 12 = 0) :
  ∃ (min : ℝ), min = 5 - Real.sqrt 5 ∧ ∀ (x y : ℝ), x^2 + y^2 - 4*x + 6*y + 12 = 0 → |2*x - y - 2| ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1035_103566


namespace NUMINAMATH_CALUDE_max_power_of_two_divides_l1035_103555

theorem max_power_of_two_divides (n : ℕ) (hn : n > 0) :
  (∃ m : ℕ, 3^(2*n+3) + 40*n - 27 = 2^6 * m) ∧
  (∃ n₀ : ℕ, n₀ > 0 ∧ ∀ m : ℕ, 3^(2*n₀+3) + 40*n₀ - 27 ≠ 2^7 * m) :=
sorry

end NUMINAMATH_CALUDE_max_power_of_two_divides_l1035_103555


namespace NUMINAMATH_CALUDE_x_power_24_equals_one_l1035_103593

theorem x_power_24_equals_one (x : ℝ) (h : x + 1/x = Real.sqrt 5) : x^24 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_power_24_equals_one_l1035_103593


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1035_103541

theorem quadratic_inequality_solution_range (d : ℝ) : 
  (d > 0 ∧ ∃ x : ℝ, x^2 - 8*x + d < 0) ↔ 0 < d ∧ d < 16 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1035_103541


namespace NUMINAMATH_CALUDE_sine_shift_to_cosine_l1035_103595

open Real

theorem sine_shift_to_cosine (x : ℝ) :
  let f (t : ℝ) := sin (2 * t + π / 6)
  let g (t : ℝ) := f (t + π / 6)
  g x = cos (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_sine_shift_to_cosine_l1035_103595


namespace NUMINAMATH_CALUDE_room_dimension_proof_l1035_103527

/-- Proves that given the room dimensions and whitewashing costs, the unknown dimension is 15 feet -/
theorem room_dimension_proof (x : ℝ) : 
  let room_length : ℝ := 25
  let room_height : ℝ := 12
  let door_area : ℝ := 6 * 3
  let window_area : ℝ := 4 * 3
  let num_windows : ℕ := 3
  let whitewash_cost_per_sqft : ℝ := 3
  let total_cost : ℝ := 2718
  let wall_area : ℝ := 2 * (room_length * room_height) + 2 * (x * room_height)
  let non_whitewash_area : ℝ := door_area + num_windows * window_area
  let whitewash_area : ℝ := wall_area - non_whitewash_area
  whitewash_area * whitewash_cost_per_sqft = total_cost → x = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_room_dimension_proof_l1035_103527


namespace NUMINAMATH_CALUDE_abc_inequality_l1035_103579

theorem abc_inequality : 
  let a : ℝ := (2/5)^(3/5)
  let b : ℝ := (2/5)^(2/5)
  let c : ℝ := (3/5)^(2/5)
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l1035_103579


namespace NUMINAMATH_CALUDE_equation_solution_l1035_103531

theorem equation_solution : ∃ y : ℤ, (2010 + 2*y)^2 = 4*y^2 ∧ y = -1005 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1035_103531


namespace NUMINAMATH_CALUDE_sequence_equality_l1035_103528

theorem sequence_equality (a b : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a (n + 1) = 2 * b n - a n)
  (h2 : ∀ n : ℕ, b (n + 1) = 2 * a n - b n)
  (h3 : ∀ n : ℕ, a n > 0) :
  a 1 = b 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_equality_l1035_103528


namespace NUMINAMATH_CALUDE_outfits_with_restrictions_l1035_103538

/-- The number of unique outfits that can be made with shirts and pants, with restrictions -/
def uniqueOutfits (shirts : ℕ) (pants : ℕ) (restrictedShirts : ℕ) (restrictedPants : ℕ) : ℕ :=
  shirts * pants - restrictedShirts * restrictedPants

/-- Theorem stating the number of unique outfits under given conditions -/
theorem outfits_with_restrictions :
  uniqueOutfits 5 6 1 2 = 28 := by
  sorry

#eval uniqueOutfits 5 6 1 2

end NUMINAMATH_CALUDE_outfits_with_restrictions_l1035_103538


namespace NUMINAMATH_CALUDE_ball_max_height_l1035_103583

-- Define the function representing the height of the ball
def f (t : ℝ) : ℝ := -5 * t^2 + 20 * t + 10

-- Theorem stating that the maximum value of f is 30
theorem ball_max_height : ∃ (t : ℝ), ∀ (s : ℝ), f s ≤ f t ∧ f t = 30 := by
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l1035_103583


namespace NUMINAMATH_CALUDE_scientific_notation_748_million_l1035_103573

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Converts a real number to scientific notation with given significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- Rounds a real number to a given number of significant figures -/
def roundToSigFigs (x : ℝ) (sigFigs : ℕ) : ℝ :=
  sorry

theorem scientific_notation_748_million :
  let original := (748 : ℝ) * 1000000
  let scientificForm := toScientificNotation original 2
  scientificForm = ScientificNotation.mk 7.5 8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_748_million_l1035_103573


namespace NUMINAMATH_CALUDE_count_valid_pairs_l1035_103588

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def valid_pair (a b : ℕ) : Prop :=
  is_odd a ∧ is_odd b ∧ a > 1 ∧ b > 1 ∧ a * b = 315

theorem count_valid_pairs :
  ∃! (pairs : Finset (ℕ × ℕ)), 
    (∀ p ∈ pairs, valid_pair p.1 p.2) ∧ 
    pairs.card = 5 :=
sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l1035_103588


namespace NUMINAMATH_CALUDE_estate_value_l1035_103510

/-- Represents the estate distribution problem --/
def EstateDistribution (total : ℝ) : Prop :=
  ∃ (elder_niece younger_niece brother caretaker : ℝ),
    -- The two nieces together received half of the estate
    elder_niece + younger_niece = total / 2 ∧
    -- The nieces' shares are in the ratio of 3 to 2
    elder_niece = (3/5) * (total / 2) ∧
    younger_niece = (2/5) * (total / 2) ∧
    -- The brother got three times as much as the elder niece
    brother = 3 * elder_niece ∧
    -- The caretaker received $800
    caretaker = 800 ∧
    -- The sum of all shares equals the total estate
    elder_niece + younger_niece + brother + caretaker = total

/-- Theorem stating that the estate value is $2000 --/
theorem estate_value : EstateDistribution 2000 :=
sorry

end NUMINAMATH_CALUDE_estate_value_l1035_103510


namespace NUMINAMATH_CALUDE_min_c_squared_l1035_103551

theorem min_c_squared (a b c : ℝ) (A B C : ℝ) : 
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + 2*b = 4 →
  a * Real.sin A + 4*b * Real.sin B = 6*a * Real.sin B * Real.sin C →
  c^2 ≥ 5 - (4 * Real.sqrt 5) / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_c_squared_l1035_103551


namespace NUMINAMATH_CALUDE_sin_960_degrees_l1035_103582

theorem sin_960_degrees : Real.sin (960 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_960_degrees_l1035_103582


namespace NUMINAMATH_CALUDE_max_value_and_k_range_l1035_103509

def f (x : ℝ) : ℝ := -3 * x^2 - 3 * x + 18

theorem max_value_and_k_range :
  (∀ x > -1, (f x - 21) / (x + 1) ≤ -3) ∧
  (∀ k : ℝ, (∀ x ∈ Set.Ioo 1 4, -3 * x^2 + k * x - 5 > 0) → k < 2 * Real.sqrt 15) := by
  sorry

end NUMINAMATH_CALUDE_max_value_and_k_range_l1035_103509


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1035_103524

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 8 = 0 ∧ x₂^2 + m*x₂ - 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1035_103524


namespace NUMINAMATH_CALUDE_integer_sum_and_square_is_ten_l1035_103581

theorem integer_sum_and_square_is_ten (N : ℤ) : N^2 + N = 10 → N = 2 ∨ N = -5 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_and_square_is_ten_l1035_103581


namespace NUMINAMATH_CALUDE_rotten_apples_smell_percentage_l1035_103526

theorem rotten_apples_smell_percentage 
  (total_apples : ℕ) 
  (rotten_percentage : ℚ) 
  (non_smelling_rotten : ℕ) 
  (h1 : total_apples = 200)
  (h2 : rotten_percentage = 40 / 100)
  (h3 : non_smelling_rotten = 24) : 
  (total_apples * rotten_percentage - non_smelling_rotten : ℚ) / (total_apples * rotten_percentage) * 100 = 70 := by
sorry

end NUMINAMATH_CALUDE_rotten_apples_smell_percentage_l1035_103526


namespace NUMINAMATH_CALUDE_dance_circle_partition_l1035_103587

/-- The number of ways to partition n distinguishable objects into k indistinguishable,
    non-empty subsets, where rotations within subsets are considered identical. -/
def partition_count (n k : ℕ) : ℕ :=
  if k > n ∨ k = 0 then 0
  else
    (Finset.range (n - k + 1)).sum (λ i =>
      Nat.choose n (i + 1) * Nat.factorial i * Nat.factorial (n - i - 2))
    / 2

/-- Theorem stating that there are 50 ways to partition 5 children into 2 dance circles. -/
theorem dance_circle_partition :
  partition_count 5 2 = 50 := by
  sorry


end NUMINAMATH_CALUDE_dance_circle_partition_l1035_103587


namespace NUMINAMATH_CALUDE_census_contradiction_l1035_103557

/-- Represents a family in the house -/
structure Family where
  boys : ℕ
  girls : ℕ

/-- The census data for the house -/
structure CensusData where
  families : List Family

/-- Conditions from the problem -/
def ValidCensus (data : CensusData) : Prop :=
  ∀ f ∈ data.families,
    (f.boys > 0 → f.girls > 0) ∧  -- Every boy has a sister
    (f.boys + f.girls > 0)  -- No families without children

/-- Total number of boys in the house -/
def TotalBoys (data : CensusData) : ℕ :=
  (data.families.map (λ f => f.boys)).sum

/-- Total number of girls in the house -/
def TotalGirls (data : CensusData) : ℕ :=
  (data.families.map (λ f => f.girls)).sum

/-- Total number of children in the house -/
def TotalChildren (data : CensusData) : ℕ :=
  TotalBoys data + TotalGirls data

/-- Total number of adults in the house -/
def TotalAdults (data : CensusData) : ℕ :=
  2 * data.families.length

/-- The main theorem to prove -/
theorem census_contradiction (data : CensusData) 
  (h_valid : ValidCensus data)
  (h_more_boys : TotalBoys data > TotalGirls data) :
  TotalChildren data > TotalAdults data :=
sorry

end NUMINAMATH_CALUDE_census_contradiction_l1035_103557


namespace NUMINAMATH_CALUDE_evenProductProbabilityFor6And4_l1035_103568

/-- Represents a spinner with n equal segments numbered from 1 to n -/
structure Spinner :=
  (n : ℕ)

/-- The probability of getting an even product when spinning two spinners -/
def evenProductProbability (spinnerA spinnerB : Spinner) : ℚ :=
  sorry

/-- Theorem stating that the probability of getting an even product
    when spinning a 6-segment spinner and a 4-segment spinner is 1/2 -/
theorem evenProductProbabilityFor6And4 :
  evenProductProbability (Spinner.mk 6) (Spinner.mk 4) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_evenProductProbabilityFor6And4_l1035_103568


namespace NUMINAMATH_CALUDE_train_passing_time_l1035_103512

/-- The time it takes for a train to pass a person moving in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (person_speed : ℝ) :
  train_length = 240 →
  train_speed = 100 * (5/18) →
  person_speed = 8 * (5/18) →
  (train_length / (train_speed + person_speed)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l1035_103512


namespace NUMINAMATH_CALUDE_paint_usage_l1035_103558

theorem paint_usage (initial_paint : ℝ) (first_week_fraction : ℝ) (second_week_fraction : ℝ)
  (h1 : initial_paint = 360)
  (h2 : first_week_fraction = 1/4)
  (h3 : second_week_fraction = 1/3) :
  let first_week_usage := first_week_fraction * initial_paint
  let remaining_paint := initial_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  let total_usage := first_week_usage + second_week_usage
  total_usage = 180 := by
  sorry

end NUMINAMATH_CALUDE_paint_usage_l1035_103558


namespace NUMINAMATH_CALUDE_acai_juice_cost_per_litre_l1035_103504

/-- The cost per litre of the superfruit juice cocktail -/
def cocktail_cost : ℝ := 1399.45

/-- The cost per litre of the mixed fruit juice -/
def mixed_juice_cost : ℝ := 262.85

/-- The volume of mixed fruit juice in litres -/
def mixed_juice_volume : ℝ := 32

/-- The volume of açaí berry juice in litres -/
def acai_juice_volume : ℝ := 21.333333333333332

/-- The total volume of the cocktail in litres -/
def total_volume : ℝ := mixed_juice_volume + acai_juice_volume

theorem acai_juice_cost_per_litre : 
  ∃ (acai_cost : ℝ),
    acai_cost = 3105.00 ∧
    mixed_juice_cost * mixed_juice_volume + acai_cost * acai_juice_volume = 
    cocktail_cost * total_volume :=
by sorry

end NUMINAMATH_CALUDE_acai_juice_cost_per_litre_l1035_103504


namespace NUMINAMATH_CALUDE_corner_sum_is_sixteen_l1035_103585

/-- Represents a 3x3 grid with integer entries -/
def Grid := Matrix (Fin 3) (Fin 3) ℤ

/-- The sum of elements in a given row -/
def row_sum (g : Grid) (i : Fin 3) : ℤ :=
  g i 0 + g i 1 + g i 2

/-- The sum of elements in a given column -/
def col_sum (g : Grid) (j : Fin 3) : ℤ :=
  g 0 j + g 1 j + g 2 j

/-- The sum of elements in the main diagonal -/
def main_diag_sum (g : Grid) : ℤ :=
  g 0 0 + g 1 1 + g 2 2

/-- The sum of elements in the anti-diagonal -/
def anti_diag_sum (g : Grid) : ℤ :=
  g 0 2 + g 1 1 + g 2 0

/-- A grid is magic if all rows, columns, and diagonals sum to 12 -/
def is_magic (g : Grid) : Prop :=
  (∀ i : Fin 3, row_sum g i = 12) ∧
  (∀ j : Fin 3, col_sum g j = 12) ∧
  main_diag_sum g = 12 ∧
  anti_diag_sum g = 12

theorem corner_sum_is_sixteen (g : Grid) 
  (h_magic : is_magic g)
  (h_corners : g 0 0 = 4 ∧ g 0 2 = 3 ∧ g 2 0 = 5 ∧ g 2 2 = 4) :
  g 0 0 + g 0 2 + g 2 0 + g 2 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_is_sixteen_l1035_103585


namespace NUMINAMATH_CALUDE_range_of_g_l1035_103511

noncomputable def g (x : ℝ) : ℝ := Real.arctan (x^2) + Real.arctan ((2 - 2*x^2) / (1 + 2*x^2))

theorem range_of_g : ∀ x : ℝ, g x = Real.arctan 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l1035_103511


namespace NUMINAMATH_CALUDE_number_of_persons_working_prove_number_of_persons_working_l1035_103513

/-- The number of days it takes for some persons to finish the job -/
def group_days : ℕ := 8

/-- The number of days it takes for the first person to finish the job -/
def first_person_days : ℕ := 24

/-- The number of days it takes for the second person to finish the job -/
def second_person_days : ℕ := 12

/-- The work rate of a person is the fraction of the job they can complete in one day -/
def work_rate (days : ℕ) : ℚ := 1 / days

/-- The theorem stating that the number of persons working on the job is 2 -/
theorem number_of_persons_working : ℕ :=
  2

/-- Proof that the number of persons working on the job is 2 -/
theorem prove_number_of_persons_working :
  work_rate group_days = work_rate first_person_days + work_rate second_person_days →
  number_of_persons_working = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_of_persons_working_prove_number_of_persons_working_l1035_103513


namespace NUMINAMATH_CALUDE_pythagorean_triple_5_12_13_l1035_103533

theorem pythagorean_triple_5_12_13 : 5^2 + 12^2 = 13^2 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_5_12_13_l1035_103533


namespace NUMINAMATH_CALUDE_carries_shopping_money_l1035_103544

theorem carries_shopping_money (initial_amount sweater_cost tshirt_cost shoes_cost : ℕ) 
  (h1 : initial_amount = 91)
  (h2 : sweater_cost = 24)
  (h3 : tshirt_cost = 6)
  (h4 : shoes_cost = 11) :
  initial_amount - (sweater_cost + tshirt_cost + shoes_cost) = 50 := by
  sorry

end NUMINAMATH_CALUDE_carries_shopping_money_l1035_103544


namespace NUMINAMATH_CALUDE_solution_product_l1035_103554

/-- Given that p and q are the two distinct solutions of the equation
    (x - 5)(3x + 9) = x^2 - 16x + 55, prove that (p + 4)(q + 4) = -54 -/
theorem solution_product (p q : ℝ) : 
  (p - 5) * (3 * p + 9) = p^2 - 16 * p + 55 →
  (q - 5) * (3 * q + 9) = q^2 - 16 * q + 55 →
  p ≠ q →
  (p + 4) * (q + 4) = -54 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l1035_103554


namespace NUMINAMATH_CALUDE_least_n_divisible_by_77_l1035_103597

theorem least_n_divisible_by_77 (n : ℕ) : 
  (n ≥ 100 ∧ 
   77 ∣ (2^(n+1) - 1) ∧ 
   ∀ m, m ≥ 100 ∧ m < n → ¬(77 ∣ (2^(m+1) - 1))) → 
  n = 119 :=
by sorry

end NUMINAMATH_CALUDE_least_n_divisible_by_77_l1035_103597


namespace NUMINAMATH_CALUDE_smallest_c_inequality_l1035_103507

theorem smallest_c_inequality (c : ℝ) : 
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) + c * |x^2 - y^2| ≥ (x + y) / 2) ↔ c ≥ (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_inequality_l1035_103507


namespace NUMINAMATH_CALUDE_square_sum_minus_one_le_zero_l1035_103515

theorem square_sum_minus_one_le_zero (a b : ℝ) :
  a^2 + b^2 - 1 - a^2 * b^2 ≤ 0 ↔ (a^2 - 1) * (b^2 - 1) ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_square_sum_minus_one_le_zero_l1035_103515


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1035_103552

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : arithmetic_sequence a) 
  (h1 : a 3 + a 8 = 10) : 3 * a 5 + a 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1035_103552


namespace NUMINAMATH_CALUDE_bacteria_exceeds_200_on_day_4_l1035_103550

-- Define the bacteria population function
def bacteria_population (initial_population : ℕ) (growth_factor : ℕ) (days : ℕ) : ℕ :=
  initial_population * growth_factor ^ days

-- Theorem statement
theorem bacteria_exceeds_200_on_day_4 :
  let initial_population := 5
  let growth_factor := 3
  let threshold := 200
  (∀ d : ℕ, d < 4 → bacteria_population initial_population growth_factor d ≤ threshold) ∧
  (bacteria_population initial_population growth_factor 4 > threshold) :=
by sorry

end NUMINAMATH_CALUDE_bacteria_exceeds_200_on_day_4_l1035_103550


namespace NUMINAMATH_CALUDE_extra_coverage_area_l1035_103522

/-- Represents the area covered by one bag of grass seed in square feet. -/
def bag_coverage : ℕ := 250

/-- Represents the length of the lawn from house to curb in feet. -/
def lawn_length : ℕ := 22

/-- Represents the width of the lawn from side to side in feet. -/
def lawn_width : ℕ := 36

/-- Represents the number of bags of grass seed bought. -/
def bags_bought : ℕ := 4

/-- Calculates the extra area that can be covered by leftover grass seed after reseeding the lawn. -/
theorem extra_coverage_area : 
  bags_bought * bag_coverage - lawn_length * lawn_width = 208 := by
  sorry

end NUMINAMATH_CALUDE_extra_coverage_area_l1035_103522


namespace NUMINAMATH_CALUDE_petes_journey_distance_l1035_103506

/-- Represents the distance of each segment of Pete's journey in blocks -/
structure JourneySegments where
  toGarage : ℕ
  toPostOffice : ℕ
  toLibrary : ℕ
  toFriend : ℕ

/-- Calculates the total distance of Pete's round trip journey -/
def totalDistance (segments : JourneySegments) : ℕ :=
  2 * (segments.toGarage + segments.toPostOffice + segments.toLibrary + segments.toFriend)

/-- Pete's actual journey segments -/
def petesJourney : JourneySegments :=
  { toGarage := 5
  , toPostOffice := 20
  , toLibrary := 8
  , toFriend := 10 }

/-- Theorem stating that Pete's total journey distance is 86 blocks -/
theorem petes_journey_distance : totalDistance petesJourney = 86 := by
  sorry


end NUMINAMATH_CALUDE_petes_journey_distance_l1035_103506


namespace NUMINAMATH_CALUDE_smallest_positive_integer_3003m_66666n_l1035_103589

theorem smallest_positive_integer_3003m_66666n : 
  (∃ (k : ℕ), k > 0 ∧ ∀ (x : ℕ), x > 0 → (∃ (m n : ℤ), x = 3003 * m + 66666 * n) → k ≤ x) ∧
  (∃ (m n : ℤ), 3 = 3003 * m + 66666 * n) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_3003m_66666n_l1035_103589


namespace NUMINAMATH_CALUDE_number_ordering_l1035_103575

theorem number_ordering : 10^5 < 2^20 ∧ 2^20 < 5^10 := by
  sorry

end NUMINAMATH_CALUDE_number_ordering_l1035_103575


namespace NUMINAMATH_CALUDE_comic_book_percentage_l1035_103501

theorem comic_book_percentage (total_books : ℕ) (novel_percentage : ℚ) (graphic_novels : ℕ) : 
  total_books = 120 →
  novel_percentage = 65 / 100 →
  graphic_novels = 18 →
  (total_books - (total_books * novel_percentage).floor - graphic_novels) / total_books = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_comic_book_percentage_l1035_103501


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_and_24_l1035_103576

theorem smallest_divisible_by_15_and_24 : 
  ∃ n : ℕ, n > 0 ∧ 15 ∣ n ∧ 24 ∣ n ∧ ∀ m : ℕ, m > 0 → 15 ∣ m → 24 ∣ m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_and_24_l1035_103576


namespace NUMINAMATH_CALUDE_interior_angle_sum_regular_polygon_l1035_103563

theorem interior_angle_sum_regular_polygon (n : ℕ) (exterior_angle : ℝ) :
  n > 2 →
  exterior_angle = 45 →
  n * exterior_angle = 360 →
  (n - 2) * 180 = 1080 :=
by sorry

end NUMINAMATH_CALUDE_interior_angle_sum_regular_polygon_l1035_103563


namespace NUMINAMATH_CALUDE_polynomial_value_equivalence_l1035_103578

theorem polynomial_value_equivalence (x y : ℝ) :
  3 * x^2 + 4 * y + 9 = 8 → 9 * x^2 + 12 * y + 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_equivalence_l1035_103578


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1035_103590

theorem polynomial_remainder (x : ℝ) : 
  (x^4 - x + 1) % (x + 3) = 85 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1035_103590


namespace NUMINAMATH_CALUDE_square_digit_sum_99999_l1035_103549

/-- Given a natural number n, returns the sum of its digits -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Checks if a number consists of all nines -/
def is_all_nines (n : ℕ) : Prop := sorry

theorem square_digit_sum_99999 (n : ℕ) :
  n = 99999 → is_all_nines n → sum_of_digits (n^2) = 45 := by sorry

end NUMINAMATH_CALUDE_square_digit_sum_99999_l1035_103549


namespace NUMINAMATH_CALUDE_max_consecutive_odds_is_five_l1035_103574

/-- Represents a natural number as a list of its digits -/
def Digits := List Nat

/-- Returns the largest digit in a number -/
def largestDigit (n : Digits) : Nat :=
  n.foldl max 0

/-- Adds the largest digit to the number -/
def addLargestDigit (n : Digits) : Digits :=
  sorry

/-- Checks if a number is odd -/
def isOdd (n : Digits) : Bool :=
  sorry

/-- Generates the sequence of numbers following the given rule -/
def generateSequence (start : Digits) : List Digits :=
  sorry

/-- Counts the maximum number of consecutive odd numbers in a list -/
def maxConsecutiveOdds (seq : List Digits) : Nat :=
  sorry

/-- The main theorem stating that the maximum number of consecutive odd numbers is 5 -/
theorem max_consecutive_odds_is_five :
  ∀ start : Digits, maxConsecutiveOdds (generateSequence start) ≤ 5 ∧
  ∃ start : Digits, maxConsecutiveOdds (generateSequence start) = 5 :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_odds_is_five_l1035_103574


namespace NUMINAMATH_CALUDE_cubic_sum_from_linear_and_quadratic_sum_l1035_103596

theorem cubic_sum_from_linear_and_quadratic_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 17) : 
  x^3 + y^3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_from_linear_and_quadratic_sum_l1035_103596


namespace NUMINAMATH_CALUDE_complex_expression_equality_l1035_103571

theorem complex_expression_equality : 
  let x := (3 + 3/8)^(2/3) - (5 + 4/9)^(1/2) + 0.008^(2/3) / 0.02^(1/2) * 0.32^(1/2)
  x / 0.0625^(1/4) = 23/150 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l1035_103571


namespace NUMINAMATH_CALUDE_division_problem_l1035_103520

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 167 →
  quotient = 9 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  divisor = 18 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1035_103520


namespace NUMINAMATH_CALUDE_fractional_inequality_solution_set_l1035_103534

theorem fractional_inequality_solution_set :
  {x : ℝ | (x + 2) / (x - 1) > 0} = {x : ℝ | x > 1 ∨ x < -2} := by sorry

end NUMINAMATH_CALUDE_fractional_inequality_solution_set_l1035_103534


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1035_103532

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_quadratic_equation : 
  (¬ ∃ x : ℝ, x^2 + x + 1 = 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1035_103532
