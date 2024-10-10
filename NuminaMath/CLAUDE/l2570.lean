import Mathlib

namespace sales_volume_function_correct_profit_at_95_yuan_max_profit_at_110_yuan_max_profit_value_l2570_257094

/-- Represents the weekly sales volume as a function of selling price -/
def sales_volume (x : ℝ) : ℝ := -10 * x + 1500

/-- Represents the weekly profit as a function of selling price -/
def profit (x : ℝ) : ℝ := (x - 80) * (sales_volume x)

/-- The cost price of each shirt -/
def cost_price : ℝ := 80

/-- The minimum allowed selling price -/
def min_price : ℝ := 90

/-- The maximum allowed selling price -/
def max_price : ℝ := 110

theorem sales_volume_function_correct :
  ∀ x, sales_volume x = -10 * x + 1500 := by sorry

theorem profit_at_95_yuan :
  profit 95 = 8250 := by sorry

theorem max_profit_at_110_yuan :
  ∀ x, min_price ≤ x ∧ x ≤ max_price → profit x ≤ profit 110 := by sorry

theorem max_profit_value :
  profit 110 = 12000 := by sorry

end sales_volume_function_correct_profit_at_95_yuan_max_profit_at_110_yuan_max_profit_value_l2570_257094


namespace henrys_age_l2570_257079

/-- Given that the sum of Henry and Jill's present ages is 41, and 7 years ago Henry was twice the age of Jill, prove that Henry's present age is 25. -/
theorem henrys_age (h_age j_age : ℕ) 
  (sum_condition : h_age + j_age = 41)
  (past_condition : h_age - 7 = 2 * (j_age - 7)) :
  h_age = 25 := by
  sorry

end henrys_age_l2570_257079


namespace garden_width_is_correct_garden_area_is_correct_l2570_257077

/-- Represents a rectangular flower garden -/
structure FlowerGarden where
  length : ℝ
  width : ℝ
  area : ℝ

/-- The flower garden has the given dimensions -/
def garden : FlowerGarden where
  length := 4
  width := 35.8
  area := 143.2

/-- Theorem: The width of the flower garden is 35.8 meters -/
theorem garden_width_is_correct : garden.width = 35.8 := by
  sorry

/-- Theorem: The area of the garden is equal to length times width -/
theorem garden_area_is_correct : garden.area = garden.length * garden.width := by
  sorry

end garden_width_is_correct_garden_area_is_correct_l2570_257077


namespace triangle_problem_l2570_257022

open Real

theorem triangle_problem (a b c A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle
  A + B + C = π ∧
  sin A / a = sin B / b ∧ sin A / a = sin C / c ∧  -- Law of sines
  sqrt 3 * c * cos A - a * cos C + b - 2 * c = 0 →
  A = π / 3 ∧ 
  sqrt 3 / 2 < cos B + cos C ∧ cos B + cos C ≤ 1 := by
sorry

end triangle_problem_l2570_257022


namespace orange_pill_cost_l2570_257060

/-- The cost of an orange pill given the conditions of Bob's treatment --/
theorem orange_pill_cost : 
  ∀ (duration : ℕ) (total_cost : ℚ) (blue_pill_cost : ℚ),
  duration = 21 →
  total_cost = 735 →
  blue_pill_cost + 3 + blue_pill_cost = total_cost / duration →
  blue_pill_cost + 3 = 19 := by
  sorry

end orange_pill_cost_l2570_257060


namespace study_seminar_selection_l2570_257020

theorem study_seminar_selection (n m k : ℕ) (h1 : n = 10) (h2 : m = 6) (h3 : k = 2) :
  (n.choose m) - ((n - k).choose (m - k)) = 140 := by
  sorry

end study_seminar_selection_l2570_257020


namespace max_length_special_arithmetic_progression_l2570_257037

/-- An arithmetic progression of natural numbers with common difference 2 -/
def ArithmeticProgression (a₁ : ℕ) (n : ℕ) : Fin n → ℕ :=
  λ i => a₁ + 2 * i.val

/-- Predicate to check if a number is prime -/
def IsPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The maximum length of the special arithmetic progression -/
def MaxLength : ℕ := 3

theorem max_length_special_arithmetic_progression :
  ∀ a₁ n : ℕ,
    (∀ k : Fin n, IsPrime ((ArithmeticProgression a₁ n k)^2 + 1)) →
    n ≤ MaxLength :=
by sorry

end max_length_special_arithmetic_progression_l2570_257037


namespace set_operation_equality_l2570_257005

def U : Set ℝ := Set.univ

def A : Set ℝ := {x : ℝ | x > 0}

def B : Set ℝ := {x : ℝ | x ≤ -1}

theorem set_operation_equality : 
  (A ∩ (U \ B)) ∪ (B ∩ (U \ A)) = {x : ℝ | x > 0 ∨ x ≤ -1} := by sorry

end set_operation_equality_l2570_257005


namespace campers_hiking_morning_l2570_257023

theorem campers_hiking_morning (morning_rowers afternoon_rowers total_rowers : ℕ)
  (h1 : morning_rowers = 13)
  (h2 : afternoon_rowers = 21)
  (h3 : total_rowers = 34)
  (h4 : morning_rowers + afternoon_rowers = total_rowers) :
  total_rowers - (morning_rowers + afternoon_rowers) = 0 :=
by sorry

end campers_hiking_morning_l2570_257023


namespace polynomial_factorization_l2570_257006

theorem polynomial_factorization (y : ℝ) :
  (16 * y^7 - 36 * y^5 + 8 * y) - (4 * y^7 - 12 * y^5 - 8 * y) = 8 * y * (3 * y^6 - 6 * y^4 + 4) := by
  sorry

end polynomial_factorization_l2570_257006


namespace bed_price_ratio_bed_to_frame_ratio_l2570_257039

/-- Given a bed frame price, a bed price multiple, a discount rate, and a final price,
    calculate the ratio of the bed's price to the bed frame's price. -/
theorem bed_price_ratio
  (bed_frame_price : ℝ)
  (bed_price_multiple : ℝ)
  (discount_rate : ℝ)
  (final_price : ℝ)
  (h1 : bed_frame_price = 75)
  (h2 : discount_rate = 0.2)
  (h3 : final_price = 660)
  (h4 : (1 - discount_rate) * (bed_frame_price + bed_frame_price * bed_price_multiple) = final_price) :
  bed_price_multiple = 10 := by
sorry

/-- The ratio of the bed's price to the bed frame's price is 10:1. -/
theorem bed_to_frame_ratio (bed_price_multiple : ℝ) 
  (h : bed_price_multiple = 10) : 
  bed_price_multiple / 1 = 10 / 1 := by
sorry

end bed_price_ratio_bed_to_frame_ratio_l2570_257039


namespace f_zero_gt_f_one_l2570_257061

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def isEvenOn (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, -a ≤ x ∧ x ≤ a → f x = f (-x)

def isMonotonicOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → (f x ≤ f y ∨ f y ≤ f x)

-- State the theorem
theorem f_zero_gt_f_one
  (h_even : isEvenOn f 5)
  (h_mono : isMonotonicOn f 0 5)
  (h_ineq : f (-3) < f (-1)) :
  f 0 > f 1 := by
  sorry

end f_zero_gt_f_one_l2570_257061


namespace train_speed_problem_l2570_257030

/-- Given two trains traveling in opposite directions, this theorem proves
    the speed of the second train given the conditions of the problem. -/
theorem train_speed_problem (v : ℝ) : v = 50 := by
  -- Define the speed of the first train
  let speed1 : ℝ := 64
  -- Define the time of travel
  let time : ℝ := 2.5
  -- Define the total distance between trains after the given time
  let total_distance : ℝ := 285
  
  -- The equation representing the problem:
  -- speed1 * time + v * time = total_distance
  have h : speed1 * time + v * time = total_distance := by sorry
  
  -- Prove that v = 50 given the above equation
  sorry

end train_speed_problem_l2570_257030


namespace inequalities_satisfied_l2570_257009

theorem inequalities_satisfied (a b c x y z : ℝ) 
  (h1 : x ≤ a) (h2 : y ≤ b) (h3 : z ≤ c) : 
  (x * y + y * z + z * x ≤ a * b + b * c + c * a) ∧ 
  (x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) ∧ 
  (x * y * z ≤ a * b * c) := by
  sorry

end inequalities_satisfied_l2570_257009


namespace quadratic_coefficient_l2570_257070

/-- Given a quadratic equation of the form (kx^2 + 5kx + k) = 0 with equal roots when k = 0.64,
    the coefficient of x^2 is 0.64 -/
theorem quadratic_coefficient (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 5 * k * x + k = 0) ∧ 
  (∀ x y : ℝ, k * x^2 + 5 * k * x + k = 0 ∧ k * y^2 + 5 * k * y + k = 0 → x = y) ∧
  k = 0.64 → 
  k = 0.64 := by sorry

end quadratic_coefficient_l2570_257070


namespace sum_after_changes_l2570_257048

theorem sum_after_changes (A B : ℤ) (h : A + B = 100) : 
  (A - 35) + (B + 15) = 80 := by
  sorry

end sum_after_changes_l2570_257048


namespace isabel_piggy_bank_l2570_257036

def initial_amount : ℚ := 204
def spend_half (x : ℚ) : ℚ := x / 2

theorem isabel_piggy_bank :
  spend_half (spend_half initial_amount) = 51 := by
  sorry

end isabel_piggy_bank_l2570_257036


namespace total_apples_packed_l2570_257067

/-- Calculates the total number of apples packed in two weeks under specific conditions -/
theorem total_apples_packed (apples_per_box : ℕ) (boxes_per_day : ℕ) (days_per_week : ℕ) (reduced_apples : ℕ) : 
  apples_per_box = 40 →
  boxes_per_day = 50 →
  days_per_week = 7 →
  reduced_apples = 500 →
  (apples_per_box * boxes_per_day * days_per_week) + 
  ((apples_per_box * boxes_per_day - reduced_apples) * days_per_week) = 24500 := by
sorry

end total_apples_packed_l2570_257067


namespace jimmys_coffee_bean_weight_l2570_257017

/-- Proves the weight of Jimmy's coffee bean bags given the problem conditions -/
theorem jimmys_coffee_bean_weight 
  (suki_bags : ℝ) 
  (suki_weight_per_bag : ℝ) 
  (jimmy_bags : ℝ) 
  (container_weight : ℝ) 
  (num_containers : ℕ) 
  (h1 : suki_bags = 6.5)
  (h2 : suki_weight_per_bag = 22)
  (h3 : jimmy_bags = 4.5)
  (h4 : container_weight = 8)
  (h5 : num_containers = 28) :
  (↑num_containers * container_weight - suki_bags * suki_weight_per_bag) / jimmy_bags = 18 := by
  sorry

#check jimmys_coffee_bean_weight

end jimmys_coffee_bean_weight_l2570_257017


namespace cos_minus_sin_2pi_non_decreasing_l2570_257087

def T_non_decreasing (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) ≥ f x

theorem cos_minus_sin_2pi_non_decreasing :
  T_non_decreasing (fun x => Real.cos x - Real.sin x) (2 * Real.pi) := by
  sorry

end cos_minus_sin_2pi_non_decreasing_l2570_257087


namespace frank_candy_weight_l2570_257029

/-- Frank's candy weight in pounds -/
def frank_candy : ℕ := 10

/-- Gwen's candy weight in pounds -/
def gwen_candy : ℕ := 7

/-- Total candy weight in pounds -/
def total_candy : ℕ := 17

theorem frank_candy_weight : 
  frank_candy + gwen_candy = total_candy :=
by sorry

end frank_candy_weight_l2570_257029


namespace inequality_proof_l2570_257063

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 3 / (1 + a * b * c) := by
  sorry

end inequality_proof_l2570_257063


namespace total_wax_required_l2570_257059

/-- Given the amount of wax already available and the additional amount needed,
    calculate the total wax required for the feathers. -/
theorem total_wax_required 
  (wax_available : ℕ) 
  (wax_needed : ℕ) 
  (h1 : wax_available = 331) 
  (h2 : wax_needed = 22) : 
  wax_available + wax_needed = 353 := by
  sorry

end total_wax_required_l2570_257059


namespace complex_square_one_plus_i_l2570_257043

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_square_one_plus_i : (1 + i)^2 = 2*i :=
sorry

end complex_square_one_plus_i_l2570_257043


namespace parabola_chord_length_l2570_257027

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the line passing through the focus with slope 45°
def line (x y : ℝ) : Prop := y = x - 2

-- Define the chord length
def chord_length : ℝ := 16

-- Theorem statement
theorem parabola_chord_length :
  ∀ (A B : ℝ × ℝ),
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
  line x₁ y₁ ∧ line x₂ y₂ ∧
  A ≠ B →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = chord_length :=
by sorry

end parabola_chord_length_l2570_257027


namespace halfway_fraction_l2570_257038

theorem halfway_fraction (a b c d : ℤ) (h1 : a = 3 ∧ b = 4) (h2 : c = 5 ∧ d = 6) :
  (a : ℚ) / b + ((c : ℚ) / d - (a : ℚ) / b) / 2 = 19 / 24 := by
  sorry

end halfway_fraction_l2570_257038


namespace consecutive_integers_sqrt_l2570_257058

theorem consecutive_integers_sqrt (x y : ℤ) : 
  (y = x + 1) →  -- x and y are consecutive integers
  (x < Real.sqrt 30) →  -- x < √30
  (Real.sqrt 30 < y) →  -- √30 < y
  Real.sqrt (2 * x + y) = 4 ∨ Real.sqrt (2 * x + y) = -4 := by
  sorry

end consecutive_integers_sqrt_l2570_257058


namespace omega_range_for_four_zeros_l2570_257078

/-- Given a function f(x) = cos(ωx) - 1 with ω > 0, if f has exactly 4 zeros 
    in the interval [0, 2π], then 3 ≤ ω < 4. -/
theorem omega_range_for_four_zeros (ω : ℝ) (h_pos : ω > 0) : 
  (∃! (s : Finset ℝ), s.card = 4 ∧ 
    (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.cos (ω * x) = 1) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.cos (ω * x) = 1 → x ∈ s)) →
  3 ≤ ω ∧ ω < 4 := by
sorry

end omega_range_for_four_zeros_l2570_257078


namespace candies_left_after_event_l2570_257055

/-- Calculates the number of candies left after a carousel event --/
theorem candies_left_after_event (
  num_clowns : ℕ
  ) (num_children : ℕ
  ) (initial_supply : ℕ
  ) (candies_per_clown : ℕ
  ) (candies_per_child : ℕ
  ) (candies_as_prizes : ℕ
  ) (h1 : num_clowns = 4
  ) (h2 : num_children = 30
  ) (h3 : initial_supply = 1200
  ) (h4 : candies_per_clown = 10
  ) (h5 : candies_per_child = 15
  ) (h6 : candies_as_prizes = 100
  ) : initial_supply - (num_clowns * candies_per_clown + num_children * candies_per_child + candies_as_prizes) = 610 := by
  sorry

end candies_left_after_event_l2570_257055


namespace smallest_nonprime_with_large_factors_l2570_257011

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def has_no_prime_factor_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, p < 20 → is_prime p → ¬(n % p = 0)

def is_nonprime (n : ℕ) : Prop := n > 1 ∧ ¬(is_prime n)

theorem smallest_nonprime_with_large_factors :
  ∃ n : ℕ, is_nonprime n ∧ 
           has_no_prime_factor_less_than_20 n ∧
           (∀ m : ℕ, m < n → ¬(is_nonprime m ∧ has_no_prime_factor_less_than_20 m)) ∧
           n = 529 :=
sorry

end smallest_nonprime_with_large_factors_l2570_257011


namespace remainder_theorem_l2570_257050

theorem remainder_theorem (n : ℤ) (h : ∃ k : ℤ, n = 60 * k + 1) :
  (n^2 + 2*n + 3) % 60 = 6 := by
sorry

end remainder_theorem_l2570_257050


namespace larger_circle_radius_l2570_257057

/-- The radius of a circle that is internally tangent to four externally tangent circles of radius 2 -/
theorem larger_circle_radius : ℝ := by
  -- Define the radius of the smaller circles
  let small_radius : ℝ := 2

  -- Define the number of smaller circles
  let num_small_circles : ℕ := 4

  -- Define the angle between the centers of adjacent smaller circles
  let angle_between_centers : ℝ := 360 / num_small_circles

  -- Define the radius of the larger circle
  let large_radius : ℝ := small_radius * (1 + Real.sqrt 2)

  -- Prove that the radius of the larger circle is 2(1 + √2)
  sorry

end larger_circle_radius_l2570_257057


namespace rahul_deepak_age_ratio_l2570_257054

/-- Proves that the ratio of Rahul's present age to Deepak's present age is 4:3 -/
theorem rahul_deepak_age_ratio :
  let rahul_future_age : ℕ := 26
  let years_to_future : ℕ := 6
  let deepak_present_age : ℕ := 15
  let rahul_present_age : ℕ := rahul_future_age - years_to_future
  (rahul_present_age : ℚ) / deepak_present_age = 4 / 3 := by
  sorry

end rahul_deepak_age_ratio_l2570_257054


namespace more_customers_left_than_stayed_l2570_257018

theorem more_customers_left_than_stayed (initial_customers remaining_customers : ℕ) :
  initial_customers = 25 →
  remaining_customers = 7 →
  (initial_customers - remaining_customers) - remaining_customers = 11 := by
  sorry

end more_customers_left_than_stayed_l2570_257018


namespace cafeteria_lasagnas_l2570_257025

/-- The number of lasagnas made by the school cafeteria -/
def num_lasagnas : ℕ := sorry

/-- The amount of ground mince used for each lasagna (in pounds) -/
def mince_per_lasagna : ℕ := 2

/-- The amount of ground mince used for each cottage pie (in pounds) -/
def mince_per_cottage_pie : ℕ := 3

/-- The total amount of ground mince used (in pounds) -/
def total_mince_used : ℕ := 500

/-- The number of cottage pies made -/
def num_cottage_pies : ℕ := 100

/-- Theorem stating that the number of lasagnas made is 100 -/
theorem cafeteria_lasagnas : num_lasagnas = 100 := by sorry

end cafeteria_lasagnas_l2570_257025


namespace octahedron_edge_length_is_four_l2570_257095

/-- A regular octahedron circumscribed around four identical balls -/
structure OctahedronWithBalls where
  /-- The radius of each ball -/
  ball_radius : ℝ
  /-- The edge length of the octahedron -/
  edge_length : ℝ
  /-- The condition that three balls are touching each other on the floor -/
  balls_touching : ball_radius = 2
  /-- The condition that the fourth ball rests on top of the other three -/
  fourth_ball_on_top : True

/-- The theorem stating that the edge length of the octahedron is 4 units -/
theorem octahedron_edge_length_is_four (o : OctahedronWithBalls) : o.edge_length = 4 := by
  sorry

end octahedron_edge_length_is_four_l2570_257095


namespace symmetric_point_x_axis_l2570_257047

/-- Given a point M with coordinates (2,3), prove that its symmetric point N 
    with respect to the x-axis has coordinates (2, -3) -/
theorem symmetric_point_x_axis : 
  let M : ℝ × ℝ := (2, 3)
  let N : ℝ × ℝ := (M.1, -M.2)
  N = (2, -3) := by sorry

end symmetric_point_x_axis_l2570_257047


namespace marks_total_votes_l2570_257083

/-- Calculates the total votes Mark received in an election given specific conditions --/
theorem marks_total_votes (first_area_voters : ℕ) 
  (first_area_undecided_percent : ℚ)
  (first_area_mark_percent : ℚ)
  (remaining_area_mark_multiplier : ℕ)
  (remaining_area_undecided_percent : ℚ)
  (remaining_area_population_increase : ℚ) :
  first_area_voters = 100000 →
  first_area_undecided_percent = 5 / 100 →
  first_area_mark_percent = 70 / 100 →
  remaining_area_mark_multiplier = 2 →
  remaining_area_undecided_percent = 7 / 100 →
  remaining_area_population_increase = 20 / 100 →
  ∃ (total_votes : ℕ), total_votes = 199500 ∧
    total_votes = 
      (first_area_voters * (1 - first_area_undecided_percent) * first_area_mark_percent).floor +
      (remaining_area_mark_multiplier * 
        (first_area_voters * (1 - first_area_undecided_percent) * first_area_mark_percent).floor) :=
by
  sorry


end marks_total_votes_l2570_257083


namespace football_original_price_l2570_257066

theorem football_original_price : 
  ∀ (original_price : ℝ), 
  (original_price * 0.8 + 25 = original_price) → 
  original_price = 125 := by
sorry

end football_original_price_l2570_257066


namespace simplify_expression_l2570_257098

theorem simplify_expression (n : ℕ) :
  (2^(n+5) - 4*(2^(n+1))) / (4*(2^(n+4))) = 3/8 := by
  sorry

end simplify_expression_l2570_257098


namespace physics_class_grade_distribution_l2570_257053

theorem physics_class_grade_distribution (total_students : ℕ) 
  (prob_A prob_B prob_C prob_D : ℚ) : 
  total_students = 40 →
  prob_A = (1/2) * prob_B →
  prob_C = 2 * prob_B →
  prob_D = (3/10) * prob_B →
  prob_A + prob_B + prob_C + prob_D = 1 →
  (prob_B * total_students : ℚ) = 200/19 :=
by sorry

end physics_class_grade_distribution_l2570_257053


namespace alvin_age_l2570_257056

/-- Alvin's age -/
def A : ℕ := 30

/-- Simon's age -/
def S : ℕ := 10

/-- Theorem stating that Alvin's age is 30, given the conditions -/
theorem alvin_age : 
  (S + 5 = A / 2) → A = 30 := by
  sorry

end alvin_age_l2570_257056


namespace vector_perpendicular_condition_l2570_257013

/-- Given vectors a and b in R², if (a - b) is perpendicular to b, then the x-coordinate of b is either -1 or 3. -/
theorem vector_perpendicular_condition (x : ℝ) :
  let a : ℝ × ℝ := (2, 4)
  let b : ℝ × ℝ := (x, 3)
  (a.1 - b.1) * b.1 + (a.2 - b.2) * b.2 = 0 → x = -1 ∨ x = 3 := by
  sorry

end vector_perpendicular_condition_l2570_257013


namespace largest_five_digit_divisible_by_8_l2570_257003

theorem largest_five_digit_divisible_by_8 :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 8 = 0 → n ≤ 99992 :=
by sorry

end largest_five_digit_divisible_by_8_l2570_257003


namespace range_of_a_for_three_roots_l2570_257034

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x + a

-- State the theorem
theorem range_of_a_for_three_roots (a : ℝ) :
  (∃ m n p : ℝ, m ≠ n ∧ n ≠ p ∧ m ≠ p ∧ 
    f a m = 2024 ∧ f a n = 2024 ∧ f a p = 2024) →
  2022 < a ∧ a < 2026 := by
sorry

end range_of_a_for_three_roots_l2570_257034


namespace max_strong_boys_is_ten_l2570_257099

/-- A type representing a boy with height and weight -/
structure Boy where
  height : ℕ
  weight : ℕ

/-- A group of 10 boys -/
def Boys := Fin 10 → Boy

/-- Predicate to check if one boy is not inferior to another -/
def not_inferior (a b : Boy) : Prop :=
  a.height > b.height ∨ a.weight > b.weight

/-- Predicate to check if a boy is strong (not inferior to any other boy) -/
def is_strong (boys : Boys) (i : Fin 10) : Prop :=
  ∀ j : Fin 10, j ≠ i → not_inferior (boys i) (boys j)

/-- Theorem stating that it's possible to have 10 strong boys -/
theorem max_strong_boys_is_ten :
  ∃ (boys : Boys), (∀ i j : Fin 10, i ≠ j → boys i ≠ boys j) ∧
                   (∀ i : Fin 10, is_strong boys i) := by
  sorry

end max_strong_boys_is_ten_l2570_257099


namespace selectThreePeopleIs600_l2570_257044

/-- The number of ways to select 3 people from a 5×5 matrix,
    such that no two selected people are in the same row or column. -/
def selectThreePeople : ℕ :=
  let numColumns : ℕ := 5
  let numRows : ℕ := 5
  let numPeopleToSelect : ℕ := 3
  let waysToChooseColumns : ℕ := Nat.choose numColumns numPeopleToSelect
  let waysToChooseFirstPerson : ℕ := numRows
  let waysToChooseSecondPerson : ℕ := numRows - 1
  let waysToChooseThirdPerson : ℕ := numRows - 2
  waysToChooseColumns * waysToChooseFirstPerson * waysToChooseSecondPerson * waysToChooseThirdPerson

/-- Theorem stating that the number of ways to select 3 people
    from a 5×5 matrix, such that no two selected people are in
    the same row or column, is equal to 600. -/
theorem selectThreePeopleIs600 : selectThreePeople = 600 := by
  sorry

end selectThreePeopleIs600_l2570_257044


namespace square_rectangle_area_relation_l2570_257004

theorem square_rectangle_area_relation : 
  ∀ x : ℝ,
  let square_side : ℝ := x - 5
  let rect_length : ℝ := x - 4
  let rect_width : ℝ := x + 3
  let square_area : ℝ := square_side ^ 2
  let rect_area : ℝ := rect_length * rect_width
  (rect_area = 3 * square_area) →
  (∃ y : ℝ, y ≠ x ∧ 
    let square_side' : ℝ := y - 5
    let rect_length' : ℝ := y - 4
    let rect_width' : ℝ := y + 3
    let square_area' : ℝ := square_side' ^ 2
    let rect_area' : ℝ := rect_length' * rect_width'
    (rect_area' = 3 * square_area')) →
  x + y = 33/2 :=
by sorry

end square_rectangle_area_relation_l2570_257004


namespace quadratic_roots_real_distinct_l2570_257014

theorem quadratic_roots_real_distinct :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 + 6*x₁ + 8 = 0) ∧ (x₂^2 + 6*x₂ + 8 = 0) :=
by
  sorry

end quadratic_roots_real_distinct_l2570_257014


namespace sin_thirteen_pi_sixths_l2570_257002

theorem sin_thirteen_pi_sixths : Real.sin (13 * π / 6) = 1 / 2 := by
  sorry

end sin_thirteen_pi_sixths_l2570_257002


namespace distance_from_origin_l2570_257072

theorem distance_from_origin (x : ℝ) : |x| = 4 → x = 4 ∨ x = -4 := by sorry

end distance_from_origin_l2570_257072


namespace polynomial_root_relation_l2570_257012

/-- Two monic cubic polynomials with specified roots and a relation between them -/
theorem polynomial_root_relation (s : ℝ) (h j : ℝ → ℝ) : 
  (∀ x, h x = (x - (s + 2)) * (x - (s + 6)) * (x - c)) →
  (∀ x, j x = (x - (s + 4)) * (x - (s + 8)) * (x - d)) →
  (∀ x, 2 * (h x - j x) = s) →
  s = 64 := by sorry

end polynomial_root_relation_l2570_257012


namespace square_of_complex_fraction_l2570_257075

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem square_of_complex_fraction : (2 * i / (1 - i)) ^ 2 = -2 * i := by
  sorry

end square_of_complex_fraction_l2570_257075


namespace cylinder_height_relationship_l2570_257081

theorem cylinder_height_relationship (r1 h1 r2 h2 : ℝ) :
  r1 > 0 → h1 > 0 → r2 > 0 → h2 > 0 →
  r2 = 1.1 * r1 →
  π * r1^2 * h1 = π * r2^2 * h2 →
  h1 = 1.21 * h2 := by sorry

end cylinder_height_relationship_l2570_257081


namespace largest_integer_for_binary_op_l2570_257026

def binary_op (n : ℤ) : ℤ := n - (n * 5)

theorem largest_integer_for_binary_op :
  ∃ m : ℤ, m = -19 ∧
  (∀ n : ℤ, n > 0 → binary_op n < m → n ≤ 5) ∧
  (∀ m' : ℤ, m' > m → ∃ n : ℤ, n > 0 ∧ n > 5 ∧ binary_op n < m') :=
sorry

end largest_integer_for_binary_op_l2570_257026


namespace shortest_distance_to_origin_l2570_257024

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 2 = 1

-- Define the left focus F
def left_focus : ℝ × ℝ := sorry

-- Define a point P on the right branch of the hyperbola
def point_P : ℝ × ℝ := sorry

-- Define point A satisfying the orthogonality condition
def point_A : ℝ × ℝ := sorry

-- State the theorem
theorem shortest_distance_to_origin :
  ∀ (A : ℝ × ℝ),
    (∃ (P : ℝ × ℝ), hyperbola P.1 P.2 ∧ 
      ((A.1 - P.1) * (A.1 - left_focus.1) + (A.2 - P.2) * (A.2 - left_focus.2) = 0)) →
    (∃ (d : ℝ), d = Real.sqrt 3 ∧ 
      ∀ (B : ℝ × ℝ), Real.sqrt (B.1^2 + B.2^2) ≥ d) :=
by sorry

end shortest_distance_to_origin_l2570_257024


namespace oil_bottles_volume_l2570_257019

theorem oil_bottles_volume :
  let total_bottles : ℕ := 60
  let bottles_250ml : ℕ := 20
  let bottles_300ml : ℕ := 25
  let bottles_350ml : ℕ := total_bottles - bottles_250ml - bottles_300ml
  let volume_250ml : ℕ := 250
  let volume_300ml : ℕ := 300
  let volume_350ml : ℕ := 350
  let total_volume_ml : ℕ := bottles_250ml * volume_250ml + bottles_300ml * volume_300ml + bottles_350ml * volume_350ml
  let ml_per_liter : ℕ := 1000
  total_volume_ml / ml_per_liter = (17750 : ℚ) / 1000 := by
  sorry

end oil_bottles_volume_l2570_257019


namespace cow_husk_consumption_l2570_257046

/-- Given that 50 cows eat 50 bags of husk in 50 days, 
    prove that one cow will eat one bag of husk in the same number of days. -/
theorem cow_husk_consumption (days : ℕ) 
  (h : 50 * 50 = 50 * days) : 
  1 * 1 = 1 * days :=
by
  sorry

end cow_husk_consumption_l2570_257046


namespace watch_cost_price_l2570_257008

theorem watch_cost_price (CP : ℝ) : 
  (1.04 * CP - 0.90 * CP = 280) → CP = 2000 := by
  sorry

end watch_cost_price_l2570_257008


namespace complex_magnitude_equation_l2570_257062

theorem complex_magnitude_equation (t : ℝ) (h : t > 0) :
  Complex.abs (2 + t * Complex.I) = 4 * Real.sqrt 10 → t = 2 * Real.sqrt 39 := by
  sorry

end complex_magnitude_equation_l2570_257062


namespace integer_set_condition_l2570_257033

theorem integer_set_condition (a : ℕ+) : 
  (∃ X : Finset ℤ, X.card = 6 ∧ 
    ∀ k : ℕ, 1 ≤ k ∧ k ≤ 36 → 
      ∃ x y : ℤ, x ∈ X ∧ y ∈ X ∧ 37 ∣ (a.val * x + y - k))
  ↔ (a.val % 37 = 6 ∨ a.val % 37 = 31) :=
by sorry

end integer_set_condition_l2570_257033


namespace set_equality_l2570_257042

def A : Set ℕ := {0, 1, 2, 4, 5, 7}
def B : Set ℕ := {1, 3, 6, 8, 9}
def C : Set ℕ := {3, 7, 8}

theorem set_equality : (A ∩ B) ∪ C = (A ∪ C) ∩ (B ∪ C) := by
  sorry

end set_equality_l2570_257042


namespace disjoint_triangles_probability_disjoint_triangles_probability_proof_l2570_257052

/-- The probability that two triangles formed by six points chosen sequentially at random on a circle's circumference are disjoint -/
theorem disjoint_triangles_probability : ℚ :=
  3/10

/-- Total number of distinct arrangements with one point fixed -/
def total_arrangements : ℕ := 120

/-- Number of favorable outcomes where the triangles are disjoint -/
def favorable_outcomes : ℕ := 36

theorem disjoint_triangles_probability_proof :
  disjoint_triangles_probability = (favorable_outcomes : ℚ) / total_arrangements :=
by sorry

end disjoint_triangles_probability_disjoint_triangles_probability_proof_l2570_257052


namespace cube_sum_equals_thirteen_l2570_257041

theorem cube_sum_equals_thirteen (a b : ℝ) 
  (h1 : a^3 - 3*a*b^2 = 39)
  (h2 : b^3 - 3*a^2*b = 26) :
  a^2 + b^2 = 13 := by
sorry

end cube_sum_equals_thirteen_l2570_257041


namespace tea_cost_price_l2570_257045

/-- Represents the cost price per kg of the 80 kg tea -/
def x : ℝ := 15

/-- The total amount of tea in kg -/
def total_tea : ℝ := 100

/-- The amount of tea with known cost price in kg -/
def known_tea : ℝ := 20

/-- The amount of tea with unknown cost price in kg -/
def unknown_tea : ℝ := 80

/-- The cost price per kg of the known tea -/
def known_tea_price : ℝ := 20

/-- The sale price per kg of the mixed tea -/
def sale_price : ℝ := 21.6

/-- The profit percentage as a decimal -/
def profit_percentage : ℝ := 0.35

theorem tea_cost_price : 
  (unknown_tea * x + known_tea * known_tea_price) * (1 + profit_percentage) = 
  total_tea * sale_price := by sorry

end tea_cost_price_l2570_257045


namespace andrew_stickers_l2570_257090

theorem andrew_stickers (daniel_stickers fred_stickers andrew_kept : ℕ) 
  (h1 : daniel_stickers = 250)
  (h2 : fred_stickers = daniel_stickers + 120)
  (h3 : andrew_kept = 130) :
  andrew_kept + daniel_stickers + fred_stickers = 750 :=
by sorry

end andrew_stickers_l2570_257090


namespace circle_intersection_theorem_l2570_257073

-- Define the circles and line
def C₁ (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 2*y + 7 = 0
def l (x : ℝ) : Prop := x = 1

-- Define the intersection points
def intersection_points (x y : ℝ) : Prop := C₁ x y ∧ C₂ x y

-- Define the line y = x
def y_eq_x (x y : ℝ) : Prop := y = x

-- Define circle C₃
def C₃ (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

theorem circle_intersection_theorem :
  (∀ x y, intersection_points x y → l x) ∧
  (∃ x₀ y₀, C₃ x₀ y₀ ∧ y_eq_x x₀ y₀ ∧ (∀ x y, intersection_points x y → C₃ x y)) :=
sorry

end circle_intersection_theorem_l2570_257073


namespace nested_radical_value_l2570_257065

/-- The value of the infinite nested radical sqrt(16 + sqrt(16 + sqrt(16 + ...))) -/
noncomputable def nestedRadical : ℝ :=
  Real.sqrt (16 + Real.sqrt (16 + Real.sqrt (16 + Real.sqrt (16 + Real.sqrt (16 + Real.sqrt 16)))))

/-- Theorem stating that the nested radical equals (1 + sqrt(65)) / 2 -/
theorem nested_radical_value : nestedRadical = (1 + Real.sqrt 65) / 2 := by
  sorry

end nested_radical_value_l2570_257065


namespace factorial_equation_solutions_l2570_257097

def is_solution (x y z : ℕ) : Prop :=
  (x + y) / z = (Nat.factorial x + Nat.factorial y) / Nat.factorial z

theorem factorial_equation_solutions :
  ∀ x y z : ℕ, is_solution x y z →
    ((x = 1 ∧ y = 1 ∧ z = 2) ∨
     (x = 2 ∧ y = 2 ∧ z = 1) ∨
     (x = y ∧ y = z ∧ z ≥ 3)) :=
by sorry

end factorial_equation_solutions_l2570_257097


namespace regression_line_mean_y_l2570_257021

theorem regression_line_mean_y (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ = 1) (h₂ : x₂ = 5) (h₃ : x₃ = 7) (h₄ : x₄ = 13) (h₅ : x₅ = 19)
  (regression_eq : ℝ → ℝ) (h_reg : ∀ x, regression_eq x = 1.5 * x + 45) : 
  let x_mean := (x₁ + x₂ + x₃ + x₄ + x₅) / 5
  regression_eq x_mean = 58.5 := by
sorry

end regression_line_mean_y_l2570_257021


namespace ratio_a_to_c_l2570_257031

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 9) :
  a / c = 135 / 16 := by
sorry

end ratio_a_to_c_l2570_257031


namespace magnitude_of_complex_power_l2570_257080

theorem magnitude_of_complex_power : 
  Complex.abs ((2 : ℂ) + 2 * Complex.I * Real.sqrt 2) ^ 6 = 1728 := by sorry

end magnitude_of_complex_power_l2570_257080


namespace probability_is_sqrt_two_over_fifteen_l2570_257089

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The probability of x^2 < y for a point (x,y) randomly picked from the given rectangle --/
def probability_x_squared_less_than_y (rect : Rectangle) : ℝ :=
  sorry

/-- The specific rectangle in the problem --/
def problem_rectangle : Rectangle :=
  { x_min := 0
  , x_max := 5
  , y_min := 0
  , y_max := 2
  , h_x := by norm_num
  , h_y := by norm_num
  }

theorem probability_is_sqrt_two_over_fifteen :
  probability_x_squared_less_than_y problem_rectangle = Real.sqrt 2 / 15 := by
  sorry

end probability_is_sqrt_two_over_fifteen_l2570_257089


namespace lowest_digit_change_l2570_257032

/-- The correct sum of the addition -/
def correct_sum : ℕ := 1179

/-- The first addend in the incorrect addition -/
def addend1 : ℕ := 374

/-- The second addend in the incorrect addition -/
def addend2 : ℕ := 519

/-- The third addend in the incorrect addition -/
def addend3 : ℕ := 286

/-- The incorrect sum displayed in the problem -/
def incorrect_sum : ℕ := 1229

/-- Function to check if a digit change makes the addition correct -/
def is_correct_change (digit : ℕ) (position : ℕ) : Prop :=
  ∃ (new_addend : ℕ),
    (position = 1 ∧ new_addend + addend2 + addend3 = correct_sum) ∨
    (position = 2 ∧ addend1 + new_addend + addend3 = correct_sum) ∨
    (position = 3 ∧ addend1 + addend2 + new_addend = correct_sum)

/-- The lowest digit that can be changed to make the addition correct -/
def lowest_changeable_digit : ℕ := 4

theorem lowest_digit_change :
  (∀ d : ℕ, d < lowest_changeable_digit → ¬∃ p : ℕ, is_correct_change d p) ∧
  (∃ p : ℕ, is_correct_change lowest_changeable_digit p) :=
sorry

end lowest_digit_change_l2570_257032


namespace complex_multiplication_sum_l2570_257000

theorem complex_multiplication_sum (a b : ℝ) (i : ℂ) : 
  (1 + i) * (2 + i) = a + b * i → i * i = -1 → a + b = 4 := by
  sorry

end complex_multiplication_sum_l2570_257000


namespace function_properties_l2570_257074

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 2

-- Theorem statement
theorem function_properties :
  (∀ x y, x < y → x < -1 → y < -1 → f x < f y) ∧
  (∀ x y, x < y → x > 3 → y > 3 → f x < f y) ∧
  (∀ x ∈ interval, f x ≤ 5) ∧
  (∃ x ∈ interval, f x = 5) ∧
  (∀ x ∈ interval, f x ≥ -22) ∧
  (∃ x ∈ interval, f x = -22) :=
by sorry

end function_properties_l2570_257074


namespace combined_eighth_grade_percentage_l2570_257088

/-- Represents the percentage of 8th grade students in a school -/
structure School :=
  (total_students : ℕ)
  (eighth_grade_percentage : ℚ)

/-- Calculates the total number of 8th grade students in both schools -/
def total_eighth_graders (oakwood pinecrest : School) : ℚ :=
  (oakwood.total_students : ℚ) * oakwood.eighth_grade_percentage / 100 +
  (pinecrest.total_students : ℚ) * pinecrest.eighth_grade_percentage / 100

/-- Calculates the total number of students in both schools -/
def total_students (oakwood pinecrest : School) : ℕ :=
  oakwood.total_students + pinecrest.total_students

/-- Theorem stating that the percentage of 8th graders in both schools combined is 57% -/
theorem combined_eighth_grade_percentage 
  (oakwood : School) 
  (pinecrest : School)
  (h1 : oakwood.total_students = 150)
  (h2 : pinecrest.total_students = 250)
  (h3 : oakwood.eighth_grade_percentage = 60)
  (h4 : pinecrest.eighth_grade_percentage = 55) :
  (total_eighth_graders oakwood pinecrest) / (total_students oakwood pinecrest : ℚ) * 100 = 57 :=
sorry

end combined_eighth_grade_percentage_l2570_257088


namespace x_plus_y_value_l2570_257076

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.sin y = 2008)
  (h2 : x + 2008 * Real.cos y = 2007)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 := by
sorry

end x_plus_y_value_l2570_257076


namespace arithmetic_geometric_sequence_ratio_l2570_257015

-- Define arithmetic sequence
def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

-- Define geometric sequence
def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

theorem arithmetic_geometric_sequence_ratio :
  ∀ (x y a b c : ℝ),
  is_arithmetic_sequence 1 x y 4 →
  is_geometric_sequence (-2) a b c (-8) →
  (y - x) / b = -1/4 := by sorry

end arithmetic_geometric_sequence_ratio_l2570_257015


namespace salt_solution_volume_l2570_257091

/-- Given a mixture of pure water and a salt solution, calculates the volume of the salt solution needed to achieve a specific concentration. -/
theorem salt_solution_volume 
  (pure_water_volume : ℝ)
  (salt_solution_concentration : ℝ)
  (final_concentration : ℝ)
  (h1 : pure_water_volume = 1)
  (h2 : salt_solution_concentration = 0.75)
  (h3 : final_concentration = 0.15) :
  ∃ x : ℝ, x = 0.25 ∧ 
    salt_solution_concentration * x = final_concentration * (pure_water_volume + x) :=
by sorry

end salt_solution_volume_l2570_257091


namespace unique_quadratic_solution_l2570_257092

/-- A function that checks if a quadratic equation with coefficients based on A has positive integer solutions -/
def has_positive_integer_solutions (A : ℕ) : Prop :=
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 - (2*A)*x + (A+1)*10 = 0 ∧ y^2 - (2*A)*y + (A+1)*10 = 0

/-- The theorem stating that there is exactly one single-digit positive integer A that satisfies the condition -/
theorem unique_quadratic_solution : 
  ∃! A : ℕ, 1 ≤ A ∧ A ≤ 9 ∧ has_positive_integer_solutions A :=
sorry

end unique_quadratic_solution_l2570_257092


namespace Q_when_b_is_one_Q_subset_P_iff_b_in_range_l2570_257096

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - 5*x + 4 ≤ 0}
def Q (b : ℝ) : Set ℝ := {x | x^2 - (b+2)*x + 2*b ≤ 0}

-- Theorem 1: When b = 1, Q = {x | 1 ≤ x ≤ 2}
theorem Q_when_b_is_one : Q 1 = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem 2: Q ⊆ P if and only if b ∈ [1, 4]
theorem Q_subset_P_iff_b_in_range : ∀ b : ℝ, Q b ⊆ P ↔ 1 ≤ b ∧ b ≤ 4 := by sorry

end Q_when_b_is_one_Q_subset_P_iff_b_in_range_l2570_257096


namespace red_straws_per_mat_l2570_257035

theorem red_straws_per_mat (orange_per_mat green_per_mat total_straws mats : ℕ)
  (h1 : orange_per_mat = 30)
  (h2 : green_per_mat = orange_per_mat / 2)
  (h3 : total_straws = 650)
  (h4 : mats = 10) :
  (total_straws - (orange_per_mat + green_per_mat) * mats) / mats = 20 :=
by sorry

end red_straws_per_mat_l2570_257035


namespace expression_evaluation_l2570_257082

theorem expression_evaluation :
  let x : ℕ := 3
  let y : ℕ := 4
  5 * x^y + 2 * y^x = 533 := by
sorry

end expression_evaluation_l2570_257082


namespace correct_expression_l2570_257093

theorem correct_expression (x y : ℚ) (h : x / y = 5 / 6) : 
  (x + 3 * y) / x = 23 / 5 := by
  sorry

end correct_expression_l2570_257093


namespace equal_distances_l2570_257049

def circular_distance (n : ℕ) (a b : ℕ) : ℕ :=
  (b - a + n) % n

theorem equal_distances : ∃ n : ℕ, 
  n > 0 ∧ 
  circular_distance n 31 7 = circular_distance n 31 14 ∧ 
  n = 41 := by
  sorry

end equal_distances_l2570_257049


namespace derivative_at_two_l2570_257016

-- Define f as a real-valued function
variable (f : ℝ → ℝ)

-- Define the conditions
def tangent_coincide (f : ℝ → ℝ) : Prop :=
  ∃ (m : ℝ), (∀ x, x ≠ 0 → (f x) / x - 1 = m * (x - 2)) ∧
             (∀ x, f x = m * x)

-- Theorem statement
theorem derivative_at_two (f : ℝ → ℝ) 
  (h1 : tangent_coincide f) 
  (h2 : f 0 = 0) :
  deriv f 2 = 2 := by
  sorry

end derivative_at_two_l2570_257016


namespace bicycle_selection_l2570_257071

theorem bicycle_selection (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 3) :
  (n * (n - 1) * (n - 2)) = 2730 :=
sorry

end bicycle_selection_l2570_257071


namespace triangle_inequality_l2570_257010

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define points D, E, F on the sides of the triangle
structure TriangleWithPoints extends Triangle :=
  (D : ℝ × ℝ)
  (E : ℝ × ℝ)
  (F : ℝ × ℝ)

-- Define the condition DC + CE = EA + AF = FB + BD
def satisfiesCondition (t : TriangleWithPoints) : Prop :=
  let distAB := Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2)
  let distBC := Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)
  let distCA := Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)
  let distDC := Real.sqrt ((t.D.1 - t.C.1)^2 + (t.D.2 - t.C.2)^2)
  let distCE := Real.sqrt ((t.C.1 - t.E.1)^2 + (t.C.2 - t.E.2)^2)
  let distEA := Real.sqrt ((t.E.1 - t.A.1)^2 + (t.E.2 - t.A.2)^2)
  let distAF := Real.sqrt ((t.A.1 - t.F.1)^2 + (t.A.2 - t.F.2)^2)
  let distFB := Real.sqrt ((t.F.1 - t.B.1)^2 + (t.F.2 - t.B.2)^2)
  let distBD := Real.sqrt ((t.B.1 - t.D.1)^2 + (t.B.2 - t.D.2)^2)
  distDC + distCE = distEA + distAF ∧ distEA + distAF = distFB + distBD

-- State the theorem
theorem triangle_inequality (t : TriangleWithPoints) (h : satisfiesCondition t) :
  let distDE := Real.sqrt ((t.D.1 - t.E.1)^2 + (t.D.2 - t.E.2)^2)
  let distEF := Real.sqrt ((t.E.1 - t.F.1)^2 + (t.E.2 - t.F.2)^2)
  let distFD := Real.sqrt ((t.F.1 - t.D.1)^2 + (t.F.2 - t.D.2)^2)
  let distAB := Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2)
  let distBC := Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)
  let distCA := Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)
  distDE + distEF + distFD ≥ (1/2) * (distAB + distBC + distCA) :=
by
  sorry

end triangle_inequality_l2570_257010


namespace no_five_integers_with_prime_triples_l2570_257084

theorem no_five_integers_with_prime_triples : ¬ ∃ (a b c d e : ℕ+),
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
  (∀ (x y z : ℕ+), (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) →
                   (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) →
                   (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) →
                   (x ≠ y ∧ x ≠ z ∧ y ≠ z) →
                   Nat.Prime (x.val + y.val + z.val)) :=
by sorry

end no_five_integers_with_prime_triples_l2570_257084


namespace parallel_vectors_imply_x_value_l2570_257068

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b, prove that if they are parallel, then x = -2 -/
theorem parallel_vectors_imply_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, -4)
  are_parallel a b → x = -2 :=
by
  sorry


end parallel_vectors_imply_x_value_l2570_257068


namespace dans_age_proof_l2570_257051

/-- Dan's present age in years -/
def dans_present_age : ℕ := 16

/-- Theorem stating that Dan's present age satisfies the given condition -/
theorem dans_age_proof :
  dans_present_age + 16 = 4 * (dans_present_age - 8) :=
by sorry

end dans_age_proof_l2570_257051


namespace nested_sum_value_l2570_257069

def nested_sum (n : ℕ) : ℚ :=
  if n = 0 then 0
  else (n + 1000 : ℚ) + (1/3) * nested_sum (n-1)

theorem nested_sum_value :
  nested_sum 999 = 999.5 + 1498.5 * 3^997 :=
sorry

end nested_sum_value_l2570_257069


namespace reciprocal_roots_l2570_257028

theorem reciprocal_roots (p q : ℝ) (r₁ r₂ : ℂ) : 
  (r₁^2 + p*r₁ + q = 0 ∧ r₂^2 + p*r₂ + q = 0) → 
  ((1/r₁)^2 * q + (1/r₁) * p + 1 = 0 ∧ (1/r₂)^2 * q + (1/r₂) * p + 1 = 0) :=
sorry

end reciprocal_roots_l2570_257028


namespace a_seven_value_l2570_257040

/-- An arithmetic sequence where the reciprocals of terms form an arithmetic sequence -/
def reciprocal_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, 1 / a (n + 1) - 1 / a n = d

theorem a_seven_value (a : ℕ → ℝ) 
  (h_seq : reciprocal_arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a4 : a 4 = 4) :
  a 7 = -2 :=
sorry

end a_seven_value_l2570_257040


namespace oxford_high_school_teachers_l2570_257064

theorem oxford_high_school_teachers (num_classes : ℕ) (students_per_class : ℕ) (total_people : ℕ) :
  num_classes = 15 →
  students_per_class = 20 →
  total_people = 349 →
  total_people = num_classes * students_per_class + 1 + 48 :=
by
  sorry

end oxford_high_school_teachers_l2570_257064


namespace expression_simplification_and_evaluation_l2570_257086

theorem expression_simplification_and_evaluation (x : ℝ) 
  (hx_neq_neg1 : x ≠ -1) (hx_neq_0 : x ≠ 0) (hx_neq_1 : x ≠ 1) :
  (1 / (x + 1) + 1 / (x^2 - 1)) / (x / (x - 1)) = 1 / (x + 1) ∧
  (1 / (Real.sqrt 3 + 1) = (Real.sqrt 3 - 1) / 2) := by
  sorry

end expression_simplification_and_evaluation_l2570_257086


namespace expression_evaluation_l2570_257007

theorem expression_evaluation : 
  3 + Real.sqrt 3 + (3 + Real.sqrt 3)⁻¹ + (Real.sqrt 3 - 3)⁻¹ = 3 := by
  sorry

end expression_evaluation_l2570_257007


namespace second_year_interest_rate_l2570_257085

/-- Proves that given an initial investment of $15,000 with a 10% simple annual interest rate
    for the first year, and a final amount of $17,325 after two years, the interest rate of
    the second year's investment is 5%. -/
theorem second_year_interest_rate
  (initial_investment : ℝ)
  (first_year_rate : ℝ)
  (final_amount : ℝ)
  (h1 : initial_investment = 15000)
  (h2 : first_year_rate = 0.1)
  (h3 : final_amount = 17325)
  : ∃ (second_year_rate : ℝ),
    final_amount = initial_investment * (1 + first_year_rate) * (1 + second_year_rate) ∧
    second_year_rate = 0.05 := by
  sorry

end second_year_interest_rate_l2570_257085


namespace line_problem_l2570_257001

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_problem (a b : ℝ) :
  let l0 : Line := ⟨1, -1, 1⟩
  let l1 : Line := ⟨a, -2, 1⟩
  let l2 : Line := ⟨1, b, 3⟩
  perpendicular l0 l1 → parallel l0 l2 → a + b = -3 := by
  sorry

end line_problem_l2570_257001
