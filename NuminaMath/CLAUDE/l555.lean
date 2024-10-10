import Mathlib

namespace polynomial_expansion_l555_55588

theorem polynomial_expansion (x : ℝ) : 
  (x^3 - 3*x^2 + 3*x - 1) * (x^2 + 3*x + 3) = x^5 - 3*x^3 - x^2 + 3*x := by
  sorry

end polynomial_expansion_l555_55588


namespace sequence_properties_l555_55517

def sequence_a (n : ℕ) : ℚ := 1/10 * (3/2)^(n-1) - 2/5 * (-1)^n

def partial_sum (n : ℕ) : ℚ := 3 * sequence_a n + (-1)^n

theorem sequence_properties :
  (sequence_a 1 = 1/2) ∧
  (sequence_a 2 = -1/4) ∧
  (sequence_a 3 = 5/8) ∧
  (∀ n : ℕ, n ≥ 2 → sequence_a n + 2/5 * (-1)^n = 3/2 * (sequence_a (n-1) + 2/5 * (-1)^(n-1))) ∧
  (∀ n : ℕ, partial_sum n = 3 * sequence_a n + (-1)^n) :=
sorry

end sequence_properties_l555_55517


namespace odd_induction_l555_55573

theorem odd_induction (P : ℕ → Prop) 
  (base : P 1) 
  (step : ∀ k : ℕ, k ≥ 1 → P k → P (k + 2)) : 
  ∀ n : ℕ, n > 0 ∧ Odd n → P n :=
sorry

end odd_induction_l555_55573


namespace special_sequence_sixth_term_l555_55552

/-- A sequence of positive integers where each term after the first is 1/3 of the sum of the term that precedes it and the term that follows it. -/
def SpecialSequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n > 0 → a n > 0 ∧ a (n + 1) = (a n + a (n + 2)) / 3

theorem special_sequence_sixth_term
  (a : ℕ → ℚ)
  (h_seq : SpecialSequence a)
  (h_first : a 1 = 3)
  (h_fifth : a 5 = 54) :
  a 6 = 1133 / 7 := by
  sorry

end special_sequence_sixth_term_l555_55552


namespace jack_cookies_needed_l555_55564

/-- Represents the sales data and goals for Jack's bake sale -/
structure BakeSale where
  brownies_sold : Nat
  brownies_price : Nat
  lemon_squares_sold : Nat
  lemon_squares_price : Nat
  cookie_price : Nat
  bulk_pack_size : Nat
  bulk_pack_price : Nat
  sales_goal : Nat

/-- Calculates the minimum number of cookies needed to reach the sales goal -/
def min_cookies_needed (sale : BakeSale) : Nat :=
  sorry

/-- Theorem stating that Jack needs to sell 8 cookies to reach his goal -/
theorem jack_cookies_needed (sale : BakeSale) 
  (h1 : sale.brownies_sold = 4)
  (h2 : sale.brownies_price = 3)
  (h3 : sale.lemon_squares_sold = 5)
  (h4 : sale.lemon_squares_price = 2)
  (h5 : sale.cookie_price = 4)
  (h6 : sale.bulk_pack_size = 5)
  (h7 : sale.bulk_pack_price = 17)
  (h8 : sale.sales_goal = 50) :
  min_cookies_needed sale = 8 := by
  sorry

end jack_cookies_needed_l555_55564


namespace organization_member_count_l555_55597

/-- Represents an organization with committees and members -/
structure Organization where
  num_committees : Nat
  num_members : Nat
  member_committee_count : Nat
  pair_common_member_count : Nat

/-- The specific organization described in the problem -/
def specific_org : Organization :=
  { num_committees := 5
  , num_members := 10
  , member_committee_count := 2
  , pair_common_member_count := 1
  }

/-- Theorem stating that the organization with the given properties has 10 members -/
theorem organization_member_count :
  ∀ (org : Organization),
    org.num_committees = 5 ∧
    org.member_committee_count = 2 ∧
    org.pair_common_member_count = 1 →
    org.num_members = 10 := by
  sorry

#check organization_member_count

end organization_member_count_l555_55597


namespace sqrt_three_fourths_equals_sqrt_three_over_two_l555_55572

theorem sqrt_three_fourths_equals_sqrt_three_over_two :
  Real.sqrt (3 / 4) = Real.sqrt 3 / 2 := by
  sorry

end sqrt_three_fourths_equals_sqrt_three_over_two_l555_55572


namespace polynomial_simplification_l555_55596

theorem polynomial_simplification (p q : ℝ) :
  (4 * q^4 + 2 * p^3 - 7 * p + 8) + (3 * q^4 - 2 * p^3 + 3 * p^2 - 5 * p + 6) =
  7 * q^4 + 3 * p^2 - 12 * p + 14 := by
sorry

end polynomial_simplification_l555_55596


namespace binary_110_eq_6_l555_55565

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110_eq_6 :
  binary_to_decimal [true, true, false] = 6 := by
  sorry

end binary_110_eq_6_l555_55565


namespace range_of_k_when_q_range_of_a_when_p_necessary_not_sufficient_l555_55542

-- Define propositions p and q
def p (k a : ℝ) : Prop := ∃ x y : ℝ, x^2/(k-1) + y^2/(7-a) = 1 ∧ k ≠ 1 ∧ a ≠ 7

def q (k : ℝ) : Prop := ¬∃ x y : ℝ, (4-k)*x^2 + (k-2)*y^2 = 1 ∧ (4-k)*(k-2) < 0

-- Theorem 1: Range of k when q is true
theorem range_of_k_when_q (k : ℝ) : q k → 2 ≤ k ∧ k ≤ 4 :=
sorry

-- Theorem 2: Range of a when p is a necessary but not sufficient condition for q
theorem range_of_a_when_p_necessary_not_sufficient (a : ℝ) :
  (∀ k : ℝ, q k → (∃ k', p k' a)) ∧ (∃ k : ℝ, p k a ∧ ¬q k) → a < 4 :=
sorry

end range_of_k_when_q_range_of_a_when_p_necessary_not_sufficient_l555_55542


namespace polly_tweets_l555_55516

/-- Represents the tweet rate per minute for different states of Polly --/
structure TweetRates where
  happy : Nat
  hungry : Nat
  mirror : Nat

/-- Represents the duration in minutes for different activities --/
structure ActivityDurations where
  happy : Nat
  hungry : Nat
  mirror : Nat

/-- Calculates the total number of tweets based on rates and durations --/
def totalTweets (rates : TweetRates) (durations : ActivityDurations) : Nat :=
  rates.happy * durations.happy +
  rates.hungry * durations.hungry +
  rates.mirror * durations.mirror

/-- Theorem stating that Polly's total tweets equal 1340 --/
theorem polly_tweets (rates : TweetRates) (durations : ActivityDurations)
    (h1 : rates.happy = 18)
    (h2 : rates.hungry = 4)
    (h3 : rates.mirror = 45)
    (h4 : durations.happy = 20)
    (h5 : durations.hungry = 20)
    (h6 : durations.mirror = 20) :
    totalTweets rates durations = 1340 := by
  sorry


end polly_tweets_l555_55516


namespace max_profit_is_120_l555_55532

def profit_A (x : ℕ) : ℚ := -x^2 + 21*x
def profit_B (x : ℕ) : ℚ := 2*x
def total_profit (x : ℕ) : ℚ := profit_A x + profit_B (15 - x)

theorem max_profit_is_120 :
  ∃ x : ℕ, x > 0 ∧ x ≤ 15 ∧
  total_profit x = 120 ∧
  ∀ y : ℕ, y > 0 → y ≤ 15 → total_profit y ≤ total_profit x :=
sorry

end max_profit_is_120_l555_55532


namespace division_reduction_l555_55580

theorem division_reduction (x : ℝ) (h : x > 0) : 54 / x = 54 - 36 → x = 3 := by
  sorry

end division_reduction_l555_55580


namespace sum_in_base5_l555_55530

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : ℕ) : ℕ := sorry

/-- Represents a number in base 5 -/
structure Base5 where
  value : ℕ

theorem sum_in_base5 : 
  let a := Base5.mk 132
  let b := Base5.mk 214
  let c := Base5.mk 341
  let sum := base10ToBase5 (base5ToBase10 a.value + base5ToBase10 b.value + base5ToBase10 c.value)
  sum = 1242 := by
  sorry

end sum_in_base5_l555_55530


namespace infiniteSum_eq_one_l555_55519

/-- Sequence F defined recursively -/
def F : ℕ → ℚ
  | 0 => 0
  | 1 => 3/2
  | (n+2) => 5/2 * F (n+1) - F n

/-- The sum of 1/F(2^n) from n=0 to infinity -/
noncomputable def infiniteSum : ℚ := ∑' n, 1 / F (2^n)

/-- Theorem stating that the infinite sum is equal to 1 -/
theorem infiniteSum_eq_one : infiniteSum = 1 := by sorry

end infiniteSum_eq_one_l555_55519


namespace min_overlap_blue_eyes_lunch_box_l555_55557

theorem min_overlap_blue_eyes_lunch_box 
  (total_students : ℕ) 
  (blue_eyes : ℕ) 
  (lunch_box : ℕ) 
  (h1 : total_students = 25) 
  (h2 : blue_eyes = 15) 
  (h3 : lunch_box = 18) :
  blue_eyes + lunch_box - total_students = 8 := by
  sorry

end min_overlap_blue_eyes_lunch_box_l555_55557


namespace surface_area_of_specific_solid_l555_55590

/-- A right prism with equilateral triangle bases -/
structure RightPrism where
  height : ℝ
  base_side : ℝ

/-- Midpoint of an edge -/
structure Midpoint where
  edge : String

/-- The solid formed by slicing off the top part of the prism -/
structure SlicedSolid where
  prism : RightPrism
  x : Midpoint
  y : Midpoint
  z : Midpoint

/-- Calculate the surface area of the sliced solid -/
noncomputable def surface_area (solid : SlicedSolid) : ℝ :=
  sorry

/-- Theorem stating the surface area of the specific sliced solid -/
theorem surface_area_of_specific_solid :
  let prism := RightPrism.mk 20 10
  let x := Midpoint.mk "AC"
  let y := Midpoint.mk "BC"
  let z := Midpoint.mk "DF"
  let solid := SlicedSolid.mk prism x y z
  surface_area solid = 100 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 418.75) / 2 :=
sorry

end surface_area_of_specific_solid_l555_55590


namespace P_subset_M_l555_55533

def M : Set ℕ := {0, 2}

def P : Set ℕ := {x | x ∈ M}

theorem P_subset_M : P ⊆ M := by
  sorry

end P_subset_M_l555_55533


namespace odd_even_function_sum_l555_55546

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem odd_even_function_sum (f g : ℝ → ℝ) 
  (h_odd : is_odd f) (h_even : is_even g) 
  (h_sum : ∀ x, f x + g x = x^3 - x^2 + x - 3) :
  ∀ x, f x = x^3 + x := by
sorry

end odd_even_function_sum_l555_55546


namespace football_cost_l555_55575

/-- The cost of a football given the total cost of a football and baseball, and the cost of the baseball. -/
theorem football_cost (total_cost baseball_cost : ℚ) : 
  total_cost = 20 - (4 + 5/100) → 
  baseball_cost = 6 + 81/100 → 
  total_cost - baseball_cost = 9 + 14/100 := by
sorry

end football_cost_l555_55575


namespace solve_and_prove_l555_55576

-- Given that |x+a| ≤ b has the solution set [-6, 2]
def has_solution_set (a b : ℝ) : Prop :=
  ∀ x, |x + a| ≤ b ↔ -6 ≤ x ∧ x ≤ 2

-- Define the conditions |am+n| < 1/3 and |m-bn| < 1/6
def conditions (a b m n : ℝ) : Prop :=
  |a * m + n| < 1/3 ∧ |m - b * n| < 1/6

theorem solve_and_prove (a b m n : ℝ) 
  (h1 : has_solution_set a b) 
  (h2 : conditions a b m n) : 
  (a = 2 ∧ b = 4) ∧ |n| < 2/27 :=
sorry

end solve_and_prove_l555_55576


namespace function_difference_constant_l555_55512

open Function Real

theorem function_difference_constant 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g)
  (h_second_deriv : ∀ x, deriv (deriv f) x = deriv (deriv g) x) :
  ∃ C, ∀ x, f x - g x = C :=
sorry

end function_difference_constant_l555_55512


namespace fraction_multiplication_and_subtraction_l555_55599

theorem fraction_multiplication_and_subtraction :
  (5 : ℚ) / 6 * ((2 : ℚ) / 3 - (1 : ℚ) / 9) = (25 : ℚ) / 54 := by
  sorry

end fraction_multiplication_and_subtraction_l555_55599


namespace ninth_day_skating_time_l555_55523

def minutes_per_hour : ℕ := 60

def skating_time_first_5_days : ℕ := 75
def skating_time_next_3_days : ℕ := 90
def total_days : ℕ := 9
def target_average : ℕ := 85

def total_skating_time : ℕ := 
  (skating_time_first_5_days * 5) + (skating_time_next_3_days * 3)

theorem ninth_day_skating_time :
  (total_skating_time + 120) / total_days = target_average :=
sorry

end ninth_day_skating_time_l555_55523


namespace triangle_angle_calculation_l555_55505

theorem triangle_angle_calculation (A B C : ℝ) : 
  A = 88 → B - C = 20 → A + B + C = 180 → C = 36 := by
  sorry

end triangle_angle_calculation_l555_55505


namespace autumn_outing_problem_l555_55506

/-- Autumn Outing Problem -/
theorem autumn_outing_problem 
  (bus_seats : ℕ) 
  (public_bus_seats : ℕ) 
  (bus_count : ℕ) 
  (teachers_per_bus : ℕ) 
  (extra_seats_buses : ℕ) 
  (extra_teachers_public : ℕ) 
  (h1 : bus_seats = 39)
  (h2 : public_bus_seats = 27)
  (h3 : bus_count + 2 = public_bus_count)
  (h4 : teachers_per_bus = 2)
  (h5 : extra_seats_buses = 3)
  (h6 : extra_teachers_public = 3)
  (h7 : bus_seats * bus_count = teachers_per_bus * bus_count + students + extra_seats_buses)
  (h8 : public_bus_seats * public_bus_count = teachers + students)
  (h9 : teachers = public_bus_count + extra_teachers_public) :
  teachers = 18 ∧ students = 330 := by
  sorry


end autumn_outing_problem_l555_55506


namespace college_student_count_l555_55579

theorem college_student_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 160) :
  boys + girls = 416 := by
sorry

end college_student_count_l555_55579


namespace min_value_of_y_l555_55559

def y (x : ℝ) : ℝ :=
  |x - 1| + |x - 2| + |x - 3| + |x - 4| + |x - 5| + |x - 6| + |x - 7| + |x - 8| + |x - 9| + |x - 10|

theorem min_value_of_y :
  ∃ (x : ℝ), ∀ (z : ℝ), y z ≥ y x ∧ y x = 25 :=
sorry

end min_value_of_y_l555_55559


namespace f_properties_l555_55587

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a * Real.exp x

theorem f_properties :
  ∀ a : ℝ,
  (a = -1 →
    (∀ x y : ℝ, x < y → x < 0 → y < 0 → f a x < f a y) ∧
    (∀ x y : ℝ, x < y → x > 0 → y > 0 → f a x > f a y)) ∧
  (a ≥ 0 →
    ∀ x : ℝ, ¬∃ y : ℝ, (∀ z : ℝ, f a z ≤ f a y) ∨ (∀ z : ℝ, f a z ≥ f a y)) ∧
  (a < 0 →
    (∃ x : ℝ, ∀ y : ℝ, f a y ≤ f a x) ∧
    (¬∃ x : ℝ, ∀ y : ℝ, f a y ≥ f a x) ∧
    (∃ x : ℝ, x = Real.log (-1/a) ∧ ∀ y : ℝ, f a y ≤ f a x) ∧
    (∃ x : ℝ, x = Real.log (-1/a) ∧ f a x = Real.log (-1/a) - 1)) :=
by sorry

end f_properties_l555_55587


namespace cube_root_equation_solution_l555_55595

theorem cube_root_equation_solution : 
  {x : ℝ | ∃ y : ℝ, y^3 = 4*x - 1 ∧ ∃ z : ℝ, z^3 = 4*x + 1 ∧ ∃ w : ℝ, w^3 = 8*x ∧ y + z = w} = 
  {0, 1/4, -1/4} := by
  sorry

end cube_root_equation_solution_l555_55595


namespace simplify_and_evaluate_l555_55537

theorem simplify_and_evaluate : 
  let x : ℚ := 3/2
  (3 + x)^2 - (x + 5) * (x - 1) = 17 := by
sorry

end simplify_and_evaluate_l555_55537


namespace simplify_and_evaluate_expression_l555_55502

theorem simplify_and_evaluate_expression :
  let x : ℝ := Real.sqrt 5 - 2
  (2 / (x^2 - 4)) / (1 - x / (x - 2)) = -Real.sqrt 5 / 5 := by
  sorry

end simplify_and_evaluate_expression_l555_55502


namespace rectangle_13_squares_ratio_l555_55543

/-- A rectangle that can be divided into 13 equal squares -/
structure Rectangle13Squares where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_divisible : ∃ s : ℝ, 0 < s ∧ (a = 13 * s ∧ b = s) ∨ (a = s ∧ b = 13 * s)

/-- The ratio of the longer side to the shorter side is 13:1 -/
theorem rectangle_13_squares_ratio (rect : Rectangle13Squares) :
  (max rect.a rect.b) / (min rect.a rect.b) = 13 := by
  sorry

end rectangle_13_squares_ratio_l555_55543


namespace f_properties_l555_55539

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval [-1, 3]
def interval : Set ℝ := Set.Icc (-1) 3

-- Theorem for monotonicity and extreme values
theorem f_properties :
  (∀ x y, x < y ∧ x < -1 → f x < f y) ∧  -- Increasing on (-∞, -1)
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y) ∧  -- Decreasing on (-1, 1)
  (∀ x y, 1 < x ∧ x < y → f x < f y) ∧  -- Increasing on (1, +∞)
  (∀ x ∈ interval, f x ≤ 18) ∧  -- Maximum value
  (∀ x ∈ interval, f x ≥ -2) ∧  -- Minimum value
  (∃ x ∈ interval, f x = 18) ∧  -- Maximum is attained
  (∃ x ∈ interval, f x = -2) :=  -- Minimum is attained
by sorry

end f_properties_l555_55539


namespace hat_scarf_game_theorem_l555_55569

/-- Represents the maximum guaranteed points in the hat-scarf game -/
def max_guaranteed_points (n k : ℕ) : ℕ :=
  n / k

theorem hat_scarf_game_theorem :
  (∀ n k : ℕ, max_guaranteed_points n k = n / k) ∧
  (max_guaranteed_points 2 2 = 1) := by
  sorry

#check hat_scarf_game_theorem

end hat_scarf_game_theorem_l555_55569


namespace sqrt_x_plus_inverse_sqrt_x_l555_55515

theorem sqrt_x_plus_inverse_sqrt_x (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) : 
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end sqrt_x_plus_inverse_sqrt_x_l555_55515


namespace f_3_equals_neg_26_l555_55540

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_3_equals_neg_26 (a b : ℝ) :
  f a b (-3) = 10 → f a b 3 = -26 := by
  sorry

end f_3_equals_neg_26_l555_55540


namespace count_scalene_triangles_l555_55548

def is_valid_scalene_triangle (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + b + c < 13 ∧ a + b > c ∧ a + c > b ∧ b + c > a

theorem count_scalene_triangles :
  ∃! (S : Finset (ℕ × ℕ × ℕ)), 
    (∀ (t : ℕ × ℕ × ℕ), t ∈ S ↔ is_valid_scalene_triangle t.1 t.2.1 t.2.2) ∧
    S.card = 3 :=
sorry

end count_scalene_triangles_l555_55548


namespace classroom_composition_l555_55570

/-- In a class, each boy is friends with exactly two girls, and each girl is friends with exactly three boys. -/
structure Classroom where
  boys : ℕ
  girls : ℕ
  total_students : boys + girls = 31
  boy_girl_connections : 2 * boys = 3 * girls

/-- The number of boys and girls in the classroom satisfies the given conditions. -/
theorem classroom_composition : ∃ (c : Classroom), c.boys = 19 ∧ c.girls = 12 := by
  sorry

end classroom_composition_l555_55570


namespace annas_number_l555_55577

theorem annas_number : ∃ x : ℚ, 5 * ((3 * x + 20) - 5) = 200 ∧ x = 25 / 3 := by sorry

end annas_number_l555_55577


namespace arnold_protein_consumption_l555_55520

-- Define the protein content of each food item
def collagen_protein_per_2_scoops : ℕ := 18
def protein_powder_per_scoop : ℕ := 21
def steak_protein : ℕ := 56

-- Define Arnold's consumption
def collagen_scoops : ℕ := 1
def protein_powder_scoops : ℕ := 1
def steak_portions : ℕ := 1

-- Theorem to prove
theorem arnold_protein_consumption :
  (collagen_scoops * collagen_protein_per_2_scoops / 2) +
  (protein_powder_scoops * protein_powder_per_scoop) +
  (steak_portions * steak_protein) = 86 := by
  sorry

end arnold_protein_consumption_l555_55520


namespace fixed_point_of_exponential_translation_l555_55509

/-- The function f(x) = ax - 3 + 3 always passes through the point (3, 4) for any real number a. -/
theorem fixed_point_of_exponential_translation (a : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x - 3 + 3
  f 3 = 4 := by
  sorry

end fixed_point_of_exponential_translation_l555_55509


namespace power_problem_l555_55501

theorem power_problem (a m n : ℕ) (h1 : a ^ m = 3) (h2 : a ^ n = 2) :
  a ^ (3 * m + 2 * n) = 108 := by
  sorry

end power_problem_l555_55501


namespace sufficient_condition_for_perpendicular_l555_55531

-- Define the types for planes and lines
def Plane : Type := Unit
def Line : Type := Unit

-- Define the perpendicular relation between a line and a plane
def perp_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the perpendicular relation between two planes
def perp_plane_plane (p1 : Plane) (p2 : Plane) : Prop := sorry

-- Theorem statement
theorem sufficient_condition_for_perpendicular 
  (α β : Plane) (m n : Line) :
  perp_line_plane n α → 
  perp_line_plane n β → 
  perp_line_plane m α → 
  perp_line_plane m β :=
sorry

end sufficient_condition_for_perpendicular_l555_55531


namespace angle_bisector_d_value_l555_55574

-- Define the triangle ABC
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (-4, -2)
def C : ℝ × ℝ := (7, -1)

-- Define the angle bisector equation
def angleBisectorEq (x y d : ℝ) : Prop := x - 3*y + d = 0

-- Theorem statement
theorem angle_bisector_d_value :
  ∃ d : ℝ, (∀ x y : ℝ, angleBisectorEq x y d ↔ 
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
      x = B.1 + t * (C.1 - B.1) ∧
      y = B.2 + t * (C.2 - B.2))) ∧
    angleBisectorEq B.1 B.2 d :=
by sorry

end angle_bisector_d_value_l555_55574


namespace tan_sum_specific_angles_l555_55514

theorem tan_sum_specific_angles (α β : Real) 
  (h1 : Real.tan α = 2) 
  (h2 : Real.tan β = 3) : 
  Real.tan (α + β) = -1 := by
  sorry

end tan_sum_specific_angles_l555_55514


namespace value_of_a_l555_55545

theorem value_of_a (a b c : ℚ) 
  (eq1 : a + b = c)
  (eq2 : b + c + 2 * b = 11)
  (eq3 : c = 7) :
  a = 17 / 3 := by
sorry

end value_of_a_l555_55545


namespace inequality_system_solution_l555_55525

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (x > -a ∧ x > -b) ↔ x > -b) → a ≥ b := by
  sorry

end inequality_system_solution_l555_55525


namespace scientific_notation_218_million_l555_55585

theorem scientific_notation_218_million :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧ 
    218000000 = a * (10 : ℝ) ^ n ∧
    a = 2.18 ∧ n = 8 := by
  sorry

end scientific_notation_218_million_l555_55585


namespace quadratic_factorization_l555_55591

theorem quadratic_factorization (x : ℝ) : 12 * x^2 + 8 * x - 4 = 4 * (3 * x - 1) * (x + 1) := by
  sorry

end quadratic_factorization_l555_55591


namespace ratio_sum_problem_l555_55578

theorem ratio_sum_problem (a b : ℝ) : 
  a / b = 3 / 8 → b - a = 20 → a + b = 44 := by sorry

end ratio_sum_problem_l555_55578


namespace range_of_a_l555_55536

-- Define the propositions p and q
def p (x : ℝ) : Prop := abs (4 * x - 3) ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the condition that ¬p is a necessary but not sufficient condition for ¬q
def condition (a : ℝ) : Prop :=
  (∀ x : ℝ, q x a → p x) ∧ (∃ x : ℝ, p x ∧ ¬(q x a))

-- State the theorem
theorem range_of_a (a : ℝ) :
  condition a ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end range_of_a_l555_55536


namespace cubic_polynomial_property_l555_55592

theorem cubic_polynomial_property (n : ℕ+) : 
  ∃ k : ℤ, (n^3 : ℚ) + (3/2) * n^2 + (1/2) * n - 1 = k ∧ k % 3 = 2 := by
  sorry

end cubic_polynomial_property_l555_55592


namespace truck_travel_distance_l555_55508

/-- Represents the distance a truck can travel -/
def distance_traveled (diesel_amount : ℚ) : ℚ :=
  150 * diesel_amount / 5

/-- The theorem states that the truck travels 210 miles on 7 gallons of diesel -/
theorem truck_travel_distance : distance_traveled 7 = 210 := by
  sorry

end truck_travel_distance_l555_55508


namespace housing_boom_result_l555_55510

/-- Represents the housing development in Lawrence County -/
structure LawrenceCountyHousing where
  initial_houses : ℕ
  developer_a_rate : ℕ
  developer_a_months : ℕ
  developer_b_rate : ℕ
  developer_b_months : ℕ
  developer_c_rate : ℕ
  developer_c_months : ℕ
  final_houses : ℕ

/-- Calculates the total number of houses built by developers -/
def total_houses_built (h : LawrenceCountyHousing) : ℕ :=
  h.developer_a_rate * h.developer_a_months +
  h.developer_b_rate * h.developer_b_months +
  h.developer_c_rate * h.developer_c_months

/-- Theorem stating that the total houses built by developers is 405 -/
theorem housing_boom_result (h : LawrenceCountyHousing)
  (h_initial : h.initial_houses = 1426)
  (h_dev_a : h.developer_a_rate = 25 ∧ h.developer_a_months = 6)
  (h_dev_b : h.developer_b_rate = 15 ∧ h.developer_b_months = 9)
  (h_dev_c : h.developer_c_rate = 30 ∧ h.developer_c_months = 4)
  (h_final : h.final_houses = 2000) :
  total_houses_built h = 405 := by
  sorry


end housing_boom_result_l555_55510


namespace range_of_f_l555_55541

def f (x : ℝ) := -x^2 + 2*x + 3

theorem range_of_f : 
  ∀ y ∈ Set.Icc (-5 : ℝ) 4, ∃ x ∈ Set.Icc (-2 : ℝ) 3, f x = y ∧
  ∀ x ∈ Set.Icc (-2 : ℝ) 3, f x ∈ Set.Icc (-5 : ℝ) 4 :=
by
  sorry

end range_of_f_l555_55541


namespace waiter_customers_l555_55558

/-- Calculates the total number of customers for a waiter given the number of tables and customers per table. -/
def total_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) : ℕ :=
  num_tables * (women_per_table + men_per_table)

/-- Proves that a waiter with 9 tables, each having 7 women and 3 men, has 90 customers in total. -/
theorem waiter_customers : total_customers 9 7 3 = 90 := by
  sorry

end waiter_customers_l555_55558


namespace function_identically_zero_l555_55560

/-- A function satisfying f(a · b) = a f(b) + b f(a) and |f(x)| ≤ 1 is identically zero -/
theorem function_identically_zero (f : ℝ → ℝ) 
  (h1 : ∀ a b : ℝ, f (a * b) = a * f b + b * f a) 
  (h2 : ∀ x : ℝ, |f x| ≤ 1) : 
  ∀ x : ℝ, f x = 0 := by
  sorry

end function_identically_zero_l555_55560


namespace simultaneous_equations_solution_l555_55524

theorem simultaneous_equations_solution (m : ℝ) : 
  (m ≠ 1) ↔ (∃ x y : ℝ, y = m * x + 3 ∧ y = (2 * m - 1) * x + 4) :=
by sorry

end simultaneous_equations_solution_l555_55524


namespace sum_of_reciprocals_of_roots_l555_55566

theorem sum_of_reciprocals_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 14*x₁ + 8 = 0 → 
  x₂^2 - 14*x₂ + 8 = 0 → 
  x₁ ≠ x₂ →
  1/x₁ + 1/x₂ = 7/4 := by
sorry

end sum_of_reciprocals_of_roots_l555_55566


namespace exists_unrepresentable_group_l555_55593

/-- Represents a person in the group -/
structure Person :=
  (id : ℕ)

/-- Represents the acquaintance relationship between two people -/
def Acquainted (p1 p2 : Person) : Prop := sorry

/-- Represents a chord in a circle -/
structure Chord :=
  (person : Person)

/-- Represents the intersection of two chords -/
def Intersects (c1 c2 : Chord) : Prop := sorry

/-- The main theorem stating that there exists a group of people whose acquaintance relationships
    cannot be represented by intersecting chords in a circle -/
theorem exists_unrepresentable_group :
  ∃ (group : Set Person) (acquaintance : Person → Person → Prop),
    ¬∃ (chord_assignment : Person → Chord),
      ∀ (p1 p2 : Person),
        p1 ∈ group → p2 ∈ group → p1 ≠ p2 →
          (acquaintance p1 p2 ↔ Intersects (chord_assignment p1) (chord_assignment p2)) :=
sorry

end exists_unrepresentable_group_l555_55593


namespace louisa_travel_time_l555_55528

theorem louisa_travel_time 
  (distance_day1 : ℝ) 
  (distance_day2 : ℝ) 
  (time_difference : ℝ) 
  (h1 : distance_day1 = 200)
  (h2 : distance_day2 = 360)
  (h3 : time_difference = 4)
  (h4 : ∃ v : ℝ, v > 0 ∧ distance_day1 / v + time_difference = distance_day2 / v) :
  ∃ total_time : ℝ, total_time = 14 ∧ total_time = distance_day1 / (distance_day2 - distance_day1) * time_difference + distance_day2 / (distance_day2 - distance_day1) * time_difference :=
by sorry

end louisa_travel_time_l555_55528


namespace min_value_of_f_l555_55584

noncomputable def f (x : ℝ) := 12 * x - x^3

theorem min_value_of_f :
  ∃ (m : ℝ), m = -16 ∧ ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≥ m :=
by sorry

end min_value_of_f_l555_55584


namespace investment_problem_l555_55553

/-- Proves that given the conditions of the investment problem, the initial sum invested was $900 -/
theorem investment_problem (P : ℝ) : 
  P > 0 → 
  (P * (4.5 / 100) * 7) - (P * (4 / 100) * 7) = 31.5 → 
  P = 900 := by
sorry

end investment_problem_l555_55553


namespace problem_statement_l555_55518

-- Definition of an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Definition of an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

-- Definition of a periodic function
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem problem_statement :
  (∀ (f : ℝ → ℝ), IsEven (fun x ↦ f x + f (-x))) ∧
  (∀ (f : ℝ → ℝ), IsOdd f → IsOdd (fun x ↦ f (x + 2)) → IsPeriodic f 4) := by
  sorry

end problem_statement_l555_55518


namespace six_digit_divisibility_l555_55561

theorem six_digit_divisibility (a b c d e f : ℕ) 
  (h_six_digit : 100000 ≤ a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f ∧ 
                 a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f < 1000000)
  (h_sum_equal : a + d = b + e ∧ b + e = c + f) : 
  ∃ k : ℕ, a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f = 37 * k :=
by sorry

end six_digit_divisibility_l555_55561


namespace f_vertex_f_at_zero_f_expression_f_monotonic_interval_l555_55586

/-- A quadratic function with vertex at (1, 1) and f(0) = 3 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

/-- The vertex of f is at (1, 1) -/
theorem f_vertex : ∀ x : ℝ, f x ≥ f 1 := sorry

/-- f(0) = 3 -/
theorem f_at_zero : f 0 = 3 := sorry

/-- f(x) = 2x^2 - 4x + 3 -/
theorem f_expression : ∀ x : ℝ, f x = 2 * x^2 - 4 * x + 3 := sorry

/-- f(x) is monotonic in [a, a+1] iff a ≤ 0 or a ≥ 1 -/
theorem f_monotonic_interval (a : ℝ) :
  (∀ x y : ℝ, a ≤ x ∧ x ≤ y ∧ y ≤ a + 1 → f x ≤ f y) ↔ (a ≤ 0 ∨ a ≥ 1) := sorry

end f_vertex_f_at_zero_f_expression_f_monotonic_interval_l555_55586


namespace fred_onions_l555_55589

/-- Proves that Fred grew 9 onions given the conditions of the problem -/
theorem fred_onions (sara : ℕ) (sally : ℕ) (fred : ℕ) (total : ℕ)
  (h1 : sara = 4)
  (h2 : sally = 5)
  (h3 : total = 18)
  (h4 : sara + sally + fred = total) :
  fred = 9 := by
  sorry

end fred_onions_l555_55589


namespace half_quarter_difference_l555_55562

theorem half_quarter_difference (n : ℝ) (h : n = 8) : 0.5 * n - 0.25 * n = 2 := by
  sorry

end half_quarter_difference_l555_55562


namespace three_digit_powers_of_three_l555_55594

theorem three_digit_powers_of_three :
  (∃! (s : Finset ℕ), s = {n : ℕ | 100 ≤ 3^n ∧ 3^n ≤ 999} ∧ Finset.card s = 2) :=
by sorry

end three_digit_powers_of_three_l555_55594


namespace hours_minutes_conversion_tons_kilograms_conversion_seconds_conversion_square_meters_conversion_l555_55544

-- Define conversion rates
def minutes_per_hour : ℕ := 60
def kilograms_per_ton : ℕ := 1000
def seconds_per_minute : ℕ := 60
def square_meters_per_hectare : ℕ := 10000

-- Define the conversion functions
def hours_minutes_to_minutes (hours minutes : ℕ) : ℕ :=
  hours * minutes_per_hour + minutes

def tons_kilograms_to_kilograms (tons kilograms : ℕ) : ℕ :=
  tons * kilograms_per_ton + kilograms

def seconds_to_minutes_seconds (total_seconds : ℕ) : ℕ × ℕ :=
  (total_seconds / seconds_per_minute, total_seconds % seconds_per_minute)

def square_meters_to_hectares (square_meters : ℕ) : ℕ :=
  square_meters / square_meters_per_hectare

-- State the theorems
theorem hours_minutes_conversion :
  hours_minutes_to_minutes 4 35 = 275 := by sorry

theorem tons_kilograms_conversion :
  tons_kilograms_to_kilograms 4 35 = 4035 := by sorry

theorem seconds_conversion :
  seconds_to_minutes_seconds 678 = (11, 18) := by sorry

theorem square_meters_conversion :
  square_meters_to_hectares 120000 = 12 := by sorry

end hours_minutes_conversion_tons_kilograms_conversion_seconds_conversion_square_meters_conversion_l555_55544


namespace optimal_path_to_island_l555_55551

/-- Represents the optimal path problem for Hagrid to reach Harry Potter --/
theorem optimal_path_to_island (island_distance : ℝ) (shore_distance : ℝ) 
  (shore_speed : ℝ) (sea_speed : ℝ) :
  island_distance = 9 →
  shore_distance = 15 →
  shore_speed = 50 →
  sea_speed = 40 →
  ∃ (x : ℝ), x = 3 ∧ 
    ∀ (y : ℝ), y ≥ 0 → 
      (x / shore_speed + (Real.sqrt ((island_distance^2) + (shore_distance - x)^2)) / sea_speed) ≤
      (y / shore_speed + (Real.sqrt ((island_distance^2) + (shore_distance - y)^2)) / sea_speed) :=
by sorry


end optimal_path_to_island_l555_55551


namespace sum_of_20th_and_30th_triangular_l555_55563

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the 20th and 30th triangular numbers is 675 -/
theorem sum_of_20th_and_30th_triangular : triangular_number 20 + triangular_number 30 = 675 := by
  sorry

end sum_of_20th_and_30th_triangular_l555_55563


namespace parabola_translation_l555_55554

/-- Represents a parabola in the form y = -(x - h)² + k -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { h := p.h + dx, k := p.k + dy }

theorem parabola_translation :
  let original := Parabola.mk 1 0
  let translated := translate original 1 2
  translated = Parabola.mk 2 2 := by sorry

end parabola_translation_l555_55554


namespace circle_and_max_z_l555_55598

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 4}

-- Define the line on which the center lies
def center_line (x y : ℝ) : Prop :=
  x + y - 3 = 0

-- Define the function z
def z (p : ℝ × ℝ) : ℝ :=
  p.1 + p.2

-- Theorem statement
theorem circle_and_max_z :
  (∀ p : ℝ × ℝ, p ∈ circle_C → (p.1 = 1 ∧ p.2 = 4) ∨ (p.1 = 3 ∧ p.2 = 2)) ∧
  (∃ c : ℝ × ℝ, c ∈ circle_C ∧ center_line c.1 c.2) →
  (∀ p : ℝ × ℝ, p ∈ circle_C → (p.1 - 1)^2 + (p.2 - 2)^2 = 4) ∧
  (∀ p : ℝ × ℝ, p ∈ circle_C → z p ≤ 3 + 2 * Real.sqrt 2) ∧
  (∃ p : ℝ × ℝ, p ∈ circle_C ∧ z p = 3 + 2 * Real.sqrt 2) :=
by sorry


end circle_and_max_z_l555_55598


namespace inequality_equivalence_l555_55522

theorem inequality_equivalence (a : ℝ) : (a + 1 < 0) ↔ (a < -1) := by
  sorry

end inequality_equivalence_l555_55522


namespace cubes_fill_box_completely_l555_55550

def box_length : ℕ := 12
def box_width : ℕ := 6
def box_height : ℕ := 9
def cube_side : ℕ := 3

def cubes_per_length : ℕ := box_length / cube_side
def cubes_per_width : ℕ := box_width / cube_side
def cubes_per_height : ℕ := box_height / cube_side

def total_cubes : ℕ := cubes_per_length * cubes_per_width * cubes_per_height

def box_volume : ℕ := box_length * box_width * box_height
def cube_volume : ℕ := cube_side ^ 3
def total_cube_volume : ℕ := total_cubes * cube_volume

theorem cubes_fill_box_completely :
  total_cube_volume = box_volume := by sorry

end cubes_fill_box_completely_l555_55550


namespace largest_angle_convex_pentagon_l555_55529

theorem largest_angle_convex_pentagon (x : ℝ) : 
  (x + 2) + (2 * x + 3) + (3 * x + 4) + (4 * x + 5) + (5 * x + 6) = 540 →
  max (x + 2) (max (2 * x + 3) (max (3 * x + 4) (max (4 * x + 5) (5 * x + 6)))) = 538 / 3 :=
by sorry

end largest_angle_convex_pentagon_l555_55529


namespace turnip_bag_weights_l555_55526

def bag_weights : List Nat := [13, 15, 16, 17, 21, 24]

def is_valid_turnip_weight (t : Nat) : Prop :=
  t ∈ bag_weights ∧
  ∃ (o c : Nat),
    o + c = (bag_weights.sum - t) ∧
    c = 2 * o

theorem turnip_bag_weights :
  ∀ t, is_valid_turnip_weight t ↔ t = 13 ∨ t = 16 :=
by sorry

end turnip_bag_weights_l555_55526


namespace unique_six_digit_number_divisibility_l555_55549

def is_valid_digit (d : Nat) : Prop := d ≥ 1 ∧ d ≤ 6

def all_digits_unique (p q r s t u : Nat) : Prop :=
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧
  r ≠ s ∧ r ≠ t ∧ r ≠ u ∧
  s ≠ t ∧ s ≠ u ∧
  t ≠ u

def three_digit_number (a b c : Nat) : Nat :=
  100 * a + 10 * b + c

theorem unique_six_digit_number_divisibility (p q r s t u : Nat) :
  is_valid_digit p ∧ is_valid_digit q ∧ is_valid_digit r ∧
  is_valid_digit s ∧ is_valid_digit t ∧ is_valid_digit u ∧
  all_digits_unique p q r s t u ∧
  (three_digit_number p q r) % 4 = 0 ∧
  (three_digit_number q r s) % 6 = 0 ∧
  (three_digit_number r s t) % 3 = 0 →
  p = 5 := by
  sorry

end unique_six_digit_number_divisibility_l555_55549


namespace divisibility_condition_l555_55547

theorem divisibility_condition (m n : ℤ) : 
  m > 1 → n > 1 → (m * n - 1 ∣ n^3 - 1) ↔ (m = n^2 ∨ n = m^2) :=
by sorry

end divisibility_condition_l555_55547


namespace total_cost_is_thirteen_l555_55521

/-- The cost of a pencil in dollars -/
def pencil_cost : ℝ := 2

/-- The additional cost of a pen compared to a pencil in dollars -/
def pen_additional_cost : ℝ := 9

/-- The cost of a pen in dollars -/
def pen_cost : ℝ := pencil_cost + pen_additional_cost

/-- The total cost of both items in dollars -/
def total_cost : ℝ := pen_cost + pencil_cost

theorem total_cost_is_thirteen : total_cost = 13 := by
  sorry

end total_cost_is_thirteen_l555_55521


namespace smallest_integer_l555_55556

theorem smallest_integer (a b : ℕ) (ha : a = 60) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 44) :
  b ≥ 165 ∧ ∃ (b' : ℕ), b' = 165 ∧ Nat.lcm a b' / Nat.gcd a b' = 44 := by
  sorry

end smallest_integer_l555_55556


namespace visitors_yesterday_l555_55527

def total_visitors : ℕ := 829
def visitors_today : ℕ := 784

theorem visitors_yesterday (total : ℕ) (today : ℕ) (h1 : total = total_visitors) (h2 : today = visitors_today) :
  total - today = 45 := by
  sorry

end visitors_yesterday_l555_55527


namespace two_distinct_integer_roots_l555_55571

theorem two_distinct_integer_roots (r : ℝ) : 
  (∃ x y : ℤ, x ≠ y ∧ r^2 * x^2 + 2*r*x + 4 = 28*r^2 ∧ r^2 * y^2 + 2*r*y + 4 = 28*r^2) ↔ 
  (r = 1 ∨ r = -1 ∨ r = 1/2 ∨ r = -1/2 ∨ r = 1/3 ∨ r = -1/3) :=
sorry

end two_distinct_integer_roots_l555_55571


namespace chalkboard_width_l555_55567

theorem chalkboard_width (width : ℝ) (length : ℝ) (area : ℝ) : 
  length = 2 * width →
  area = width * length →
  area = 18 →
  width = 3 :=
by sorry

end chalkboard_width_l555_55567


namespace exists_same_color_rectangle_l555_55504

/-- A color representation --/
inductive Color
| Black
| White

/-- A grid of colors --/
def Grid := Fin 3 → Fin 7 → Color

/-- A rectangle in the grid --/
structure Rectangle where
  x1 : Fin 7
  x2 : Fin 7
  y1 : Fin 3
  y2 : Fin 3
  h_distinct : x1 ≠ x2 ∧ y1 ≠ y2

/-- Check if a rectangle has vertices of the same color --/
def sameColorVertices (g : Grid) (r : Rectangle) : Prop :=
  g r.y1 r.x1 = g r.y1 r.x2 ∧
  g r.y1 r.x1 = g r.y2 r.x1 ∧
  g r.y1 r.x1 = g r.y2 r.x2

/-- Theorem: In any 3x7 grid coloring, there exists a rectangle with vertices of the same color --/
theorem exists_same_color_rectangle (g : Grid) : ∃ r : Rectangle, sameColorVertices g r := by
  sorry

end exists_same_color_rectangle_l555_55504


namespace power_mod_eleven_l555_55535

theorem power_mod_eleven : 3^251 % 11 = 3 := by
  sorry

end power_mod_eleven_l555_55535


namespace parking_lot_problem_l555_55568

theorem parking_lot_problem (total_vehicles : ℕ) (total_wheels : ℕ) 
  (car_wheels : ℕ) (motorcycle_wheels : ℕ) :
  total_vehicles = 30 →
  total_wheels = 84 →
  car_wheels = 4 →
  motorcycle_wheels = 2 →
  ∃ (cars : ℕ) (motorcycles : ℕ),
    cars + motorcycles = total_vehicles ∧
    car_wheels * cars + motorcycle_wheels * motorcycles = total_wheels ∧
    motorcycles = 18 := by
  sorry

end parking_lot_problem_l555_55568


namespace mechanic_job_duration_l555_55507

/-- Proves that given a mechanic's hourly rate, parts cost, and total bill, the job duration can be calculated. -/
theorem mechanic_job_duration 
  (hourly_rate : ℝ) 
  (parts_cost : ℝ) 
  (total_bill : ℝ) 
  (h : hourly_rate = 45) 
  (p : parts_cost = 225) 
  (t : total_bill = 450) : 
  (total_bill - parts_cost) / hourly_rate = 5 := by
  sorry

end mechanic_job_duration_l555_55507


namespace gcd_21_and_number_between_50_60_l555_55581

theorem gcd_21_and_number_between_50_60 :
  ∃! n : ℕ, 50 ≤ n ∧ n ≤ 60 ∧ Nat.gcd 21 n = 7 :=
by
  -- The proof goes here
  sorry

end gcd_21_and_number_between_50_60_l555_55581


namespace softball_team_savings_l555_55513

/-- Calculates the savings for a softball team buying uniforms with a group discount. -/
theorem softball_team_savings 
  (team_size : ℕ) 
  (regular_shirt_price regular_pants_price regular_socks_price : ℚ)
  (discount_shirt_price discount_pants_price discount_socks_price : ℚ)
  (h1 : team_size = 12)
  (h2 : regular_shirt_price = 7.5)
  (h3 : regular_pants_price = 15)
  (h4 : regular_socks_price = 4.5)
  (h5 : discount_shirt_price = 6.75)
  (h6 : discount_pants_price = 13.5)
  (h7 : discount_socks_price = 3.75) :
  (team_size : ℚ) * ((regular_shirt_price + regular_pants_price + regular_socks_price) - 
  (discount_shirt_price + discount_pants_price + discount_socks_price)) = 36 :=
by sorry

end softball_team_savings_l555_55513


namespace limit_of_sequence_l555_55555

def a (n : ℕ) : ℚ := (3 * n - 2) / (2 * n - 1)

theorem limit_of_sequence : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 3/2| < ε :=
by sorry

end limit_of_sequence_l555_55555


namespace simplify_expression_1_simplify_and_evaluate_expression_2_l555_55583

/-- Proof of the first simplification -/
theorem simplify_expression_1 (a b : ℝ) : 2 * a^2 + 9 * b - 5 * a^2 - 4 * b = -3 * a^2 + 5 * b := by
  sorry

/-- Proof of the second simplification and evaluation -/
theorem simplify_and_evaluate_expression_2 : 3 * 1 * (-2)^2 + 1^2 * (-2) - 2 * (2 * 1 * (-2)^2 - 1^2 * (-2)) = -10 := by
  sorry

end simplify_expression_1_simplify_and_evaluate_expression_2_l555_55583


namespace caroline_lassis_l555_55511

/-- Given that Caroline can make 15 lassis from 3 mangoes, prove that she can make 90 lassis from 18 mangoes. -/
theorem caroline_lassis (mangoes_small : ℕ) (lassis_small : ℕ) (mangoes_large : ℕ) :
  mangoes_small = 3 →
  lassis_small = 15 →
  mangoes_large = 18 →
  (lassis_small * mangoes_large) / mangoes_small = 90 :=
by sorry

end caroline_lassis_l555_55511


namespace remaining_problems_to_grade_l555_55534

theorem remaining_problems_to_grade 
  (total_worksheets : ℕ) 
  (graded_worksheets : ℕ) 
  (problems_per_worksheet : ℕ) 
  (h1 : total_worksheets = 25) 
  (h2 : graded_worksheets = 12) 
  (h3 : problems_per_worksheet = 15) : 
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 195 := by
sorry

end remaining_problems_to_grade_l555_55534


namespace rival_to_jessie_award_ratio_l555_55500

/-- Given that Scott won 4 awards, Jessie won 3 times as many awards as Scott,
    and the rival won 24 awards, prove that the ratio of awards won by the rival
    to Jessie is 2:1. -/
theorem rival_to_jessie_award_ratio :
  let scott_awards : ℕ := 4
  let jessie_awards : ℕ := 3 * scott_awards
  let rival_awards : ℕ := 24
  (rival_awards : ℚ) / jessie_awards = 2 := by sorry

end rival_to_jessie_award_ratio_l555_55500


namespace soccer_team_win_percentage_l555_55582

/-- Given a soccer team that played 140 games and won 70 games, 
    prove that the percentage of games won is 50%. -/
theorem soccer_team_win_percentage 
  (total_games : ℕ) 
  (games_won : ℕ) 
  (h1 : total_games = 140) 
  (h2 : games_won = 70) : 
  (games_won : ℚ) / total_games * 100 = 50 := by
  sorry

#check soccer_team_win_percentage

end soccer_team_win_percentage_l555_55582


namespace parabola_line_intersection_l555_55503

/-- Parabola intersecting with a line -/
theorem parabola_line_intersection 
  (a : ℝ) 
  (h_a : a ≠ 0) 
  (b : ℝ) 
  (h_intersection : b = 2 * 1 - 3 ∧ b = a * 1^2) :
  (a = -1 ∧ b = -1) ∧ 
  ∃ x y : ℝ, x = -3 ∧ y = -9 ∧ y = a * x^2 ∧ y = 2 * x - 3 :=
sorry

end parabola_line_intersection_l555_55503


namespace perfect_squares_digit_parity_l555_55538

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def units_digit (n : ℕ) : ℕ := n % 10

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem perfect_squares_digit_parity (a b x y : ℕ) :
  is_perfect_square a →
  is_perfect_square b →
  units_digit a = 1 →
  tens_digit a = x →
  units_digit b = 6 →
  tens_digit b = y →
  Even x ∧ Odd y :=
sorry

end perfect_squares_digit_parity_l555_55538
