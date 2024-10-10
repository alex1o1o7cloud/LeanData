import Mathlib

namespace fountain_area_l1000_100025

theorem fountain_area (diameter : Real) (radius : Real) :
  diameter = 20 →
  radius * 2 = diameter →
  radius ^ 2 = 244 →
  π * radius ^ 2 = 244 * π :=
by sorry

end fountain_area_l1000_100025


namespace problem_1_problem_2_problem_3_l1000_100091

-- Problem 1
theorem problem_1 : 0.25 * 1.25 * 32 = 10 := by sorry

-- Problem 2
theorem problem_2 : 4/5 * 5/11 + 5/11 / 5 = 5/11 := by sorry

-- Problem 3
theorem problem_3 : 7 - 4/9 - 5/9 = 6 := by sorry

end problem_1_problem_2_problem_3_l1000_100091


namespace max_lateral_area_inscribed_prism_l1000_100048

noncomputable section

-- Define the sphere's surface area
def sphere_surface_area : ℝ := 8 * Real.pi

-- Define the sphere's radius
def sphere_radius : ℝ := Real.sqrt (sphere_surface_area / (4 * Real.pi))

-- Define the base edge length of the prism
def base_edge_length : ℝ := Real.sqrt 3

-- Define the lateral area of the prism as a function of the base edge length
def lateral_area (x : ℝ) : ℝ := 
  6 * Real.sqrt (-(1/3) * (x^2 - 3)^2 + 3)

-- Theorem statement
theorem max_lateral_area_inscribed_prism :
  (lateral_area base_edge_length = 6 * Real.sqrt 3) ∧
  (∀ x : ℝ, 0 < x → x < Real.sqrt 6 → lateral_area x ≤ lateral_area base_edge_length) := by
  sorry

end

end max_lateral_area_inscribed_prism_l1000_100048


namespace modular_inverse_13_mod_101_l1000_100072

theorem modular_inverse_13_mod_101 : ∃ x : ℤ, (13 * x) % 101 = 1 ∧ 0 ≤ x ∧ x < 101 :=
by
  use 70
  sorry

end modular_inverse_13_mod_101_l1000_100072


namespace jack_email_difference_l1000_100081

theorem jack_email_difference : 
  ∀ (morning_emails afternoon_emails morning_letters afternoon_letters : ℕ),
  morning_emails = 10 →
  afternoon_emails = 3 →
  morning_letters = 12 →
  afternoon_letters = 44 →
  morning_emails - afternoon_emails = 7 :=
by sorry

end jack_email_difference_l1000_100081


namespace geometric_sequence_ratio_l1000_100078

/-- Given a geometric sequence with positive terms where a₁, ½a₃, 2a₂ form an arithmetic sequence,
    the ratio (a₁₃ + a₁₄) / (a₁₄ + a₁₅) equals √2 - 1. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
    (h_geom : ∃ q : ℝ, ∀ n, a (n + 1) = q * a n) 
    (h_arith : ∃ d : ℝ, a 1 + d = (1/2) * a 3 ∧ (1/2) * a 3 + d = 2 * a 2) :
  (a 13 + a 14) / (a 14 + a 15) = Real.sqrt 2 - 1 := by
sorry

end geometric_sequence_ratio_l1000_100078


namespace car_max_acceleration_l1000_100043

theorem car_max_acceleration
  (g : ℝ) -- acceleration due to gravity
  (θ : ℝ) -- angle of the hill
  (μ : ℝ) -- coefficient of static friction
  (h1 : 0 < g)
  (h2 : 0 ≤ θ)
  (h3 : θ < π / 2)
  (h4 : μ > Real.tan θ) :
  ∃ a : ℝ,
    a = g * (μ * Real.cos θ - Real.sin θ) ∧
    ∀ a' : ℝ,
      (∃ m : ℝ, 0 < m ∧
        m * a' ≤ μ * (m * g * Real.cos θ) - m * g * Real.sin θ) →
      a' ≤ a :=
by sorry

end car_max_acceleration_l1000_100043


namespace autograph_value_change_l1000_100007

theorem autograph_value_change (initial_value : ℝ) : 
  initial_value = 100 → 
  (initial_value * (1 - 0.3) * (1 + 0.4)) = 98 := by
  sorry

end autograph_value_change_l1000_100007


namespace partial_fraction_decomposition_l1000_100097

theorem partial_fraction_decomposition :
  ∀ (x : ℝ), x ≠ 10 → x ≠ -2 →
  (6 * x - 4) / (x^2 - 8 * x - 20) = 
  (14 / 3) / (x - 10) + (4 / 3) / (x + 2) :=
by sorry

end partial_fraction_decomposition_l1000_100097


namespace burger_share_inches_l1000_100054

-- Define the length of the burger in feet
def burger_length_feet : ℝ := 1

-- Define the number of people sharing the burger
def num_people : ℕ := 2

-- Define the conversion factor from feet to inches
def feet_to_inches : ℝ := 12

-- Theorem to prove
theorem burger_share_inches : 
  (burger_length_feet * feet_to_inches) / num_people = 6 := by
  sorry

end burger_share_inches_l1000_100054


namespace sqrt_diff_inequality_l1000_100039

theorem sqrt_diff_inequality (k : ℕ) (h : k ≥ 2) :
  Real.sqrt k - Real.sqrt (k - 1) > Real.sqrt (k + 1) - Real.sqrt k := by
  sorry

end sqrt_diff_inequality_l1000_100039


namespace square_division_l1000_100067

theorem square_division (s : ℝ) (x : ℝ) : 
  s = 2 →  -- side length of the square
  (4 * (1/2 * s * x) + (s^2 - 4 * (1/2 * s * x))) = (s^2 / 5) →  -- equal areas condition
  x = 4/5 :=
by sorry

end square_division_l1000_100067


namespace negation_of_forall_exp_gt_x_l1000_100014

theorem negation_of_forall_exp_gt_x :
  (¬ ∀ x : ℝ, Real.exp x > x) ↔ (∃ x : ℝ, Real.exp x ≤ x) := by sorry

end negation_of_forall_exp_gt_x_l1000_100014


namespace gcd_digit_bound_l1000_100037

theorem gcd_digit_bound (a b : ℕ) : 
  (10^6 ≤ a ∧ a < 10^7) →
  (10^6 ≤ b ∧ b < 10^7) →
  (10^12 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^13) →
  Nat.gcd a b < 10^2 :=
sorry

end gcd_digit_bound_l1000_100037


namespace greatest_number_of_fruit_baskets_l1000_100086

theorem greatest_number_of_fruit_baskets : Nat.gcd (Nat.gcd 18 27) 12 = 3 := by
  sorry

end greatest_number_of_fruit_baskets_l1000_100086


namespace negative_angle_quadrant_l1000_100018

/-- If an angle α is in the third quadrant, then -α is in the second quadrant -/
theorem negative_angle_quadrant (α : Real) : 
  (∃ k : ℤ, k * 2 * π + π < α ∧ α < k * 2 * π + 3 * π / 2) → 
  (∃ m : ℤ, m * 2 * π + π / 2 < -α ∧ -α < m * 2 * π + π) :=
by sorry

end negative_angle_quadrant_l1000_100018


namespace mean_equality_implies_z_value_l1000_100059

theorem mean_equality_implies_z_value : ∃ z : ℝ,
  (6 + 15 + 9 + 20) / 4 = (13 + z) / 2 → z = 12 := by
  sorry

end mean_equality_implies_z_value_l1000_100059


namespace triangle_side_length_l1000_100053

theorem triangle_side_length (A B C : Real) (tanA : Real) (angleC : Real) (BC : Real) :
  tanA = 1 / 3 →
  angleC = 150 * π / 180 →
  BC = 1 →
  let sinA := Real.sqrt (1 - 1 / (1 + tanA^2))
  let AB := BC * Real.sin angleC / sinA
  AB = Real.sqrt 10 / 2 :=
by sorry

end triangle_side_length_l1000_100053


namespace quadratic_coefficient_l1000_100045

theorem quadratic_coefficient (x : ℝ) : 
  (3 * x^2 = 8 * x + 10) → 
  ∃ a b c : ℝ, (a * x^2 + b * x + c = 0 ∧ b = -8) := by
sorry

end quadratic_coefficient_l1000_100045


namespace rahul_share_l1000_100089

/-- Calculates the share of payment for a worker given the total payment and the time taken by each worker --/
def calculate_share (total_payment : ℚ) (time_worker1 time_worker2 : ℚ) : ℚ :=
  let work_rate1 := 1 / time_worker1
  let work_rate2 := 1 / time_worker2
  let combined_rate := work_rate1 + work_rate2
  let share_ratio := work_rate1 / combined_rate
  total_payment * share_ratio

/-- Proves that Rahul's share of the payment is 900 given the conditions --/
theorem rahul_share :
  let total_payment : ℚ := 2250
  let rahul_time : ℚ := 3
  let rajesh_time : ℚ := 2
  calculate_share total_payment rahul_time rajesh_time = 900 := by
  sorry

#eval calculate_share 2250 3 2

end rahul_share_l1000_100089


namespace john_chores_time_l1000_100022

/-- Calculates the number of minutes of chores John has to do based on his cartoon watching time -/
def chores_minutes (cartoon_hours : ℕ) : ℕ :=
  let cartoon_minutes := cartoon_hours * 60
  let chore_blocks := cartoon_minutes / 10
  chore_blocks * 8

/-- Theorem: John has to do 96 minutes of chores when he watches 2 hours of cartoons -/
theorem john_chores_time : chores_minutes 2 = 96 := by
  sorry

end john_chores_time_l1000_100022


namespace greatest_whole_number_satisfying_inequality_l1000_100046

theorem greatest_whole_number_satisfying_inequality :
  ∀ x : ℤ, (5 * x - 4 < 3 - 2 * x) → x ≤ 0 :=
by sorry

end greatest_whole_number_satisfying_inequality_l1000_100046


namespace max_trio_sum_l1000_100040

/-- A trio is a set of three distinct integers where two are divisors or multiples of the third -/
def is_trio (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ((a ∣ c ∧ b ∣ c) ∨ (a ∣ b ∧ c ∣ b) ∨ (b ∣ a ∧ c ∣ a))

/-- The set of integers from 1 to 2002 -/
def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 2002}

theorem max_trio_sum :
  ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → is_trio a b c →
    a + b + c ≤ 4004 ∧
    (a + b + c = 4004 ↔ c = 2002 ∧ a ∣ 2002 ∧ b = 2002 - a) :=
sorry

end max_trio_sum_l1000_100040


namespace julia_tag_game_l1000_100021

theorem julia_tag_game (monday tuesday wednesday : ℕ) 
  (h1 : monday = 17) 
  (h2 : tuesday = 15) 
  (h3 : wednesday = 2) : 
  monday + tuesday + wednesday = 34 := by
  sorry

end julia_tag_game_l1000_100021


namespace fraction_to_decimal_l1000_100071

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end fraction_to_decimal_l1000_100071


namespace system_solution_proof_l1000_100036

theorem system_solution_proof (x y z : ℝ) : 
  (2 * x + y = 3) ∧ 
  (3 * x - z = 7) ∧ 
  (x - y + 3 * z = 0) → 
  (x = 2 ∧ y = -1 ∧ z = -1) :=
by sorry

end system_solution_proof_l1000_100036


namespace f_of_10_l1000_100026

theorem f_of_10 (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f (2*x + y) + 5*x*y = f (3*x - y) + 2*x^2 + 1) :
  f 10 = -49 := by
sorry

end f_of_10_l1000_100026


namespace employed_female_percentage_l1000_100070

/-- Represents the percentage of a population --/
def Percentage := Finset (Fin 100)

theorem employed_female_percentage
  (total_employed : Percentage)
  (employed_males : Percentage)
  (h1 : total_employed.card = 60)
  (h2 : employed_males.card = 48) :
  (total_employed.card - employed_males.card : ℚ) / total_employed.card * 100 = 20 := by
  sorry

end employed_female_percentage_l1000_100070


namespace triangle_abc_properties_l1000_100049

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  2 * Real.cos A * (b * Real.cos C + c * Real.cos B) = a →
  Real.cos B = 3 / 5 →
  A = π / 3 ∧ Real.sin (B - C) = (7 * Real.sqrt 3 - 24) / 50 := by
  sorry

end triangle_abc_properties_l1000_100049


namespace simplify_expression_l1000_100001

theorem simplify_expression (x t : ℝ) : (x^2 * t^3) * (x^3 * t^4) = x^5 * t^7 := by
  sorry

end simplify_expression_l1000_100001


namespace gummy_bear_cost_l1000_100035

theorem gummy_bear_cost
  (total_cost : ℝ)
  (chocolate_chip_cost : ℝ)
  (chocolate_bar_cost : ℝ)
  (num_chocolate_bars : ℕ)
  (num_gummy_bears : ℕ)
  (num_chocolate_chips : ℕ)
  (h1 : total_cost = 150)
  (h2 : chocolate_chip_cost = 5)
  (h3 : chocolate_bar_cost = 3)
  (h4 : num_chocolate_bars = 10)
  (h5 : num_gummy_bears = 10)
  (h6 : num_chocolate_chips = 20)
  : (total_cost - num_chocolate_bars * chocolate_bar_cost - num_chocolate_chips * chocolate_chip_cost) / num_gummy_bears = 2 := by
  sorry

end gummy_bear_cost_l1000_100035


namespace circle_radius_problem_l1000_100084

-- Define the points
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (5, -2)

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem statement
theorem circle_radius_problem (M N : Circle) : 
  -- Conditions
  (M.center.1 = 0) →  -- Center of M is on y-axis
  (N.center.1 = 2) →  -- x-coordinate of N's center is 2
  (N.center.2 = 4 - M.center.2) →  -- y-coordinate of N's center
  (M.radius = N.radius) →  -- Equal radii
  (M.radius^2 = (B.1 - M.center.1)^2 + (B.2 - M.center.2)^2) →  -- M passes through B
  (N.radius^2 = (B.1 - N.center.1)^2 + (B.2 - N.center.2)^2) →  -- N passes through B
  (N.radius^2 = (C.1 - N.center.1)^2 + (C.2 - N.center.2)^2) →  -- N passes through C
  -- Conclusion
  M.radius = Real.sqrt 10 :=
by sorry

end circle_radius_problem_l1000_100084


namespace number_plus_expression_l1000_100057

theorem number_plus_expression (x : ℝ) : x + 2 * (8 - 3) = 15 → x = 5 := by
  sorry

end number_plus_expression_l1000_100057


namespace math_majors_consecutive_seats_probability_l1000_100050

/-- The number of people sitting at the round table. -/
def totalPeople : ℕ := 12

/-- The number of math majors. -/
def mathMajors : ℕ := 5

/-- The number of physics majors. -/
def physicsMajors : ℕ := 4

/-- The number of biology majors. -/
def biologyMajors : ℕ := 3

/-- The probability of all math majors sitting in consecutive seats. -/
def probabilityConsecutiveSeats : ℚ := 1 / 66

theorem math_majors_consecutive_seats_probability :
  probabilityConsecutiveSeats = (totalPeople : ℚ) / (totalPeople.choose mathMajors) := by
  sorry

end math_majors_consecutive_seats_probability_l1000_100050


namespace odd_function_sum_l1000_100090

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_sum (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : f 4 = 5) :
  f 4 + f (-4) = 0 := by
  sorry

end odd_function_sum_l1000_100090


namespace heart_equation_solution_l1000_100004

/-- The heart operation defined on two real numbers -/
def heart (A B : ℝ) : ℝ := 4*A + A*B + 3*B + 6

/-- Theorem stating that 60/7 is the unique solution to A ♥ 3 = 75 -/
theorem heart_equation_solution :
  ∃! A : ℝ, heart A 3 = 75 ∧ A = 60/7 := by sorry

end heart_equation_solution_l1000_100004


namespace lecture_schedules_count_l1000_100028

/-- Represents the number of lecturers --/
def num_lecturers : ℕ := 8

/-- Represents the number of lecturer pairs with order requirements --/
def num_ordered_pairs : ℕ := 2

/-- Calculates the number of valid lecture schedules --/
def num_valid_schedules : ℕ := (Nat.factorial num_lecturers) / (2^num_ordered_pairs)

/-- Theorem stating the number of valid lecture schedules --/
theorem lecture_schedules_count : num_valid_schedules = 10080 := by
  sorry

end lecture_schedules_count_l1000_100028


namespace simplify_fraction_l1000_100058

theorem simplify_fraction : (45 : ℚ) / 75 = 3 / 5 := by
  sorry

end simplify_fraction_l1000_100058


namespace matches_arrangement_count_l1000_100033

/-- The number of ways to arrange matches for n players with some interchangeable players -/
def arrangeMatches (n : ℕ) (interchangeablePairs : ℕ) : ℕ :=
  Nat.factorial n * (2 ^ interchangeablePairs)

/-- Theorem: For 7 players with 3 pairs of interchangeable players, there are 40320 ways to arrange matches -/
theorem matches_arrangement_count :
  arrangeMatches 7 3 = 40320 := by
  sorry

end matches_arrangement_count_l1000_100033


namespace circumscribed_circle_radius_of_special_triangle_l1000_100098

/-- Given a triangle ABC where side b = 2√3 and angles A, B, C form an arithmetic sequence,
    the radius of the circumscribed circle is 2. -/
theorem circumscribed_circle_radius_of_special_triangle (A B C : Real) (a b c : Real) :
  b = 2 * Real.sqrt 3 →
  ∃ (d : Real), B = (A + C) / 2 ∧ A + d = B ∧ B + d = C →
  A + B + C = Real.pi →
  2 * Real.sin B = b / 2 →
  2 = 2 * Real.sin B / b * 2 * Real.sqrt 3 := by
  sorry

#check circumscribed_circle_radius_of_special_triangle

end circumscribed_circle_radius_of_special_triangle_l1000_100098


namespace triangle_area_l1000_100032

/-- Given a triangle ABC where:
    - The side opposite to angle B has length 2
    - The side opposite to angle C has length 2√3
    - Angle C measures 2π/3 radians
    Prove that the area of the triangle is 3 -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  b = 2 →
  c = 2 * Real.sqrt 3 →
  C = 2 * π / 3 →
  (1/2) * b * c * Real.sin A = 3 := by
sorry

end triangle_area_l1000_100032


namespace pulley_centers_distance_l1000_100069

theorem pulley_centers_distance (r₁ r₂ contact_distance : ℝ) 
  (h₁ : r₁ = 10)
  (h₂ : r₂ = 6)
  (h₃ : contact_distance = 20) :
  Real.sqrt ((r₁ - r₂)^2 + contact_distance^2) = 2 * Real.sqrt 104 :=
by sorry

end pulley_centers_distance_l1000_100069


namespace prob_different_colors_l1000_100030

/-- The probability of drawing two balls of different colors from a box containing 3 red balls and 2 yellow balls. -/
theorem prob_different_colors (total : ℕ) (red : ℕ) (yellow : ℕ) : 
  total = 5 → red = 3 → yellow = 2 → 
  (red.choose 1 * yellow.choose 1 : ℚ) / total.choose 2 = 3 / 5 := by
  sorry

end prob_different_colors_l1000_100030


namespace parabola_equation_l1000_100023

/-- A parabola with vertex at the origin and focus on the positive x-axis -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  h_p_pos : 0 < p
  h_focus : focus = (p / 2, 0)

/-- Two points on the parabola -/
structure ParabolaPoints (C : Parabola) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_on_parabola : A.2^2 = 2 * C.p * A.1 ∧ B.2^2 = 2 * C.p * B.1
  h_line_through_focus : ∃ k : ℝ, A.2 = k * (A.1 - C.p / 2) ∧ B.2 = k * (B.1 - C.p / 2)

/-- The dot product condition -/
def dot_product_condition (C : Parabola) (P : ParabolaPoints C) : Prop :=
  P.A.1 * P.B.1 + P.A.2 * P.B.2 = -12

/-- The main theorem -/
theorem parabola_equation (C : Parabola) (P : ParabolaPoints C)
  (h_dot : dot_product_condition C P) :
  C.p = 4 :=
sorry

end parabola_equation_l1000_100023


namespace intersection_of_sets_l1000_100092

theorem intersection_of_sets (A B : Set ℝ) (a : ℝ) : 
  A = {-1, 0, 1} → 
  B = {a + 1, 2 * a} → 
  A ∩ B = {0} → 
  a = -1 := by sorry

end intersection_of_sets_l1000_100092


namespace largest_n_value_l1000_100077

/-- The largest possible value of n for regular polygons Q1 (m-gon) and Q2 (n-gon) 
    satisfying the given conditions -/
theorem largest_n_value (m n : ℕ) : m ≥ n → n ≥ 3 → 
  (m - 2) * n = (n - 2) * m * 8 / 7 → 
  (∀ k, k > n → (k - 2) * m ≠ (m - 2) * k * 8 / 7) →
  n = 112 :=
by sorry

end largest_n_value_l1000_100077


namespace books_needed_l1000_100063

/-- The number of books each person has -/
structure BookCounts where
  darryl : ℕ
  lamont : ℕ
  loris : ℕ

/-- The conditions of the problem -/
def book_problem (b : BookCounts) : Prop :=
  b.darryl = 20 ∧
  b.lamont = 2 * b.darryl ∧
  b.darryl + b.lamont + b.loris = 97

/-- The theorem to prove -/
theorem books_needed (b : BookCounts) (h : book_problem b) : 
  b.lamont - b.loris = 3 := by
  sorry

end books_needed_l1000_100063


namespace sum_of_squares_bound_l1000_100034

theorem sum_of_squares_bound (a b : ℝ) (h : a + b = 1) : a^2 + b^2 ≥ 1/2 := by
  sorry

end sum_of_squares_bound_l1000_100034


namespace set_inclusion_range_l1000_100011

theorem set_inclusion_range (a : ℝ) : 
  let P : Set ℝ := {x | |x - 1| > 2}
  let S : Set ℝ := {x | x^2 - (a + 1)*x + a > 0}
  (P ⊆ S) → ((-1 ≤ a ∧ a < 1) ∨ (1 < a ∧ a ≤ 3)) := by
  sorry

end set_inclusion_range_l1000_100011


namespace simplify_expression_l1000_100060

theorem simplify_expression (b : ℝ) : ((3 * b + 6) - 6 * b) / 3 = -b + 2 := by
  sorry

end simplify_expression_l1000_100060


namespace complex_division_theorem_l1000_100066

theorem complex_division_theorem : 
  let z₁ : ℂ := Complex.mk 1 (-1)
  let z₂ : ℂ := Complex.mk 3 1
  z₂ / z₁ = Complex.mk 1 2 := by
sorry

end complex_division_theorem_l1000_100066


namespace cake_bread_weight_difference_l1000_100093

/-- Given that 4 cakes weigh 800 g and 3 cakes plus 5 pieces of bread weigh 1100 g,
    prove that a cake is 100 g heavier than a piece of bread. -/
theorem cake_bread_weight_difference :
  ∀ (cake_weight bread_weight : ℕ),
    4 * cake_weight = 800 →
    3 * cake_weight + 5 * bread_weight = 1100 →
    cake_weight - bread_weight = 100 := by
  sorry

end cake_bread_weight_difference_l1000_100093


namespace jacket_discount_percentage_l1000_100016

/-- Calculates the discount percentage on a jacket sale --/
theorem jacket_discount_percentage
  (purchase_price : ℝ)
  (markup_percentage : ℝ)
  (gross_profit : ℝ)
  (h1 : purchase_price = 48)
  (h2 : markup_percentage = 0.4)
  (h3 : gross_profit = 16) :
  let selling_price := purchase_price / (1 - markup_percentage)
  let sale_price := purchase_price + gross_profit
  (selling_price - sale_price) / selling_price = 0.2 := by
  sorry

end jacket_discount_percentage_l1000_100016


namespace f_properties_l1000_100010

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * x - (1 / π) * x^2 + cos x

theorem f_properties :
  (∀ x ∈ Set.Icc 0 (π / 2), Monotone f) ∧
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < π → f x1 = f x2 → 
    deriv f ((x1 + x2) / 2) < 0) :=
by sorry

end f_properties_l1000_100010


namespace rationalize_denominator_l1000_100008

theorem rationalize_denominator :
  (1 : ℝ) / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2 := by
  sorry

end rationalize_denominator_l1000_100008


namespace f_geq_6_iff_min_value_sum_reciprocals_min_value_sum_reciprocals_equality_l1000_100015

-- Part 1
def f (x : ℝ) : ℝ := |x + 1| + |2*x - 4|

theorem f_geq_6_iff (x : ℝ) : f x ≥ 6 ↔ x ≤ -1 ∨ x ≥ 3 := by sorry

-- Part 2
theorem min_value_sum_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + 2*b + 4*c = 8) : 
  (1/a + 1/b + 1/c) ≥ (11 + 6*Real.sqrt 2) / 8 := by sorry

theorem min_value_sum_reciprocals_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + 2*b + 4*c = 8) : 
  (1/a + 1/b + 1/c) = (11 + 6*Real.sqrt 2) / 8 ↔ a = Real.sqrt 2 * b ∧ b = 2 * c := by sorry

end f_geq_6_iff_min_value_sum_reciprocals_min_value_sum_reciprocals_equality_l1000_100015


namespace complex_simplification_l1000_100029

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_simplification :
  3 * (4 - 2*i) + 2*i*(3 + 2*i) - (1 + i)*(2 - i) = 5 - i :=
by sorry

end complex_simplification_l1000_100029


namespace molecular_weight_proof_l1000_100061

/-- Given a compound where 9 moles weigh 8100 grams, prove that its molecular weight is 900 grams/mole. -/
theorem molecular_weight_proof (compound : Type) 
  (moles : ℕ) (total_weight : ℝ) (molecular_weight : ℝ) 
  (h1 : moles = 9) 
  (h2 : total_weight = 8100) 
  (h3 : total_weight = moles * molecular_weight) : 
  molecular_weight = 900 := by
  sorry

end molecular_weight_proof_l1000_100061


namespace trig_identity_l1000_100020

theorem trig_identity (α β : Real) : 
  (1 / Real.tan α)^2 + (1 / Real.tan β)^2 - 2 * Real.cos (β - α) / (Real.sin α * Real.sin β) + 2 = 
  Real.sin (α - β)^2 / (Real.sin α^2 * Real.sin β^2) := by
  sorry

end trig_identity_l1000_100020


namespace polynomial_factor_implies_coefficients_l1000_100096

theorem polynomial_factor_implies_coefficients 
  (p q : ℚ) 
  (h : ∃ (a b : ℚ), px^4 + qx^3 + 45*x^2 - 25*x + 10 = (5*x^2 - 3*x + 2)*(a*x^2 + b*x + 5)) :
  p = 25/2 ∧ q = -65/2 := by
sorry

end polynomial_factor_implies_coefficients_l1000_100096


namespace special_numbers_l1000_100068

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def satisfies_condition (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → is_prime (d^2 - d + 1) ∧ is_prime (d^2 + d + 1)

theorem special_numbers :
  ∀ n : ℕ, satisfies_condition n ↔ n = 2 ∨ n = 3 ∨ n = 6 :=
sorry

end special_numbers_l1000_100068


namespace quadratic_inequality_theorem_l1000_100055

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) := {x : ℝ | x ≤ -3 ∨ x ≥ 4}

-- Theorem statement
theorem quadratic_inequality_theorem (a b c : ℝ) 
  (h : ∀ x, f a b c x ≥ 0 ↔ x ∈ solution_set a b c) : 
  (a > 0) ∧ 
  (∀ x, f c (-b) a x < 0 ↔ x < -1/4 ∨ x > 1/3) := by
sorry

end quadratic_inequality_theorem_l1000_100055


namespace product_of_fractions_equals_nine_l1000_100000

theorem product_of_fractions_equals_nine (a b c : ℝ) 
  (sum_zero : a + b + c = 0)
  (a_nonzero : a ≠ 0)
  (b_nonzero : b ≠ 0)
  (c_nonzero : c ≠ 0)
  (a_neq_b : a ≠ b)
  (a_neq_c : a ≠ c)
  (b_neq_c : b ≠ c) :
  ((a - b) / c + (b - c) / a + (c - a) / b) * (c / (a - b) + a / (b - c) + b / (c - a)) = 9 := by
  sorry

end product_of_fractions_equals_nine_l1000_100000


namespace smallest_solution_quadratic_l1000_100006

theorem smallest_solution_quadratic (y : ℝ) : 
  (3 * y^2 + 33 * y - 90 = y * (y + 16)) → y ≥ -10 :=
by
  sorry

end smallest_solution_quadratic_l1000_100006


namespace bigger_part_of_52_l1000_100052

theorem bigger_part_of_52 (x y : ℕ) (h1 : x + y = 52) (h2 : 10 * x + 22 * y = 780) :
  max x y = 30 := by sorry

end bigger_part_of_52_l1000_100052


namespace cookies_eaten_l1000_100073

theorem cookies_eaten (initial_cookies bought_cookies final_cookies : ℕ) :
  initial_cookies = 40 →
  bought_cookies = 37 →
  final_cookies = 75 →
  initial_cookies + bought_cookies - final_cookies = 2 :=
by
  sorry

end cookies_eaten_l1000_100073


namespace remainder_sum_l1000_100094

theorem remainder_sum (a b : ℤ) 
  (ha : a % 60 = 53) 
  (hb : b % 50 = 24) : 
  (a + b) % 20 = 17 := by
sorry

end remainder_sum_l1000_100094


namespace convex_polygon_coverage_l1000_100042

-- Define a convex polygon
def ConvexPolygon : Type := sorry

-- Define a function to check if a polygon can cover a triangle of given area
def can_cover (M : ConvexPolygon) (area : ℝ) : Prop := sorry

-- Define a function to check if a polygon can be covered by a triangle of given area
def can_be_covered (M : ConvexPolygon) (area : ℝ) : Prop := sorry

-- Theorem statement
theorem convex_polygon_coverage (M : ConvexPolygon) :
  (¬ can_cover M 1) → can_be_covered M 4 := by sorry

end convex_polygon_coverage_l1000_100042


namespace variance_linear_transform_l1000_100099

-- Define the variance of a dataset
def variance (data : List ℝ) : ℝ := sorry

-- Define a linear transformation of a dataset
def linearTransform (a b : ℝ) (data : List ℝ) : List ℝ := 
  data.map (fun x => a * x + b)

theorem variance_linear_transform (data : List ℝ) :
  variance data = 2 → variance (linearTransform 3 (-2) data) = 18 := by
  sorry

end variance_linear_transform_l1000_100099


namespace cobys_speed_l1000_100002

/-- Coby's road trip problem -/
theorem cobys_speed (d_WI d_IN : ℝ) (v_WI : ℝ) (t_total : ℝ) (h1 : d_WI = 640) (h2 : d_IN = 550) (h3 : v_WI = 80) (h4 : t_total = 19) :
  (d_IN / (t_total - d_WI / v_WI)) = 50 := by
  sorry

end cobys_speed_l1000_100002


namespace estimate_fish_population_l1000_100079

/-- Estimates the number of fish in a lake using the mark-recapture method. -/
theorem estimate_fish_population (initial_marked : ℕ) (second_catch : ℕ) (marked_in_second : ℕ) :
  initial_marked = 100 →
  second_catch = 200 →
  marked_in_second = 25 →
  (initial_marked * second_catch) / marked_in_second = 800 :=
by
  sorry

end estimate_fish_population_l1000_100079


namespace square_roots_problem_l1000_100082

theorem square_roots_problem (a : ℝ) (n : ℝ) : 
  (2*a + 3)^2 = n ∧ (a - 18)^2 = n → n = 169 := by
  sorry

end square_roots_problem_l1000_100082


namespace power_mod_seven_l1000_100005

theorem power_mod_seven : 3^1995 % 7 = 6 := by sorry

end power_mod_seven_l1000_100005


namespace circle_area_increase_l1000_100087

theorem circle_area_increase (r : ℝ) (h : r > 0) : 
  (π * (2 * r)^2 - π * r^2) / (π * r^2) = 3 := by
  sorry

end circle_area_increase_l1000_100087


namespace greaterElementSumOfS_l1000_100064

def S : Finset ℕ := {8, 5, 1, 13, 34, 3, 21, 2}

def greaterElementSum (s : Finset ℕ) : ℕ :=
  s.sum (λ x => (s.filter (λ y => y < x)).card * x)

theorem greaterElementSumOfS : greaterElementSum S = 484 := by
  sorry

end greaterElementSumOfS_l1000_100064


namespace sally_bought_twenty_cards_l1000_100085

/-- The number of cards Sally bought -/
def cards_bought (initial : ℕ) (received : ℕ) (total : ℕ) : ℕ :=
  total - (initial + received)

/-- Theorem: Sally bought 20 cards -/
theorem sally_bought_twenty_cards :
  cards_bought 27 41 88 = 20 := by
  sorry

end sally_bought_twenty_cards_l1000_100085


namespace problem_statement_l1000_100019

theorem problem_statement (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) := by
  sorry

end problem_statement_l1000_100019


namespace quadratic_bound_l1000_100038

theorem quadratic_bound (a b c : ℝ) 
  (h : ∀ x : ℝ, |x| ≤ 1 → |a*x^2 + b*x + c| ≤ 1) : 
  ∀ x : ℝ, |x| ≤ 1 → |2*a*x + b| ≤ 4 := by
sorry

end quadratic_bound_l1000_100038


namespace max_silver_tokens_l1000_100031

/-- Represents the number of tokens of each color --/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents the exchange rates for the two booths --/
structure ExchangeRates where
  redToSilver : TokenCount → TokenCount
  blueToSilver : TokenCount → TokenCount

/-- The initial token count --/
def initialTokens : TokenCount :=
  { red := 60, blue := 90, silver := 0 }

/-- The exchange rates for the two booths --/
def boothRates : ExchangeRates :=
  { redToSilver := λ tc => { red := tc.red - 3, blue := tc.blue + 2, silver := tc.silver + 1 },
    blueToSilver := λ tc => { red := tc.red + 1, blue := tc.blue - 4, silver := tc.silver + 2 } }

/-- Determines if further exchanges are possible --/
def canExchange (tc : TokenCount) : Bool :=
  tc.red ≥ 3 ∨ tc.blue ≥ 4

/-- The main theorem to prove --/
theorem max_silver_tokens :
  ∃ (finalTokens : TokenCount),
    (¬canExchange finalTokens) ∧
    (finalTokens.silver = 101) ∧
    (∃ (exchanges : List (TokenCount → TokenCount)),
      exchanges.foldl (λ acc f => f acc) initialTokens = finalTokens) :=
  sorry

end max_silver_tokens_l1000_100031


namespace rotated_ellipse_sum_l1000_100074

/-- Represents an ellipse rotated 90 degrees around its center. -/
structure RotatedEllipse where
  h' : ℝ  -- x-coordinate of the center
  k' : ℝ  -- y-coordinate of the center
  a' : ℝ  -- length of the semi-major axis
  b' : ℝ  -- length of the semi-minor axis

/-- Theorem stating the sum of parameters for a specific rotated ellipse. -/
theorem rotated_ellipse_sum (e : RotatedEllipse) 
  (center_x : e.h' = 3) 
  (center_y : e.k' = -5) 
  (major_axis : e.a' = 4) 
  (minor_axis : e.b' = 2) : 
  e.h' + e.k' + e.a' + e.b' = 4 := by
  sorry

end rotated_ellipse_sum_l1000_100074


namespace no_solution_iff_n_eq_neg_one_l1000_100083

/-- A system of equations parameterized by n -/
def system (n : ℝ) (x y z : ℝ) : Prop :=
  n * x + y = 2 ∧ n * y + z = 2 ∧ x + n^2 * z = 2

/-- The system has no solution if and only if n = -1 -/
theorem no_solution_iff_n_eq_neg_one :
  ∀ n : ℝ, (∀ x y z : ℝ, ¬system n x y z) ↔ n = -1 := by
  sorry

end no_solution_iff_n_eq_neg_one_l1000_100083


namespace train_distance_problem_l1000_100047

theorem train_distance_problem (speed1 speed2 distance_difference : ℝ) 
  (h1 : speed1 = 16)
  (h2 : speed2 = 21)
  (h3 : distance_difference = 60)
  (time : ℝ)
  (h4 : time > 0)
  (distance1 : ℝ)
  (h5 : distance1 = speed1 * time)
  (distance2 : ℝ)
  (h6 : distance2 = speed2 * time)
  (h7 : distance2 = distance1 + distance_difference) :
  distance1 + distance2 = 444 := by
sorry

end train_distance_problem_l1000_100047


namespace tenth_term_of_sequence_l1000_100095

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem tenth_term_of_sequence (a₁ a₂ : ℤ) (h₁ : a₁ = 2) (h₂ : a₂ = 1) :
  arithmetic_sequence a₁ (a₂ - a₁) 10 = -7 := by
sorry

end tenth_term_of_sequence_l1000_100095


namespace range_of_k_l1000_100051

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < k + 1}

-- Define the complement of A in ℝ
def C_ℝA : Set ℝ := {x | 1 < x ∧ x < 3}

-- Theorem statement
theorem range_of_k (k : ℝ) :
  (C_ℝA ∩ B k = ∅) → (k ≤ 0 ∨ k ≥ 3) :=
by sorry

end range_of_k_l1000_100051


namespace plot_perimeter_l1000_100076

/-- Proves that the perimeter of a rectangular plot is 300 meters given specific conditions -/
theorem plot_perimeter : 
  ∀ (width length perimeter : ℝ),
  length = width + 10 →
  1950 = (perimeter * 6.5) →
  perimeter = 2 * (length + width) →
  perimeter = 300 := by
sorry

end plot_perimeter_l1000_100076


namespace sin_cos_difference_sin_negative_main_theorem_l1000_100027

theorem sin_cos_difference (x y : Real) : 
  Real.sin (x * π / 180) * Real.cos (y * π / 180) - Real.cos (x * π / 180) * Real.sin (y * π / 180) = 
  Real.sin ((x - y) * π / 180) :=
sorry

theorem sin_negative (x : Real) : Real.sin (-x) = -Real.sin x :=
sorry

theorem main_theorem : 
  Real.sin (24 * π / 180) * Real.cos (54 * π / 180) - Real.cos (24 * π / 180) * Real.sin (54 * π / 180) = -1/2 :=
sorry

end sin_cos_difference_sin_negative_main_theorem_l1000_100027


namespace identity_function_divisibility_l1000_100012

theorem identity_function_divisibility (f : ℕ+ → ℕ+) :
  (∀ a b : ℕ+, (a.val ^ 2 + (f a).val * (f b).val) % ((f a).val + b.val) = 0) →
  (∀ n : ℕ+, f n = n) :=
by sorry

end identity_function_divisibility_l1000_100012


namespace smallest_three_digit_multiple_of_17_l1000_100062

-- Define the properties of our target number
def is_valid (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ ∃ k : ℕ, n = 17 * k

-- State the theorem
theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, is_valid n → n ≥ 102 :=
by
  sorry

end smallest_three_digit_multiple_of_17_l1000_100062


namespace catman_do_whiskers_l1000_100056

theorem catman_do_whiskers (princess_puff_whiskers : ℕ) (catman_do_whiskers : ℕ) : 
  princess_puff_whiskers = 14 →
  catman_do_whiskers = 2 * princess_puff_whiskers - 6 →
  catman_do_whiskers = 22 := by
  sorry

end catman_do_whiskers_l1000_100056


namespace appended_number_cube_sum_l1000_100044

theorem appended_number_cube_sum (a b c : ℕ) : 
  b ≥ 10 ∧ b < 100 ∧ c ≥ 10 ∧ c < 100 →
  10000 * a + 100 * b + c = (a + b + c)^3 →
  a = 9 ∧ b = 11 ∧ c = 25 :=
by sorry

end appended_number_cube_sum_l1000_100044


namespace train_speed_problem_l1000_100075

-- Define the speeds and times
def speed_A : ℝ := 90
def time_A : ℝ := 9
def time_B : ℝ := 4

-- Theorem statement
theorem train_speed_problem :
  ∃ (speed_B : ℝ),
    speed_B > 0 ∧
    speed_A * time_A = speed_B * time_B ∧
    speed_B = 202.5 := by
  sorry

end train_speed_problem_l1000_100075


namespace cube_side_length_l1000_100009

theorem cube_side_length (v : Real) (s : Real) :
  v = 8 →
  v = s^3 →
  ∃ (x : Real), 
    6 * x^2 = 3 * (6 * s^2) ∧
    x = 2 * Real.sqrt 3 :=
by sorry

end cube_side_length_l1000_100009


namespace problem_statement_l1000_100041

theorem problem_statement (a b k : ℝ) 
  (h1 : 2^a = k) 
  (h2 : 3^b = k) 
  (h3 : k ≠ 1) 
  (h4 : 1/a + 2/b = 1) : 
  k = 18 := by sorry

end problem_statement_l1000_100041


namespace complex_modulus_seven_l1000_100024

theorem complex_modulus_seven (x : ℝ) : 
  x > 0 → (Complex.abs (3 + x * Complex.I) = 7 ↔ x = 2 * Real.sqrt 10) := by
  sorry

end complex_modulus_seven_l1000_100024


namespace message_spread_time_l1000_100080

theorem message_spread_time (n : ℕ) : ∃ (m : ℕ), m ≥ 5 ∧ 2^(m+1) - 2 > 55 ∧ ∀ (k : ℕ), k < m → 2^(k+1) - 2 ≤ 55 := by
  sorry

end message_spread_time_l1000_100080


namespace smallest_n_for_sqrt_50n_two_satisfies_condition_two_is_smallest_l1000_100017

theorem smallest_n_for_sqrt_50n (n : ℕ) : (∃ k : ℕ, k * k = 50 * n) → n ≥ 2 := by
  sorry

theorem two_satisfies_condition : ∃ k : ℕ, k * k = 50 * 2 := by
  sorry

theorem two_is_smallest : ∀ n : ℕ, n > 0 → n < 2 → ¬(∃ k : ℕ, k * k = 50 * n) := by
  sorry

end smallest_n_for_sqrt_50n_two_satisfies_condition_two_is_smallest_l1000_100017


namespace inscribed_rectangle_perimeter_l1000_100013

theorem inscribed_rectangle_perimeter (circle_area : ℝ) (rect_area : ℝ) :
  circle_area = 32 * Real.pi →
  rect_area = 34 →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    a * b = rect_area ∧
    a^2 + b^2 = 2 * circle_area / Real.pi ∧
    2 * (a + b) = 28 := by
  sorry

end inscribed_rectangle_perimeter_l1000_100013


namespace expression_simplification_l1000_100088

theorem expression_simplification (x : ℝ) : 
  2*x - 3*(2 - x) + 5*(2 + x) - 7*(1 - 3*x) = 31*x - 3 := by
  sorry

end expression_simplification_l1000_100088


namespace toy_sale_proof_l1000_100003

/-- Calculates the total selling price of toys given the number of toys sold,
    the cost price per toy, and the gain in terms of number of toys. -/
def totalSellingPrice (numToysSold : ℕ) (costPrice : ℕ) (gainInToys : ℕ) : ℕ :=
  (numToysSold + gainInToys) * costPrice

/-- Proves that the total selling price for 18 toys with a cost price of 1200
    and a gain equal to the cost of 3 toys is 25200. -/
theorem toy_sale_proof :
  totalSellingPrice 18 1200 3 = 25200 := by
  sorry

end toy_sale_proof_l1000_100003


namespace product_expansion_l1000_100065

theorem product_expansion (x : ℝ) : (x + 3) * (x + 7) * (x - 2) = x^3 + 8*x^2 + x - 42 := by
  sorry

end product_expansion_l1000_100065
