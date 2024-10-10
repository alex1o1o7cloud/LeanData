import Mathlib

namespace pyramid_edges_count_l3603_360351

/-- A prism is a polyhedron with two congruent bases and rectangular faces connecting corresponding edges of the bases. -/
structure Prism where
  vertices : ℕ
  faces : ℕ
  edges : ℕ
  sum_property : vertices + faces + edges = 50
  euler_formula : vertices - edges + faces = 2

/-- A pyramid is a polyhedron with a polygonal base and triangular faces meeting at a point (apex). -/
structure Pyramid where
  base_edges : ℕ

/-- Given a prism, construct a pyramid with the same base shape. -/
def pyramid_from_prism (p : Prism) : Pyramid :=
  { base_edges := (p.edges / 3) }

theorem pyramid_edges_count (p : Prism) : 
  (pyramid_from_prism p).base_edges * 2 = 16 := by
  sorry

end pyramid_edges_count_l3603_360351


namespace f_properties_l3603_360332

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Theorem stating that f is an odd function and monotonically increasing
theorem f_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
by
  sorry

end f_properties_l3603_360332


namespace abc_sum_eq_three_l3603_360327

theorem abc_sum_eq_three (a b c : ℕ+) 
  (h1 : c = b^2)
  (h2 : (a + b + c)^3 - a^3 - b^3 - c^3 = 210) :
  a + b + c = 3 := by
  sorry

end abc_sum_eq_three_l3603_360327


namespace snow_leopard_lineup_l3603_360353

/-- The number of ways to arrange 9 distinct objects in a row, 
    where 3 specific objects must be placed at the ends and middle -/
def arrangement_count : ℕ := 4320

/-- The number of ways to arrange 3 objects in 3 specific positions -/
def short_leopard_arrangements : ℕ := 6

/-- The number of ways to arrange the remaining 6 objects -/
def remaining_leopard_arrangements : ℕ := 720

theorem snow_leopard_lineup : 
  arrangement_count = short_leopard_arrangements * remaining_leopard_arrangements :=
sorry

end snow_leopard_lineup_l3603_360353


namespace number_machine_input_l3603_360341

/-- A number machine that adds 15 and then subtracts 6 -/
def number_machine (x : ℤ) : ℤ := x + 15 - 6

/-- Theorem stating that if the number machine outputs 77, the input must have been 68 -/
theorem number_machine_input (x : ℤ) : number_machine x = 77 → x = 68 := by
  sorry

end number_machine_input_l3603_360341


namespace evaluate_expression_l3603_360321

theorem evaluate_expression (c : ℕ) (h : c = 3) : (c^c - c*(c-1)^c)^c = 27 := by
  sorry

end evaluate_expression_l3603_360321


namespace willy_stuffed_animals_l3603_360300

/-- The number of stuffed animals Willy's mom gave him for his birthday -/
def moms_gift : ℕ := 2

/-- Willy's initial number of stuffed animals -/
def initial_count : ℕ := 10

/-- The factor by which Willy's dad increases his stuffed animal count -/
def dad_factor : ℕ := 3

/-- The total number of stuffed animals Willy has at the end -/
def final_count : ℕ := 48

theorem willy_stuffed_animals :
  initial_count + moms_gift + dad_factor * (initial_count + moms_gift) = final_count :=
by sorry

end willy_stuffed_animals_l3603_360300


namespace not_mysterious_consecutive_odd_squares_diff_l3603_360303

/-- A positive integer that can be expressed as the difference of squares of two consecutive even numbers. -/
def MysteriousNumber (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2*k + 2)^2 - (2*k)^2 ∧ n > 0

/-- The difference of squares of two consecutive odd numbers. -/
def ConsecutiveOddSquaresDiff (k : ℤ) : ℤ :=
  (2*k + 1)^2 - (2*k - 1)^2

theorem not_mysterious_consecutive_odd_squares_diff :
  ∀ k : ℤ, ¬(MysteriousNumber (ConsecutiveOddSquaresDiff k).natAbs) :=
by sorry

end not_mysterious_consecutive_odd_squares_diff_l3603_360303


namespace first_item_is_five_l3603_360365

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  total_items : ℕ
  sample_size : ℕ
  item_16 : ℕ

/-- The first item in a systematic sampling scheme -/
def first_item (s : SystematicSampling) : ℕ :=
  s.item_16 - (16 - 1) * (s.total_items / s.sample_size)

/-- Theorem: In the given systematic sampling scheme, the first item is 5 -/
theorem first_item_is_five :
  let s : SystematicSampling := ⟨160, 20, 125⟩
  first_item s = 5 := by sorry

end first_item_is_five_l3603_360365


namespace count_palindrome_pairs_l3603_360324

def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ 
  (n / 1000 = n % 10) ∧ 
  ((n / 100) % 10 = (n / 10) % 10)

def palindrome_pair (p1 p2 : ℕ) : Prop :=
  is_four_digit_palindrome p1 ∧ 
  is_four_digit_palindrome p2 ∧ 
  p1 - p2 = 3674

theorem count_palindrome_pairs : 
  ∃ (S : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ S ↔ palindrome_pair p.1 p.2) ∧ 
    Finset.card S = 35 := by
  sorry

end count_palindrome_pairs_l3603_360324


namespace shortest_altitude_of_triangle_l3603_360392

theorem shortest_altitude_of_triangle (a b c : ℝ) (h1 : a = 12) (h2 : b = 16) (h3 : c = 20) :
  ∃ h : ℝ, h = 9.6 ∧ h ≤ min a b ∧ h ≤ (2 * (a * b) / c) := by
  sorry

end shortest_altitude_of_triangle_l3603_360392


namespace tv_and_radio_clients_l3603_360361

def total_clients : ℕ := 180
def tv_clients : ℕ := 115
def radio_clients : ℕ := 110
def magazine_clients : ℕ := 130
def tv_and_magazine : ℕ := 85
def radio_and_magazine : ℕ := 95
def all_three : ℕ := 80

theorem tv_and_radio_clients : 
  total_clients = tv_clients + radio_clients + magazine_clients - tv_and_magazine - radio_and_magazine - (tv_clients + radio_clients - total_clients) + all_three := by
  sorry

end tv_and_radio_clients_l3603_360361


namespace sqrt_sum_equality_l3603_360395

theorem sqrt_sum_equality : Real.sqrt 8 + Real.sqrt 18 = 5 * Real.sqrt 2 := by
  sorry

end sqrt_sum_equality_l3603_360395


namespace factorial_14_mod_17_l3603_360337

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem factorial_14_mod_17 : 
  factorial 14 % 17 = 8 :=
by
  sorry

end factorial_14_mod_17_l3603_360337


namespace fifth_month_sale_proof_l3603_360396

/-- Calculates the sale in the fifth month given the sales of other months and the average -/
def fifth_month_sale (m1 m2 m3 m4 m6 avg : ℕ) : ℕ :=
  6 * avg - (m1 + m2 + m3 + m4 + m6)

/-- Proves that the sale in the fifth month is 3562 given the specified conditions -/
theorem fifth_month_sale_proof :
  fifth_month_sale 3435 3927 3855 4230 1991 3500 = 3562 := by
  sorry

end fifth_month_sale_proof_l3603_360396


namespace survey_result_l3603_360352

/-- Calculates the percentage of the surveyed population that supports a new environmental policy. -/
def survey_support_percentage (men_support_rate : ℚ) (women_support_rate : ℚ) (men_count : ℕ) (women_count : ℕ) : ℚ :=
  let total_count := men_count + women_count
  let supporting_count := men_support_rate * men_count + women_support_rate * women_count
  supporting_count / total_count

/-- Theorem stating that given the survey conditions, 74% of the population supports the policy. -/
theorem survey_result :
  let men_support_rate : ℚ := 70 / 100
  let women_support_rate : ℚ := 75 / 100
  let men_count : ℕ := 200
  let women_count : ℕ := 800
  survey_support_percentage men_support_rate women_support_rate men_count women_count = 74 / 100 := by
  sorry

#eval survey_support_percentage (70 / 100) (75 / 100) 200 800

end survey_result_l3603_360352


namespace cube_ending_in_eight_and_nine_l3603_360334

theorem cube_ending_in_eight_and_nine :
  ∀ a b : ℕ,
  (10 ≤ a ∧ a < 100) →
  (10 ≤ b ∧ b < 100) →
  (1000 ≤ a^3 ∧ a^3 < 10000) →
  (1000 ≤ b^3 ∧ b^3 < 10000) →
  a^3 % 10 = 8 →
  b^3 % 10 = 9 →
  a = 12 ∧ b = 19 :=
by sorry

end cube_ending_in_eight_and_nine_l3603_360334


namespace trivia_team_tryouts_l3603_360382

theorem trivia_team_tryouts (not_picked : ℕ) (num_groups : ℕ) (students_per_group : ℕ) : 
  not_picked = 9 → num_groups = 3 → students_per_group = 9 → 
  not_picked + num_groups * students_per_group = 36 := by
sorry

end trivia_team_tryouts_l3603_360382


namespace intersection_of_H_and_G_l3603_360335

def H : Set ℕ := {2, 3, 4}
def G : Set ℕ := {1, 3}

theorem intersection_of_H_and_G : H ∩ G = {3} := by sorry

end intersection_of_H_and_G_l3603_360335


namespace intersection_equidistant_l3603_360305

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the condition AB = CD
def equal_sides (q : Quadrilateral) : Prop :=
  dist q.A q.B = dist q.C q.D

-- Define the intersection point O of diagonals AC and BD
def intersection_point (q : Quadrilateral) : ℝ × ℝ :=
  sorry

-- Define a line passing through O
structure Line :=
  (slope : ℝ)
  (point : ℝ × ℝ)

-- Define the intersection points of a line with the quadrilateral sides
def intersection_points (q : Quadrilateral) (l : Line) :
  (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry

-- Define the intersection of a line with BD
def intersection_with_diagonal (q : Quadrilateral) (l : Line) : ℝ × ℝ :=
  sorry

-- Main theorem
theorem intersection_equidistant (q : Quadrilateral) (l1 l2 : Line)
  (h : equal_sides q) :
  let O := intersection_point q
  let I := intersection_with_diagonal q l1
  let J := intersection_with_diagonal q l2
  dist O I = dist O J :=
sorry

end intersection_equidistant_l3603_360305


namespace work_completion_time_l3603_360313

/-- Work rates and completion times for a team project -/
theorem work_completion_time 
  (man_rate : ℚ) 
  (woman_rate : ℚ) 
  (girl_rate : ℚ) 
  (team_rate : ℚ) 
  (h1 : man_rate = 1/6) 
  (h2 : woman_rate = 1/18) 
  (h3 : girl_rate = 1/12) 
  (h4 : team_rate = 1/3) 
  (h5 : man_rate + woman_rate + girl_rate + (team_rate - man_rate - woman_rate - girl_rate) = team_rate) : 
  (1 / ((team_rate - man_rate - woman_rate - girl_rate) + 2 * girl_rate)) = 36/7 := by
  sorry

end work_completion_time_l3603_360313


namespace cyclic_sum_inequality_l3603_360315

theorem cyclic_sum_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_one : a + b + c + d = 1) :
  (a / (a^2 + 1)) + (b / (b^2 + 1)) + (c / (c^2 + 1)) + (d / (d^2 + 1)) ≤ 16/17 := by
  sorry

end cyclic_sum_inequality_l3603_360315


namespace donation_growth_rate_l3603_360322

theorem donation_growth_rate 
  (initial_donation : ℝ) 
  (third_day_donation : ℝ) 
  (h1 : initial_donation = 10000)
  (h2 : third_day_donation = 12100) :
  ∃ (rate : ℝ), 
    initial_donation * (1 + rate)^2 = third_day_donation ∧ 
    rate = 0.1 := by
sorry

end donation_growth_rate_l3603_360322


namespace octagon_perimeter_is_six_l3603_360360

/-- A pentagon formed by removing a right-angled isosceles triangle from a unit square -/
structure Pentagon where
  /-- The side length of the original square -/
  squareSide : ℝ
  /-- The length of the leg of the removed right-angled isosceles triangle -/
  triangleLeg : ℝ
  /-- Assertion that the square is a unit square -/
  squareIsUnit : squareSide = 1
  /-- Assertion that the removed triangle is right-angled isosceles with leg length equal to the square side -/
  triangleIsRightIsosceles : triangleLeg = squareSide

/-- An octagon formed by fitting together two congruent pentagons -/
structure Octagon where
  /-- The first pentagon used to form the octagon -/
  pentagon1 : Pentagon
  /-- The second pentagon used to form the octagon -/
  pentagon2 : Pentagon
  /-- Assertion that the two pentagons are congruent -/
  pentagonsAreCongruent : pentagon1 = pentagon2

/-- The perimeter of the octagon -/
def octagonPerimeter (o : Octagon) : ℝ :=
  -- Definition of perimeter calculation goes here
  sorry

/-- Theorem: The perimeter of the octagon is 6 -/
theorem octagon_perimeter_is_six (o : Octagon) : octagonPerimeter o = 6 := by
  sorry

end octagon_perimeter_is_six_l3603_360360


namespace calculate_savings_l3603_360399

/-- Given a person's income and expenditure ratio, and their income, calculate their savings. -/
theorem calculate_savings (income_ratio : ℕ) (expenditure_ratio : ℕ) (income : ℕ) 
    (h1 : income_ratio = 8)
    (h2 : expenditure_ratio = 7)
    (h3 : income = 40000) :
  income - (expenditure_ratio * income / income_ratio) = 5000 := by
  sorry

#check calculate_savings

end calculate_savings_l3603_360399


namespace jack_second_half_time_is_six_l3603_360389

/-- The time Jack took to run up the hill -/
def jack_total_time (jill_time first_half_time time_diff : ℕ) : ℕ :=
  jill_time - time_diff

/-- The time Jack took to run up the second half of the hill -/
def jack_second_half_time (total_time first_half_time : ℕ) : ℕ :=
  total_time - first_half_time

/-- Proof that Jack took 6 seconds to run up the second half of the hill -/
theorem jack_second_half_time_is_six :
  ∀ (jill_time first_half_time time_diff : ℕ),
    jill_time = 32 →
    first_half_time = 19 →
    time_diff = 7 →
    jack_second_half_time (jack_total_time jill_time first_half_time time_diff) first_half_time = 6 :=
by
  sorry

end jack_second_half_time_is_six_l3603_360389


namespace bug_flower_consumption_l3603_360344

theorem bug_flower_consumption (num_bugs : ℝ) (flowers_per_bug : ℝ) : 
  num_bugs = 2.0 → flowers_per_bug = 1.5 → num_bugs * flowers_per_bug = 3.0 := by
  sorry

end bug_flower_consumption_l3603_360344


namespace complex_circle_equation_l3603_360310

theorem complex_circle_equation (z : ℂ) (h : Complex.abs (z - 1) = 5) :
  ∃ (x y : ℝ), z = Complex.mk x y ∧ 
  -4 ≤ x ∧ x ≤ 6 ∧ 
  (y = Real.sqrt (25 - (x - 1)^2) ∨ y = -Real.sqrt (25 - (x - 1)^2)) :=
by sorry

end complex_circle_equation_l3603_360310


namespace apple_boxes_l3603_360316

theorem apple_boxes (apples_per_crate : ℕ) (crates : ℕ) (rotten_apples : ℕ) (apples_per_box : ℕ) :
  apples_per_crate = 42 →
  crates = 12 →
  rotten_apples = 4 →
  apples_per_box = 10 →
  (crates * apples_per_crate - rotten_apples) / apples_per_box = 50 := by
  sorry

end apple_boxes_l3603_360316


namespace remainder_problem_l3603_360368

theorem remainder_problem (N : ℤ) : N % 899 = 63 → N % 29 = 10 := by
  sorry

end remainder_problem_l3603_360368


namespace tangent_slope_at_2_l3603_360362

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 3*x

-- State the theorem
theorem tangent_slope_at_2 :
  (deriv f) 2 = 7 := by sorry

end tangent_slope_at_2_l3603_360362


namespace cube_sum_over_product_equals_three_l3603_360372

theorem cube_sum_over_product_equals_three
  (p q r : ℝ)
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h_sum : p + q + r = 6) :
  (p^3 + q^3 + r^3) / (p * q * r) = 3 :=
by sorry

end cube_sum_over_product_equals_three_l3603_360372


namespace square_binomial_constant_l3603_360375

theorem square_binomial_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 200*x + c = (x + a)^2) → c = 10000 := by
  sorry

end square_binomial_constant_l3603_360375


namespace flower_pots_on_path_l3603_360343

/-- Calculates the number of flower pots on a path -/
def flowerPots (pathLength : ℕ) (interval : ℕ) : ℕ :=
  pathLength / interval + 1

/-- Theorem: On a 15-meter path with flower pots every 3 meters, there are 6 flower pots -/
theorem flower_pots_on_path : flowerPots 15 3 = 6 := by
  sorry

end flower_pots_on_path_l3603_360343


namespace hyperbola_properties_l3603_360358

/-- A hyperbola with equation y²/2 - x²/4 = 1 -/
def hyperbola (x y : ℝ) : Prop := y^2 / 2 - x^2 / 4 = 1

/-- The reference hyperbola with equation x²/2 - y² = 1 -/
def reference_hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 = 1

theorem hyperbola_properties :
  (∃ (x y : ℝ), hyperbola x y ∧ x = 2 ∧ y = -2) ∧
  (∀ (x y : ℝ), ∃ (k : ℝ), hyperbola x y ↔ reference_hyperbola (x * k) (y * k)) :=
sorry

end hyperbola_properties_l3603_360358


namespace max_p_value_l3603_360379

/-- Given a function f(x) = e^x and real numbers m, n, p satisfying certain conditions,
    the maximum value of p is 2ln(2) - ln(3). -/
theorem max_p_value (f : ℝ → ℝ) (m n p : ℝ) 
    (h1 : ∀ x, f x = Real.exp x)
    (h2 : f (m + n) = f m + f n)
    (h3 : f (m + n + p) = f m + f n + f p) :
    p ≤ 2 * Real.log 2 - Real.log 3 := by
  sorry

end max_p_value_l3603_360379


namespace triangle_cos_C_eq_neg_one_fourth_l3603_360378

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C opposite to these sides respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The law of sines for a triangle -/
axiom law_of_sines (t : Triangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- The law of cosines for a triangle -/
axiom law_of_cosines (t : Triangle) : Real.cos t.C = (t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b)

theorem triangle_cos_C_eq_neg_one_fourth (t : Triangle) 
  (ha : t.a = 2)
  (hc : t.c = 4)
  (h_sin : 3 * Real.sin t.A = 2 * Real.sin t.B) :
  Real.cos t.C = -1/4 := by
  sorry

end triangle_cos_C_eq_neg_one_fourth_l3603_360378


namespace method_two_more_swims_at_300_method_one_cheaper_for_40_plus_swims_l3603_360381

-- Define the cost functions for both methods
def cost_method_one (x : ℕ) : ℕ := 120 + 10 * x
def cost_method_two (x : ℕ) : ℕ := 15 * x

-- Theorem 1: For a total cost of 300 yuan, Method two allows more swims
theorem method_two_more_swims_at_300 :
  ∃ (x y : ℕ), cost_method_one x = 300 ∧ cost_method_two y = 300 ∧ y > x :=
sorry

-- Theorem 2: For 40 or more swims, Method one is less expensive
theorem method_one_cheaper_for_40_plus_swims :
  ∀ x : ℕ, x ≥ 40 → cost_method_one x < cost_method_two x :=
sorry

end method_two_more_swims_at_300_method_one_cheaper_for_40_plus_swims_l3603_360381


namespace kids_waiting_swings_is_three_l3603_360339

/-- The number of kids waiting for the swings -/
def kids_waiting_swings : ℕ := sorry

/-- The number of kids waiting for the slide -/
def kids_waiting_slide : ℕ := 2 * kids_waiting_swings

/-- The wait time for the swings in seconds -/
def wait_time_swings : ℕ := 120 * kids_waiting_swings

/-- The wait time for the slide in seconds -/
def wait_time_slide : ℕ := 15 * kids_waiting_slide

/-- The difference between the longer and shorter wait times -/
def wait_time_difference : ℕ := 270

theorem kids_waiting_swings_is_three :
  kids_waiting_swings = 3 ∧
  kids_waiting_slide = 2 * kids_waiting_swings ∧
  wait_time_swings = 120 * kids_waiting_swings ∧
  wait_time_slide = 15 * kids_waiting_slide ∧
  wait_time_swings - wait_time_slide = wait_time_difference :=
by sorry

end kids_waiting_swings_is_three_l3603_360339


namespace unique_five_digit_number_l3603_360307

/-- Represents a five-digit number -/
def FiveDigitNumber := { n : ℕ // 10000 ≤ n ∧ n < 100000 }

/-- Returns the four-digit number formed by removing the digit at position i -/
def removeDigit (n : FiveDigitNumber) (i : Fin 5) : ℕ :=
  sorry

/-- The theorem to be proved -/
theorem unique_five_digit_number :
  ∃! (n : FiveDigitNumber),
    ∃ (i : Fin 5),
      n.val + removeDigit n i = 54321 :=
by
  sorry

end unique_five_digit_number_l3603_360307


namespace condition_necessary_not_sufficient_l3603_360342

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ x ≤ 1) := by
  sorry

end condition_necessary_not_sufficient_l3603_360342


namespace cheolsu_number_problem_l3603_360371

theorem cheolsu_number_problem (x : ℚ) : 
  x + (-5/12) - (-5/2) = 1/3 → x = -7/4 := by
  sorry

end cheolsu_number_problem_l3603_360371


namespace division_result_l3603_360325

theorem division_result : (2014 : ℕ) / (2 * 2 + 2 * 3 + 3 * 3) = 106 := by
  sorry

end division_result_l3603_360325


namespace paul_candy_count_l3603_360345

theorem paul_candy_count :
  ∀ (chocolate_boxes caramel_boxes pieces_per_box : ℕ),
    chocolate_boxes = 6 →
    caramel_boxes = 4 →
    pieces_per_box = 9 →
    chocolate_boxes * pieces_per_box + caramel_boxes * pieces_per_box = 90 :=
by
  sorry

end paul_candy_count_l3603_360345


namespace min_value_reciprocal_l3603_360387

theorem min_value_reciprocal (a b : ℝ) (h1 : a + a * b + 2 * b = 30) (h2 : a > 0) (h3 : b > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + x * y + 2 * y = 30 → 1 / (a * b) ≤ 1 / (x * y)) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + x * y + 2 * y = 30 ∧ 1 / (x * y) = 1 / 18) :=
by sorry

end min_value_reciprocal_l3603_360387


namespace interval_length_theorem_l3603_360350

theorem interval_length_theorem (a b : ℝ) : 
  (∃ x : ℝ, a ≤ 2*x + 3 ∧ 2*x + 3 ≤ b) ∧ 
  ((b - 3) / 2 - (a - 3) / 2 = 10) → 
  b - a = 20 := by
sorry

end interval_length_theorem_l3603_360350


namespace projection_equality_l3603_360309

/-- Given two vectors in R^2 that project to the same vector, 
    prove that the projection is (16/5, 8/5) -/
theorem projection_equality (v : ℝ × ℝ) :
  let a : ℝ × ℝ := (5, -2)
  let b : ℝ × ℝ := (2, 4)
  let proj (x : ℝ × ℝ) := 
    let dot_prod := x.1 * v.1 + x.2 * v.2
    let v_norm_sq := v.1 * v.1 + v.2 * v.2
    ((dot_prod / v_norm_sq) * v.1, (dot_prod / v_norm_sq) * v.2)
  proj a = proj b → proj a = (16/5, 8/5) :=
by
  sorry

#check projection_equality

end projection_equality_l3603_360309


namespace problem_solution_l3603_360356

theorem problem_solution :
  (∃ m_max : ℝ, 
    (∀ m : ℝ, (∀ x : ℝ, |x + 3| + |x + m| ≥ 2 * m) → m ≤ m_max) ∧
    (∀ x : ℝ, |x + 3| + |x + m_max| ≥ 2 * m_max) ∧
    m_max = 1) ∧
  (∀ a b c : ℝ, 
    a > 0 → b > 0 → c > 0 → a + b + c = 1 →
    2 * a^2 + 3 * b^2 + 4 * c^2 ≥ 12/13 ∧
    (2 * a^2 + 3 * b^2 + 4 * c^2 = 12/13 ↔ a = 6/13 ∧ b = 4/13 ∧ c = 3/13)) :=
by sorry

end problem_solution_l3603_360356


namespace solution_difference_l3603_360398

theorem solution_difference (r s : ℝ) : 
  r ≠ s ∧ 
  (r - 5) * (r + 5) = 25 * r - 125 ∧
  (s - 5) * (s + 5) = 25 * s - 125 ∧
  r > s →
  r - s = 15 := by sorry

end solution_difference_l3603_360398


namespace solution_replacement_l3603_360336

theorem solution_replacement (initial_volume : ℝ) (initial_concentration : ℝ) 
  (replacement_concentration : ℝ) (final_concentration : ℝ) 
  (h1 : initial_volume = 100)
  (h2 : initial_concentration = 0.4)
  (h3 : replacement_concentration = 0.25)
  (h4 : final_concentration = 0.35) :
  ∃ (replaced_volume : ℝ), 
    replaced_volume / initial_volume = 1 / 3 ∧
    initial_volume * initial_concentration - replaced_volume * initial_concentration + 
    replaced_volume * replacement_concentration = 
    initial_volume * final_concentration :=
by sorry

end solution_replacement_l3603_360336


namespace total_rabbits_l3603_360333

theorem total_rabbits (white_rabbits black_rabbits : ℕ) 
  (hw : white_rabbits = 15) 
  (hb : black_rabbits = 37) : 
  white_rabbits + black_rabbits = 52 := by
  sorry

end total_rabbits_l3603_360333


namespace distance_between_points_l3603_360354

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (-4, 2)
  let p2 : ℝ × ℝ := (3, 6)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 65 :=
by sorry

end distance_between_points_l3603_360354


namespace p_or_q_true_not_imply_p_and_q_true_l3603_360331

theorem p_or_q_true_not_imply_p_and_q_true (p q : Prop) : 
  (p ∨ q) → ¬(p ∧ q → True) :=
by sorry

end p_or_q_true_not_imply_p_and_q_true_l3603_360331


namespace apple_distribution_l3603_360330

theorem apple_distribution (boxes : Nat) (apples_per_box : Nat) (rotten_apples : Nat) (people : Nat) :
  boxes = 7 →
  apples_per_box = 9 →
  rotten_apples = 7 →
  people = 8 →
  (boxes * apples_per_box - rotten_apples) / people = 7 := by
  sorry

end apple_distribution_l3603_360330


namespace min_sum_of_squares_min_sum_of_squares_achievable_l3603_360369

theorem min_sum_of_squares (a b : ℝ) (h : a * b = -6) : a^2 + b^2 ≥ 12 := by
  sorry

theorem min_sum_of_squares_achievable : ∃ (a b : ℝ), a * b = -6 ∧ a^2 + b^2 = 12 := by
  sorry

end min_sum_of_squares_min_sum_of_squares_achievable_l3603_360369


namespace vasya_driving_distance_l3603_360397

theorem vasya_driving_distance 
  (total_distance : ℝ) 
  (anton_distance vasya_distance sasha_distance dima_distance : ℝ) 
  (h1 : anton_distance = vasya_distance / 2)
  (h2 : sasha_distance = anton_distance + dima_distance)
  (h3 : dima_distance = total_distance / 10)
  (h4 : anton_distance + vasya_distance + sasha_distance + dima_distance = total_distance)
  : vasya_distance = (2 / 5) * total_distance :=
by sorry

end vasya_driving_distance_l3603_360397


namespace towel_purchase_cost_is_correct_l3603_360311

/-- Calculates the total cost of Bailey's towel purchase --/
def towel_purchase_cost : ℝ :=
  let guest_price := 40
  let master_price := 50
  let hand_price := 30
  let kitchen_price := 20
  
  let guest_discount := 0.15
  let master_discount := 0.20
  let hand_discount := 0.15
  let kitchen_discount := 0.10
  
  let sales_tax := 0.08
  
  let guest_discounted := guest_price * (1 - guest_discount)
  let master_discounted := master_price * (1 - master_discount)
  let hand_discounted := hand_price * (1 - hand_discount)
  let kitchen_discounted := kitchen_price * (1 - kitchen_discount)
  
  let total_before_tax := 
    2 * guest_discounted + 
    4 * master_discounted + 
    3 * hand_discounted + 
    5 * kitchen_discounted
  
  total_before_tax * (1 + sales_tax)

/-- Theorem stating that the total cost of Bailey's towel purchase is $426.06 --/
theorem towel_purchase_cost_is_correct : 
  towel_purchase_cost = 426.06 := by sorry

end towel_purchase_cost_is_correct_l3603_360311


namespace negative_three_is_rational_l3603_360319

theorem negative_three_is_rational : ℚ :=
  sorry

end negative_three_is_rational_l3603_360319


namespace boat_upstream_time_l3603_360385

/-- Proves that the time taken by a boat to cover a distance upstream is 1.5 hours,
    given the conditions of the problem. -/
theorem boat_upstream_time (distance : ℝ) (stream_speed : ℝ) (boat_speed : ℝ) : 
  stream_speed = 3 →
  boat_speed = 15 →
  distance = (boat_speed + stream_speed) * 1 →
  (distance / (boat_speed - stream_speed)) = 1.5 := by
sorry

end boat_upstream_time_l3603_360385


namespace f_strictly_increasing_l3603_360349

-- Define the function f(x) = x³ + x
def f (x : ℝ) : ℝ := x^3 + x

-- Theorem statement
theorem f_strictly_increasing :
  StrictMono f := by sorry

end f_strictly_increasing_l3603_360349


namespace number_of_sides_interior_angle_measure_l3603_360326

/-- 
A regular polygon where the sum of interior angles is 4 times the sum of exterior angles.
-/
structure RegularPolygon where
  n : ℕ  -- number of sides
  sum_interior_angles : ℝ
  sum_exterior_angles : ℝ
  h1 : sum_interior_angles = (n - 2) * 180
  h2 : sum_exterior_angles = 360
  h3 : sum_interior_angles = 4 * sum_exterior_angles

/-- The number of sides of the regular polygon is 10. -/
theorem number_of_sides (p : RegularPolygon) : p.n = 10 := by
  sorry

/-- The measure of each interior angle of the regular polygon is 144°. -/
theorem interior_angle_measure (p : RegularPolygon) : 
  (p.n - 2) * 180 / p.n = 144 := by
  sorry

end number_of_sides_interior_angle_measure_l3603_360326


namespace max_value_of_fraction_l3603_360384

theorem max_value_of_fraction (k : ℝ) (h : k > 0) :
  (3 * k^3 + 3 * k) / ((3/2 * k^2 + 14) * (14 * k^2 + 3/2)) ≤ Real.sqrt 21 / 175 :=
by sorry

end max_value_of_fraction_l3603_360384


namespace positive_solution_equation_l3603_360323

theorem positive_solution_equation (x : ℝ) :
  x = 20 + Real.sqrt 409 →
  x > 0 ∧
  (1 / 3) * (2 * x^2 + 3) = (x^2 - 40 * x - 8) * (x^2 + 20 * x + 4) :=
by sorry

end positive_solution_equation_l3603_360323


namespace class_overlap_difference_l3603_360318

theorem class_overlap_difference (total : ℕ) (geometry : ℕ) (biology : ℕ)
  (h1 : total = 232)
  (h2 : geometry = 144)
  (h3 : biology = 119)
  (h4 : geometry ≤ total)
  (h5 : biology ≤ total) :
  min geometry biology - max 0 (geometry + biology - total) = 88 :=
by sorry

end class_overlap_difference_l3603_360318


namespace vowel_writing_count_l3603_360377

theorem vowel_writing_count (num_vowels : ℕ) (total_alphabets : ℕ) : 
  num_vowels = 5 → 
  total_alphabets = 10 → 
  ∃ (times_written : ℕ), times_written * num_vowels = total_alphabets ∧ times_written = 2 :=
by sorry

end vowel_writing_count_l3603_360377


namespace product_increase_by_three_times_l3603_360308

theorem product_increase_by_three_times : 
  ∃ (a b c d : ℕ), (a + 1) * (b + 1) * (c + 1) * (d + 1) = 3 * (a * b * c * d) := by
  sorry

end product_increase_by_three_times_l3603_360308


namespace rectangular_frame_properties_l3603_360373

/-- Calculates the total length of wire needed for a rectangular frame --/
def total_wire_length (a b c : ℕ) : ℕ := 4 * (a + b + c)

/-- Calculates the total area of paper needed to cover a rectangular frame --/
def total_paper_area (a b c : ℕ) : ℕ := 2 * (a * b + a * c + b * c)

theorem rectangular_frame_properties :
  total_wire_length 3 4 5 = 48 ∧ total_paper_area 3 4 5 = 94 := by
  sorry

end rectangular_frame_properties_l3603_360373


namespace three_digit_powers_of_two_l3603_360393

theorem three_digit_powers_of_two (n : ℕ) : 
  (∃ k, 100 ≤ 2^k ∧ 2^k ≤ 999) ↔ (n = 7 ∨ n = 8 ∨ n = 9) :=
sorry

end three_digit_powers_of_two_l3603_360393


namespace women_in_room_l3603_360317

theorem women_in_room (initial_men : ℕ) (initial_women : ℕ) : 
  initial_men * 5 = initial_women * 4 →
  (initial_men + 2) = 14 →
  (2 * (initial_women - 3)) = 24 :=
by
  sorry

end women_in_room_l3603_360317


namespace bus_cost_proof_l3603_360394

-- Define the cost of a bus ride
def bus_cost : ℝ := 3.75

-- Define the cost of a train ride
def train_cost : ℝ := bus_cost + 2.35

-- Theorem stating the conditions and the result to be proved
theorem bus_cost_proof :
  (train_cost = bus_cost + 2.35) ∧
  (train_cost + bus_cost = 9.85) →
  bus_cost = 3.75 :=
by
  sorry

end bus_cost_proof_l3603_360394


namespace L₂_equations_l3603_360366

noncomputable section

-- Define the line L₁
def L₁ : Set (ℝ × ℝ) := {p : ℝ × ℝ | 6 * p.1 - p.2 + 6 = 0}

-- Define points P and Q
def P : ℝ × ℝ := (-1, 0)
def Q : ℝ × ℝ := (0, 6)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define a general line L₂ passing through (1,0)
def L₂ (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = m * p.1 - m}

-- Define point R as the intersection of L₂ and y-axis
def R (m : ℝ) : ℝ × ℝ := (0, -m)

-- Define point S as the intersection of L₁ and L₂
def S (m : ℝ) : ℝ × ℝ := ((-m - 6) / (6 - m), (-12 * m) / (6 - m))

-- Define the area of a triangle given three points
def triangleArea (a b c : ℝ × ℝ) : ℝ :=
  (1/2) * abs (a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2))

-- State the theorem
theorem L₂_equations : 
  ∀ m : ℝ, (triangleArea O P Q = 6 * triangleArea Q (R m) (S m)) → 
  (m = -3 ∨ m = -10) :=
sorry

end L₂_equations_l3603_360366


namespace f_monotone_increasing_l3603_360383

noncomputable def f (x : ℝ) := Real.log (x^2 + 2*x - 3) / Real.log (1/2)

theorem f_monotone_increasing :
  StrictMonoOn f (Set.Iio (-3)) := by sorry

end f_monotone_increasing_l3603_360383


namespace one_and_one_third_of_number_is_48_l3603_360376

theorem one_and_one_third_of_number_is_48 :
  ∃ x : ℚ, (4 / 3) * x = 48 ∧ x = 36 := by
  sorry

end one_and_one_third_of_number_is_48_l3603_360376


namespace rectangle_triangle_area_ratio_l3603_360312

/-- The ratio of the area of a rectangle to the area of a triangle -/
theorem rectangle_triangle_area_ratio 
  (rectangle_length : ℝ) 
  (rectangle_width : ℝ) 
  (triangle_area : ℝ) 
  (h1 : rectangle_length = 6)
  (h2 : rectangle_width = 4)
  (h3 : triangle_area = 60) :
  (rectangle_length * rectangle_width) / triangle_area = 2 / 5 := by
  sorry

end rectangle_triangle_area_ratio_l3603_360312


namespace powderman_distance_powderman_runs_185_yards_l3603_360388

/-- The distance in yards that a powderman runs when he hears a blast, given specific conditions -/
theorem powderman_distance (fuse_time reaction_time : ℝ) (run_speed : ℝ) (sound_speed : ℝ) : ℝ :=
  let blast_time := fuse_time
  let powderman_speed_ft_per_sec := run_speed * 3 -- Convert yards/sec to feet/sec
  let time_of_hearing := (sound_speed * blast_time + powderman_speed_ft_per_sec * reaction_time) / (sound_speed - powderman_speed_ft_per_sec)
  let distance_ft := powderman_speed_ft_per_sec * (time_of_hearing - reaction_time)
  let distance_yd := distance_ft / 3
  distance_yd

/-- The powderman runs 185 yards before hearing the blast under the given conditions -/
theorem powderman_runs_185_yards : 
  powderman_distance 20 2 10 1100 = 185 := by
  sorry


end powderman_distance_powderman_runs_185_yards_l3603_360388


namespace fraction_equality_l3603_360329

theorem fraction_equality : (5 * 3 + 4) / 7 = 19 / 7 := by
  sorry

end fraction_equality_l3603_360329


namespace rectangle_area_l3603_360370

/-- Given a rectangle where the length is five times the width and the perimeter is 180 cm,
    prove that its area is 1125 square centimeters. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 5 * w
  2 * l + 2 * w = 180 → l * w = 1125 := by sorry

end rectangle_area_l3603_360370


namespace diameter_in_scientific_notation_l3603_360374

def scientific_notation (n : ℝ) (a : ℝ) (b : ℤ) : Prop :=
  n = a * (10 : ℝ) ^ b ∧ 1 ≤ a ∧ a < 10

theorem diameter_in_scientific_notation :
  scientific_notation 0.0000077 7.7 (-6) :=
sorry

end diameter_in_scientific_notation_l3603_360374


namespace grapes_cost_proof_l3603_360314

/-- The amount Alyssa paid for grapes -/
def grapes_cost : ℝ := 12.08

/-- The amount Alyssa paid for cherries -/
def cherries_cost : ℝ := 9.85

/-- The total amount Alyssa spent -/
def total_cost : ℝ := 21.93

/-- Theorem: Given the total cost and the cost of cherries, prove that the cost of grapes is correct -/
theorem grapes_cost_proof : grapes_cost = total_cost - cherries_cost := by
  sorry

end grapes_cost_proof_l3603_360314


namespace distinct_parenthesizations_l3603_360380

-- Define a function to represent exponentiation
def exp (a : ℕ) (b : ℕ) : ℕ := a ^ b

-- Define the five possible parenthesizations
def p1 : ℕ := exp 3 (exp 3 (exp 3 3))
def p2 : ℕ := exp 3 ((exp 3 3) ^ 3)
def p3 : ℕ := ((exp 3 3) ^ 3) ^ 3
def p4 : ℕ := (exp 3 (exp 3 3)) ^ 3
def p5 : ℕ := (exp 3 3) ^ (exp 3 3)

-- Theorem stating that there are exactly 5 distinct values
theorem distinct_parenthesizations :
  ∃! (s : Finset ℕ), s = {p1, p2, p3, p4, p5} ∧ s.card = 5 :=
sorry

end distinct_parenthesizations_l3603_360380


namespace divisible_by_nine_l3603_360364

theorem divisible_by_nine (k : ℕ+) : 9 ∣ (3 * (2 + 7^(k : ℕ))) := by sorry

end divisible_by_nine_l3603_360364


namespace max_k_value_l3603_360328

theorem max_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 - 8*x + 15 = 0 ∧ 
   y = k*x - 2 ∧ 
   ∃ cx cy : ℝ, cy = k*cx - 2 ∧ 
   (cx - x)^2 + (cy - y)^2 ≤ 1) → 
  k ≤ 4/3 :=
sorry

end max_k_value_l3603_360328


namespace fraction_to_decimal_decimal_representation_three_twentieths_decimal_l3603_360363

theorem fraction_to_decimal :
  (3 : ℚ) / 20 = (15 : ℚ) / 100 := by sorry

theorem decimal_representation :
  (15 : ℚ) / 100 = 0.15 := by sorry

theorem three_twentieths_decimal :
  (3 : ℚ) / 20 = 0.15 := by sorry

end fraction_to_decimal_decimal_representation_three_twentieths_decimal_l3603_360363


namespace regular_star_n_value_l3603_360302

/-- Represents an n-pointed regular star diagram -/
structure RegularStar where
  n : ℕ
  edge_length : ℝ
  angle_A : ℝ
  angle_B : ℝ

/-- The properties of the regular star diagram -/
def is_valid_regular_star (star : RegularStar) : Prop :=
  star.n > 0 ∧
  star.edge_length > 0 ∧
  star.angle_A > 0 ∧
  star.angle_B > 0 ∧
  star.angle_A = (5 / 14) * star.angle_B ∧
  star.n * (star.angle_A + star.angle_B) = 360

theorem regular_star_n_value (star : RegularStar) 
  (h : is_valid_regular_star star) : star.n = 133 := by
  sorry

#check regular_star_n_value

end regular_star_n_value_l3603_360302


namespace students_doing_hula_hoops_l3603_360338

theorem students_doing_hula_hoops 
  (jumping_rope : ℕ) 
  (hula_hoop_ratio : ℕ) 
  (h1 : jumping_rope = 7)
  (h2 : hula_hoop_ratio = 5) :
  jumping_rope * hula_hoop_ratio = 35 := by
  sorry

end students_doing_hula_hoops_l3603_360338


namespace accessory_time_is_ten_l3603_360304

/-- Represents the production details of a doll factory --/
structure DollFactory where
  num_dolls : ℕ
  time_per_doll : ℕ
  total_time : ℕ
  shoes_per_doll : ℕ
  bags_per_doll : ℕ
  cosmetics_per_doll : ℕ
  hats_per_doll : ℕ

/-- Calculates the time taken to make each accessory --/
def time_per_accessory (factory : DollFactory) : ℕ :=
  let total_accessories := factory.num_dolls * (factory.shoes_per_doll + factory.bags_per_doll + 
                           factory.cosmetics_per_doll + factory.hats_per_doll)
  let time_for_dolls := factory.num_dolls * factory.time_per_doll
  let time_for_accessories := factory.total_time - time_for_dolls
  time_for_accessories / total_accessories

/-- Theorem stating that the time to make each accessory is 10 seconds --/
theorem accessory_time_is_ten (factory : DollFactory) 
  (h1 : factory.num_dolls = 12000)
  (h2 : factory.time_per_doll = 45)
  (h3 : factory.total_time = 1860000)
  (h4 : factory.shoes_per_doll = 2)
  (h5 : factory.bags_per_doll = 3)
  (h6 : factory.cosmetics_per_doll = 1)
  (h7 : factory.hats_per_doll = 5) :
  time_per_accessory factory = 10 := by
  sorry

#eval time_per_accessory { 
  num_dolls := 12000, 
  time_per_doll := 45, 
  total_time := 1860000, 
  shoes_per_doll := 2, 
  bags_per_doll := 3, 
  cosmetics_per_doll := 1, 
  hats_per_doll := 5 
}

end accessory_time_is_ten_l3603_360304


namespace quadrilateral_is_rhombus_l3603_360340

theorem quadrilateral_is_rhombus (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 = a*b + b*c + c*d + d*a) : 
  a = b ∧ b = c ∧ c = d := by
  sorry

-- The theorem states that if the given condition is true,
-- then all sides of the quadrilateral are equal,
-- which is the definition of a rhombus.

end quadrilateral_is_rhombus_l3603_360340


namespace gcd_lcm_1729_867_l3603_360359

theorem gcd_lcm_1729_867 :
  (Nat.gcd 1729 867 = 1) ∧ (Nat.lcm 1729 867 = 1499003) := by
  sorry

end gcd_lcm_1729_867_l3603_360359


namespace inscribed_square_area_bound_l3603_360348

-- Define an acute triangle
def AcuteTriangle (A B C : Point) : Prop := sorry

-- Define a square
def Square (M N P Q : Point) : Prop := sorry

-- Define a point being on a line segment
def PointOnSegment (P A B : Point) : Prop := sorry

-- Define the area of a polygon
def Area (polygon : Set Point) : ℝ := sorry

theorem inscribed_square_area_bound 
  (A B C M N P Q : Point) 
  (h_acute : AcuteTriangle A B C)
  (h_square : Square M N P Q)
  (h_inscribed : PointOnSegment M B C ∧ PointOnSegment N B C ∧ 
                 PointOnSegment P A C ∧ PointOnSegment Q A B) :
  Area {M, N, P, Q} ≤ (1/2) * Area {A, B, C} := by
  sorry

end inscribed_square_area_bound_l3603_360348


namespace eight_and_half_minutes_in_seconds_l3603_360386

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes we're converting -/
def minutes : ℚ := 8.5

/-- Converts minutes to seconds -/
def minutes_to_seconds (m : ℚ) : ℚ := m * seconds_per_minute

theorem eight_and_half_minutes_in_seconds :
  minutes_to_seconds minutes = 510 := by
  sorry

end eight_and_half_minutes_in_seconds_l3603_360386


namespace apple_difference_l3603_360306

theorem apple_difference (total : ℕ) (red : ℕ) (h1 : total = 44) (h2 : red = 16) :
  total > red → total - red - red = 12 := by
  sorry

end apple_difference_l3603_360306


namespace cubic_factorization_l3603_360346

theorem cubic_factorization (m : ℝ) : m^3 - 4*m = m*(m - 2)*(m + 2) := by
  sorry

end cubic_factorization_l3603_360346


namespace inequality_proof_l3603_360357

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) := by
  sorry

end inequality_proof_l3603_360357


namespace geometric_arithmetic_ratio_l3603_360301

/-- Given a geometric sequence {a_n} with common ratio q (q ≠ 1),
    if a_1, a_3, a_2 form an arithmetic sequence, then q = -1/2 -/
theorem geometric_arithmetic_ratio (a : ℕ → ℝ) (q : ℝ) (h1 : q ≠ 1)
    (h2 : ∀ n, a (n + 1) = a n * q)  -- geometric sequence condition
    (h3 : 2 * a 3 = a 1 + a 2)       -- arithmetic sequence condition
    : q = -1/2 := by
  sorry

end geometric_arithmetic_ratio_l3603_360301


namespace fifth_group_sample_l3603_360355

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population : ℕ
  num_groups : ℕ
  group_size : ℕ
  first_sample : ℕ

/-- Calculates the sample number for a given group in a systematic sampling scenario -/
def sample_number (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.first_sample + (group - 1) * s.group_size

/-- Theorem: In the given systematic sampling scenario, the sample number in the fifth group is 43 -/
theorem fifth_group_sample (s : SystematicSampling) 
  (h1 : s.population = 60)
  (h2 : s.num_groups = 6)
  (h3 : s.group_size = s.population / s.num_groups)
  (h4 : s.first_sample = 3) :
  sample_number s 5 = 43 := by
  sorry


end fifth_group_sample_l3603_360355


namespace wilmas_garden_rows_l3603_360367

/-- The number of rows in Wilma's garden --/
def garden_rows : ℕ :=
  let yellow_flowers : ℕ := 12
  let green_flowers : ℕ := 2 * yellow_flowers
  let red_flowers : ℕ := 42
  let total_flowers : ℕ := yellow_flowers + green_flowers + red_flowers
  let flowers_per_row : ℕ := 13
  total_flowers / flowers_per_row

/-- Theorem stating that the number of rows in Wilma's garden is 6 --/
theorem wilmas_garden_rows :
  garden_rows = 6 := by
  sorry

end wilmas_garden_rows_l3603_360367


namespace sequence_min_value_and_ratio_l3603_360390

/-- Given a positive integer m ≥ 3, an arithmetic sequence {a_n} with positive terms,
    and a geometric sequence {b_n} with positive terms, such that:
    1. The first term of {a_n} equals the common ratio of {b_n}
    2. The first term of {b_n} equals the common difference of {a_n}
    3. a_m = b_m
    This theorem proves the minimum value of a_m and the ratio of a_1 to b_1 when a_m is minimum. -/
theorem sequence_min_value_and_ratio (m : ℕ) (a b : ℝ → ℝ) (h_m : m ≥ 3) 
  (h_a_pos : ∀ n, a n > 0) (h_b_pos : ∀ n, b n > 0)
  (h_a_arith : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_b_geom : ∀ n, b (n + 1) / b n = b 2 / b 1)
  (h_first_term : a 1 = b 2 / b 1)
  (h_common_diff : b 1 = a 2 - a 1)
  (h_m_equal : a m = b m) :
  ∃ (min_am : ℝ) (ratio : ℝ),
    min_am = ((m^m : ℝ) / ((m - 1 : ℝ)^(m - 2)))^(1 / (m - 1 : ℝ)) ∧
    ratio = (m - 1 : ℝ)^2 ∧
    a m ≥ min_am ∧
    (a m = min_am → a 1 / b 1 = ratio) := by
  sorry

end sequence_min_value_and_ratio_l3603_360390


namespace geometric_sequence_sum_l3603_360391

/-- A geometric sequence with real number terms -/
def GeometricSequence := ℕ → ℝ

/-- Sum of the first n terms of a geometric sequence -/
def SumN (a : GeometricSequence) (n : ℕ) : ℝ := sorry

theorem geometric_sequence_sum (a : GeometricSequence) :
  SumN a 10 = 10 →
  SumN a 30 = 70 →
  SumN a 40 = 150 := by sorry

end geometric_sequence_sum_l3603_360391


namespace equilateral_triangle_on_parabola_l3603_360347

/-- Given points A and B on the parabola y = -x^2 forming an equilateral triangle with the origin,
    prove that their x-coordinates are ±√3 and the side length is 2√3. -/
theorem equilateral_triangle_on_parabola :
  ∀ (a : ℝ),
  let A : ℝ × ℝ := (a, -a^2)
  let B : ℝ × ℝ := (-a, -a^2)
  let O : ℝ × ℝ := (0, 0)
  -- Distance between two points (x₁, y₁) and (x₂, y₂)
  let dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  -- Condition for equilateral triangle
  (dist A O = dist B O ∧ dist A O = dist A B) →
  (a = Real.sqrt 3 ∧ dist A O = 2 * Real.sqrt 3) :=
by sorry

end equilateral_triangle_on_parabola_l3603_360347


namespace medicine_price_reduction_l3603_360320

/-- Represents the price reduction equation for a medicine that undergoes two
    successive price reductions of the same percentage. -/
theorem medicine_price_reduction (x : ℝ) : 
  (58 : ℝ) * (1 - x)^2 = 43 ↔ 
  (∃ (initial_price final_price : ℝ),
    initial_price = 58 ∧
    final_price = 43 ∧
    final_price = initial_price * (1 - x)^2) :=
by sorry

end medicine_price_reduction_l3603_360320
