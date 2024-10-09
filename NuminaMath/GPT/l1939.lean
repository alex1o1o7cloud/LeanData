import Mathlib

namespace rulers_left_l1939_193942

variable (rulers_in_drawer : Nat)
variable (rulers_taken : Nat)

theorem rulers_left (h1 : rulers_in_drawer = 46) (h2 : rulers_taken = 25) : 
  rulers_in_drawer - rulers_taken = 21 := by
  sorry

end rulers_left_l1939_193942


namespace age_difference_l1939_193995

theorem age_difference :
  let x := 5
  let prod_today := x * x
  let prod_future := (x + 1) * (x + 1)
  prod_future - prod_today = 11 :=
by
  sorry

end age_difference_l1939_193995


namespace laptop_total_selling_price_l1939_193958

-- Define the original price of the laptop
def originalPrice : ℝ := 1200

-- Define the discount rate
def discountRate : ℝ := 0.30

-- Define the redemption coupon amount
def coupon : ℝ := 50

-- Define the tax rate
def taxRate : ℝ := 0.15

-- Calculate the discount amount
def discountAmount : ℝ := originalPrice * discountRate

-- Calculate the sale price after discount
def salePrice : ℝ := originalPrice - discountAmount

-- Calculate the new sale price after applying the coupon
def newSalePrice : ℝ := salePrice - coupon

-- Calculate the tax amount
def taxAmount : ℝ := newSalePrice * taxRate

-- Calculate the total selling price after tax
def totalSellingPrice : ℝ := newSalePrice + taxAmount

-- Prove that the total selling price is 908.5 dollars
theorem laptop_total_selling_price : totalSellingPrice = 908.5 := by
  unfold totalSellingPrice newSalePrice taxAmount salePrice discountAmount
  norm_num
  sorry

end laptop_total_selling_price_l1939_193958


namespace sheets_paper_150_l1939_193977

def num_sheets_of_paper (S : ℕ) (E : ℕ) : Prop :=
  (S - E = 50) ∧ (3 * E - S = 150)

theorem sheets_paper_150 (S E : ℕ) : num_sheets_of_paper S E → S = 150 :=
by
  sorry

end sheets_paper_150_l1939_193977


namespace probability_at_least_one_admitted_l1939_193935

-- Define the events and probabilities
variables (A B : Prop)
variables (P_A : ℝ) (P_B : ℝ)
variables (independent : Prop)

-- Assume the given conditions
def P_A_def : Prop := P_A = 0.6
def P_B_def : Prop := P_B = 0.7
def independent_def : Prop := independent = true  -- simplistic representation for independence

-- Statement: Prove the probability that at least one of them is admitted is 0.88
theorem probability_at_least_one_admitted : 
  P_A = 0.6 → P_B = 0.7 → independent = true →
  (1 - (1 - P_A) * (1 - P_B)) = 0.88 :=
by
  intros
  sorry

end probability_at_least_one_admitted_l1939_193935


namespace sum_of_two_digit_divisors_l1939_193992

theorem sum_of_two_digit_divisors (d : ℕ) (h1 : 145 % d = 4) (h2 : 10 ≤ d ∧ d < 100) :
  d = 47 :=
by
  have hd : d ∣ 141 := sorry
  exact sorry

end sum_of_two_digit_divisors_l1939_193992


namespace tangent_line_and_curve_l1939_193988

theorem tangent_line_and_curve (a x0 : ℝ) 
  (h1 : ∀ (x : ℝ), x0 + a = 1) 
  (h2 : ∀ (y : ℝ), y = x0 + 1) 
  (h3 : ∀ (y : ℝ), y = Real.log (x0 + a)) 
  : a = 2 := 
by 
  sorry

end tangent_line_and_curve_l1939_193988


namespace volume_conversion_l1939_193927

theorem volume_conversion (a : Nat) (b : Nat) (c : Nat) (d : Nat) (e : Nat) (f : Nat)
  (h1 : a = 1) (h2 : b = 3) (h3 : c = a^3) (h4 : d = b^3) (h5 : c = 1) (h6 : d = 27) 
  (h7 : 1 = 1) (h8 : 27 = 27) (h9 : e = 5) 
  (h10 : f = e * d) : 
  f = 135 := 
sorry

end volume_conversion_l1939_193927


namespace construct_rectangle_l1939_193956

structure Rectangle where
  side1 : ℝ
  side2 : ℝ
  diagonal : ℝ
  sum_diag_side : ℝ := side2 + diagonal

theorem construct_rectangle (b a d : ℝ) (r : Rectangle) :
  r.side2 = a ∧ r.side1 = b ∧ r.sum_diag_side = a + d :=
by
  sorry

end construct_rectangle_l1939_193956


namespace m_over_n_add_one_l1939_193984

theorem m_over_n_add_one (m n : ℕ) (h : (m : ℚ) / n = 3 / 7) : (m + n : ℚ) / n = 10 / 7 :=
by
  sorry

end m_over_n_add_one_l1939_193984


namespace joan_remaining_kittens_l1939_193998

-- Definitions based on the given conditions
def original_kittens : Nat := 8
def kittens_given_away : Nat := 2

-- Statement to prove
theorem joan_remaining_kittens : original_kittens - kittens_given_away = 6 := 
by
  -- Proof skipped
  sorry

end joan_remaining_kittens_l1939_193998


namespace contradiction_of_distinct_roots_l1939_193902

theorem contradiction_of_distinct_roots
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (distinct_abc : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (H : ¬ (∃ x1 x2, x1 ≠ x2 ∧ (a * x1^2 + 2 * b * x1 + c = 0 ∨ b * x1^2 + 2 * c * x1 + a = 0 ∨ c * x1^2 + 2 * a * x1 + b = 0))) :
  False := 
sorry

end contradiction_of_distinct_roots_l1939_193902


namespace inequality_solution_l1939_193924

theorem inequality_solution (x : ℝ) : 
  (0 < x ∧ x ≤ 3) ∨ (4 ≤ x) ↔ (3 * (x - 3) * (x - 4)) / x ≥ 0 := 
sorry

end inequality_solution_l1939_193924


namespace smallest_n_l1939_193919

theorem smallest_n (n : ℕ) (h₁ : n > 2016) (h₂ : n % 4 = 0) : 
  ¬(1^n + 2^n + 3^n + 4^n) % 10 = 0 → n = 2020 :=
by
  sorry

end smallest_n_l1939_193919


namespace steak_and_egg_meal_cost_is_16_l1939_193950

noncomputable def steak_and_egg_cost (x : ℝ) := 
  (x + 14) / 2 + 0.20 * (x + 14) = 21

theorem steak_and_egg_meal_cost_is_16 (x : ℝ) (h : steak_and_egg_cost x) : x = 16 := 
by 
  sorry

end steak_and_egg_meal_cost_is_16_l1939_193950


namespace baker_cakes_l1939_193930

theorem baker_cakes (P x : ℝ) (h1 : P * x = 320) (h2 : 0.80 * P * (x + 2) = 320) : x = 8 :=
by
  sorry

end baker_cakes_l1939_193930


namespace intersection_P_Q_l1939_193903

def setP : Set ℝ := {1, 2, 3, 4}
def setQ : Set ℝ := {x | abs x ≤ 2}

theorem intersection_P_Q : (setP ∩ setQ) = {1, 2} :=
by
  sorry

end intersection_P_Q_l1939_193903


namespace maximum_rubles_l1939_193953

-- We define the initial number of '1' and '2' cards
def num_ones : ℕ := 2013
def num_twos : ℕ := 2013
def total_digits : ℕ := num_ones + num_twos

-- Definition of the problem statement
def problem_statement : Prop :=
  ∃ (max_rubles : ℕ), 
    max_rubles = 5 ∧
    ∀ (current_k : ℕ), 
      current_k = 5 → 
      ∃ (moves : ℕ), 
        moves ≤ max_rubles ∧
        (current_k - moves * 2) % 11 = 0

-- The expected solution is proving the maximum rubles is 5
theorem maximum_rubles : problem_statement :=
by
  sorry

end maximum_rubles_l1939_193953


namespace max_side_length_of_triangle_l1939_193923

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end max_side_length_of_triangle_l1939_193923


namespace regular_dodecahedron_edges_l1939_193960

-- Define a regular dodecahedron as a type
inductive RegularDodecahedron : Type
| mk : RegularDodecahedron

-- Define a function that returns the number of edges for a regular dodecahedron
def numberOfEdges (d : RegularDodecahedron) : Nat :=
  30

-- The mathematical statement to be proved
theorem regular_dodecahedron_edges (d : RegularDodecahedron) : numberOfEdges d = 30 := by
  sorry

end regular_dodecahedron_edges_l1939_193960


namespace half_difference_donation_l1939_193970

def margoDonation : ℝ := 4300
def julieDonation : ℝ := 4700

theorem half_difference_donation : (julieDonation - margoDonation) / 2 = 200 := by
  sorry

end half_difference_donation_l1939_193970


namespace largest_possible_m_l1939_193967

theorem largest_possible_m (x y : ℕ) (h1 : x > y) (hx : Nat.Prime x) (hy : Nat.Prime y) (hxy : x < 10) (hyy : y < 10) (h_prime_10xy : Nat.Prime (10 * x + y)) : ∃ m : ℕ, m = x * y * (10 * x + y) ∧ 1000 ≤ m ∧ m ≤ 9999 ∧ ∀ n : ℕ, (n = x * y * (10 * x + y) ∧ 1000 ≤ n ∧ n ≤ 9999) → n ≤ 1533 :=
by
  sorry

end largest_possible_m_l1939_193967


namespace triangle_with_ratio_is_right_triangle_l1939_193900

/-- If the ratio of the interior angles of a triangle is 1:2:3, then the triangle is a right triangle. -/
theorem triangle_with_ratio_is_right_triangle (x : ℝ) (h : x + 2*x + 3*x = 180) : 
  3*x = 90 :=
sorry

end triangle_with_ratio_is_right_triangle_l1939_193900


namespace calculate_expression_l1939_193922

theorem calculate_expression :
  2 * Real.sin (60 * Real.pi / 180) + abs (Real.sqrt 3 - 3) + (Real.pi - 1)^0 = 4 :=
by
  sorry

end calculate_expression_l1939_193922


namespace gcd_of_polynomials_l1939_193917

theorem gcd_of_polynomials (b : ℤ) (h : 2460 ∣ b) : 
  Int.gcd (b^2 + 6 * b + 30) (b + 5) = 30 :=
sorry

end gcd_of_polynomials_l1939_193917


namespace radius_of_outer_circle_l1939_193993

theorem radius_of_outer_circle (C_inner : ℝ) (width : ℝ) (h : C_inner = 880) (w : width = 25) :
  ∃ r_outer : ℝ, r_outer = 165 :=
by
  have r_inner := C_inner / (2 * Real.pi)
  have r_outer := r_inner + width
  use r_outer
  sorry

end radius_of_outer_circle_l1939_193993


namespace lucy_flour_used_l1939_193905

theorem lucy_flour_used
  (initial_flour : ℕ := 500)
  (final_flour : ℕ := 130)
  (flour_needed_to_buy : ℤ := 370)
  (used_flour : ℕ) :
  initial_flour - used_flour = 2 * final_flour → used_flour = 240 :=
by
  sorry

end lucy_flour_used_l1939_193905


namespace sandwich_count_l1939_193980

-- Define the given conditions
def meats : ℕ := 8
def cheeses : ℕ := 12
def cheese_combination_count : ℕ := Nat.choose cheeses 3

-- Define the total sandwich count based on the conditions
def total_sandwiches : ℕ := meats * cheese_combination_count

-- The theorem we want to prove
theorem sandwich_count : total_sandwiches = 1760 := by
  -- Mathematical steps here are omitted
  sorry

end sandwich_count_l1939_193980


namespace subtraction_problem_l1939_193918

variable (x : ℕ) -- Let's assume x is a natural number for this problem

theorem subtraction_problem (h : x - 46 = 15) : x - 29 = 32 := 
by 
  sorry -- Proof to be filled in

end subtraction_problem_l1939_193918


namespace pencil_cost_is_correct_l1939_193907

-- Defining the cost of a pen as x and the cost of a pencil as y in cents
def cost_of_pen_and_pencil (x y : ℕ) : Prop :=
  3 * x + 5 * y = 345 ∧ 4 * x + 2 * y = 280

-- Stating the theorem that proves y = 39
theorem pencil_cost_is_correct (x y : ℕ) (h : cost_of_pen_and_pencil x y) : y = 39 :=
by
  sorry

end pencil_cost_is_correct_l1939_193907


namespace binom_comb_always_integer_l1939_193991

theorem binom_comb_always_integer (k n : ℕ) (h1 : 1 ≤ k) (h2 : k < n) (h3 : (k + 2) ∣ n) : 
  ∃ m : ℤ, ((n - 3 * k - 2) / (k + 2)) * Nat.choose n k = m := 
sorry

end binom_comb_always_integer_l1939_193991


namespace contrapositive_example_l1939_193913

theorem contrapositive_example (a b : ℝ) (h : a^2 + b^2 < 4) : a + b ≠ 3 :=
sorry

end contrapositive_example_l1939_193913


namespace domain_of_function_l1939_193957

-- Definitions based on conditions
def function_domain (x : ℝ) : Prop := (x > -1) ∧ (x ≠ 1)

-- Prove the domain is the desired set
theorem domain_of_function :
  ∀ x, function_domain x ↔ ((-1 < x ∧ x < 1) ∨ (1 < x)) :=
  by
    sorry

end domain_of_function_l1939_193957


namespace find_y_l1939_193949

theorem find_y (t y : ℝ) (h1 : -3 = 2 - t) (h2 : y = 4 * t + 7) : y = 27 :=
sorry

end find_y_l1939_193949


namespace farm_field_ploughing_l1939_193940

theorem farm_field_ploughing (A D : ℕ) 
  (h1 : ∀ farmerA_initial_capacity: ℕ, farmerA_initial_capacity = 120)
  (h2 : ∀ farmerB_initial_capacity: ℕ, farmerB_initial_capacity = 100)
  (h3 : ∀ farmerA_adjustment: ℕ, farmerA_adjustment = 10)
  (h4 : ∀ farmerA_reduced_capacity: ℕ, farmerA_reduced_capacity = farmerA_initial_capacity - (farmerA_adjustment * farmerA_initial_capacity / 100))
  (h5 : ∀ farmerB_reduced_capacity: ℕ, farmerB_reduced_capacity = 90)
  (h6 : ∀ extra_days: ℕ, extra_days = 3)
  (h7 : ∀ remaining_hectares: ℕ, remaining_hectares = 60)
  (h8 : ∀ initial_combined_effort: ℕ, initial_combined_effort = (farmerA_initial_capacity + farmerB_initial_capacity) * D)
  (h9 : ∀ total_combined_effort: ℕ, total_combined_effort = (farmerA_reduced_capacity + farmerB_reduced_capacity) * (D + extra_days))
  (h10 : ∀ area_covered: ℕ, area_covered = total_combined_effort + remaining_hectares)
  : initial_combined_effort = A ∧ D = 30 ∧ A = 6600 :=
by
  sorry

end farm_field_ploughing_l1939_193940


namespace smallest_area_right_triangle_l1939_193933

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l1939_193933


namespace train_crosses_bridge_in_30_seconds_l1939_193979

theorem train_crosses_bridge_in_30_seconds
    (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ)
    (h1 : train_length = 110)
    (h2 : train_speed_kmh = 45)
    (h3 : bridge_length = 265) : 
    (train_length + bridge_length) / (train_speed_kmh * (1000 / 3600)) = 30 := 
by
  sorry

end train_crosses_bridge_in_30_seconds_l1939_193979


namespace total_length_of_board_l1939_193951

-- Define variables for the lengths
variable (S L : ℝ)

-- Given conditions as Lean definitions
def condition1 : Prop := 2 * S = L + 4
def condition2 : Prop := S = 8.0

-- The goal is to prove the total length of the board is 20.0 feet
theorem total_length_of_board (h1 : condition1 S L) (h2 : condition2 S) : S + L = 20.0 := by
  sorry

end total_length_of_board_l1939_193951


namespace oak_grove_total_books_l1939_193999

theorem oak_grove_total_books (public_library_books : ℕ) (school_library_books : ℕ)
  (h1 : public_library_books = 1986) (h2 : school_library_books = 5106) :
  public_library_books + school_library_books = 7092 := by
  sorry

end oak_grove_total_books_l1939_193999


namespace decorations_per_box_l1939_193911

-- Definitions based on given conditions
def used_decorations : ℕ := 35
def given_away_decorations : ℕ := 25
def number_of_boxes : ℕ := 4

-- Theorem stating the problem
theorem decorations_per_box : (used_decorations + given_away_decorations) / number_of_boxes = 15 := by
  sorry

end decorations_per_box_l1939_193911


namespace inequality_subtraction_real_l1939_193965

theorem inequality_subtraction_real (a b c : ℝ) (h : a > b) : a - c > b - c :=
sorry

end inequality_subtraction_real_l1939_193965


namespace line_through_midpoint_bisects_chord_eqn_l1939_193915

theorem line_through_midpoint_bisects_chord_eqn :
  ∀ (x y : ℝ), (x^2 - 4*y^2 = 4) ∧ (∃ x1 y1 x2 y2 : ℝ, 
    (x1^2 - 4 * y1^2 = 4) ∧ (x2^2 - 4 * y2^2 = 4) ∧ 
    (x1 + x2) / 2 = 3 ∧ (y1 + y2) / 2 = -1) → 
    3 * x + 4 * y - 5 = 0 :=
by
  intros x y h
  sorry

end line_through_midpoint_bisects_chord_eqn_l1939_193915


namespace members_playing_both_l1939_193961

theorem members_playing_both
  (N B T Neither : ℕ)
  (hN : N = 40)
  (hB : B = 20)
  (hT : T = 18)
  (hNeither : Neither = 5) :
  (B + T) - (N - Neither) = 3 := by
-- to complete the proof
sorry

end members_playing_both_l1939_193961


namespace need_to_sell_more_rolls_l1939_193994

variable (goal sold_grandmother sold_uncle_1 sold_uncle_additional sold_neighbor_1 returned_neighbor sold_mothers_friend sold_cousin_1 sold_cousin_additional : ℕ)

theorem need_to_sell_more_rolls
  (h_goal : goal = 100)
  (h_sold_grandmother : sold_grandmother = 5)
  (h_sold_uncle_1 : sold_uncle_1 = 12)
  (h_sold_uncle_additional : sold_uncle_additional = 10)
  (h_sold_neighbor_1 : sold_neighbor_1 = 8)
  (h_returned_neighbor : returned_neighbor = 4)
  (h_sold_mothers_friend : sold_mothers_friend = 25)
  (h_sold_cousin_1 : sold_cousin_1 = 3)
  (h_sold_cousin_additional : sold_cousin_additional = 5) :
  goal - (sold_grandmother + (sold_uncle_1 + sold_uncle_additional) + (sold_neighbor_1 - returned_neighbor) + sold_mothers_friend + (sold_cousin_1 + sold_cousin_additional)) = 36 := by
  sorry

end need_to_sell_more_rolls_l1939_193994


namespace range_of_a_l1939_193985

def operation (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, operation (x - a) (x + 1) < 1) ↔ -2 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l1939_193985


namespace percent_deficit_in_width_l1939_193901

theorem percent_deficit_in_width (L W : ℝ) (h : 1.08 * (1 - (d : ℝ) / W) = 1.0044) : d = 0.07 * W :=
by sorry

end percent_deficit_in_width_l1939_193901


namespace sum_of_acute_angles_l1939_193921

theorem sum_of_acute_angles (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ)
  (h1 : angle1 = 30) (h2 : angle2 = 30) (h3 : angle3 = 30) (h4 : angle4 = 30) (h5 : angle5 = 30) (h6 : angle6 = 30) :
  (angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + 
  (angle1 + angle2) + (angle2 + angle3) + (angle3 + angle4) + (angle4 + angle5) + (angle5 + angle6)) = 480 :=
  sorry

end sum_of_acute_angles_l1939_193921


namespace cost_of_two_dogs_l1939_193959

theorem cost_of_two_dogs (original_price : ℤ) (profit_margin : ℤ) (num_dogs : ℤ) (final_price : ℤ) :
  original_price = 1000 →
  profit_margin = 30 →
  num_dogs = 2 →
  final_price = original_price + (profit_margin * original_price / 100) →
  num_dogs * final_price = 2600 :=
by
  sorry

end cost_of_two_dogs_l1939_193959


namespace min_diagonal_length_of_trapezoid_l1939_193964

theorem min_diagonal_length_of_trapezoid (a b h d1 d2 : ℝ) 
  (h_area : a * h + b * h = 2)
  (h_diag : d1^2 + d2^2 = h^2 + (a + b)^2) 
  : d1 ≥ Real.sqrt 2 :=
sorry

end min_diagonal_length_of_trapezoid_l1939_193964


namespace f_is_decreasing_max_k_value_l1939_193908

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x + 1)) / x

theorem f_is_decreasing : ∀ x > 0, (∃ y > x, f y < f x) :=
by
  sorry

theorem max_k_value : ∃ k : ℕ, (∀ x > 0, f x > k / (x + 1)) ∧ k = 3 :=
by
  sorry

end f_is_decreasing_max_k_value_l1939_193908


namespace multiply_vars_l1939_193989

variables {a b : ℝ}

theorem multiply_vars : -3 * a * b * 2 * a = -6 * a^2 * b := by
  sorry

end multiply_vars_l1939_193989


namespace find_triangle_sides_l1939_193910

theorem find_triangle_sides (x y : ℕ) : 
  (x * y = 200) ∧ (x + 2 * y = 50) → ((x = 40 ∧ y = 5) ∨ (x = 10 ∧ y = 20)) := 
by
  intro h
  sorry

end find_triangle_sides_l1939_193910


namespace maximum_smallest_angle_l1939_193973

-- Definition of points on the plane
structure Point2D :=
  (x : ℝ)
  (y : ℝ)

-- Function to calculate the angle between three points (p1, p2, p3)
def angle (p1 p2 p3 : Point2D) : ℝ := 
  -- Placeholder for the actual angle calculation
  sorry

-- Condition: Given five points on a plane
variables (A B C D E : Point2D)

-- Maximum value of the smallest angle formed by any triple is 36 degrees
theorem maximum_smallest_angle :
  ∃ α : ℝ, (∀ p1 p2 p3 : Point2D, α ≤ angle p1 p2 p3) ∧ α = 36 :=
sorry

end maximum_smallest_angle_l1939_193973


namespace even_combinations_result_in_486_l1939_193928

-- Define the operations possible (increase by 2, increase by 3, multiply by 2)
inductive Operation
| inc2
| inc3
| mul2

open Operation

-- Function to apply an operation to a number
def applyOperation : Operation → ℕ → ℕ
| inc2, n => n + 2
| inc3, n => n + 3
| mul2, n => n * 2

-- Function to apply a list of operations to the initial number 1
def applyOperationsList (ops : List Operation) : ℕ :=
ops.foldl (fun acc op => applyOperation op acc) 1

-- Count the number of combinations that result in an even number
noncomputable def evenCombosCount : ℕ :=
(List.replicate 6 [inc2, inc3, mul2]).foldl (fun acc x => acc * x.length) 1
  |> λ _ => (3 ^ 5) * 2

theorem even_combinations_result_in_486 :
  evenCombosCount = 486 :=
sorry

end even_combinations_result_in_486_l1939_193928


namespace sum_a_b_eq_5_l1939_193938

theorem sum_a_b_eq_5 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * b = a - 2) (h4 : (-2)^2 = b * (2 * b + 2)) : a + b = 5 :=
sorry

end sum_a_b_eq_5_l1939_193938


namespace radius_of_larger_circle_l1939_193932

theorem radius_of_larger_circle (r : ℝ) (R : ℝ) 
  (h1 : ∀ a b c : ℝ, a = 2 ∧ b = 2 ∧ c = 2) 
  (h2 : ∀ x y z : ℝ, (x = 4) ∧ (y = 4) ∧ (z = 4) ) 
  (h3 : ∀ A B : ℝ, A * 2 = 2) : 
  R = 2 + 2 * Real.sqrt 3 :=
by
  sorry

end radius_of_larger_circle_l1939_193932


namespace calculation1_calculation2_calculation3_calculation4_l1939_193936

-- Proving the first calculation: 3 * 232 + 456 = 1152
theorem calculation1 : 3 * 232 + 456 = 1152 := 
by 
  sorry

-- Proving the second calculation: 760 * 5 - 2880 = 920
theorem calculation2 : 760 * 5 - 2880 = 920 :=
by 
  sorry

-- Proving the third calculation: 805 / 7 = 115 (integer division)
theorem calculation3 : 805 / 7 = 115 :=
by 
  sorry

-- Proving the fourth calculation: 45 + 255 / 5 = 96
theorem calculation4 : 45 + 255 / 5 = 96 :=
by 
  sorry

end calculation1_calculation2_calculation3_calculation4_l1939_193936


namespace pqrs_l1939_193945

theorem pqrs(p q r s t u : ℤ) :
  (729 * (x : ℤ) * x * x + 64 = (p * x * x + q * x + r) * (s * x * x + t * x + u)) →
  p = 9 → q = 4 → r = 0 → s = 81 → t = -36 → u = 16 →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210 := by
  intros h1 hp hq hr hs ht hu
  sorry

end pqrs_l1939_193945


namespace residue_of_7_pow_2023_mod_19_l1939_193963

theorem residue_of_7_pow_2023_mod_19 : (7^2023) % 19 = 3 :=
by 
  -- The main goal is to construct the proof that matches our explanation.
  sorry

end residue_of_7_pow_2023_mod_19_l1939_193963


namespace shell_highest_point_time_l1939_193962

theorem shell_highest_point_time (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : a * 7^2 + b * 7 + c = a * 14^2 + b * 14 + c) :
  (-b / (2 * a)) = 10.5 :=
by
  -- The proof is omitted as per the instructions
  sorry

end shell_highest_point_time_l1939_193962


namespace quadratic_solution_l1939_193914

theorem quadratic_solution (x : ℝ) : x ^ 2 - 4 * x + 3 = 0 ↔ x = 1 ∨ x = 3 :=
by
  sorry

end quadratic_solution_l1939_193914


namespace hannahs_brothers_l1939_193982

theorem hannahs_brothers (B : ℕ) (h1 : ∀ (b : ℕ), b = 8) (h2 : 48 = 2 * (8 * B)) : B = 3 :=
by
  sorry

end hannahs_brothers_l1939_193982


namespace work_completion_time_l1939_193934

theorem work_completion_time (W : ℝ) : 
  let A_effort := 1 / 11
  let B_effort := 1 / 20
  let C_effort := 1 / 55
  (2 * A_effort + B_effort + C_effort) = 1 / 4 → 
  8 * (2 * A_effort + B_effort + C_effort) = 1 :=
by { sorry }

end work_completion_time_l1939_193934


namespace minimum_value_of_3x_plus_4y_exists_x_y_for_minimum_value_l1939_193920

theorem minimum_value_of_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 3 * x + 4 * y ≥ 5 :=
sorry

theorem exists_x_y_for_minimum_value : ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 3 * y = 5 * x * y ∧ 3 * x + 4 * y = 5 :=
sorry

end minimum_value_of_3x_plus_4y_exists_x_y_for_minimum_value_l1939_193920


namespace bricks_needed_for_wall_l1939_193904

noncomputable def brick_volume (length width height : ℝ) : ℝ :=
  length * width * height

noncomputable def wall_volume (length height thickness : ℝ) : ℝ :=
  length * height * thickness

theorem bricks_needed_for_wall :
  let length_wall := 800
  let height_wall := 600
  let thickness_wall := 22.5
  let length_brick := 100
  let width_brick := 11.25
  let height_brick := 6
  let vol_wall := wall_volume length_wall height_wall thickness_wall
  let vol_brick := brick_volume length_brick width_brick height_brick
  vol_wall / vol_brick = 1600 :=
by
  sorry

end bricks_needed_for_wall_l1939_193904


namespace packs_of_blue_tshirts_l1939_193986

theorem packs_of_blue_tshirts (total_tshirts white_packs white_per_pack blue_per_pack : ℕ) 
  (h_white_packs : white_packs = 3) 
  (h_white_per_pack : white_per_pack = 6) 
  (h_blue_per_pack : blue_per_pack = 4) 
  (h_total_tshirts : total_tshirts = 26) : 
  (total_tshirts - white_packs * white_per_pack) / blue_per_pack = 2 := 
by
  -- Proof omitted
  sorry

end packs_of_blue_tshirts_l1939_193986


namespace ratio_x_y_z_w_l1939_193947

theorem ratio_x_y_z_w (x y z w : ℝ) 
(h1 : 0.10 * x = 0.20 * y)
(h2 : 0.30 * y = 0.40 * z)
(h3 : 0.50 * z = 0.60 * w) : 
  (x / w) = 8 
  ∧ (y / w) = 4 
  ∧ (z / w) = 3
  ∧ (w / w) = 2.5 := 
sorry

end ratio_x_y_z_w_l1939_193947


namespace piglet_balloons_l1939_193941

theorem piglet_balloons (n w o total_balloons: ℕ) (H1: w = 2 * n) (H2: o = 4 * n) (H3: n + w + o = total_balloons) (H4: total_balloons = 44) : n - (7 * n - total_balloons) = 2 :=
by
  sorry

end piglet_balloons_l1939_193941


namespace tangent_circle_line_radius_l1939_193972

theorem tangent_circle_line_radius (m : ℝ) :
  (∀ x y : ℝ, (x - 1)^2 + y^2 = m → x + y = 1 → dist (1, 0) (x, y) = Real.sqrt m) →
  m = 1 / 2 :=
by
  sorry

end tangent_circle_line_radius_l1939_193972


namespace trivia_team_points_l1939_193983

theorem trivia_team_points (total_members absent_members total_points : ℕ) 
    (h1 : total_members = 5) 
    (h2 : absent_members = 2) 
    (h3 : total_points = 18) 
    (h4 : total_members - absent_members = present_members) 
    (h5 : total_points = present_members * points_per_member) : 
    points_per_member = 6 :=
  sorry

end trivia_team_points_l1939_193983


namespace find_a_l1939_193971

noncomputable def angle := 30 * Real.pi / 180 -- In radians

noncomputable def tan_angle : ℝ := Real.tan angle

theorem find_a (a : ℝ) (h1 : tan_angle = 1 / Real.sqrt 3) : 
  x - a * y + 3 = 0 → a = Real.sqrt 3 :=
by
  sorry

end find_a_l1939_193971


namespace solve_puzzle_l1939_193939

theorem solve_puzzle
  (EH OY AY OH : ℕ)
  (h1 : EH = 4 * OY)
  (h2 : AY = 4 * OH) :
  EH + OY + AY + OH = 150 :=
sorry

end solve_puzzle_l1939_193939


namespace value_of_y_l1939_193978

theorem value_of_y (x y : ℝ) (cond1 : 1.5 * x = 0.75 * y) (cond2 : x = 20) : y = 40 :=
by
  sorry

end value_of_y_l1939_193978


namespace total_ages_l1939_193943

theorem total_ages (Xavier Yasmin : ℕ) (h1 : Xavier = 2 * Yasmin) (h2 : Xavier + 6 = 30) : Xavier + Yasmin = 36 :=
by
  sorry

end total_ages_l1939_193943


namespace dot_product_eq_l1939_193952

def vector1 : ℝ × ℝ := (-3, 0)
def vector2 : ℝ × ℝ := (7, 9)

theorem dot_product_eq :
  (vector1.1 * vector2.1 + vector1.2 * vector2.2) = -21 :=
by
  sorry

end dot_product_eq_l1939_193952


namespace widgets_made_per_week_l1939_193937

theorem widgets_made_per_week
  (widgets_per_hour : Nat)
  (hours_per_day : Nat)
  (days_per_week : Nat)
  (total_widgets : Nat) :
  widgets_per_hour = 20 →
  hours_per_day = 8 →
  days_per_week = 5 →
  total_widgets = widgets_per_hour * hours_per_day * days_per_week →
  total_widgets = 800 :=
by
  intros h1 h2 h3 h4
  sorry

end widgets_made_per_week_l1939_193937


namespace find_eccentricity_of_ellipse_l1939_193944

noncomputable def ellipseEccentricity (k : ℝ) : ℝ :=
  let a := Real.sqrt (k + 2)
  let b := Real.sqrt (k + 1)
  let c := Real.sqrt (a^2 - b^2)
  c / a

theorem find_eccentricity_of_ellipse (k : ℝ) (h1 : k + 2 = 4) (h2 : Real.sqrt (k + 2) = 2) :
  ellipseEccentricity k = 1 / 2 := by
  sorry

end find_eccentricity_of_ellipse_l1939_193944


namespace small_rectangular_prisms_intersect_diagonal_l1939_193976

def lcm (a b c : Nat) : Nat :=
  Nat.lcm a (Nat.lcm b c)

def inclusion_exclusion (n : Nat) : Nat :=
  n / 2 + n / 3 + n / 5 - n / (2 * 3) - n / (3 * 5) - n / (5 * 2) + n / (2 * 3 * 5)

theorem small_rectangular_prisms_intersect_diagonal :
  ∀ (a b c : Nat) (L : Nat), a = 2 → b = 3 → c = 5 → L = 90 →
  lcm a b c = 30 → 3 * inclusion_exclusion (lcm a b c) = 66 :=
by
  intros
  sorry

end small_rectangular_prisms_intersect_diagonal_l1939_193976


namespace parallel_line_distance_l1939_193954

theorem parallel_line_distance 
    (A_upper : ℝ) (A_middle : ℝ) (A_lower : ℝ)
    (A_total : ℝ) (A_half : ℝ)
    (h_upper : A_upper = 3)
    (h_middle : A_middle = 5)
    (h_lower : A_lower = 2) 
    (h_total : A_total = A_upper + A_middle + A_lower)
    (h_half : A_half = A_total / 2) :
    ∃ d : ℝ, d = 2 + 0.6 ∧ A_middle * 0.6 = 3 := 
sorry

end parallel_line_distance_l1939_193954


namespace xiaoguang_advances_l1939_193990

theorem xiaoguang_advances (x1 x2 x3 x4 : ℝ) (h1 : 96 ≤ (x1 + x2 + x3 + x4) / 4) (hx1 : x1 = 95) (hx2 : x2 = 97) (hx3 : x3 = 94) : 
  98 ≤ x4 := 
by 
  sorry

end xiaoguang_advances_l1939_193990


namespace relationship_between_x_and_y_l1939_193981

theorem relationship_between_x_and_y (a b : ℝ) (x y : ℝ)
  (h1 : x = a^2 + b^2 + 20)
  (h2 : y = 4 * (2 * b - a)) :
  x ≥ y :=
by 
-- we need to prove x ≥ y
sorry

end relationship_between_x_and_y_l1939_193981


namespace max_expression_value_l1939_193925

noncomputable def max_value : ℕ := 17

theorem max_expression_value 
  (x y z : ℕ) 
  (hx : 10 ≤ x ∧ x < 100) 
  (hy : 10 ≤ y ∧ y < 100) 
  (hz : 10 ≤ z ∧ z < 100) 
  (mean_eq : (x + y + z) / 3 = 60) : 
  (x + y) / z ≤ max_value :=
sorry

end max_expression_value_l1939_193925


namespace smallest_n_sqrt_12n_integer_l1939_193948

theorem smallest_n_sqrt_12n_integer : ∃ n : ℕ, (n > 0) ∧ (∃ k : ℕ, 12 * n = k^2) ∧ n = 3 := by
  sorry

end smallest_n_sqrt_12n_integer_l1939_193948


namespace parabola_standard_form_l1939_193912

theorem parabola_standard_form (a : ℝ) (x y : ℝ) :
  (∀ a : ℝ, (2 * a + 3) * x + y - 4 * a + 2 = 0 → 
  x = 2 ∧ y = -8) → 
  (y^2 = 32 * x ∨ x^2 = - (1/2) * y) :=
by 
  intros h
  sorry

end parabola_standard_form_l1939_193912


namespace age_sum_proof_l1939_193968

noncomputable def leilei_age : ℝ := 30 -- Age of Leilei this year
noncomputable def feifei_age (R : ℝ) : ℝ := 1 / 2 * R + 12 -- Age of Feifei this year defined in terms of R

theorem age_sum_proof (R F : ℝ)
  (h1 : F = 1 / 2 * R + 12)
  (h2 : F + 1 = 2 * (R + 1) - 34) :
  R + F = 57 :=
by 
  -- Proof steps would go here
  sorry

end age_sum_proof_l1939_193968


namespace find_angle_B_find_sin_C_l1939_193906

-- Statement for proving B = π / 4 given the conditions
theorem find_angle_B (a b c : ℝ) (A B C : ℝ) 
  (h1 : a * Real.sin A + c * Real.sin C - Real.sqrt 2 * a * Real.sin C = b * Real.sin B) 
  (hABC : A + B + C = Real.pi) :
  B = Real.pi / 4 := 
sorry

-- Statement for proving sin C when cos A = 1 / 3
theorem find_sin_C (A C : ℝ) 
  (hA : Real.cos A = 1 / 3)
  (hABC : A + Real.pi / 4 + C = Real.pi) :
  Real.sin C = (4 + Real.sqrt 2) / 6 := 
sorry

end find_angle_B_find_sin_C_l1939_193906


namespace sequence_v5_value_l1939_193926

theorem sequence_v5_value (v : ℕ → ℝ) (h_rec : ∀ n, v (n + 2) = 3 * v (n + 1) - v n)
  (h_v3 : v 3 = 17) (h_v6 : v 6 = 524) : v 5 = 198.625 :=
sorry

end sequence_v5_value_l1939_193926


namespace zamena_solution_l1939_193966

def digit_assignment (A M E H Z N : ℕ) : Prop :=
  1 ≤ A ∧ A ≤ 5 ∧ 1 ≤ M ∧ M ≤ 5 ∧ 1 ≤ E ∧ E ≤ 5 ∧ 1 ≤ H ∧ H ≤ 5 ∧ 1 ≤ Z ∧ Z ≤ 5 ∧ 1 ≤ N ∧ N ≤ 5 ∧
  A ≠ M ∧ A ≠ E ∧ A ≠ H ∧ A ≠ Z ∧ A ≠ N ∧
  M ≠ E ∧ M ≠ H ∧ M ≠ Z ∧ M ≠ N ∧
  E ≠ H ∧ E ≠ Z ∧ E ≠ N ∧
  H ≠ Z ∧ H ≠ N ∧
  Z ≠ N ∧
  3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A 

theorem zamena_solution : 
  ∃ (A M E H Z N : ℕ), digit_assignment A M E H Z N ∧ (Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) := 
by
  sorry

end zamena_solution_l1939_193966


namespace union_M_N_l1939_193929

def M : Set ℝ := { x | x^2 - x = 0 }
def N : Set ℝ := { y | y^2 + y = 0 }

theorem union_M_N : (M ∪ N) = {-1, 0, 1} := 
by 
  sorry

end union_M_N_l1939_193929


namespace farmer_feed_full_price_l1939_193969

theorem farmer_feed_full_price
  (total_spent : ℕ)
  (chicken_feed_discount_percent : ℕ)
  (chicken_feed_percent : ℕ)
  (goat_feed_percent : ℕ)
  (total_spent_val : total_spent = 35)
  (chicken_feed_discount_percent_val : chicken_feed_discount_percent = 50)
  (chicken_feed_percent_val : chicken_feed_percent = 40)
  (goat_feed_percent_val : goat_feed_percent = 60) :
  (total_spent * chicken_feed_percent / 100 * 2) + (total_spent * goat_feed_percent / 100) = 49 := 
by
  -- Placeholder for proof.
  sorry

end farmer_feed_full_price_l1939_193969


namespace negation_of_zero_product_l1939_193975

theorem negation_of_zero_product (x y : ℝ) : (xy ≠ 0) → (x ≠ 0) ∧ (y ≠ 0) :=
sorry

end negation_of_zero_product_l1939_193975


namespace fruit_seller_apples_l1939_193996

theorem fruit_seller_apples (x : ℝ) (h : 0.60 * x = 420) : x = 700 :=
sorry

end fruit_seller_apples_l1939_193996


namespace find_x_l1939_193931

theorem find_x (x : ℝ) : (x / (x + 2) + 3 / (x + 2) + 2 * x / (x + 2) = 4) → x = -5 :=
by
  sorry

end find_x_l1939_193931


namespace original_price_l1939_193916

theorem original_price (P : ℝ) (h1 : ∃ P : ℝ, (120 : ℝ) = P + 0.2 * P) : P = 100 :=
by
  obtain ⟨P, h⟩ := h1
  sorry

end original_price_l1939_193916


namespace expected_value_X_correct_prob_1_red_ball_B_correct_l1939_193974

-- Boxes configuration
structure BoxConfig where
  white_A : ℕ -- Number of white balls in box A
  red_A : ℕ -- Number of red balls in box A
  white_B : ℕ -- Number of white balls in box B
  red_B : ℕ -- Number of red balls in box B

-- Given the problem configuration
def initialConfig : BoxConfig := {
  white_A := 2,
  red_A := 2,
  white_B := 1,
  red_B := 3,
}

-- Define random variable X (number of red balls drawn from box A)
def prob_X (X : ℕ) (cfg : BoxConfig) : ℚ :=
  if X = 0 then 1 / 6
  else if X = 1 then 2 / 3
  else if X = 2 then 1 / 6
  else 0

-- Expected value of X
noncomputable def expected_value_X (cfg : BoxConfig) : ℚ :=
  0 * (prob_X 0 cfg) + 1 * (prob_X 1 cfg) + 2 * (prob_X 2 cfg)

-- Probability of drawing 1 red ball from box B
noncomputable def prob_1_red_ball_B (cfg : BoxConfig) (X : ℕ) : ℚ :=
  if X = 0 then 1 / 2
  else if X = 1 then 2 / 3
  else if X = 2 then 5 / 6
  else 0

-- Total probability of drawing 1 red ball from box B
noncomputable def total_prob_1_red_ball_B (cfg : BoxConfig) : ℚ :=
  (prob_X 0 cfg * (prob_1_red_ball_B cfg 0))
  + (prob_X 1 cfg * (prob_1_red_ball_B cfg 1))
  + (prob_X 2 cfg * (prob_1_red_ball_B cfg 2))


theorem expected_value_X_correct : expected_value_X initialConfig = 1 := by
  sorry

theorem prob_1_red_ball_B_correct : total_prob_1_red_ball_B initialConfig = 2 / 3 := by
  sorry

end expected_value_X_correct_prob_1_red_ball_B_correct_l1939_193974


namespace a_n_nonzero_l1939_193987

/-- Recurrence relation for the sequence a_n --/
def a : ℕ → ℤ
| 0 => 1
| 1 => 2
| (n + 2) => if (a n * a (n + 1)) % 2 = 1 then 5 * a (n + 1) - 3 * a n else a (n + 1) - a n

/-- Proof that for all n, a_n is non-zero --/
theorem a_n_nonzero : ∀ n : ℕ, a n ≠ 0 := 
sorry

end a_n_nonzero_l1939_193987


namespace hyperbola_is_given_equation_l1939_193946

noncomputable def hyperbola_equation : Prop :=
  ∃ a b : ℝ, 
    (a > 0 ∧ b > 0) ∧ 
    (4^2 = a^2 + b^2) ∧ 
    (a = b) ∧ 
    (∀ x y : ℝ, (x^2 / 8 - y^2 / 8 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1))

theorem hyperbola_is_given_equation : hyperbola_equation :=
sorry

end hyperbola_is_given_equation_l1939_193946


namespace distance_from_focus_to_asymptote_l1939_193909

theorem distance_from_focus_to_asymptote
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : a = b)
  (h2 : |a| / Real.sqrt 2 = 2) :
  Real.sqrt 2 * 2 = 2 * Real.sqrt 2 :=
by
  sorry

end distance_from_focus_to_asymptote_l1939_193909


namespace triangle_angle_identity_l1939_193955

theorem triangle_angle_identity
  (α β γ : ℝ)
  (h_triangle : α + β + γ = π)
  (sin_α_ne_zero : Real.sin α ≠ 0)
  (sin_β_ne_zero : Real.sin β ≠ 0)
  (sin_γ_ne_zero : Real.sin γ ≠ 0) :
  (Real.cos α / (Real.sin β * Real.sin γ) +
   Real.cos β / (Real.sin α * Real.sin γ) +
   Real.cos γ / (Real.sin α * Real.sin β) = 2) := by
  sorry

end triangle_angle_identity_l1939_193955


namespace ratio_p_q_l1939_193997

-- Definitions of probabilities p and q based on combinatorial choices and probabilities described.
noncomputable def p : ℚ :=
  (Nat.choose 6 1) * (Nat.choose 5 2) * (Nat.choose 24 2) * (Nat.choose 22 4) * (Nat.choose 18 4) * (Nat.choose 14 5) * (Nat.choose 9 5) * (Nat.choose 4 5) / (6 ^ 24)

noncomputable def q : ℚ :=
  (Nat.choose 6 2) * (Nat.choose 24 3) * (Nat.choose 21 3) * (Nat.choose 18 4) * (Nat.choose 14 4) * (Nat.choose 10 4) * (Nat.choose 6 4) / (6 ^ 24)

-- Lean statement to prove p / q = 6
theorem ratio_p_q : p / q = 6 := by
  sorry

end ratio_p_q_l1939_193997
