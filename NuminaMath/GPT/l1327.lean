import Mathlib

namespace NUMINAMATH_GPT_appropriate_length_of_presentation_l1327_132758

theorem appropriate_length_of_presentation (wpm : ℕ) (min_time min_words max_time max_words total_words : ℕ) 
  (h1 : total_words = 160) 
  (h2 : min_time = 45) 
  (h3 : min_words = min_time * wpm) 
  (h4 : max_time = 60) 
  (h5 : max_words = max_time * wpm) : 
  7200 ≤ 9400 ∧ 9400 ≤ 9600 :=
by 
  sorry

end NUMINAMATH_GPT_appropriate_length_of_presentation_l1327_132758


namespace NUMINAMATH_GPT_sum_last_two_digits_l1327_132746

theorem sum_last_two_digits (a b : ℕ) (h₁ : a = 7) (h₂ : b = 13) : 
  (a^25 + b^25) % 100 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_last_two_digits_l1327_132746


namespace NUMINAMATH_GPT_min_value_square_distance_l1327_132722

theorem min_value_square_distance (x y : ℝ) (h : x^2 + y^2 - 4*x + 2 = 0) : 
  ∃ c, (∀ x y : ℝ, x^2 + y^2 - 4*x + 2 = 0 → x^2 + (y - 2)^2 ≥ c) ∧ c = 2 :=
sorry

end NUMINAMATH_GPT_min_value_square_distance_l1327_132722


namespace NUMINAMATH_GPT_find_f_2_l1327_132754

noncomputable def f : ℝ → ℝ := sorry

axiom f_additive (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x + y) = f x + f y
axiom f_8 : f 8 = 3

theorem find_f_2 : f 2 = 3 / 4 := 
by sorry

end NUMINAMATH_GPT_find_f_2_l1327_132754


namespace NUMINAMATH_GPT_distance_traveled_downstream_l1327_132756

noncomputable def boat_speed_in_still_water : ℝ := 12
noncomputable def current_speed : ℝ := 4
noncomputable def travel_time_in_minutes : ℝ := 18
noncomputable def travel_time_in_hours : ℝ := travel_time_in_minutes / 60

theorem distance_traveled_downstream :
  let effective_speed := boat_speed_in_still_water + current_speed
  let distance := effective_speed * travel_time_in_hours
  distance = 4.8 := 
by
  sorry

end NUMINAMATH_GPT_distance_traveled_downstream_l1327_132756


namespace NUMINAMATH_GPT_find_ABC_l1327_132703

theorem find_ABC (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) 
  (hA : A < 5) (hB : B < 5) (hC : C < 5) (h_nonzeroA : A ≠ 0) (h_nonzeroB : B ≠ 0) (h_nonzeroC : C ≠ 0)
  (h4 : B + C = 5) (h5 : A + 1 = C) (h6 : A + B = C) : A = 3 ∧ B = 1 ∧ C = 4 := 
by
  sorry

end NUMINAMATH_GPT_find_ABC_l1327_132703


namespace NUMINAMATH_GPT_cost_per_gift_l1327_132765

theorem cost_per_gift (a b c : ℕ) (hc : c = 70) (ha : a = 3) (hb : b = 4) :
  c / (a + b) = 10 :=
by sorry

end NUMINAMATH_GPT_cost_per_gift_l1327_132765


namespace NUMINAMATH_GPT_divisor_is_679_l1327_132774

noncomputable def x : ℕ := 8
noncomputable def y : ℕ := 9
noncomputable def z : ℝ := 549.7025036818851
noncomputable def p : ℕ := x^3
noncomputable def q : ℕ := y^3
noncomputable def r : ℕ := p * q

theorem divisor_is_679 (k : ℝ) (h : r / k = z) : k = 679 := by
  sorry

end NUMINAMATH_GPT_divisor_is_679_l1327_132774


namespace NUMINAMATH_GPT_babylon_game_proof_l1327_132776

section BabylonGame

-- Defining the number of holes on the sphere
def number_of_holes : Nat := 26

-- The number of 45° angles formed by the pairs of rays
def num_45_degree_angles : Nat := 40

-- The number of 60° angles formed by the pairs of rays
def num_60_degree_angles : Nat := 48

-- The other angles that can occur between pairs of rays
def other_angles : List Real := [31.4, 81.6, 90]

-- Constructs possible given the conditions
def constructible (shape : String) : Bool :=
  shape = "regular tetrahedron" ∨ shape = "regular octahedron"

-- Constructs not possible given the conditions
def non_constructible (shape : String) : Bool :=
  shape = "joined regular tetrahedrons"

-- Proof problem statement
theorem babylon_game_proof :
  (number_of_holes = 26) →
  (num_45_degree_angles = 40) →
  (num_60_degree_angles = 48) →
  (other_angles = [31.4, 81.6, 90]) →
  (constructible "regular tetrahedron" = True) →
  (constructible "regular octahedron" = True) →
  (non_constructible "joined regular tetrahedrons" = True) :=
  by
    sorry

end BabylonGame

end NUMINAMATH_GPT_babylon_game_proof_l1327_132776


namespace NUMINAMATH_GPT_part_a_part_b_l1327_132769

def g (n : ℕ) : ℕ := (n.digits 10).prod

theorem part_a : ∀ n : ℕ, g n ≤ n :=
by
  -- Proof omitted
  sorry

theorem part_b : {n : ℕ | n^2 - 12*n + 36 = g n} = {4, 9} :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_part_a_part_b_l1327_132769


namespace NUMINAMATH_GPT_comparison_a_b_c_l1327_132735

theorem comparison_a_b_c :
  let a := (1 / 2) ^ (1 / 3)
  let b := (1 / 3) ^ (1 / 2)
  let c := Real.log (3 / Real.pi)
  c < b ∧ b < a :=
by
  sorry

end NUMINAMATH_GPT_comparison_a_b_c_l1327_132735


namespace NUMINAMATH_GPT_algebraic_expression_domain_l1327_132717

theorem algebraic_expression_domain (x : ℝ) : (∃ y : ℝ, y = 1 / (x + 2)) ↔ (x ≠ -2) := 
sorry

end NUMINAMATH_GPT_algebraic_expression_domain_l1327_132717


namespace NUMINAMATH_GPT_bill_due_in_months_l1327_132706

noncomputable def true_discount_time (TD A R : ℝ) : ℝ :=
  let P := A - TD
  let T := TD / (P * R / 100)
  12 * T

theorem bill_due_in_months :
  ∀ (TD A R : ℝ), TD = 189 → A = 1764 → R = 16 →
  abs (true_discount_time TD A R - 10.224) < 1 :=
by
  intros TD A R hTD hA hR
  sorry

end NUMINAMATH_GPT_bill_due_in_months_l1327_132706


namespace NUMINAMATH_GPT_solve_inequality_l1327_132751

theorem solve_inequality (x : ℝ) (h : x ≠ -1) : (2 - x) / (x + 1) ≥ 0 ↔ -1 < x ∧ x ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1327_132751


namespace NUMINAMATH_GPT_area_of_rectangular_field_l1327_132777

theorem area_of_rectangular_field (W D : ℝ) (hW : W = 15) (hD : D = 17) :
  ∃ L : ℝ, (W * L = 120) ∧ D^2 = L^2 + W^2 :=
by 
  use 8
  sorry

end NUMINAMATH_GPT_area_of_rectangular_field_l1327_132777


namespace NUMINAMATH_GPT_smallest_n_is_1770_l1327_132712

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

def is_smallest_n (n : ℕ) : Prop :=
  n = sum_of_digits n + 1755 ∧ (∀ m : ℕ, (m < n → m ≠ sum_of_digits m + 1755))

theorem smallest_n_is_1770 : is_smallest_n 1770 :=
sorry

end NUMINAMATH_GPT_smallest_n_is_1770_l1327_132712


namespace NUMINAMATH_GPT_total_amount_shared_l1327_132724

theorem total_amount_shared (x y z : ℝ) (h1 : x = 1.25 * y) (h2 : y = 1.2 * z) (h3 : z = 400) :
  x + y + z = 1480 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_shared_l1327_132724


namespace NUMINAMATH_GPT_total_surface_area_space_l1327_132744

theorem total_surface_area_space (h r1 : ℝ) (h_cond : h = 8) (r1_cond : r1 = 3) : 
  (2 * π * (r1 + 1) * h - 2 * π * r1 * h) = 16 * π := 
by
  sorry

end NUMINAMATH_GPT_total_surface_area_space_l1327_132744


namespace NUMINAMATH_GPT_at_least_two_equal_l1327_132762

theorem at_least_two_equal (x y z : ℝ) (h : (x - y) / (2 + x * y) + (y - z) / (2 + y * z) + (z - x) / (2 + z * x) = 0) : 
x = y ∨ y = z ∨ z = x := 
by
  sorry

end NUMINAMATH_GPT_at_least_two_equal_l1327_132762


namespace NUMINAMATH_GPT_sum_of_series_l1327_132730

theorem sum_of_series : 
  (3 + 13 + 23 + 33 + 43) + (11 + 21 + 31 + 41 + 51) = 270 := by
  sorry

end NUMINAMATH_GPT_sum_of_series_l1327_132730


namespace NUMINAMATH_GPT_distance_comparison_l1327_132772

def distance_mart_to_home : ℕ := 800
def distance_home_to_academy : ℕ := 1300
def distance_academy_to_restaurant : ℕ := 1700

theorem distance_comparison :
  (distance_mart_to_home + distance_home_to_academy) - distance_academy_to_restaurant = 400 :=
by
  sorry

end NUMINAMATH_GPT_distance_comparison_l1327_132772


namespace NUMINAMATH_GPT_find_c_l1327_132794

def f (x c : ℝ) : ℝ := x * (x - c) ^ 2

theorem find_c (c : ℝ) :
  (∀ x, f x c ≤ f 2 c) → c = 6 :=
sorry

end NUMINAMATH_GPT_find_c_l1327_132794


namespace NUMINAMATH_GPT_squirrel_walnuts_l1327_132713

theorem squirrel_walnuts :
  let boy_gathered := 6
  let boy_dropped := 1
  let initial_in_burrow := 12
  let girl_brought := 5
  let girl_ate := 2
  initial_in_burrow + (boy_gathered - boy_dropped) + girl_brought - girl_ate = 20 :=
by
  sorry

end NUMINAMATH_GPT_squirrel_walnuts_l1327_132713


namespace NUMINAMATH_GPT_rainfall_difference_l1327_132757

-- Defining the conditions
def march_rainfall : ℝ := 0.81
def april_rainfall : ℝ := 0.46

-- Stating the theorem
theorem rainfall_difference : march_rainfall - april_rainfall = 0.35 := by
  -- insert proof steps here
  sorry

end NUMINAMATH_GPT_rainfall_difference_l1327_132757


namespace NUMINAMATH_GPT_bottle_ratio_l1327_132727

theorem bottle_ratio (C1 C2 : ℝ)  
  (h1 : (C1 / 2) + (C2 / 4) = (C1 + C2) / 3) :
  C2 = 2 * C1 :=
sorry

end NUMINAMATH_GPT_bottle_ratio_l1327_132727


namespace NUMINAMATH_GPT_quadratic_polynomial_l1327_132739

noncomputable def p (x : ℝ) : ℝ := (14 * x^2 + 4 * x + 12) / 15

theorem quadratic_polynomial :
  p (-2) = 4 ∧ p 1 = 2 ∧ p 3 = 10 :=
by
  have : p (-2) = (14 * (-2 : ℝ) ^ 2 + 4 * (-2 : ℝ) + 12) / 15 := rfl
  have : p 1 = (14 * (1 : ℝ) ^ 2 + 4 * (1 : ℝ) + 12) / 15 := rfl
  have : p 3 = (14 * (3 : ℝ) ^ 2 + 4 * (3 : ℝ) + 12) / 15 := rfl
  -- You can directly state the equalities or keep track of the computation steps.
  sorry

end NUMINAMATH_GPT_quadratic_polynomial_l1327_132739


namespace NUMINAMATH_GPT_infinite_grid_coloring_l1327_132720

theorem infinite_grid_coloring (color : ℕ × ℕ → Fin 4)
  (h_coloring_condition : ∀ (i j : ℕ), color (i, j) ≠ color (i + 1, j) ∧
                                      color (i, j) ≠ color (i, j + 1) ∧
                                      color (i, j) ≠ color (i + 1, j + 1) ∧
                                      color (i + 1, j) ≠ color (i, j + 1)) :
  ∃ m : ℕ, ∃ a b : Fin 4, ∀ n : ℕ, color (m, n) = a ∨ color (m, n) = b :=
sorry

end NUMINAMATH_GPT_infinite_grid_coloring_l1327_132720


namespace NUMINAMATH_GPT_max_squares_covered_by_card_l1327_132786

noncomputable def card_coverage_max_squares (card_side : ℝ) (square_side : ℝ) : ℕ :=
  if card_side = 2 ∧ square_side = 1 then 9 else 0

theorem max_squares_covered_by_card : card_coverage_max_squares 2 1 = 9 := by
  sorry

end NUMINAMATH_GPT_max_squares_covered_by_card_l1327_132786


namespace NUMINAMATH_GPT_find_M_plus_N_l1327_132778

theorem find_M_plus_N (M N : ℕ) 
  (h1 : 5 / 7 = M / 63) 
  (h2 : 5 / 7 = 70 / N) : 
  M + N = 143 :=
by
  sorry

end NUMINAMATH_GPT_find_M_plus_N_l1327_132778


namespace NUMINAMATH_GPT_simplify_expression_l1327_132791

-- We need to prove that the simplified expression is equal to the expected form
theorem simplify_expression (y : ℝ) : (3 * y - 7 * y^2 + 4 - (5 + 3 * y - 7 * y^2)) = (0 * y^2 + 0 * y - 1) :=
by
  -- The detailed proof steps will go here
  sorry

end NUMINAMATH_GPT_simplify_expression_l1327_132791


namespace NUMINAMATH_GPT_student_ticket_price_l1327_132781

-- Define the conditions
variables (S T : ℝ)
def condition1 := 4 * S + 3 * T = 79
def condition2 := 12 * S + 10 * T = 246

-- Prove that the price of a student ticket is 9 dollars, given the equations above
theorem student_ticket_price (h1 : condition1 S T) (h2 : condition2 S T) : T = 9 :=
sorry

end NUMINAMATH_GPT_student_ticket_price_l1327_132781


namespace NUMINAMATH_GPT_total_blossoms_l1327_132710

theorem total_blossoms (first second third : ℕ) (h1 : first = 2) (h2 : second = 2 * first) (h3 : third = 4 * second) : first + second + third = 22 :=
by
  sorry

end NUMINAMATH_GPT_total_blossoms_l1327_132710


namespace NUMINAMATH_GPT_line_through_chord_with_midpoint_l1327_132736

theorem line_through_chord_with_midpoint (x y : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ,
    (x = x1 ∧ y = y1 ∨ x = x2 ∧ y = y2) ∧
    x = -1 ∧ y = 1 ∧
    x1^2 / 4 + y1^2 / 3 = 1 ∧
    x2^2 / 4 + y2^2 / 3 = 1) →
  3 * x - 4 * y + 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_line_through_chord_with_midpoint_l1327_132736


namespace NUMINAMATH_GPT_total_fish_l1327_132753

theorem total_fish (x y : ℕ) : (19 - 2 * x) + (27 - 4 * y) = 46 - 2 * x - 4 * y :=
  by
    sorry

end NUMINAMATH_GPT_total_fish_l1327_132753


namespace NUMINAMATH_GPT_weight_of_each_hardcover_book_l1327_132745

theorem weight_of_each_hardcover_book
  (weight_limit : ℕ := 80)
  (hardcover_books : ℕ := 70)
  (textbooks : ℕ := 30)
  (knick_knacks : ℕ := 3)
  (textbook_weight : ℕ := 2)
  (knick_knack_weight : ℕ := 6)
  (over_weight : ℕ := 33)
  (total_weight : ℕ := hardcover_books * x + textbooks * textbook_weight + knick_knacks * knick_knack_weight)
  (weight_eq : total_weight = weight_limit + over_weight) :
  x = 1 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_weight_of_each_hardcover_book_l1327_132745


namespace NUMINAMATH_GPT_intersection_is_correct_l1327_132763

-- Define the sets A and B based on given conditions
def setA : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}
def setB : Set ℝ := {y | ∃ x, y = Real.sqrt x + 4}

-- Define the intersection of sets A and B
def intersection : Set ℝ := {z | z ≥ 4}

-- The theorem stating that the intersection of A and B is exactly the set [4, +∞)
theorem intersection_is_correct : {x | ∃ y, y = Real.log (x - 2)} ∩ {y | ∃ x, y = Real.sqrt x + 4} = {z | z ≥ 4} :=
by
  sorry

end NUMINAMATH_GPT_intersection_is_correct_l1327_132763


namespace NUMINAMATH_GPT_A_alone_time_l1327_132702

theorem A_alone_time (x : ℕ) (h1 : 3 * x / 4  = 12) : x / 3 = 16 := by
  sorry

end NUMINAMATH_GPT_A_alone_time_l1327_132702


namespace NUMINAMATH_GPT_temperature_comparison_l1327_132779

theorem temperature_comparison: ¬ (-3 > -0.3) :=
by
  sorry -- Proof goes here, skipped for now.

end NUMINAMATH_GPT_temperature_comparison_l1327_132779


namespace NUMINAMATH_GPT_find_a4_plus_a6_l1327_132704

variable {a : ℕ → ℝ}

-- Geometric sequence definition
def is_geometric_seq (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Conditions for the problem
axiom seq_geometric : is_geometric_seq a
axiom seq_positive : ∀ n : ℕ, n > 0 → a n > 0
axiom given_equation : a 3 * a 5 + 2 * a 4 * a 6 + a 5 * a 7 = 81

-- The problem to prove
theorem find_a4_plus_a6 : a 4 + a 6 = 9 :=
sorry

end NUMINAMATH_GPT_find_a4_plus_a6_l1327_132704


namespace NUMINAMATH_GPT_buckets_required_l1327_132709

theorem buckets_required (C : ℕ) (h : C > 0) : 
  let original_buckets := 25
  let reduced_capacity := 2 / 5
  let total_capacity := original_buckets * C
  let new_buckets := total_capacity / ((2 / 5) * C)
  new_buckets = 63 := 
by
  sorry

end NUMINAMATH_GPT_buckets_required_l1327_132709


namespace NUMINAMATH_GPT_people_per_pizza_l1327_132793

def pizza_cost := 12 -- dollars per pizza
def babysitting_earnings_per_night := 4 -- dollars per night
def nights_babysitting := 15
def total_people := 15

theorem people_per_pizza : (babysitting_earnings_per_night * nights_babysitting / pizza_cost) = (total_people / ((babysitting_earnings_per_night * nights_babysitting / pizza_cost))) := 
by
  sorry

end NUMINAMATH_GPT_people_per_pizza_l1327_132793


namespace NUMINAMATH_GPT_james_initial_marbles_l1327_132701

theorem james_initial_marbles (m n : ℕ) (h1 : n = 4) (h2 : m / (n - 1) = 21) :
  m = 28 :=
by sorry

end NUMINAMATH_GPT_james_initial_marbles_l1327_132701


namespace NUMINAMATH_GPT_fraction_to_decimal_l1327_132733

theorem fraction_to_decimal :
  (7 : ℚ) / 16 = 0.4375 :=
by sorry

end NUMINAMATH_GPT_fraction_to_decimal_l1327_132733


namespace NUMINAMATH_GPT_find_m_l1327_132796

theorem find_m (x y m : ℝ) (hx : x = 1) (hy : y = 2) (h : m * x + 2 * y = 6) : m = 2 :=
by sorry

end NUMINAMATH_GPT_find_m_l1327_132796


namespace NUMINAMATH_GPT_min_b1_b2_l1327_132734

-- Define the sequence recurrence relation
def sequence_recurrence (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = (b n + 2011) / (1 + b (n + 1))

-- Problem statement: Prove the minimum value of b₁ + b₂ is 2012
theorem min_b1_b2 (b : ℕ → ℕ) (h : ∀ n ≥ 1, 0 < b n) (rec : sequence_recurrence b) :
  b 1 + b 2 ≥ 2012 :=
sorry

end NUMINAMATH_GPT_min_b1_b2_l1327_132734


namespace NUMINAMATH_GPT_chocolates_sold_at_selling_price_l1327_132766
noncomputable def chocolates_sold (C S : ℝ) (n : ℕ) : Prop :=
  (35 * C = n * S) ∧ ((S - C) / C * 100) = 66.67

theorem chocolates_sold_at_selling_price : ∃ n : ℕ, ∀ C S : ℝ,
  chocolates_sold C S n → n = 21 :=
by
  sorry

end NUMINAMATH_GPT_chocolates_sold_at_selling_price_l1327_132766


namespace NUMINAMATH_GPT_triangle_condition_isosceles_or_right_l1327_132715

theorem triangle_condition_isosceles_or_right {A B C : ℝ} {a b c : ℝ} 
  (h_triangle : A + B + C = π) (h_cos_eq : a * Real.cos A = b * Real.cos B) : 
  (A = B) ∨ (A + B = π / 2) :=
sorry

end NUMINAMATH_GPT_triangle_condition_isosceles_or_right_l1327_132715


namespace NUMINAMATH_GPT_webinar_end_time_correct_l1327_132768

-- Define start time and duration as given conditions
def startTime : Nat := 3*60 + 15  -- 3:15 p.m. in minutes after noon
def duration : Nat := 350         -- duration of the webinar in minutes

-- Define the expected end time in minutes after noon (9:05 p.m. is 9*60 + 5 => 545 minutes after noon)
def endTimeExpected : Nat := 9*60 + 5

-- Statement to prove that the calculated end time matches the expected end time
theorem webinar_end_time_correct : startTime + duration = endTimeExpected :=
by
  sorry

end NUMINAMATH_GPT_webinar_end_time_correct_l1327_132768


namespace NUMINAMATH_GPT_distance_between_foci_l1327_132749

-- Let the hyperbola be defined by the equation xy = 4.
def hyperbola (x y : ℝ) : Prop := x * y = 4

-- Prove that the distance between the foci of this hyperbola is 8.
theorem distance_between_foci : ∀ (x y : ℝ), hyperbola x y → ∃ d, d = 8 :=
by {
    sorry
}

end NUMINAMATH_GPT_distance_between_foci_l1327_132749


namespace NUMINAMATH_GPT_simplify_expression_l1327_132738

open Complex

theorem simplify_expression :
  ((4 + 6 * I) / (4 - 6 * I) * (4 - 6 * I) / (4 + 6 * I) + (4 - 6 * I) / (4 + 6 * I) * (4 + 6 * I) / (4 - 6 * I)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1327_132738


namespace NUMINAMATH_GPT_trains_clear_each_other_in_11_seconds_l1327_132799

-- Define the lengths of the trains
def length_train1 := 100  -- in meters
def length_train2 := 120  -- in meters

-- Define the speeds of the trains (in km/h), converted to m/s
def speed_train1 := 42 * 1000 / 3600  -- 42 km/h to m/s
def speed_train2 := 30 * 1000 / 3600  -- 30 km/h to m/s

-- Calculate the total distance to be covered
def total_distance := length_train1 + length_train2  -- in meters

-- Calculate the relative speed when they are moving towards each other
def relative_speed := speed_train1 + speed_train2  -- in m/s

-- Calculate the time required for the trains to be clear of each other (in seconds)
noncomputable def clear_time := total_distance / relative_speed

-- Theorem stating the above
theorem trains_clear_each_other_in_11_seconds :
  clear_time = 11 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_trains_clear_each_other_in_11_seconds_l1327_132799


namespace NUMINAMATH_GPT_algebraic_expression_value_l1327_132718

-- Define the conditions 
variables (x y : ℝ)
def condition1 : Prop := x + y = 2
def condition2 : Prop := x - y = 4

-- State the main theorem
theorem algebraic_expression_value (h1 : condition1 x y) (h2 : condition2 x y) :
  1 + x^2 - y^2 = 9 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1327_132718


namespace NUMINAMATH_GPT_tourism_revenue_scientific_notation_l1327_132784

-- Define the conditions given in the problem.
def total_tourism_revenue := 12.41 * 10^9

-- Prove the scientific notation of the total tourism revenue.
theorem tourism_revenue_scientific_notation :
  total_tourism_revenue = 1.241 * 10^9 :=
sorry

end NUMINAMATH_GPT_tourism_revenue_scientific_notation_l1327_132784


namespace NUMINAMATH_GPT_no_positive_integer_solutions_l1327_132743

theorem no_positive_integer_solutions (x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  x^3 + 2 * y^3 ≠ 4 * z^3 :=
by
  sorry

end NUMINAMATH_GPT_no_positive_integer_solutions_l1327_132743


namespace NUMINAMATH_GPT_triangle_count_l1327_132752

def count_triangles (smallest intermediate larger even_larger whole_structure : Nat) : Nat :=
  smallest + intermediate + larger + even_larger + whole_structure

theorem triangle_count :
  count_triangles 2 6 6 6 12 = 32 :=
by
  sorry

end NUMINAMATH_GPT_triangle_count_l1327_132752


namespace NUMINAMATH_GPT_problem_l1327_132775

theorem problem (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x * y) + x = x * f y + f x)
  (h2 : f (1 / 2) = 0) : 
  f (-201) = 403 :=
sorry

end NUMINAMATH_GPT_problem_l1327_132775


namespace NUMINAMATH_GPT_minimum_ab_l1327_132750

theorem minimum_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : ab = a + 4 * b + 5) : ab ≥ 25 :=
sorry

end NUMINAMATH_GPT_minimum_ab_l1327_132750


namespace NUMINAMATH_GPT_fractions_equal_l1327_132700

theorem fractions_equal (x y z : ℝ) (hx1 : x ≠ 1) (hy1 : y ≠ 1) (hxy : x ≠ y)
  (h : (yz - x^2) / (1 - x) = (xz - y^2) / (1 - y)) : (yz - x^2) / (1 - x) = x + y + z ∧ (xz - y^2) / (1 - y) = x + y + z :=
sorry

end NUMINAMATH_GPT_fractions_equal_l1327_132700


namespace NUMINAMATH_GPT_tara_marbles_modulo_l1327_132723

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem tara_marbles_modulo : 
  let B := 6  -- number of blue marbles
  let Y := 12  -- additional yellow marbles for balance
  let total_marbles := B + Y  -- total marbles
  let arrangements := binom (total_marbles + B) B
  let N := arrangements
  N % 1000 = 564 :=
by
  let B := 6  -- number of blue marbles
  let Y := 12  -- additional yellow marbles for balance
  let total_marbles := B + Y  -- total marbles
  let arrangements := binom (total_marbles + B) B
  let N := arrangements
  have : N % 1000 = 564 := sorry
  exact this

end NUMINAMATH_GPT_tara_marbles_modulo_l1327_132723


namespace NUMINAMATH_GPT_old_manufacturing_cost_l1327_132729

theorem old_manufacturing_cost (P : ℝ) (h1 : 50 = 0.50 * P) : 0.60 * P = 60 :=
by
  sorry

end NUMINAMATH_GPT_old_manufacturing_cost_l1327_132729


namespace NUMINAMATH_GPT_simplify_trig_expression_l1327_132783

theorem simplify_trig_expression (α : ℝ) :
  (2 * Real.sin (Real.pi - α) + Real.sin (2 * α)) / (2 * Real.cos (α / 2) ^ 2) = 2 * Real.sin α :=
by
  sorry

end NUMINAMATH_GPT_simplify_trig_expression_l1327_132783


namespace NUMINAMATH_GPT_ones_digit_542_mul_3_is_6_l1327_132748

/--
Given that the ones (units) digit of 542 is 2, prove that the ones digit of 542 multiplied by 3 is 6.
-/
theorem ones_digit_542_mul_3_is_6 (h: ∃ n : ℕ, 542 = 10 * n + 2) : (542 * 3) % 10 = 6 := 
by
  sorry

end NUMINAMATH_GPT_ones_digit_542_mul_3_is_6_l1327_132748


namespace NUMINAMATH_GPT_sample_size_l1327_132797

-- Define the given conditions
def number_of_male_athletes : Nat := 42
def number_of_female_athletes : Nat := 30
def sampled_female_athletes : Nat := 5

-- Define the target total sample size
def total_sample_size (male_athletes female_athletes sample_females : Nat) : Nat :=
  sample_females * male_athletes / female_athletes + sample_females

-- State the theorem to prove
theorem sample_size (h1: number_of_male_athletes = 42) 
                    (h2: number_of_female_athletes = 30)
                    (h3: sampled_female_athletes = 5) :
  total_sample_size number_of_male_athletes number_of_female_athletes sampled_female_athletes = 12 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_sample_size_l1327_132797


namespace NUMINAMATH_GPT_visitors_on_saturday_l1327_132764

theorem visitors_on_saturday (S : ℕ) (h1 : S + (S + 40) = 440) : S = 200 := by
  sorry

end NUMINAMATH_GPT_visitors_on_saturday_l1327_132764


namespace NUMINAMATH_GPT_find_sale_in_second_month_l1327_132742

def sale_in_second_month (sale1 sale3 sale4 sale5 sale6 target_average : ℕ) (S : ℕ) : Prop :=
  sale1 + S + sale3 + sale4 + sale5 + sale6 = target_average * 6

theorem find_sale_in_second_month :
  sale_in_second_month 5420 6200 6350 6500 7070 6200 5660 :=
by
  sorry

end NUMINAMATH_GPT_find_sale_in_second_month_l1327_132742


namespace NUMINAMATH_GPT_solve_for_x_l1327_132760

theorem solve_for_x (x : ℝ) (h : x / 5 + 3 = 4) : x = 5 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1327_132760


namespace NUMINAMATH_GPT_average_annual_growth_rate_eq_l1327_132726

-- Definition of variables based on given conditions
def sales_2021 := 298 -- in 10,000 units
def sales_2023 := 850 -- in 10,000 units
def years := 2

-- Problem statement in Lean 4
theorem average_annual_growth_rate_eq :
  sales_2021 * (1 + x) ^ years = sales_2023 :=
sorry

end NUMINAMATH_GPT_average_annual_growth_rate_eq_l1327_132726


namespace NUMINAMATH_GPT_original_cost_l1327_132711

theorem original_cost (original_cost : ℝ) (h : 0.30 * original_cost = 588) : original_cost = 1960 :=
sorry

end NUMINAMATH_GPT_original_cost_l1327_132711


namespace NUMINAMATH_GPT_no_three_distinct_rational_roots_l1327_132732

theorem no_three_distinct_rational_roots (a b : ℝ) : 
  ¬ ∃ (u v w : ℚ), 
    u + v + w = -(2 * a + 1) ∧ 
    u * v + v * w + w * u = (2 * a^2 + 2 * a - 3) ∧ 
    u * v * w = b := sorry

end NUMINAMATH_GPT_no_three_distinct_rational_roots_l1327_132732


namespace NUMINAMATH_GPT_smallest_y_for_square_l1327_132788

theorem smallest_y_for_square (y M : ℕ) (h1 : 2310 * y = M^2) (h2 : 2310 = 2 * 3 * 5 * 7 * 11) : y = 2310 :=
by sorry

end NUMINAMATH_GPT_smallest_y_for_square_l1327_132788


namespace NUMINAMATH_GPT_determine_location_with_coords_l1327_132767

-- Define the conditions as a Lean structure
structure Location where
  longitude : ℝ
  latitude : ℝ

-- Define the specific location given in option ①
def location_118_40 : Location :=
  {longitude := 118, latitude := 40}

-- Define the theorem and its statement
theorem determine_location_with_coords :
  ∃ loc : Location, loc = location_118_40 := 
  by
  sorry -- Placeholder for the proof

end NUMINAMATH_GPT_determine_location_with_coords_l1327_132767


namespace NUMINAMATH_GPT_total_value_of_pile_l1327_132725

def value_of_pile (total_coins dimes : ℕ) (value_dime value_nickel : ℝ) : ℝ :=
  let nickels := total_coins - dimes
  let value_dimes := dimes * value_dime
  let value_nickels := nickels * value_nickel
  value_dimes + value_nickels

theorem total_value_of_pile :
  value_of_pile 50 14 0.10 0.05 = 3.20 := by
  sorry

end NUMINAMATH_GPT_total_value_of_pile_l1327_132725


namespace NUMINAMATH_GPT_min_b_geometric_sequence_l1327_132707

theorem min_b_geometric_sequence (a b c : ℝ) (h_geom : b^2 = a * c) (h_1_4 : (a = 1 ∨ b = 1 ∨ c = 1) ∧ (a = 4 ∨ b = 4 ∨ c = 4)) :
  b ≥ -2 ∧ (∃ b', b' < b → b' ≥ -2) :=
by {
  sorry -- Proof required
}

end NUMINAMATH_GPT_min_b_geometric_sequence_l1327_132707


namespace NUMINAMATH_GPT_three_star_five_l1327_132714

-- Definitions based on conditions
def star (a b : ℕ) : ℕ := 2 * a^2 + 3 * a * b + 2 * b^2

-- Theorem statement to be proved
theorem three_star_five : star 3 5 = 113 := by
  sorry

end NUMINAMATH_GPT_three_star_five_l1327_132714


namespace NUMINAMATH_GPT_three_digit_even_two_odd_no_repetition_l1327_132771

-- Define sets of digits
def digits : List ℕ := [0, 1, 3, 4, 5, 6]
def evens : List ℕ := [0, 4, 6]
def odds : List ℕ := [1, 3, 5]

noncomputable def total_valid_numbers : ℕ :=
  let choose_0 := 12 -- Given by A_{2}^{1} A_{3}^{2} = 12
  let without_0 := 36 -- Given by C_{2}^{1} * C_{3}^{2} * A_{3}^{3} = 36
  choose_0 + without_0

theorem three_digit_even_two_odd_no_repetition : total_valid_numbers = 48 :=
by
  -- Proof would be provided here
  sorry

end NUMINAMATH_GPT_three_digit_even_two_odd_no_repetition_l1327_132771


namespace NUMINAMATH_GPT_range_of_a2_div_a1_l1327_132705

theorem range_of_a2_div_a1 (a_1 a_2 d : ℤ) : 
  1 ≤ a_1 ∧ a_1 ≤ 3 ∧ 
  a_2 = a_1 + d ∧ 
  6 ≤ 3 * a_1 + 2 * d ∧ 
  3 * a_1 + 2 * d ≤ 15 
  → (2 / 3 : ℚ) ≤ (a_2 : ℚ) / a_1 ∧ (a_2 : ℚ) / a_1 ≤ 5 :=
sorry

end NUMINAMATH_GPT_range_of_a2_div_a1_l1327_132705


namespace NUMINAMATH_GPT_profit_calculation_l1327_132770

theorem profit_calculation (investment_john investment_mike profit_john profit_mike: ℕ) 
  (total_profit profit_shared_ratio profit_remaining_profit: ℚ)
  (h_investment_john : investment_john = 700)
  (h_investment_mike : investment_mike = 300)
  (h_total_profit : total_profit = 3000)
  (h_shared_ratio : profit_shared_ratio = total_profit / 3 / 2)
  (h_remaining_profit : profit_remaining_profit = 2 * total_profit / 3)
  (h_profit_john : profit_john = profit_shared_ratio + (7 / 10) * profit_remaining_profit)
  (h_profit_mike : profit_mike = profit_shared_ratio + (3 / 10) * profit_remaining_profit)
  (h_profit_difference : profit_john = profit_mike + 800) :
  total_profit = 3000 := 
by
  sorry

end NUMINAMATH_GPT_profit_calculation_l1327_132770


namespace NUMINAMATH_GPT_line_of_symmetry_l1327_132787

-- Definitions of the circles and the line
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 4 * y - 1 = 0
def line (x y : ℝ) : Prop := x - y - 2 = 0

-- The theorem stating the symmetry condition
theorem line_of_symmetry :
  ∀ (x y : ℝ), circle1 x y ↔ ∃ (x' y' : ℝ), line ((x + x') / 2) ((y + y') / 2) ∧ circle2 x' y' :=
sorry

end NUMINAMATH_GPT_line_of_symmetry_l1327_132787


namespace NUMINAMATH_GPT_simplify_expr_l1327_132761

theorem simplify_expr (x : ℝ) : (3 * x)^5 + (4 * x) * (x^4) = 247 * x^5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr_l1327_132761


namespace NUMINAMATH_GPT_mn_min_l1327_132795

noncomputable def min_mn_value (m n : ℝ) : ℝ := m * n

theorem mn_min : 
  (∃ m n, m = Real.sin (2 * (π / 12)) ∧ n > 0 ∧ 
            Real.cos (2 * (π / 12 + n) - π / 4) = m ∧ 
            min_mn_value m n = π * 5 / 48) := by
  sorry

end NUMINAMATH_GPT_mn_min_l1327_132795


namespace NUMINAMATH_GPT_red_chairs_count_l1327_132740

-- Given conditions
variables {R Y B : ℕ} -- Assuming the number of chairs are natural numbers

-- Main theorem statement
theorem red_chairs_count : 
  Y = 4 * R ∧ B = Y - 2 ∧ R + Y + B = 43 -> R = 5 :=
by
  sorry

end NUMINAMATH_GPT_red_chairs_count_l1327_132740


namespace NUMINAMATH_GPT_last_digit_base4_of_389_l1327_132708

theorem last_digit_base4_of_389 : (389 % 4 = 1) :=
by sorry

end NUMINAMATH_GPT_last_digit_base4_of_389_l1327_132708


namespace NUMINAMATH_GPT_question_solution_l1327_132798

variable (a b : ℝ)

theorem question_solution : 2 * a - 3 * (a - b) = -a + 3 * b := by
  sorry

end NUMINAMATH_GPT_question_solution_l1327_132798


namespace NUMINAMATH_GPT_billy_win_probability_l1327_132716

-- Definitions of states and transition probabilities
def alice_step_prob_pos : ℚ := 1 / 2
def alice_step_prob_neg : ℚ := 1 / 2
def billy_step_prob_pos : ℚ := 2 / 3
def billy_step_prob_neg : ℚ := 1 / 3

-- Definitions of states in the Markov chain
inductive State
| S0 | S1 | Sm1 | S2 | Sm2 -- Alice's states
| T0 | T1 | Tm1 | T2 | Tm2 -- Billy's states

open State

-- The theorem statement: the probability that Billy wins the game
theorem billy_win_probability : 
  ∃ (P : State → ℚ), 
  P S0 = 11 / 19 ∧ P T0 = 14 / 19 ∧ 
  P S1 = 1 / 2 * P T0 ∧
  P Sm1 = 1 / 2 * P S0 + 1 / 2 ∧
  P T0 = 2 / 3 * P T1 + 1 / 3 * P Tm1 ∧
  P T1 = 2 / 3 + 1 / 3 * P S0 ∧
  P Tm1 = 2 / 3 * P T0 ∧
  P S2 = 0 ∧ P Sm2 = 1 ∧ P T2 = 1 ∧ P Tm2 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_billy_win_probability_l1327_132716


namespace NUMINAMATH_GPT_football_team_total_players_l1327_132728

/-- Let's denote the total number of players on the football team as P.
    We know that there are 31 throwers, and all of them are right-handed.
    The rest of the team is divided so one third are left-handed and the rest are right-handed.
    There are a total of 57 right-handed players on the team.
    Prove that the total number of players on the football team is 70. -/
theorem football_team_total_players 
  (P : ℕ) -- total number of players
  (T : ℕ := 31) -- number of throwers
  (L : ℕ) -- number of left-handed players
  (R : ℕ := 57) -- total number of right-handed players
  (H_all_throwers_rhs: ∀ x : ℕ, (x < P) → (x < T) → (x = T → x < R)) -- all throwers are right-handed
  (H_rest_division: ∀ x : ℕ, (x < P - T) → (x = L) → (x = 2 * L))
  : P = 70 :=
  sorry

end NUMINAMATH_GPT_football_team_total_players_l1327_132728


namespace NUMINAMATH_GPT_find_theta_l1327_132719

theorem find_theta (Theta : ℕ) (h1 : 1 ≤ Theta ∧ Theta ≤ 9)
  (h2 : 294 / Theta = (30 + Theta) + 3 * Theta) : Theta = 6 :=
by sorry

end NUMINAMATH_GPT_find_theta_l1327_132719


namespace NUMINAMATH_GPT_number_of_integer_values_x_floor_2_sqrt_x_eq_12_l1327_132789

theorem number_of_integer_values_x_floor_2_sqrt_x_eq_12 :
  ∃! n : ℕ, n = 7 ∧ (∀ x : ℕ, (⌊2 * Real.sqrt x⌋ = 12 ↔ 36 ≤ x ∧ x < 43)) :=
by 
  sorry

end NUMINAMATH_GPT_number_of_integer_values_x_floor_2_sqrt_x_eq_12_l1327_132789


namespace NUMINAMATH_GPT_solve_absolute_value_eq_l1327_132755

theorem solve_absolute_value_eq (x : ℝ) : (|x - 3| = 5 - x) → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_absolute_value_eq_l1327_132755


namespace NUMINAMATH_GPT_arithmetic_sequence_twelfth_term_l1327_132741

theorem arithmetic_sequence_twelfth_term :
  let a1 := (1 : ℚ) / 2;
  let a2 := (5 : ℚ) / 6;
  let d := a2 - a1;
  (a1 + 11 * d) = (25 : ℚ) / 6 :=
by
  let a1 := (1 : ℚ) / 2;
  let a2 := (5 : ℚ) / 6;
  let d := a2 - a1;
  exact sorry

end NUMINAMATH_GPT_arithmetic_sequence_twelfth_term_l1327_132741


namespace NUMINAMATH_GPT_count_consecutive_sets_sum_15_l1327_132790

theorem count_consecutive_sets_sum_15 : 
  ∃ n : ℕ, 
    (n > 0 ∧
    ∃ a : ℕ, 
      (n ≥ 2 ∧ 
      ∃ s : (Finset ℕ), 
        (∀ x ∈ s, x ≥ 1) ∧ 
        (s.sum id = 15))
  ) → 
  n = 2 :=
  sorry

end NUMINAMATH_GPT_count_consecutive_sets_sum_15_l1327_132790


namespace NUMINAMATH_GPT_function_D_min_value_is_2_l1327_132782

noncomputable def function_A (x : ℝ) : ℝ := x + 2
noncomputable def function_B (x : ℝ) : ℝ := Real.sin x + 2
noncomputable def function_C (x : ℝ) : ℝ := abs x + 2
noncomputable def function_D (x : ℝ) : ℝ := x^2 + 1

theorem function_D_min_value_is_2
  (x : ℝ) :
  ∃ x, function_D x = 2 := by
  sorry
 
end NUMINAMATH_GPT_function_D_min_value_is_2_l1327_132782


namespace NUMINAMATH_GPT_caroline_socks_gift_l1327_132759

theorem caroline_socks_gift :
  ∀ (initial lost donated_fraction purchased total received),
    initial = 40 →
    lost = 4 →
    donated_fraction = 2 / 3 →
    purchased = 10 →
    total = 25 →
    received = total - (initial - lost - donated_fraction * (initial - lost) + purchased) →
    received = 3 :=
by
  intros initial lost donated_fraction purchased total received
  intro h_initial h_lost h_donated_fraction h_purchased h_total h_received
  sorry

end NUMINAMATH_GPT_caroline_socks_gift_l1327_132759


namespace NUMINAMATH_GPT_compare_fractions_l1327_132747

theorem compare_fractions (a : ℝ) : 
  (a = 0 → (1 / (1 - a)) = (1 + a)) ∧ 
  (0 < a ∧ a < 1 → (1 / (1 - a)) > (1 + a)) ∧ 
  (a > 1 → (1 / (1 - a)) < (1 + a)) := by
  sorry

end NUMINAMATH_GPT_compare_fractions_l1327_132747


namespace NUMINAMATH_GPT_initial_number_of_angelfish_l1327_132792

/-- The initial number of fish in the tank. -/
def initial_total_fish (A : ℕ) := 94 + A + 89 + 58

/-- The remaining number of fish for each species after sale. -/
def remaining_fish (A : ℕ) := 64 + (A - 48) + 72 + 34

/-- Given: 
1. The total number of remaining fish in the tank is 198.
2. The initial number of fish for each species: 94 guppies, A angelfish, 89 tiger sharks, 58 Oscar fish.
3. The number of fish sold: 30 guppies, 48 angelfish, 17 tiger sharks, 24 Oscar fish.
Prove: The initial number of angelfish is 76. -/
theorem initial_number_of_angelfish (A : ℕ) (h : remaining_fish A = 198) : A = 76 :=
sorry

end NUMINAMATH_GPT_initial_number_of_angelfish_l1327_132792


namespace NUMINAMATH_GPT_pyramid_volume_l1327_132773

theorem pyramid_volume (S : ℝ) :
  ∃ (V : ℝ),
  (∀ (a b h : ℝ), S = a * b ∧
  h = a * (Real.tan (60 * (Real.pi / 180))) ∧
  h = b * (Real.tan (30 * (Real.pi / 180))) ∧
  V = (1/3) * S * h) →
  V = (S * Real.sqrt S) / 3 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_volume_l1327_132773


namespace NUMINAMATH_GPT_minimal_value_expression_l1327_132785

theorem minimal_value_expression (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 1) :
  (a + (ab)^(1/3) + (abc)^(1/4)) ≥ (1/3 + 1/(3 * (3^(1/3))) + 1/(3 * (3^(1/4)))) :=
sorry

end NUMINAMATH_GPT_minimal_value_expression_l1327_132785


namespace NUMINAMATH_GPT_express_train_speed_ratio_l1327_132721

noncomputable def speed_ratio (c h : ℝ) (x : ℝ) : Prop :=
  let t1 := h / ((1 + x) * c)
  let t2 := h / ((x - 1) * c)
  x = t2 / t1

theorem express_train_speed_ratio 
  (c h : ℝ) (x : ℝ) 
  (hc : c > 0) (hh : h > 0) (hx : x > 1) : 
  speed_ratio c h (1 + Real.sqrt 2) := 
by
  sorry

end NUMINAMATH_GPT_express_train_speed_ratio_l1327_132721


namespace NUMINAMATH_GPT_add_base_12_l1327_132731

theorem add_base_12 :
  let a := 5*12^2 + 1*12^1 + 8*12^0
  let b := 2*12^2 + 7*12^1 + 6*12^0
  let result := 7*12^2 + 9*12^1 + 2*12^0
  a + b = result :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_add_base_12_l1327_132731


namespace NUMINAMATH_GPT_ratio_area_rectangle_to_square_l1327_132780

variable (s : ℝ)
variable (area_square : ℝ := s^2)
variable (longer_side_rectangle : ℝ := 1.2 * s)
variable (shorter_side_rectangle : ℝ := 0.85 * s)
variable (area_rectangle : ℝ := longer_side_rectangle * shorter_side_rectangle)

theorem ratio_area_rectangle_to_square :
  area_rectangle / area_square = 51 / 50 := by
  sorry

end NUMINAMATH_GPT_ratio_area_rectangle_to_square_l1327_132780


namespace NUMINAMATH_GPT_students_taking_neither_l1327_132737

-- Definitions based on conditions
def total_students : ℕ := 60
def students_CS : ℕ := 40
def students_Elec : ℕ := 35
def students_both_CS_and_Elec : ℕ := 25

-- Lean statement to prove the number of students taking neither computer science nor electronics
theorem students_taking_neither : total_students - (students_CS + students_Elec - students_both_CS_and_Elec) = 10 :=
by
  sorry

end NUMINAMATH_GPT_students_taking_neither_l1327_132737
