import Mathlib

namespace geometric_sequence_sum_l1894_189447

theorem geometric_sequence_sum (a : ℝ) (q : ℝ) (h1 : a * q^2 + a * q^5 = 6)
  (h2 : a * q^4 + a * q^7 = 9) : a * q^6 + a * q^9 = 27 / 2 :=
by
  sorry

end geometric_sequence_sum_l1894_189447


namespace farthest_vertex_label_l1894_189478

-- The vertices and their labeling
def cube_faces : List (List Nat) := [
  [1, 2, 5, 8],
  [3, 4, 6, 7],
  [2, 4, 5, 7],
  [1, 3, 6, 8],
  [2, 3, 7, 8],
  [1, 4, 5, 6]
]

-- Define the cube vertices labels
def vertices : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

-- Statement of the problem in Lean 4
theorem farthest_vertex_label (h : true) : 
  ∃ v : Nat, v ∈ vertices ∧ ∀ face ∈ cube_faces, v ∉ face → v = 6 := 
sorry

end farthest_vertex_label_l1894_189478


namespace catch_bus_probability_within_5_minutes_l1894_189485

theorem catch_bus_probability_within_5_minutes :
  (Pbus3 : ℝ) → (Pbus6 : ℝ) → (Pbus3 = 0.20) → (Pbus6 = 0.60) → (Pcatch : ℝ) → (Pcatch = Pbus3 + Pbus6) → (Pcatch = 0.80) :=
by
  intros Pbus3 Pbus6 hPbus3 hPbus6 Pcatch hPcatch
  sorry

end catch_bus_probability_within_5_minutes_l1894_189485


namespace find_fiona_experience_l1894_189496

namespace Experience

variables (d e f : ℚ)

def avg_experience_equation : Prop := d + e + f = 36
def fiona_david_equation : Prop := f - 5 = d
def emma_david_future_equation : Prop := e + 4 = (3/4) * (d + 4)

theorem find_fiona_experience (h1 : avg_experience_equation d e f) (h2 : fiona_david_equation d f) (h3 : emma_david_future_equation d e) :
  f = 183 / 11 :=
by
  sorry

end Experience

end find_fiona_experience_l1894_189496


namespace sum_proper_divisors_243_l1894_189462

theorem sum_proper_divisors_243 : (1 + 3 + 9 + 27 + 81) = 121 := by
  sorry

end sum_proper_divisors_243_l1894_189462


namespace theater_ticket_sales_l1894_189449

theorem theater_ticket_sales 
  (total_tickets : ℕ) (price_adult_ticket : ℕ) (price_senior_ticket : ℕ) (senior_tickets_sold : ℕ) 
  (Total_tickets_condition : total_tickets = 510)
  (Price_adult_ticket_condition : price_adult_ticket = 21)
  (Price_senior_ticket_condition : price_senior_ticket = 15)
  (Senior_tickets_sold_condition : senior_tickets_sold = 327) : 
  (183 * 21 + 327 * 15 = 8748) :=
by
  sorry

end theater_ticket_sales_l1894_189449


namespace petyas_square_is_larger_l1894_189465

noncomputable def side_petya_square (a b : ℝ) : ℝ :=
  a * b / (a + b)

noncomputable def side_vasya_square (a b : ℝ) : ℝ :=
  a * b * Real.sqrt (a^2 + b^2) / (a^2 + a * b + b^2)

theorem petyas_square_is_larger (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  side_petya_square a b > side_vasya_square a b := by
  sorry

end petyas_square_is_larger_l1894_189465


namespace factor_in_range_l1894_189421

-- Define the given constants
def a : ℕ := 201212200619
def lower_bound : ℕ := 6000000000
def upper_bound : ℕ := 6500000000
def m : ℕ := 6490716149

-- The Lean proof statement
theorem factor_in_range :
  m ∣ a ∧ lower_bound < m ∧ m < upper_bound :=
by
  exact ⟨sorry, sorry, sorry⟩

end factor_in_range_l1894_189421


namespace min_sum_abc_l1894_189499

theorem min_sum_abc (a b c : ℕ) (h1 : a * b * c = 3960) : a + b + c ≥ 150 :=
sorry

end min_sum_abc_l1894_189499


namespace total_cost_for_doughnuts_l1894_189452

theorem total_cost_for_doughnuts
  (num_students : ℕ)
  (num_chocolate : ℕ)
  (num_glazed : ℕ)
  (price_chocolate : ℕ)
  (price_glazed : ℕ)
  (H1 : num_students = 25)
  (H2 : num_chocolate = 10)
  (H3 : num_glazed = 15)
  (H4 : price_chocolate = 2)
  (H5 : price_glazed = 1) :
  num_chocolate * price_chocolate + num_glazed * price_glazed = 35 :=
by
  -- Proof steps would go here
  sorry

end total_cost_for_doughnuts_l1894_189452


namespace discounted_price_is_correct_l1894_189427

def marked_price : ℕ := 125
def discount_rate : ℚ := 4 / 100

def calculate_discounted_price (marked_price : ℕ) (discount_rate : ℚ) : ℚ :=
  marked_price - (discount_rate * marked_price)

theorem discounted_price_is_correct :
  calculate_discounted_price marked_price discount_rate = 120 := by
  sorry

end discounted_price_is_correct_l1894_189427


namespace sphere_radius_twice_cone_volume_l1894_189493

theorem sphere_radius_twice_cone_volume :
  ∀ (r_cone h_cone : ℝ) (r_sphere : ℝ), 
    r_cone = 2 → h_cone = 8 → 2 * (1 / 3 * Real.pi * r_cone^2 * h_cone) = (4/3 * Real.pi * r_sphere^3) → 
    r_sphere = 2^(4/3) :=
by
  intros r_cone h_cone r_sphere h_r_cone h_h_cone h_volume_equiv
  sorry

end sphere_radius_twice_cone_volume_l1894_189493


namespace expected_length_after_2012_repetitions_l1894_189490

noncomputable def expected_length_remaining (n : ℕ) := (11/18 : ℚ)^n

theorem expected_length_after_2012_repetitions :
  expected_length_remaining 2012 = (11 / 18 : ℚ) ^ 2012 :=
by
  sorry

end expected_length_after_2012_repetitions_l1894_189490


namespace angle_measure_triple_complement_l1894_189410

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end angle_measure_triple_complement_l1894_189410


namespace discriminant_of_polynomial_l1894_189437

noncomputable def polynomial_discriminant (a b c : ℚ) : ℚ :=
b^2 - 4 * a * c

theorem discriminant_of_polynomial : polynomial_discriminant 2 (4 - (1/2 : ℚ)) 1 = 17 / 4 :=
by
  sorry

end discriminant_of_polynomial_l1894_189437


namespace exactly_one_equals_xx_plus_xx_l1894_189438

theorem exactly_one_equals_xx_plus_xx (x : ℝ) (hx : x > 0) :
  let expr1 := 2 * x^x
  let expr2 := x^(2*x)
  let expr3 := (2*x)^x
  let expr4 := (2*x)^(2*x)
  (expr1 = x^x + x^x) ∧ (¬(expr2 = x^x + x^x)) ∧ (¬(expr3 = x^x + x^x)) ∧ (¬(expr4 = x^x + x^x)) := 
by
  sorry

end exactly_one_equals_xx_plus_xx_l1894_189438


namespace initial_population_first_village_equals_l1894_189415

-- Definitions of the conditions
def initial_population_second_village : ℕ := 42000
def decrease_first_village_per_year : ℕ := 1200
def increase_second_village_per_year : ℕ := 800
def years : ℕ := 13

-- Proposition we want to prove
/-- The initial population of the first village such that both villages have the same population after 13 years. -/
theorem initial_population_first_village_equals :
  ∃ (P : ℕ), (P - decrease_first_village_per_year * years) = (initial_population_second_village + increase_second_village_per_year * years) 
  := sorry

end initial_population_first_village_equals_l1894_189415


namespace total_marbles_l1894_189476

theorem total_marbles (marbles_per_row_8 : ℕ) (rows_of_9 : ℕ) (marbles_per_row_1 : ℕ) (rows_of_4 : ℕ) 
  (h1 : marbles_per_row_8 = 9) 
  (h2 : rows_of_9 = 8) 
  (h3 : marbles_per_row_1 = 4) 
  (h4 : rows_of_4 = 1) : 
  (marbles_per_row_8 * rows_of_9 + marbles_per_row_1 * rows_of_4) = 76 :=
by
  sorry

end total_marbles_l1894_189476


namespace crazy_silly_school_books_movies_correct_l1894_189411

noncomputable def crazy_silly_school_books_movies (B M : ℕ) : Prop :=
  M = 61 ∧ M = B + 2 ∧ M = 10 ∧ B = 8

theorem crazy_silly_school_books_movies_correct {B M : ℕ} :
  crazy_silly_school_books_movies B M → B = 8 :=
by
  intro h
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2.1
  have h4 := h.2.2.2
  sorry

end crazy_silly_school_books_movies_correct_l1894_189411


namespace ryan_chinese_learning_hours_l1894_189460

theorem ryan_chinese_learning_hours
    (hours_per_day : ℕ) 
    (days : ℕ) 
    (h1 : hours_per_day = 4) 
    (h2 : days = 6) : 
    hours_per_day * days = 24 := 
by 
    sorry

end ryan_chinese_learning_hours_l1894_189460


namespace number_of_merchants_l1894_189429

theorem number_of_merchants (x : ℕ) (h : 2 * x^3 = 2662) : x = 11 :=
  sorry

end number_of_merchants_l1894_189429


namespace quadrilateral_identity_l1894_189467

theorem quadrilateral_identity 
  {A B C D : Type*} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
  (AB : ℝ) (BC : ℝ) (CD : ℝ) (DA : ℝ) (AC : ℝ) (BD : ℝ)
  (angle_A : ℝ) (angle_C : ℝ) 
  (h_angle_sum : angle_A + angle_C = 120)
  : (AC * BD)^2 = (AB * CD)^2 + (BC * AD)^2 + AB * BC * CD * DA := 
by {
  sorry
}

end quadrilateral_identity_l1894_189467


namespace crazy_silly_school_movie_count_l1894_189470

theorem crazy_silly_school_movie_count
  (books : ℕ) (read_books : ℕ) (watched_movies : ℕ) (diff_books_movies : ℕ)
  (total_books : books = 8) 
  (read_movie_count : watched_movies = 19)
  (read_book_count : read_books = 16)
  (book_movie_diff : watched_movies = read_books + diff_books_movies)
  (diff_value : diff_books_movies = 3) :
  ∃ M, M ≥ 19 :=
by
  sorry

end crazy_silly_school_movie_count_l1894_189470


namespace sum_of_letters_l1894_189409

def A : ℕ := 0
def B : ℕ := 1
def C : ℕ := 2
def M : ℕ := 12

theorem sum_of_letters :
  A + B + M + C = 15 :=
by
  sorry

end sum_of_letters_l1894_189409


namespace movie_attendance_l1894_189471

theorem movie_attendance (total_seats : ℕ) (empty_seats : ℕ) (h1 : total_seats = 750) (h2 : empty_seats = 218) :
  total_seats - empty_seats = 532 := by
  sorry

end movie_attendance_l1894_189471


namespace marbles_given_to_joan_l1894_189435

def mary_original_marbles : ℝ := 9.0
def mary_marbles_left : ℝ := 6.0

theorem marbles_given_to_joan :
  mary_original_marbles - mary_marbles_left = 3 := 
by
  sorry

end marbles_given_to_joan_l1894_189435


namespace arithmetic_sequence_property_l1894_189419

variable {a : ℕ → ℝ} -- Define the arithmetic sequence
variable {S : ℕ → ℝ} -- Define the sum sequence
variable {d : ℝ} -- Define the common difference
variable {a1 : ℝ} -- Define the first term

-- Suppose the sum of the first 17 terms equals 306
axiom h1 : S 17 = 306
-- Suppose the sum of the first n terms of an arithmetic sequence formula
axiom sum_formula : ∀ n, S n = n * a1 + (n * (n - 1) / 2) * d
-- Suppose the relation between the first term, common difference and sum of the first 17 terms
axiom relation : a1 + 8 * d = 18 

theorem arithmetic_sequence_property : a 7 - (a 3) / 3 = 12 := 
by sorry

end arithmetic_sequence_property_l1894_189419


namespace scientific_notation_316000000_l1894_189445

theorem scientific_notation_316000000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 316000000 = a * 10 ^ n ∧ a = 3.16 ∧ n = 8 :=
by
  -- Proof would be here
  sorry

end scientific_notation_316000000_l1894_189445


namespace cut_out_square_possible_l1894_189459

/-- 
Formalization of cutting out eight \(2 \times 1\) rectangles from an \(8 \times 8\) 
checkered board, and checking if it is always possible to cut out a \(2 \times 2\) square
from the remaining part of the board.
-/
theorem cut_out_square_possible :
  ∀ (board : ℕ) (rectangles : ℕ), (board = 64) ∧ (rectangles = 8) → (4 ∣ board) →
  ∃ (remaining_squares : ℕ), (remaining_squares = 48) ∧ 
  (∃ (square_size : ℕ), (square_size = 4) ∧ (remaining_squares ≥ square_size)) :=
by {
  sorry
}

end cut_out_square_possible_l1894_189459


namespace minimize_segment_sum_l1894_189450

theorem minimize_segment_sum (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  ∃ x y : ℝ, x = Real.sqrt (a * b) ∧ y = Real.sqrt (a * b) ∧ x * y = a * b ∧ x + y = 2 * Real.sqrt (a * b) := 
by
  sorry

end minimize_segment_sum_l1894_189450


namespace imaginary_part_z1z2_l1894_189406

open Complex

-- Define the complex numbers z1 and z2
def z1 : ℂ := (1 : ℂ) - I
def z2 : ℂ := (2 : ℂ) + 4 * I

-- Define the product of z1 and z2
def z1z2 : ℂ := z1 * z2

-- State the theorem that the imaginary part of z1z2 is 2
theorem imaginary_part_z1z2 : z1z2.im = 2 := by
  -- Proof steps would go here
  sorry

end imaginary_part_z1z2_l1894_189406


namespace vectors_perpendicular_l1894_189412

def vec (a b : ℝ) := (a, b)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

@[simp]
def a := vec (-1) 2
@[simp]
def b := vec 1 3

theorem vectors_perpendicular :
  dot_product a (vector_sub a b) = 0 := by
  sorry

end vectors_perpendicular_l1894_189412


namespace probability_white_ball_l1894_189484

theorem probability_white_ball (num_black_balls num_white_balls : ℕ) 
  (black_balls : num_black_balls = 6) 
  (white_balls : num_white_balls = 5) : 
  (num_white_balls / (num_black_balls + num_white_balls) : ℚ) = 5 / 11 :=
by
  sorry

end probability_white_ball_l1894_189484


namespace power_function_inequality_l1894_189426

theorem power_function_inequality (m : ℕ) (h : m > 0)
  (h_point : (2 : ℝ) ^ (1 / (m ^ 2 + m)) = Real.sqrt 2) :
  m = 1 ∧ ∀ a : ℝ, 1 ≤ a ∧ a < (3 / 2) → 
  (2 - a : ℝ) ^ (1 / (m ^ 2 + m)) > (a - 1 : ℝ) ^ (1 / (m ^ 2 + m)) :=
by
  sorry

end power_function_inequality_l1894_189426


namespace log_equation_solution_l1894_189457

theorem log_equation_solution (x : ℝ) (h₁ : x > 0) (h₂ : x ≠ 1) (h₃ : x ≠ 1/16) (h₄ : x ≠ 1/2) 
    (h_eq : (Real.log 2 / Real.log (4 * Real.sqrt x)) / (Real.log 2 / Real.log (2 * x)) 
            + (Real.log 2 / Real.log (2 * x)) * (Real.log (2 * x) / Real.log (1 / 2)) = 0) 
    : x = 4 := 
sorry

end log_equation_solution_l1894_189457


namespace angelina_speed_from_library_to_gym_l1894_189418

theorem angelina_speed_from_library_to_gym :
  ∃ (v : ℝ), 
    (840 / v - 510 / (1.5 * v) = 40) ∧
    (510 / (1.5 * v) - 480 / (2 * v) = 20) ∧
    (2 * v = 25) :=
by
  sorry

end angelina_speed_from_library_to_gym_l1894_189418


namespace line_through_points_l1894_189440

theorem line_through_points (m b: ℝ) 
  (h1: ∃ m, ∀ x y : ℝ, ((x, y) = (1, 3) ∨ (x, y) = (3, 7)) → y = m * x + b) 
  (h2: ∀ x y : ℝ, ((x, y) = (1, 3) ∨ (x, y) = (3, 7)) → y = m * x + b):
  m + b = 3 :=
by
  -- proof goes here
  sorry

end line_through_points_l1894_189440


namespace minimize_tangent_triangle_area_l1894_189405

open Real

theorem minimize_tangent_triangle_area {a b x y : ℝ} 
  (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1) :
  (∃ x y : ℝ, (x = a / sqrt 2 ∨ x = -a / sqrt 2) ∧ (y = b / sqrt 2 ∨ y = -b / sqrt 2)) :=
by
  -- Proof is omitted
  sorry

end minimize_tangent_triangle_area_l1894_189405


namespace arithmetic_sequence_sum_l1894_189431

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Define the conditions
def a_5 := a 5
def a_6 := a 6
def a_7 := a 7

axiom cond1 : a_5 = 11
axiom cond2 : a_6 = 17
axiom cond3 : a_7 = 23

noncomputable def sum_first_four_terms : ℤ :=
  a 1 + a 2 + a 3 + a 4

theorem arithmetic_sequence_sum :
  a_5 = 11 → a_6 = 17 → a_7 = 23 → sum_first_four_terms a = -16 :=
by
  intros h5 h6 h7
  sorry

end arithmetic_sequence_sum_l1894_189431


namespace option_d_not_equal_four_thirds_l1894_189422

theorem option_d_not_equal_four_thirds :
  1 + (2 / 7) ≠ 4 / 3 :=
by
  sorry

end option_d_not_equal_four_thirds_l1894_189422


namespace inequality_of_positive_reals_l1894_189404

variable {a b c : ℝ}

theorem inequality_of_positive_reals (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ 3 / 2 :=
sorry

end inequality_of_positive_reals_l1894_189404


namespace least_integer_with_remainders_l1894_189464

theorem least_integer_with_remainders :
  ∃ M : ℕ, 
    M % 6 = 5 ∧
    M % 7 = 6 ∧
    M % 9 = 8 ∧
    M % 10 = 9 ∧
    M % 11 = 10 ∧
    M = 6929 :=
by
  sorry

end least_integer_with_remainders_l1894_189464


namespace negation_of_universal_abs_nonneg_l1894_189480

theorem negation_of_universal_abs_nonneg :
  (¬ (∀ x : ℝ, |x| ≥ 0)) ↔ (∃ x : ℝ, |x| < 0) :=
by
  sorry

end negation_of_universal_abs_nonneg_l1894_189480


namespace complex_plane_second_quadrant_l1894_189425

theorem complex_plane_second_quadrant (x : ℝ) :
  (x ^ 2 - 6 * x + 5 < 0 ∧ x - 2 > 0) ↔ (2 < x ∧ x < 5) :=
by
  -- The proof is to be completed.
  sorry

end complex_plane_second_quadrant_l1894_189425


namespace find_point_on_line_l1894_189489

theorem find_point_on_line (x y : ℝ) (h : 4 * x + 7 * y = 28) (hx : x = 3) : y = 16 / 7 :=
by
  sorry

end find_point_on_line_l1894_189489


namespace common_tangents_l1894_189446

noncomputable def circle1 := { p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 4 }
noncomputable def circle2 := { p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 2)^2 = 9 }

theorem common_tangents (h : ∀ p : ℝ × ℝ, p ∈ circle1 → p ∈ circle2) : 
  ∃ tangents : ℕ, tangents = 2 :=
sorry

end common_tangents_l1894_189446


namespace matinee_ticket_price_l1894_189423

theorem matinee_ticket_price
  (M : ℝ)  -- Denote M as the price of a matinee ticket
  (evening_ticket_price : ℝ := 12)  -- Price of an evening ticket
  (ticket_3D_price : ℝ := 20)  -- Price of a 3D ticket
  (matinee_tickets_sold : ℕ := 200)  -- Number of matinee tickets sold
  (evening_tickets_sold : ℕ := 300)  -- Number of evening tickets sold
  (tickets_3D_sold : ℕ := 100)  -- Number of 3D tickets sold
  (total_revenue : ℝ := 6600) -- Total revenue
  (h : matinee_tickets_sold * M + evening_tickets_sold * evening_ticket_price + tickets_3D_sold * ticket_3D_price = total_revenue) :
  M = 5 :=
by
  sorry

end matinee_ticket_price_l1894_189423


namespace factor_difference_of_squares_l1894_189469

theorem factor_difference_of_squares (x : ℝ) : x^2 - 81 = (x - 9) * (x + 9) := 
by
  sorry

end factor_difference_of_squares_l1894_189469


namespace water_pump_calculation_l1894_189444

-- Define the given initial conditions
variables (f h j g k l m : ℕ)

-- Provide the correctly calculated answer
theorem water_pump_calculation (hf : f > 0) (hg : g > 0) (hk : k > 0) (hm : m > 0) : 
  (k * l * m * j * h) / (10000 * f * g) = (k * (j * h / (f * g)) * l * m) / 10000 := 
sorry

end water_pump_calculation_l1894_189444


namespace academic_integers_l1894_189408

def is_academic (n : ℕ) (h : n ≥ 2) : Prop :=
  ∃ (S P : Finset ℕ), (S ∩ P = ∅) ∧ (S ∪ P = Finset.range (n + 1)) ∧ (S.sum id = P.prod id)

theorem academic_integers :
  { n | ∃ h : n ≥ 2, is_academic n h } = { n | n = 3 ∨ n ≥ 5 } :=
by
  sorry

end academic_integers_l1894_189408


namespace evaluate_g_of_neg_one_l1894_189414

def g (x : ℤ) : ℤ :=
  x^2 - 2*x + 1

theorem evaluate_g_of_neg_one :
  g (g (g (g (g (g (-1 : ℤ)))))) = 15738504 := by
  sorry

end evaluate_g_of_neg_one_l1894_189414


namespace wendy_tooth_extraction_cost_eq_290_l1894_189486

def dentist_cleaning_cost : ℕ := 70
def dentist_filling_cost : ℕ := 120
def wendy_dentist_bill : ℕ := 5 * dentist_filling_cost
def wendy_cleaning_and_fillings_cost : ℕ := dentist_cleaning_cost + 2 * dentist_filling_cost
def wendy_tooth_extraction_cost : ℕ := wendy_dentist_bill - wendy_cleaning_and_fillings_cost

theorem wendy_tooth_extraction_cost_eq_290 : wendy_tooth_extraction_cost = 290 := by
  sorry

end wendy_tooth_extraction_cost_eq_290_l1894_189486


namespace performance_stability_l1894_189436

theorem performance_stability (avg_score : ℝ) (num_shots : ℕ) (S_A S_B : ℝ) 
  (h_avg : num_shots = 10)
  (h_same_avg : avg_score = avg_score) 
  (h_SA : S_A^2 = 0.4) 
  (h_SB : S_B^2 = 2) : 
  (S_A < S_B) :=
by
  sorry

end performance_stability_l1894_189436


namespace geometric_sequence_fraction_l1894_189451

variable (a_1 : ℝ) (q : ℝ)

theorem geometric_sequence_fraction (h : q = 2) :
  (2 * a_1 + a_1 * q) / (2 * (a_1 * q^2) + a_1 * q^3) = 1 / 4 :=
by sorry

end geometric_sequence_fraction_l1894_189451


namespace discount_amount_l1894_189455

/-- Suppose Maria received a 25% discount on DVDs, and she paid $120.
    The discount she received is $40. -/
theorem discount_amount (P : ℝ) (h : 0.75 * P = 120) : P - 120 = 40 := 
sorry

end discount_amount_l1894_189455


namespace graph_crosses_x_axis_at_origin_l1894_189442

-- Let g(x) be a quadratic function defined as ax^2 + bx
def g (a b x : ℝ) : ℝ := a * x^2 + b * x

-- Define the conditions a ≠ 0 and b ≠ 0
axiom a_ne_0 (a : ℝ) : a ≠ 0
axiom b_ne_0 (b : ℝ) : b ≠ 0

-- The problem statement
theorem graph_crosses_x_axis_at_origin (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  ∃ x : ℝ, g a b x = 0 ∧ ∀ x', g a b x' = 0 → x' = 0 ∨ x' = -b / a :=
sorry

end graph_crosses_x_axis_at_origin_l1894_189442


namespace difference_between_largest_and_smallest_quarters_l1894_189481

noncomputable def coin_collection : Prop :=
  ∃ (n d q : ℕ), 
    (n + d + q = 150) ∧ 
    (5 * n + 10 * d + 25 * q = 2000) ∧ 
    (forall (q1 q2 : ℕ), (n + d + q1 = 150) ∧ (5 * n + 10 * d + 25 * q1 = 2000) → 
     (n + d + q2 = 150) ∧ (5 * n + 10 * d + 25 * q2 = 2000) → 
     (q1 = q2))

theorem difference_between_largest_and_smallest_quarters : coin_collection :=
  sorry

end difference_between_largest_and_smallest_quarters_l1894_189481


namespace marys_garbage_bill_is_correct_l1894_189443

noncomputable def calculate_garbage_bill :=
  let weekly_trash_bin_cost := 2 * 10
  let weekly_recycling_bin_cost := 1 * 5
  let weekly_green_waste_bin_cost := 1 * 3
  let total_weekly_cost := weekly_trash_bin_cost + weekly_recycling_bin_cost + weekly_green_waste_bin_cost
  let monthly_bin_cost := total_weekly_cost * 4
  let base_monthly_cost := monthly_bin_cost + 15
  let discount := base_monthly_cost * 0.18
  let discounted_cost := base_monthly_cost - discount
  let fines := 20 + 10
  discounted_cost + fines

theorem marys_garbage_bill_is_correct :
  calculate_garbage_bill = 134.14 := 
  by {
  sorry
  }

end marys_garbage_bill_is_correct_l1894_189443


namespace krishan_money_l1894_189477

theorem krishan_money
  (R G K : ℕ)
  (h1 : R / G = 7 / 17)
  (h2 : G / K = 7 / 17)
  (h3 : R = 637) : 
  K = 3774 := 
by
  sorry

end krishan_money_l1894_189477


namespace sum_of_four_terms_l1894_189441

theorem sum_of_four_terms (a d : ℕ) (h1 : a + d > a) (h2 : a + 2 * d > a + d)
  (h3 : (a + 2 * d) * (a + 2 * d) = (a + d) * (a + 3 * d)) (h4 : (a + 3 * d) - a = 30) :
  a + (a + d) + (a + 2 * d) + (a + 3 * d) = 129 :=
sorry

end sum_of_four_terms_l1894_189441


namespace gcd_2024_2048_l1894_189400

theorem gcd_2024_2048 : Nat.gcd 2024 2048 = 8 := 
by
  sorry

end gcd_2024_2048_l1894_189400


namespace no_solution_for_inequality_l1894_189483

theorem no_solution_for_inequality (x : ℝ) : ¬ (3 * x^2 + 9 * x ≤ -12) :=
by
  sorry

end no_solution_for_inequality_l1894_189483


namespace bowling_ball_weight_l1894_189430

theorem bowling_ball_weight (b k : ℝ) (h1 : 8 * b = 5 * k) (h2 : 4 * k = 120) : b = 18.75 :=
by
  sorry

end bowling_ball_weight_l1894_189430


namespace circle_parabola_intersection_l1894_189439

theorem circle_parabola_intersection (b : ℝ) :
  (∃ c r, ∀ x y : ℝ, y = (5 / 12) * x^2 → ((x - c)^2 + (y - b)^2 = r^2) ∧ 
   (y = (5 / 12) * x + b → ((x - c)^2 + (y - b)^2 = r^2))) → b = 169 / 60 :=
by
  sorry

end circle_parabola_intersection_l1894_189439


namespace correct_transformation_l1894_189488

variable {a b c : ℝ}

-- A: \frac{a+3}{b+3} = \frac{a}{b}
def transformation_A (a b : ℝ) : Prop := (a + 3) / (b + 3) = a / b

-- B: \frac{a}{b} = \frac{ac}{bc}
def transformation_B (a b c : ℝ) : Prop := a / b = (a * c) / (b * c)

-- C: \frac{3a}{3b} = \frac{a}{b}
def transformation_C (a b : ℝ) : Prop := (3 * a) / (3 * b) = a / b

-- D: \frac{a}{b} = \frac{a^2}{b^2}
def transformation_D (a b : ℝ) : Prop := a / b = (a ^ 2) / (b ^ 2)

-- The main theorem to prove
theorem correct_transformation : transformation_C a b :=
by
  sorry

end correct_transformation_l1894_189488


namespace fraction_of_new_releases_l1894_189433

theorem fraction_of_new_releases (total_books : ℕ) (historical_fiction_percent : ℝ) (historical_new_releases_percent : ℝ) (other_new_releases_percent : ℝ)
  (h1 : total_books = 100)
  (h2 : historical_fiction_percent = 0.4)
  (h3 : historical_new_releases_percent = 0.4)
  (h4 : other_new_releases_percent = 0.2) :
  (historical_fiction_percent * historical_new_releases_percent * total_books) / 
  ((historical_fiction_percent * historical_new_releases_percent * total_books) + ((1 - historical_fiction_percent) * other_new_releases_percent * total_books)) = 4 / 7 :=
by
  have h_books : total_books = 100 := h1
  have h_fiction : historical_fiction_percent = 0.4 := h2
  have h_new_releases : historical_new_releases_percent = 0.4 := h3
  have h_other_new_releases : other_new_releases_percent = 0.2 := h4
  sorry

end fraction_of_new_releases_l1894_189433


namespace fraction_eq_zero_implies_x_eq_one_l1894_189466

theorem fraction_eq_zero_implies_x_eq_one (x : ℝ) (h1 : (x - 1) = 0) (h2 : (x - 5) ≠ 0) : x = 1 :=
sorry

end fraction_eq_zero_implies_x_eq_one_l1894_189466


namespace division_of_powers_l1894_189495

theorem division_of_powers (a : ℝ) (h : a ≠ 0) : a^10 / a^9 = a := 
by sorry

end division_of_powers_l1894_189495


namespace other_endpoint_of_diameter_l1894_189416

-- Define the basic data
def center : ℝ × ℝ := (5, 2)
def endpoint1 : ℝ × ℝ := (0, -3)
def endpoint2 : ℝ × ℝ := (10, 7)

-- State the final properties to be proved
theorem other_endpoint_of_diameter :
  ∃ (e2 : ℝ × ℝ), e2 = endpoint2 ∧
    dist center endpoint2 = dist endpoint1 center :=
sorry

end other_endpoint_of_diameter_l1894_189416


namespace find_five_digit_number_l1894_189475

theorem find_five_digit_number
  (x y : ℕ)
  (h1 : 10 * y + x - (10000 * x + y) = 34767)
  (h2 : 10 * y + x + (10000 * x + y) = 86937) :
  10000 * x + y = 26035 := by
  sorry

end find_five_digit_number_l1894_189475


namespace expression_value_is_241_l1894_189492

noncomputable def expression_value : ℕ :=
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2

theorem expression_value_is_241 : expression_value = 241 := 
by
  sorry

end expression_value_is_241_l1894_189492


namespace one_minus_repeating_decimal_three_equals_two_thirds_l1894_189413

-- Define the repeating decimal as a fraction
def repeating_decimal_three : ℚ := 1 / 3

-- Prove the desired equality
theorem one_minus_repeating_decimal_three_equals_two_thirds :
  1 - repeating_decimal_three = 2 / 3 :=
by
  sorry

end one_minus_repeating_decimal_three_equals_two_thirds_l1894_189413


namespace central_angle_radian_measure_l1894_189403

namespace SectorProof

variables (R l : ℝ)
variables (α : ℝ)

-- Given conditions
def condition1 : Prop := 2 * R + l = 20
def condition2 : Prop := 1 / 2 * l * R = 9
def α_definition : Prop := α = l / R

-- Central angle result
theorem central_angle_radian_measure (h1 : condition1 R l) (h2 : condition2 R l) :
  α_definition α l R → α = 2 / 9 :=
by
  intro h_α
  -- proof steps would be here, but we skip them with sorry
  sorry

end SectorProof

end central_angle_radian_measure_l1894_189403


namespace parabola_vertex_below_x_axis_l1894_189458

theorem parabola_vertex_below_x_axis (a : ℝ) : (∀ x : ℝ, (x^2 + 2 * x + a < 0)) → a < 1 := 
by
  intro h
  -- proof step here
  sorry

end parabola_vertex_below_x_axis_l1894_189458


namespace escalator_steps_l1894_189448

theorem escalator_steps
  (steps_ascending : ℤ)
  (steps_descending : ℤ)
  (ascend_units_time : ℤ)
  (descend_units_time : ℤ)
  (speed_ratio : ℤ)
  (equation : ((steps_ascending : ℚ) / (1 + (ascend_units_time : ℚ))) = ((steps_descending : ℚ) / ((descend_units_time : ℚ) * speed_ratio)) )
  (solution_x : (125 * 0.6 = 75)) : 
  (steps_ascending * (1 + 0.6 : ℚ) = 120) :=
by
  sorry

end escalator_steps_l1894_189448


namespace exists_pair_distinct_integers_l1894_189454

theorem exists_pair_distinct_integers :
  ∃ (a b : ℤ), a ≠ b ∧ (a / 2015 + b / 2016 = (2015 + 2016) / (2015 * 2016)) :=
by
  -- Constructing the proof or using sorry to skip it if not needed here
  sorry

end exists_pair_distinct_integers_l1894_189454


namespace probability_yellow_second_l1894_189487

section MarbleProbabilities

def bag_A := (5, 6)     -- (white marbles, black marbles)
def bag_B := (3, 7)     -- (yellow marbles, blue marbles)
def bag_C := (5, 6)     -- (yellow marbles, blue marbles)

def P_white_A := 5 / 11
def P_black_A := 6 / 11
def P_yellow_given_B := 3 / 10
def P_yellow_given_C := 5 / 11

theorem probability_yellow_second :
  P_white_A * P_yellow_given_B + P_black_A * P_yellow_given_C = 33 / 121 :=
by
  -- Proof would be provided here
  sorry

end MarbleProbabilities

end probability_yellow_second_l1894_189487


namespace correct_pair_has_integer_distance_l1894_189428

-- Define the pairs of (x, y)
def pairs : List (ℕ × ℕ) :=
  [(88209, 90288), (82098, 89028), (28098, 89082), (90882, 28809)]

-- Define the property: a pair (x, y) has the distance √(x^2 + y^2) as an integer
def is_integer_distance_pair (x y : ℕ) : Prop :=
  ∃ (n : ℕ), n * n = x * x + y * y

-- Translate the problem to the proof: Prove (88209, 90288) satisfies the given property
theorem correct_pair_has_integer_distance :
  is_integer_distance_pair 88209 90288 :=
by
  sorry

end correct_pair_has_integer_distance_l1894_189428


namespace determine_a_l1894_189472

theorem determine_a (r s a : ℝ) (h1 : r^2 = a) (h2 : 2 * r * s = 16) (h3 : s^2 = 16) : a = 4 :=
by {
  sorry
}

end determine_a_l1894_189472


namespace curve_cartesian_eq_correct_intersection_distances_sum_l1894_189461

noncomputable section

def curve_parametric_eqns (θ : ℝ) : ℝ × ℝ := 
  (1 + 3 * Real.cos θ, 3 + 3 * Real.sin θ)

def line_parametric_eqns (t : ℝ) : ℝ × ℝ := 
  (3 + (1/2) * t, 3 + (Real.sqrt 3 / 2) * t)

def curve_cartesian_eq (x y : ℝ) : Prop := 
  (x - 1)^2 + (y - 3)^2 = 9

def point_p : ℝ × ℝ := 
  (3, 3)

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem curve_cartesian_eq_correct (θ : ℝ) : 
  curve_cartesian_eq (curve_parametric_eqns θ).1 (curve_parametric_eqns θ).2 := 
by 
  sorry

theorem intersection_distances_sum (t1 t2 : ℝ) 
  (h1 : curve_cartesian_eq (line_parametric_eqns t1).1 (line_parametric_eqns t1).2) 
  (h2 : curve_cartesian_eq (line_parametric_eqns t2).1 (line_parametric_eqns t2).2) : 
  distance point_p (line_parametric_eqns t1) + distance point_p (line_parametric_eqns t2) = 2 * Real.sqrt 3 := 
by 
  sorry

end curve_cartesian_eq_correct_intersection_distances_sum_l1894_189461


namespace opposite_sides_of_line_l1894_189497

theorem opposite_sides_of_line (m : ℝ) (h : (2 * (-2 : ℝ) + m - 2) * (2 * m + 4 - 2) < 0) : -1 < m ∧ m < 6 :=
sorry

end opposite_sides_of_line_l1894_189497


namespace next_volunteer_day_l1894_189468

-- Definitions based on conditions.
def Alison_schedule := 5
def Ben_schedule := 3
def Carla_schedule := 9
def Dave_schedule := 8

-- Main theorem
theorem next_volunteer_day : Nat.lcm Alison_schedule (Nat.lcm Ben_schedule (Nat.lcm Carla_schedule Dave_schedule)) = 360 := by
  sorry

end next_volunteer_day_l1894_189468


namespace intersect_complement_A_and_B_l1894_189474

noncomputable def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x + 1 < 0}
def B : Set ℝ := {x | x - 3 < 0}

theorem intersect_complement_A_and_B : (Set.compl A ∩ B) = {x | -1 ≤ x ∧ x < 3} := by
  sorry

end intersect_complement_A_and_B_l1894_189474


namespace problem1_problem2_l1894_189417

noncomputable def f (x : Real) : Real := 
  let a := (2 * Real.cos x, Real.sqrt 3 * Real.sin (2 * x))
  let b := (Real.cos x, 1)
  a.1 * b.1 + a.2 * b.2

theorem problem1 (x : Real) : 
  ∃ k : Int, - Real.pi / 3 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 6 + k * Real.pi :=
  sorry

theorem problem2 (A B C a b c : Real)
  (h1 : a = Real.sqrt 7)
  (h2 : Real.sin B = 2 * Real.sin C)
  (h3 : f A = 2)
  : (∃ area : Real, area = (7 * Real.sqrt 3) / 6) :=
  sorry

end problem1_problem2_l1894_189417


namespace question_d_l1894_189453

variable {x a : ℝ}

theorem question_d (h1 : x < a) (h2 : a < 0) : x^3 > a * x ∧ a * x < 0 :=
  sorry

end question_d_l1894_189453


namespace ice_cubes_per_cup_l1894_189432

theorem ice_cubes_per_cup (total_ice_cubes number_of_cups : ℕ) (h1 : total_ice_cubes = 30) (h2 : number_of_cups = 6) : 
  total_ice_cubes / number_of_cups = 5 := 
by
  sorry

end ice_cubes_per_cup_l1894_189432


namespace heesu_has_greatest_sum_l1894_189420

def sum_cards (cards : List Int) : Int :=
  cards.foldl (· + ·) 0

theorem heesu_has_greatest_sum :
  let sora_cards := [4, 6]
  let heesu_cards := [7, 5]
  let jiyeon_cards := [3, 8]
  sum_cards heesu_cards > sum_cards sora_cards ∧ sum_cards heesu_cards > sum_cards jiyeon_cards :=
by
  let sora_cards := [4, 6]
  let heesu_cards := [7, 5]
  let jiyeon_cards := [3, 8]
  sorry

end heesu_has_greatest_sum_l1894_189420


namespace integer_roots_condition_l1894_189494

noncomputable def has_integer_roots (n : ℕ) : Prop :=
  ∃ x : ℤ, x * x - 4 * x + n = 0

theorem integer_roots_condition (n : ℕ) (h : n > 0) :
  has_integer_roots n ↔ n = 3 ∨ n = 4 :=
by 
  sorry

end integer_roots_condition_l1894_189494


namespace solve_for_x_l1894_189482

theorem solve_for_x (x y z : ℚ) (h1 : x * y = 2 * (x + y)) (h2 : y * z = 4 * (y + z)) (h3 : x * z = 8 * (x + z)) (hx0 : x ≠ 0) (hy0 : y ≠ 0) (hz0 : z ≠ 0) : x = 16 / 3 := 
sorry

end solve_for_x_l1894_189482


namespace fencing_cost_approx_122_52_l1894_189407

noncomputable def circumference (d : ℝ) : ℝ := Real.pi * d

noncomputable def fencing_cost (d rate : ℝ) : ℝ := circumference d * rate

theorem fencing_cost_approx_122_52 :
  let d := 26
  let rate := 1.50
  abs (fencing_cost d rate - 122.52) < 1 :=
by
  let d : ℝ := 26
  let rate : ℝ := 1.50
  let cost := fencing_cost d rate
  sorry

end fencing_cost_approx_122_52_l1894_189407


namespace work_combined_days_l1894_189479

theorem work_combined_days (A B C : ℝ) (hA : A = 1 / 4) (hB : B = 1 / 12) (hC : C = 1 / 6) :
  1 / (A + B + C) = 2 :=
by
  sorry

end work_combined_days_l1894_189479


namespace sum_of_angles_l1894_189456

theorem sum_of_angles 
    (ABC_isosceles : ∃ (A B C : Type) (angleBAC : ℝ), (AB = AC) ∧ (angleBAC = 25))
    (DEF_isosceles : ∃ (D E F : Type) (angleEDF : ℝ), (DE = DF) ∧ (angleEDF = 40)) 
    (AD_parallel_CE : Prop) : 
    ∃ (angleDAC angleADE : ℝ), angleDAC = 77.5 ∧ angleADE = 70 ∧ (angleDAC + angleADE = 147.5) :=
by {
  sorry
}

end sum_of_angles_l1894_189456


namespace value_of_business_l1894_189491

theorem value_of_business 
  (ownership : ℚ)
  (sale_fraction : ℚ)
  (sale_value : ℚ) 
  (h_ownership : ownership = 2/3) 
  (h_sale_fraction : sale_fraction = 3/4) 
  (h_sale_value : sale_value = 6500) : 
  2 * sale_value = 13000 := 
by
  -- mathematical equivalent proof here
  -- This is a placeholder.
  sorry

end value_of_business_l1894_189491


namespace volume_of_rectangular_box_l1894_189473

theorem volume_of_rectangular_box 
  (l w h : ℝ)
  (h1 : l * w = 30)
  (h2 : w * h = 20)
  (h3 : l * h = 12) : 
  l * w * h = 60 :=
sorry

end volume_of_rectangular_box_l1894_189473


namespace m_minus_t_value_l1894_189463

-- Define the sum of squares of the odd integers from 1 to 215
def sum_squares_odds (n : ℕ) : ℕ := n * (4 * n^2 - 1) / 3

-- Define the sum of squares of the even integers from 2 to 100
def sum_squares_evens (n : ℕ) : ℕ := 2 * n * (n + 1) * (2 * n + 1) / 3

-- Number of odd terms from 1 to 215
def odd_terms_count : ℕ := (215 - 1) / 2 + 1

-- Number of even terms from 2 to 100
def even_terms_count : ℕ := (100 - 2) / 2 + 1

-- Define m and t
def m : ℕ := sum_squares_odds odd_terms_count
def t : ℕ := sum_squares_evens even_terms_count

-- Prove that m - t = 1507880
theorem m_minus_t_value : m - t = 1507880 :=
by
  -- calculations to verify the proof will be here, but are omitted for now
  sorry

end m_minus_t_value_l1894_189463


namespace no_non_degenerate_triangle_l1894_189424

theorem no_non_degenerate_triangle 
  (a b c : ℕ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ c) 
  (h3 : a ≠ c) 
  (h4 : Nat.gcd a (Nat.gcd b c) = 1) 
  (h5 : a ∣ (b - c) * (b - c)) 
  (h6 : b ∣ (a - c) * (a - c)) 
  (h7 : c ∣ (a - b) * (a - b)) : 
  ¬ (a < b + c ∧ b < a + c ∧ c < a + b) := 
sorry

end no_non_degenerate_triangle_l1894_189424


namespace bugs_eat_total_flowers_l1894_189402

theorem bugs_eat_total_flowers :
  let num_A := 3
  let num_B := 2
  let num_C := 1
  let flowers_A := 2
  let flowers_B := 3
  let flowers_C := 5
  let total := (num_A * flowers_A) + (num_B * flowers_B) + (num_C * flowers_C)
  total = 17 :=
by
  -- Applying given values to compute the total flowers eaten
  let num_A := 3
  let num_B := 2
  let num_C := 1
  let flowers_A := 2
  let flowers_B := 3
  let flowers_C := 5
  let total := (num_A * flowers_A) + (num_B * flowers_B) + (num_C * flowers_C)
  
  -- Verify the total is 17
  have h_total : total = 17 := 
    by
    sorry

  -- Proving the final result
  exact h_total

end bugs_eat_total_flowers_l1894_189402


namespace highest_power_of_2_divides_l1894_189434

def a : ℕ := 17
def b : ℕ := 15
def n : ℕ := a^5 - b^5

def highestPowerOf2Divides (k : ℕ) : ℕ :=
  -- Function to find the highest power of 2 that divides k, implementation is omitted
  sorry

theorem highest_power_of_2_divides :
  highestPowerOf2Divides n = 2^5 := by
    sorry

end highest_power_of_2_divides_l1894_189434


namespace least_integer_square_double_condition_l1894_189498

theorem least_integer_square_double_condition : ∃ x : ℤ, x^2 = 2 * x + 75 ∧ ∀ y : ℤ, y^2 = 2 * y + 75 → x ≤ y :=
by
  use -8
  sorry

end least_integer_square_double_condition_l1894_189498


namespace find_integer_pairs_l1894_189401

noncomputable def satisfies_equation (x y : ℤ) :=
  12 * x ^ 2 + 6 * x * y + 3 * y ^ 2 = 28 * (x + y)

theorem find_integer_pairs (m n : ℤ) :
  satisfies_equation (3 * m - 4 * n) (4 * n) :=
sorry

end find_integer_pairs_l1894_189401
