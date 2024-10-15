import Mathlib

namespace NUMINAMATH_GPT_count_valid_numbers_l1630_163032

def digits_set : List ℕ := [0, 2, 4, 7, 8, 9]

def divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

def divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def sum_digits (digits : List ℕ) : ℕ :=
  List.sum digits

def last_two_digits_divisibility (last_two_digits : ℕ) : Prop :=
  last_two_digits % 4 = 0

def number_is_valid (digits : List ℕ) : Prop :=
  sum_digits digits % 3 = 0

theorem count_valid_numbers :
  let possible_digits := [0, 2, 4, 7, 8, 9]
  let positions := 5
  let combinations := Nat.pow (List.length possible_digits) (positions - 1)
  let last_digit_choices := [0, 4, 8]
  3888 = 3 * combinations :=
sorry

end NUMINAMATH_GPT_count_valid_numbers_l1630_163032


namespace NUMINAMATH_GPT_common_external_tangent_b_l1630_163060

def circle1_center := (1, 3)
def circle1_radius := 3
def circle2_center := (10, 6)
def circle2_radius := 7

theorem common_external_tangent_b :
  ∃ (b : ℝ), ∀ (m : ℝ), m = 3 / 4 ∧ b = 9 / 4 := sorry

end NUMINAMATH_GPT_common_external_tangent_b_l1630_163060


namespace NUMINAMATH_GPT_age_sum_proof_l1630_163038

theorem age_sum_proof (a b c : ℕ) (h1 : a - (b + c) = 16) (h2 : a^2 - (b + c)^2 = 1632) : a + b + c = 102 :=
by
  sorry

end NUMINAMATH_GPT_age_sum_proof_l1630_163038


namespace NUMINAMATH_GPT_polyhedron_volume_correct_l1630_163090

-- Definitions of geometric shapes and their properties
def is_isosceles_right_triangle (A : Type) (a b c : ℝ) := 
  a = b ∧ c = a * Real.sqrt 2

def is_square (B : Type) (side : ℝ) := 
  side = 2

def is_equilateral_triangle (G : Type) (side : ℝ) := 
  side = Real.sqrt 8

noncomputable def polyhedron_volume (A E F B C D G : Type) (a b c d e f g : ℝ) := 
  let cube_volume := 8
  let tetrahedron_volume := 2 * Real.sqrt 2 / 3
  cube_volume - tetrahedron_volume

theorem polyhedron_volume_correct (A E F B C D G : Type) (a b c d e f g : ℝ) :
  (is_isosceles_right_triangle A a b c) →
  (is_isosceles_right_triangle E a b c) →
  (is_isosceles_right_triangle F a b c) →
  (is_square B d) →
  (is_square C e) →
  (is_square D f) →
  (is_equilateral_triangle G g) →
  a = 2 → d = 2 → e = 2 → f = 2 → g = Real.sqrt 8 →
  polyhedron_volume A E F B C D G a b c d e f g =
    8 - (2 * Real.sqrt 2 / 3) :=
by
  intros hA hE hF hB hC hD hG ha hd he hf hg
  sorry

end NUMINAMATH_GPT_polyhedron_volume_correct_l1630_163090


namespace NUMINAMATH_GPT_infinite_set_divisor_l1630_163049

open Set

noncomputable def exists_divisor (A : Set ℕ) : Prop :=
  ∃ (d : ℕ), d > 1 ∧ ∀ (a : ℕ), a ∈ A → d ∣ a

theorem infinite_set_divisor (A : Set ℕ) (hA1 : ∀ (b : Finset ℕ), (↑b ⊆ A) → ∃ (d : ℕ), d > 1 ∧ ∀ (a : ℕ), a ∈ b → d ∣ a) :
  exists_divisor A :=
sorry

end NUMINAMATH_GPT_infinite_set_divisor_l1630_163049


namespace NUMINAMATH_GPT_hoopit_toes_l1630_163056

theorem hoopit_toes (h : ℕ) : 
  (7 * (4 * h) + 8 * (2 * 5) = 164) -> h = 3 :=
by
  sorry

end NUMINAMATH_GPT_hoopit_toes_l1630_163056


namespace NUMINAMATH_GPT_question_1_question_2_l1630_163074

def f (x : ℝ) : ℝ := |x + 1| - |x - 4|

theorem question_1 (m : ℝ) :
  (∀ x : ℝ, f x ≤ -m^2 + 6 * m) ↔ (1 ≤ m ∧ m ≤ 5) :=
by
  sorry

theorem question_2 (a b c : ℝ) (h1 : 3 * a + 4 * b + 5 * c = 5) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 + b^2 + c^2 ≥ 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_question_1_question_2_l1630_163074


namespace NUMINAMATH_GPT_complete_square_eq_l1630_163069

theorem complete_square_eq (b c : ℤ) (h : ∃ b c : ℤ, (∀ x : ℝ, (x - 5)^2 = b * x + c) ∧ b + c = 5) :
  b + c = 5 :=
sorry

end NUMINAMATH_GPT_complete_square_eq_l1630_163069


namespace NUMINAMATH_GPT_find_S40_l1630_163030

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ := a * (r^n - 1) / (r - 1)

theorem find_S40 (a r : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = geometric_sequence_sum a r n)
  (h2 : S 10 = 10)
  (h3 : S 30 = 70) :
  S 40 = 150 ∨ S 40 = 110 := 
sorry

end NUMINAMATH_GPT_find_S40_l1630_163030


namespace NUMINAMATH_GPT_sum_series_eq_1_div_300_l1630_163055

noncomputable def sum_series : ℝ :=
  ∑' n, (6 * (n:ℝ) + 1) / ((6 * (n:ℝ) - 1) ^ 2 * (6 * (n:ℝ) + 5) ^ 2)

theorem sum_series_eq_1_div_300 : sum_series = 1 / 300 :=
  sorry

end NUMINAMATH_GPT_sum_series_eq_1_div_300_l1630_163055


namespace NUMINAMATH_GPT_number_of_boxes_l1630_163095

-- Definitions based on conditions
def bottles_per_box := 50
def bottle_capacity := 12
def fill_fraction := 3 / 4
def total_water := 4500

-- Question rephrased as a proof problem
theorem number_of_boxes (h1 : bottles_per_box = 50)
                        (h2 : bottle_capacity = 12)
                        (h3 : fill_fraction = 3 / 4)
                        (h4 : total_water = 4500) :
  4500 / ((12 : ℝ) * (3 / 4)) / 50 = 10 := 
by {
  sorry
}

end NUMINAMATH_GPT_number_of_boxes_l1630_163095


namespace NUMINAMATH_GPT_negation_proof_l1630_163020

-- Definitions based on conditions
def Line : Type := sorry  -- Define a type for lines (using sorry for now)
def Plane : Type := sorry  -- Define a type for planes (using sorry for now)

-- Condition definition
def is_perpendicular (l : Line) (α : Plane) : Prop := sorry  -- Define what it means for a plane to be perpendicular to a line (using sorry for now)

-- Given condition
axiom condition : ∀ (l : Line), ∃ (α : Plane), is_perpendicular l α

-- Statement to prove
theorem negation_proof : (∃ (l : Line), ∀ (α : Plane), ¬is_perpendicular l α) :=
sorry

end NUMINAMATH_GPT_negation_proof_l1630_163020


namespace NUMINAMATH_GPT_solve_equation_l1630_163066

theorem solve_equation (x : ℝ) :
  (1 / (x ^ 2 + 14 * x - 10)) + (1 / (x ^ 2 + 3 * x - 10)) + (1 / (x ^ 2 - 16 * x - 10)) = 0
  ↔ (x = 5 ∨ x = -2 ∨ x = 2 ∨ x = -5) :=
sorry

end NUMINAMATH_GPT_solve_equation_l1630_163066


namespace NUMINAMATH_GPT_Shekar_average_marks_l1630_163059

theorem Shekar_average_marks 
  (math_marks : ℕ := 76)
  (science_marks : ℕ := 65)
  (social_studies_marks : ℕ := 82)
  (english_marks : ℕ := 67)
  (biology_marks : ℕ := 95)
  (num_subjects : ℕ := 5) :
  (math_marks + science_marks + social_studies_marks + english_marks + biology_marks) / num_subjects = 77 := 
sorry

end NUMINAMATH_GPT_Shekar_average_marks_l1630_163059


namespace NUMINAMATH_GPT_complete_the_square_l1630_163001

theorem complete_the_square (d e f : ℤ) (h1 : 0 < d)
    (h2 : ∀ x : ℝ, 100 * x^2 + 60 * x - 90 = 0 ↔ (d * x + e)^2 = f) :
  d + e + f = 112 := by
  sorry

end NUMINAMATH_GPT_complete_the_square_l1630_163001


namespace NUMINAMATH_GPT_algebraic_expression_value_l1630_163052

theorem algebraic_expression_value (x : ℝ) (hx : x = 2 * Real.cos 45 + 1) :
  (1 / (x - 1) - (x - 3) / (x ^ 2 - 2 * x + 1)) / (2 / (x - 1)) = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1630_163052


namespace NUMINAMATH_GPT_tank_holds_21_liters_l1630_163022

def tank_capacity (S L : ℝ) : Prop :=
  (L = 2 * S + 3) ∧
  (L = 4) ∧
  (2 * S + 5 * L = 21)

theorem tank_holds_21_liters :
  ∃ S L : ℝ, tank_capacity S L :=
by
  use 1/2, 4
  unfold tank_capacity
  simp
  sorry

end NUMINAMATH_GPT_tank_holds_21_liters_l1630_163022


namespace NUMINAMATH_GPT_problem_statement_l1630_163047

variable {f : ℝ → ℝ}

-- Assume the conditions provided in the problem statement.
def continuous_on_ℝ (f : ℝ → ℝ) : Prop := Continuous f
def condition_x_f_prime (f : ℝ → ℝ) (h : ℝ → ℝ) : Prop := ∀ x : ℝ, x * h x < 0

-- The main theorem statement based on the conditions and the correct answer.
theorem problem_statement (hf : continuous_on_ℝ f) (hf' : ∀ x : ℝ, x * (deriv f x) < 0) :
  f (-1) + f 1 < 2 * f 0 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1630_163047


namespace NUMINAMATH_GPT_vertex_of_parabola_l1630_163046

-- Define the statement of the problem
theorem vertex_of_parabola :
  ∀ (a h k : ℝ), (∀ x : ℝ, 3 * (x - 5) ^ 2 + 4 = a * (x - h) ^ 2 + k) → (h, k) = (5, 4) :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_l1630_163046


namespace NUMINAMATH_GPT_total_heads_eq_fifteen_l1630_163098

-- Definitions for types of passengers and their attributes
def cats_heads : Nat := 7
def cats_legs : Nat := 7 * 4
def total_legs : Nat := 43
def captain_heads : Nat := 1
def captain_legs : Nat := 1

noncomputable def crew_heads (C : Nat) : Nat := C
noncomputable def crew_legs (C : Nat) : Nat := 2 * C

theorem total_heads_eq_fifteen : 
  ∃ (C : Nat),
    cats_legs + crew_legs C + captain_legs = total_legs ∧
    cats_heads + crew_heads C + captain_heads = 15 :=
by
  sorry

end NUMINAMATH_GPT_total_heads_eq_fifteen_l1630_163098


namespace NUMINAMATH_GPT_solve_inequality_part1_solve_inequality_part2_l1630_163094

-- Define the first part of the problem
theorem solve_inequality_part1 (a : ℝ) (x : ℝ) :
  (x^2 - a * x - 2 * a^2 < 0) ↔ 
    (a = 0 ∧ false) ∨ 
    (a > 0 ∧ -a < x ∧ x < 2 * a) ∨ 
    (a < 0 ∧ 2 * a < x ∧ x < -a) := 
sorry

-- Define the second part of the problem
theorem solve_inequality_part2 (a b : ℝ) (x : ℝ) 
  (h : { x | x^2 - a * x - b < 0 } = { x | -1 < x ∧ x < 2 }) :
  { x | a * x^2 + x - b > 0 } = { x | x < -2 } ∪ { x | 1 < x } :=
sorry

end NUMINAMATH_GPT_solve_inequality_part1_solve_inequality_part2_l1630_163094


namespace NUMINAMATH_GPT_initial_amount_of_A_l1630_163000

variable (a b c : ℕ)

-- Conditions
axiom condition1 : a - b - c = 32
axiom condition2 : b + c = 48
axiom condition3 : a + b + c = 128

-- The goal is to prove that A had 80 cents initially.
theorem initial_amount_of_A : a = 80 :=
by
  -- We need to skip the proof here
  sorry

end NUMINAMATH_GPT_initial_amount_of_A_l1630_163000


namespace NUMINAMATH_GPT_movie_store_additional_movie_needed_l1630_163039

theorem movie_store_additional_movie_needed (movies shelves : ℕ) (h_movies : movies = 999) (h_shelves : shelves = 5) : 
  (shelves - (movies % shelves)) % shelves = 1 :=
by
  sorry

end NUMINAMATH_GPT_movie_store_additional_movie_needed_l1630_163039


namespace NUMINAMATH_GPT_exists_polynomial_h_l1630_163028

variable {R : Type} [CommRing R] [IsDomain R] [CharZero R]

noncomputable def f (x : R) : ℝ := sorry -- define the polynomial f(x) here
noncomputable def g (x : R) : ℝ := sorry -- define the polynomial g(x) here

theorem exists_polynomial_h (m n : ℕ) (a : ℕ → ℝ) (b : ℕ → ℝ) (h_mn : m + n > 0)
  (h_fg_squares : ∀ x : ℝ, (∃ k : ℤ, f x = k^2) ↔ (∃ l : ℤ, g x = l^2)) :
  ∃ h : ℝ → ℝ, ∀ x : ℝ, f x * g x = (h x)^2 :=
sorry

end NUMINAMATH_GPT_exists_polynomial_h_l1630_163028


namespace NUMINAMATH_GPT_fewer_spoons_l1630_163054

/--
Stephanie initially planned to buy 15 pieces of each type of silverware.
There are 4 types of silverware.
This totals to 60 pieces initially planned to be bought.
She only bought 44 pieces in total.
Show that she decided to purchase 4 fewer spoons.
-/
theorem fewer_spoons
  (initial_total : ℕ := 60)
  (final_total : ℕ := 44)
  (types : ℕ := 4)
  (pieces_per_type : ℕ := 15) :
  (initial_total - final_total) / types = 4 := 
by
  -- since initial_total = 60, final_total = 44, and types = 4
  -- we need to prove (60 - 44) / 4 = 4
  sorry

end NUMINAMATH_GPT_fewer_spoons_l1630_163054


namespace NUMINAMATH_GPT_total_students_l1630_163067

theorem total_students (S : ℕ) (h1 : S / 2 / 2 = 250) : S = 1000 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l1630_163067


namespace NUMINAMATH_GPT_number_of_points_l1630_163023

theorem number_of_points (a b : ℤ) : (|a| = 3 ∧ |b| = 2) → ∃! (P : ℤ × ℤ), P = (a, b) :=
by sorry

end NUMINAMATH_GPT_number_of_points_l1630_163023


namespace NUMINAMATH_GPT_perimeter_ratio_l1630_163092

def original_paper : ℕ × ℕ := (12, 8)
def folded_paper : ℕ × ℕ := (original_paper.1, original_paper.2 / 2)
def small_rectangle : ℕ × ℕ := (folded_paper.1 / 2, folded_paper.2)

def perimeter (rect : ℕ × ℕ) : ℕ :=
  2 * (rect.1 + rect.2)

theorem perimeter_ratio :
  perimeter small_rectangle = 1 / 2 * perimeter original_paper :=
by
  sorry

end NUMINAMATH_GPT_perimeter_ratio_l1630_163092


namespace NUMINAMATH_GPT_raspberry_pies_l1630_163041

theorem raspberry_pies (total_pies : ℕ) (r_peach : ℕ) (r_strawberry : ℕ) (r_raspberry : ℕ) (r_sum : ℕ) :
    total_pies = 36 → r_peach = 2 → r_strawberry = 5 → r_raspberry = 3 → r_sum = (r_peach + r_strawberry + r_raspberry) →
    (total_pies : ℝ) / (r_sum : ℝ) * (r_raspberry : ℝ) = 10.8 :=
by
    -- This theorem is intended to state the problem.
    sorry

end NUMINAMATH_GPT_raspberry_pies_l1630_163041


namespace NUMINAMATH_GPT_mangoes_count_l1630_163099

noncomputable def total_fruits : ℕ := 58
noncomputable def pears : ℕ := 10
noncomputable def pawpaws : ℕ := 12
noncomputable def lemons : ℕ := 9
noncomputable def kiwi : ℕ := 9

theorem mangoes_count (mangoes : ℕ) : 
  (pears + pawpaws + lemons + kiwi + mangoes = total_fruits) → 
  mangoes = 18 :=
by
  sorry

end NUMINAMATH_GPT_mangoes_count_l1630_163099


namespace NUMINAMATH_GPT_number_of_sides_of_polygon_l1630_163093

theorem number_of_sides_of_polygon (n : ℕ) :
  ((n - 2) * 180 = 3 * 360) → n = 8 := 
by
  sorry

end NUMINAMATH_GPT_number_of_sides_of_polygon_l1630_163093


namespace NUMINAMATH_GPT_fewest_tiles_needed_to_cover_rectangle_l1630_163002

noncomputable def height_of_equilateral_triangle (side_length : ℝ) : ℝ :=
  (Real.sqrt 3 / 2) * side_length

noncomputable def area_of_equilateral_triangle (side_length : ℝ) : ℝ :=
  (1 / 2) * side_length * height_of_equilateral_triangle side_length

noncomputable def area_of_floor_in_square_inches (length_in_feet : ℝ) (width_in_feet : ℝ) : ℝ :=
  length_in_feet * width_in_feet * (12 * 12)

noncomputable def number_of_tiles_required (floor_area : ℝ) (tile_area : ℝ) : ℝ :=
  floor_area / tile_area

theorem fewest_tiles_needed_to_cover_rectangle :
  number_of_tiles_required (area_of_floor_in_square_inches 3 4) (area_of_equilateral_triangle 2) = 997 := 
by
  sorry

end NUMINAMATH_GPT_fewest_tiles_needed_to_cover_rectangle_l1630_163002


namespace NUMINAMATH_GPT_frequency_count_l1630_163012

theorem frequency_count (n : ℕ) (f : ℝ) (h1 : n = 1000) (h2 : f = 0.4) : n * f = 400 := by
  sorry

end NUMINAMATH_GPT_frequency_count_l1630_163012


namespace NUMINAMATH_GPT_Ella_jellybeans_l1630_163008

-- Definitions based on conditions from part (a)
def Dan_volume := 10
def Dan_jellybeans := 200
def scaling_factor := 3

-- Prove that Ella's box holds 5400 jellybeans
theorem Ella_jellybeans : scaling_factor^3 * Dan_jellybeans = 5400 := 
by
  sorry

end NUMINAMATH_GPT_Ella_jellybeans_l1630_163008


namespace NUMINAMATH_GPT_find_multiple_l1630_163010

theorem find_multiple (a b m : ℤ) (h1 : b = 7) (h2 : b - a = 2) 
  (h3 : a * b = m * (a + b) + 11) : m = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_multiple_l1630_163010


namespace NUMINAMATH_GPT_number_of_students_l1630_163083

theorem number_of_students (n : ℕ)
  (h1 : ∃ n, (175 * n) / n = 175)
  (h2 : 175 * n - 40 = 173 * n) :
  n = 20 :=
sorry

end NUMINAMATH_GPT_number_of_students_l1630_163083


namespace NUMINAMATH_GPT_problem_statement_l1630_163082

/-- Define the sequence of numbers spoken by Jo and Blair. -/
def next_number (n : ℕ) : ℕ :=
if n % 2 = 1 then (n + 1) / 2 else n / 2

/-- Helper function to compute the 21st number said. -/
noncomputable def twenty_first_number : ℕ :=
(21 + 1) / 2

/-- Statement of the problem in Lean 4. -/
theorem problem_statement : twenty_first_number = 11 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1630_163082


namespace NUMINAMATH_GPT_valid_license_plates_count_l1630_163040

def validLicensePlates : Nat :=
  26 * 26 * 26 * 10 * 9 * 8

theorem valid_license_plates_count :
  validLicensePlates = 15818400 :=
by
  sorry

end NUMINAMATH_GPT_valid_license_plates_count_l1630_163040


namespace NUMINAMATH_GPT_rabbits_ate_three_potatoes_l1630_163007

variable (initial_potatoes remaining_potatoes eaten_potatoes : ℕ)

-- Definitions from the conditions
def mary_initial_potatoes : initial_potatoes = 8 := sorry
def mary_remaining_potatoes : remaining_potatoes = 5 := sorry

-- The goal to prove
theorem rabbits_ate_three_potatoes :
  initial_potatoes - remaining_potatoes = 3 := sorry

end NUMINAMATH_GPT_rabbits_ate_three_potatoes_l1630_163007


namespace NUMINAMATH_GPT_volume_of_cylindrical_block_l1630_163044

variable (h_cylindrical : ℕ) (combined_value : ℝ)

theorem volume_of_cylindrical_block (h_cylindrical : ℕ) (combined_value : ℝ):
  h_cylindrical = 3 → combined_value / 5 * h_cylindrical = 15.42 := by
suffices combined_value / 5 = 5.14 from sorry
suffices 5.14 * 3 = 15.42 from sorry
suffices h_cylindrical = 3 from sorry
suffices 25.7 = combined_value from sorry
sorry

end NUMINAMATH_GPT_volume_of_cylindrical_block_l1630_163044


namespace NUMINAMATH_GPT_two_zeros_of_cubic_polynomial_l1630_163076

theorem two_zeros_of_cubic_polynomial (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ -x1^3 + 3*x1 + m = 0 ∧ -x2^3 + 3*x2 + m = 0) →
  (m = -2 ∨ m = 2) :=
by
  sorry

end NUMINAMATH_GPT_two_zeros_of_cubic_polynomial_l1630_163076


namespace NUMINAMATH_GPT_weight_of_rod_l1630_163084

theorem weight_of_rod (w₆ : ℝ) (h₁ : w₆ = 6.1) : 
  w₆ / 6 * 12 = 12.2 := by
  sorry

end NUMINAMATH_GPT_weight_of_rod_l1630_163084


namespace NUMINAMATH_GPT_expression_is_product_l1630_163064

def not_sum (a x : Int) : Prop :=
  ¬(a + x = -7 * x)

def not_difference (a x : Int) : Prop :=
  ¬(a - x = -7 * x)

def not_quotient (a x : Int) : Prop :=
  ¬(a / x = -7 * x)

theorem expression_is_product (x : Int) : 
  not_sum (-7) x ∧ not_difference (-7) x ∧ not_quotient (-7) x → (-7 * x = -7 * x) :=
by sorry

end NUMINAMATH_GPT_expression_is_product_l1630_163064


namespace NUMINAMATH_GPT_smallest_visible_sum_of_3x3x3_cube_is_90_l1630_163033

theorem smallest_visible_sum_of_3x3x3_cube_is_90 
: ∀ (dices: Fin 27 → Fin 6 → ℕ),
    (∀ i j k, dices (3*i+j) k = 7 - dices (3*i+j) (5-k)) → 
    (∃ s, s = 90 ∧
    s = (8 * (dices 0 0 + dices 0 1 + dices 0 2)) + 
        (12 * (dices 0 0 + dices 0 1)) +
        (6 * (dices 0 0))) := sorry

end NUMINAMATH_GPT_smallest_visible_sum_of_3x3x3_cube_is_90_l1630_163033


namespace NUMINAMATH_GPT_power_sum_inequality_l1630_163043

theorem power_sum_inequality (k l m : ℕ) : 
  2 ^ (k + l) + 2 ^ (k + m) + 2 ^ (l + m) ≤ 2 ^ (k + l + m + 1) + 1 := 
by 
  sorry

end NUMINAMATH_GPT_power_sum_inequality_l1630_163043


namespace NUMINAMATH_GPT_maria_high_school_students_l1630_163026

variable (M D : ℕ)

theorem maria_high_school_students (h1 : M = 4 * D) (h2 : M - D = 1800) : M = 2400 :=
by
  sorry

end NUMINAMATH_GPT_maria_high_school_students_l1630_163026


namespace NUMINAMATH_GPT_geometric_sequence_product_l1630_163005

variable {α : Type*} [LinearOrderedField α]
variable (a : ℕ → α)

theorem geometric_sequence_product (h : a 7 * a 12 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_geometric_sequence_product_l1630_163005


namespace NUMINAMATH_GPT_percent_increase_is_equivalent_l1630_163057

variable {P : ℝ}

theorem percent_increase_is_equivalent 
  (h1 : 1.0 + 15.0 / 100.0 = 1.15)
  (h2 : 1.15 * (1.0 + 25.0 / 100.0) = 1.4375)
  (h3 : 1.4375 * (1.0 + 10.0 / 100.0) = 1.58125) :
  (1.58125 - 1) * 100 = 58.125 :=
by
  sorry

end NUMINAMATH_GPT_percent_increase_is_equivalent_l1630_163057


namespace NUMINAMATH_GPT_determine_k_for_one_real_solution_l1630_163070

theorem determine_k_for_one_real_solution (k : ℝ):
  (∃ x : ℝ, 9 * x^2 + k * x + 49 = 0 ∧ (∀ y : ℝ, 9 * y^2 + k * y + 49 = 0 → y = x)) → k = 42 :=
sorry

end NUMINAMATH_GPT_determine_k_for_one_real_solution_l1630_163070


namespace NUMINAMATH_GPT_ratio_of_pq_l1630_163011

def is_pure_imaginary (z : Complex) : Prop :=
  z.re = 0

theorem ratio_of_pq (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (H : is_pure_imaginary ((Complex.ofReal 3 - Complex.ofReal 4 * Complex.I) * (Complex.ofReal p + Complex.ofReal q * Complex.I))) :
  p / q = -4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_pq_l1630_163011


namespace NUMINAMATH_GPT_new_mean_rent_l1630_163063

theorem new_mean_rent
  (num_friends : ℕ)
  (avg_rent : ℕ)
  (original_rent_increased : ℕ)
  (increase_percentage : ℝ)
  (new_mean_rent : ℕ) :
  num_friends = 4 →
  avg_rent = 800 →
  original_rent_increased = 1400 →
  increase_percentage = 0.2 →
  new_mean_rent = 870 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_new_mean_rent_l1630_163063


namespace NUMINAMATH_GPT_smallest_sum_ending_2050306_l1630_163051

/--
Given nine consecutive natural numbers starting at n,
prove that the smallest sum of these nine numbers ending in 2050306 is 22050306.
-/
theorem smallest_sum_ending_2050306 
  (n : ℕ) 
  (hn : ∃ m : ℕ, 9 * m = (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8)) ∧ 
                 (9 * m) % 10^7 = 2050306) :
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8)) = 22050306 := 
sorry

end NUMINAMATH_GPT_smallest_sum_ending_2050306_l1630_163051


namespace NUMINAMATH_GPT_total_bottles_remaining_is_14090_l1630_163034

-- Define the constants
def total_small_bottles : ℕ := 5000
def total_big_bottles : ℕ := 12000
def small_bottles_sold_percentage : ℕ := 15
def big_bottles_sold_percentage : ℕ := 18

-- Define the remaining bottles
def calc_remaining_bottles (total_bottles sold_percentage : ℕ) : ℕ :=
  total_bottles - (sold_percentage * total_bottles / 100)

-- Define the remaining small and big bottles
def remaining_small_bottles : ℕ := calc_remaining_bottles total_small_bottles small_bottles_sold_percentage
def remaining_big_bottles : ℕ := calc_remaining_bottles total_big_bottles big_bottles_sold_percentage

-- Define the total remaining bottles
def total_remaining_bottles : ℕ := remaining_small_bottles + remaining_big_bottles

-- State the theorem
theorem total_bottles_remaining_is_14090 : total_remaining_bottles = 14090 := by
  sorry

end NUMINAMATH_GPT_total_bottles_remaining_is_14090_l1630_163034


namespace NUMINAMATH_GPT_lower_bound_third_inequality_l1630_163079

theorem lower_bound_third_inequality (x : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 18)
  (h3 : 8 > x ∧ x > 0)
  (h4 : x + 1 < 9) :
  x = 7 → ∃ l < 7, ∀ y, l < y ∧ y < 9 → y = x := 
sorry

end NUMINAMATH_GPT_lower_bound_third_inequality_l1630_163079


namespace NUMINAMATH_GPT_consecutive_integers_solution_l1630_163087

theorem consecutive_integers_solution :
  ∃ (n : ℕ), n > 0 ∧ n * (n + 1) + 91 = n^2 + (n + 1)^2 ∧ n + 1 = 10 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_integers_solution_l1630_163087


namespace NUMINAMATH_GPT_a_left_after_working_days_l1630_163077

variable (x : ℕ)  -- x represents the days A worked 

noncomputable def A_work_rate := (1 : ℚ) / 21
noncomputable def B_work_rate := (1 : ℚ) / 28
noncomputable def B_remaining_work := (3 : ℚ) / 4
noncomputable def combined_work_rate := A_work_rate + B_work_rate

theorem a_left_after_working_days 
  (h : combined_work_rate * x + B_remaining_work = 1) : x = 3 :=
by 
  sorry

end NUMINAMATH_GPT_a_left_after_working_days_l1630_163077


namespace NUMINAMATH_GPT_martha_apples_l1630_163037

theorem martha_apples (initial_apples jane_apples extra_apples final_apples : ℕ)
  (h1 : initial_apples = 20)
  (h2 : jane_apples = 5)
  (h3 : extra_apples = 2)
  (h4 : final_apples = 4) :
  initial_apples - jane_apples - (jane_apples + extra_apples) - final_apples = final_apples := 
by
  sorry

end NUMINAMATH_GPT_martha_apples_l1630_163037


namespace NUMINAMATH_GPT_factorization_divisibility_l1630_163019

theorem factorization_divisibility (n : ℤ) : 120 ∣ (n^5 - 5 * n^3 + 4 * n) :=
by
  sorry

end NUMINAMATH_GPT_factorization_divisibility_l1630_163019


namespace NUMINAMATH_GPT_stratified_sample_correct_l1630_163097

variable (popA popB popC : ℕ) (totalSample : ℕ)

def stratified_sample (popA popB popC totalSample : ℕ) : ℕ × ℕ × ℕ :=
  let totalChickens := popA + popB + popC
  let sampledA := (popA * totalSample) / totalChickens
  let sampledB := (popB * totalSample) / totalChickens
  let sampledC := (popC * totalSample) / totalChickens
  (sampledA, sampledB, sampledC)

theorem stratified_sample_correct
  (hA : popA = 12000) (hB : popB = 8000) (hC : popC = 4000) (hSample : totalSample = 120) :
  stratified_sample popA popB popC totalSample = (60, 40, 20) :=
by
  sorry

end NUMINAMATH_GPT_stratified_sample_correct_l1630_163097


namespace NUMINAMATH_GPT_trapezium_area_proof_l1630_163086

def trapeziumArea (a b h : ℕ) : ℕ :=
  (1 / 2) * (a + b) * h

theorem trapezium_area_proof :
  let a := 20
  let b := 18
  let h := 14
  trapeziumArea a b h = 266 := by
  sorry

end NUMINAMATH_GPT_trapezium_area_proof_l1630_163086


namespace NUMINAMATH_GPT_bucket_full_weight_l1630_163018

variable (p q : ℝ)

theorem bucket_full_weight (p q : ℝ) (x y: ℝ) (h1 : x + 3/4 * y = p) (h2 : x + 1/3 * y = q) :
  x + y = (8 * p - 7 * q) / 5 :=
by
  sorry

end NUMINAMATH_GPT_bucket_full_weight_l1630_163018


namespace NUMINAMATH_GPT_no_eight_consecutive_sums_in_circle_l1630_163058

theorem no_eight_consecutive_sums_in_circle :
  ¬ ∃ (arrangement : Fin 8 → ℕ) (sums : Fin 8 → ℤ),
      (∀ i, 1 ≤ arrangement i ∧ arrangement i ≤ 8) ∧
      (∀ i, sums i = arrangement i + arrangement (⟨(i + 1) % 8, sorry⟩)) ∧
      (∃ (n : ℤ), 
        (sums 0 = n - 3) ∧ 
        (sums 1 = n - 2) ∧ 
        (sums 2 = n - 1) ∧ 
        (sums 3 = n) ∧ 
        (sums 4 = n + 1) ∧ 
        (sums 5 = n + 2) ∧ 
        (sums 6 = n + 3) ∧ 
        (sums 7 = n + 4)) := 
sorry

end NUMINAMATH_GPT_no_eight_consecutive_sums_in_circle_l1630_163058


namespace NUMINAMATH_GPT_second_group_people_l1630_163017

theorem second_group_people (x : ℕ) (K : ℕ) (hK : K > 0) :
  (96 - 16 = K * (x + 16) + 6) → (x = 58 ∨ x = 21) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_second_group_people_l1630_163017


namespace NUMINAMATH_GPT_pet_store_cages_l1630_163003

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage : ℕ) (h₁ : initial_puppies = 78)
(h₂ : sold_puppies = 30) (h₃ : puppies_per_cage = 8) : (initial_puppies - sold_puppies) / puppies_per_cage = 6 :=
by
  -- assumptions: initial_puppies = 78, sold_puppies = 30, puppies_per_cage = 8
  -- goal: (initial_puppies - sold_puppies) / puppies_per_cage = 6
  sorry

end NUMINAMATH_GPT_pet_store_cages_l1630_163003


namespace NUMINAMATH_GPT_area_region_inside_but_outside_l1630_163006

noncomputable def area_diff (side_large side_small : ℝ) : ℝ :=
  (side_large ^ 2) - (side_small ^ 2)

theorem area_region_inside_but_outside (h_large : 10 > 0) (h_small : 4 > 0) :
  area_diff 10 4 = 84 :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_area_region_inside_but_outside_l1630_163006


namespace NUMINAMATH_GPT_arithmetic_sequence_condition_l1630_163091

theorem arithmetic_sequence_condition {a : ℕ → ℤ} 
  (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (m p q : ℕ) (hpq_pos : 0 < p) (hq_pos : 0 < q) (hm_pos : 0 < m) : 
  (p + q = 2 * m) → (a p + a q = 2 * a m) ∧ ¬((a p + a q = 2 * a m) → (p + q = 2 * m)) :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_condition_l1630_163091


namespace NUMINAMATH_GPT_sum_of_slopes_range_l1630_163081

theorem sum_of_slopes_range (p b : ℝ) (hpb : 2 * p > b) (hp : p > 0) 
  (K1 K2 : ℝ) (A B : ℝ × ℝ) (hA : A.2^2 = 2 * p * A.1) (hB : B.2^2 = 2 * p * B.1)
  (hl1 : A.2 = A.1 + b) (hl2 : B.2 = B.1 + b) 
  (hA_pos : A.2 > 0) (hB_pos : B.2 > 0) :
  4 < K1 + K2 :=
sorry

end NUMINAMATH_GPT_sum_of_slopes_range_l1630_163081


namespace NUMINAMATH_GPT_percentage_increase_area_rectangle_l1630_163025

theorem percentage_increase_area_rectangle (L W : ℝ) :
  let new_length := 1.20 * L
  let new_width := 1.20 * W
  let original_area := L * W
  let new_area := new_length * new_width
  let percentage_increase := ((new_area - original_area) / original_area) * 100
  percentage_increase = 44 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_area_rectangle_l1630_163025


namespace NUMINAMATH_GPT_sunflower_is_taller_l1630_163015

def sister_height_ft : Nat := 4
def sister_height_in : Nat := 3
def sunflower_height_ft : Nat := 6

def feet_to_inches (ft : Nat) : Nat := ft * 12

def sister_height := feet_to_inches sister_height_ft + sister_height_in
def sunflower_height := feet_to_inches sunflower_height_ft

def height_difference : Nat := sunflower_height - sister_height

theorem sunflower_is_taller : height_difference = 21 :=
by
  -- proof has to be provided:
  sorry

end NUMINAMATH_GPT_sunflower_is_taller_l1630_163015


namespace NUMINAMATH_GPT_hexagon_interior_angle_Q_l1630_163021

theorem hexagon_interior_angle_Q 
  (A B C D E F : ℕ)
  (hA : A = 135) (hB : B = 150) (hC : C = 120) (hD : D = 130) (hE : E = 100)
  (hex_angle_sum : A + B + C + D + E + F = 720) :
  F = 85 :=
by
  rw [hA, hB, hC, hD, hE] at hex_angle_sum
  sorry

end NUMINAMATH_GPT_hexagon_interior_angle_Q_l1630_163021


namespace NUMINAMATH_GPT_max_profit_achieved_at_180_l1630_163089

-- Definitions:
def cost (x : ℝ) : ℝ := 0.1 * x^2 - 11 * x + 3000  -- Condition 1
def selling_price_per_unit : ℝ := 25  -- Condition 2

-- Statement to prove that the maximum profit is achieved at x = 180
theorem max_profit_achieved_at_180 :
  ∃ (S : ℝ), ∀ (x : ℝ),
    S = -0.1 * (x - 180)^2 + 240 → S = 25 * 180 - cost 180 :=
by
  sorry

end NUMINAMATH_GPT_max_profit_achieved_at_180_l1630_163089


namespace NUMINAMATH_GPT_find_x_l1630_163078

theorem find_x (x : ℝ) (y : ℝ) : (∀ y, 10 * x * y - 15 * y + 2 * x - 3 = 0) → x = 3 / 2 := by
  intros h
  -- At this point, you would include the necessary proof steps, but for now we skip it.
  sorry

end NUMINAMATH_GPT_find_x_l1630_163078


namespace NUMINAMATH_GPT_domain_f_2x_minus_1_l1630_163036

theorem domain_f_2x_minus_1 (f : ℝ → ℝ) : 
  (∀ x, 1 ≤ x ∧ x ≤ 2 → 2 ≤ x + 1 ∧ x + 1 ≤ 3) → 
  (∀ z, 2 ≤ 2 * z - 1 ∧ 2 * z - 1 ≤ 3 → ∃ x, 3/2 ≤ x ∧ x ≤ 2 ∧ 2 * x - 1 = z) := 
sorry

end NUMINAMATH_GPT_domain_f_2x_minus_1_l1630_163036


namespace NUMINAMATH_GPT_distance_between_A_and_B_is_45_kilometers_l1630_163062

variable (speedA speedB : ℝ)
variable (distanceAB : ℝ)

noncomputable def problem_conditions := 
  speedA = 1.2 * speedB ∧
  ∃ (distanceMalfunction : ℝ), distanceMalfunction = 5 ∧
  ∃ (timeFixingMalfunction : ℝ), timeFixingMalfunction = (distanceAB / 6) / speedB ∧
  ∃ (increasedSpeedB : ℝ), increasedSpeedB = 1.6 * speedB ∧
  ∃ (timeA timeB timeB_new : ℝ),
    timeA = (distanceAB / speedA) ∧
    timeB = (distanceMalfunction / speedB) + timeFixingMalfunction + (distanceAB - distanceMalfunction) / increasedSpeedB ∧
    timeA = timeB

theorem distance_between_A_and_B_is_45_kilometers
  (speedA speedB distanceAB : ℝ) 
  (cond : problem_conditions speedA speedB distanceAB) :
  distanceAB = 45 :=
sorry

end NUMINAMATH_GPT_distance_between_A_and_B_is_45_kilometers_l1630_163062


namespace NUMINAMATH_GPT_unit_place_3_pow_34_l1630_163029

theorem unit_place_3_pow_34 : Nat.mod (3^34) 10 = 9 :=
by
  sorry

end NUMINAMATH_GPT_unit_place_3_pow_34_l1630_163029


namespace NUMINAMATH_GPT_max_value_isosceles_triangle_l1630_163072

theorem max_value_isosceles_triangle (a b c : ℝ) (h_isosceles : b = c) :
  ∃ B, (∀ (a b c : ℝ), b = c → (b + c) / a ≤ B) ∧ B = 2 :=
by
  sorry

end NUMINAMATH_GPT_max_value_isosceles_triangle_l1630_163072


namespace NUMINAMATH_GPT_cost_per_chicken_l1630_163088

-- Definitions for conditions
def totalBirds : ℕ := 15
def ducks : ℕ := totalBirds / 3
def chickens : ℕ := totalBirds - ducks
def feed_cost : ℕ := 20

-- Theorem stating the cost per chicken
theorem cost_per_chicken : (feed_cost / chickens) = 2 := by
  sorry

end NUMINAMATH_GPT_cost_per_chicken_l1630_163088


namespace NUMINAMATH_GPT_neg_of_univ_prop_l1630_163009

theorem neg_of_univ_prop :
  (∃ (x₀ : ℝ), 0 ≤ x₀ ∧ x₀^3 + x₀ < 0) ↔ ¬ (∀ (x : ℝ), 0 ≤ x → x^3 + x ≥ 0) := by
sorry

end NUMINAMATH_GPT_neg_of_univ_prop_l1630_163009


namespace NUMINAMATH_GPT_typing_difference_l1630_163096

theorem typing_difference (initial_speed after_speed : ℕ) (time_interval : ℕ) (h_initial : initial_speed = 10) 
  (h_after : after_speed = 8) (h_time : time_interval = 5) : 
  (initial_speed * time_interval) - (after_speed * time_interval) = 10 := 
by 
  sorry

end NUMINAMATH_GPT_typing_difference_l1630_163096


namespace NUMINAMATH_GPT_closest_points_to_A_l1630_163075

noncomputable def distance_squared (x y : ℝ) : ℝ :=
  x^2 + (y + 3)^2

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 = 9

theorem closest_points_to_A :
  ∃ (x y : ℝ),
    hyperbola x y ∧
    (distance_squared x y = distance_squared (-3 * Real.sqrt 5 / 2) (-3/2) ∨
     distance_squared x y = distance_squared (3 * Real.sqrt 5 / 2) (-3/2)) :=
sorry

end NUMINAMATH_GPT_closest_points_to_A_l1630_163075


namespace NUMINAMATH_GPT_non_working_games_l1630_163035

def total_games : ℕ := 30
def working_games : ℕ := 17

theorem non_working_games :
  total_games - working_games = 13 := 
by 
  sorry

end NUMINAMATH_GPT_non_working_games_l1630_163035


namespace NUMINAMATH_GPT_triangle_first_side_l1630_163031

theorem triangle_first_side (x : ℕ) (h1 : 10 + 15 + x = 32) : x = 7 :=
by
  sorry

end NUMINAMATH_GPT_triangle_first_side_l1630_163031


namespace NUMINAMATH_GPT_angle_bisector_length_is_5_l1630_163073

open Real

noncomputable def triangleAngleBisectorLength (a b c : ℝ) : ℝ :=
  sqrt (a * b * (1 - (c * c) / ((a + b) * (a + b))))

theorem angle_bisector_length_is_5 :
  ∀ (A B C : ℝ), A = 20 ∧ C = 40 ∧ (b - c = 5) →
  triangleAngleBisectorLength a (2 * a * cos (A * π / 180) + 5) (2 * a * cos (A * π / 180)) = 5 :=
  by
  -- you can skip this part with sorry
  sorry

end NUMINAMATH_GPT_angle_bisector_length_is_5_l1630_163073


namespace NUMINAMATH_GPT_ratio_of_part_to_whole_l1630_163016

theorem ratio_of_part_to_whole (N : ℝ) (h1 : (1/3) * (2/5) * N = 15) (h2 : (40/100) * N = 180) :
  (15 / N) = (1 / 7.5) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_part_to_whole_l1630_163016


namespace NUMINAMATH_GPT_budget_equality_year_l1630_163050

theorem budget_equality_year :
  ∃ n : ℕ, 540000 + 30000 * n = 780000 - 10000 * n ∧ 1990 + n = 1996 :=
by
  sorry

end NUMINAMATH_GPT_budget_equality_year_l1630_163050


namespace NUMINAMATH_GPT_find_stiffnesses_l1630_163071

def stiffnesses (m g x1 x2 k1 k2 : ℝ) : Prop :=
  (m = 3) ∧ (g = 10) ∧ (x1 = 0.4) ∧ (x2 = 0.075) ∧
  (k1 * k2 / (k1 + k2) * x1 = m * g) ∧
  ((k1 + k2) * x2 = m * g)

theorem find_stiffnesses (k1 k2 : ℝ) :
  stiffnesses 3 10 0.4 0.075 k1 k2 → 
  k1 = 300 ∧ k2 = 100 := 
sorry

end NUMINAMATH_GPT_find_stiffnesses_l1630_163071


namespace NUMINAMATH_GPT_probability_of_drawing_red_ball_l1630_163061

theorem probability_of_drawing_red_ball (total_balls red_balls : ℕ) (h_total : total_balls = 10) (h_red : red_balls = 7) : (red_balls : ℚ) / total_balls = 7 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_drawing_red_ball_l1630_163061


namespace NUMINAMATH_GPT_sequence_u5_value_l1630_163085

theorem sequence_u5_value (u : ℕ → ℝ) 
  (h_rec : ∀ n, u (n + 2) = 2 * u (n + 1) + u n)
  (h_u3 : u 3 = 9) 
  (h_u6 : u 6 = 128) : 
  u 5 = 53 := 
sorry

end NUMINAMATH_GPT_sequence_u5_value_l1630_163085


namespace NUMINAMATH_GPT_angle_in_fourth_quadrant_l1630_163004

variable (α : ℝ)

def is_in_first_quadrant (α : ℝ) : Prop := 0 < α ∧ α < 90

def is_in_fourth_quadrant (θ : ℝ) : Prop := 270 < θ ∧ θ < 360

theorem angle_in_fourth_quadrant (h : is_in_first_quadrant α) : is_in_fourth_quadrant (360 - α) := sorry

end NUMINAMATH_GPT_angle_in_fourth_quadrant_l1630_163004


namespace NUMINAMATH_GPT_regular_polygon_num_sides_l1630_163024

theorem regular_polygon_num_sides (angle : ℝ) (h : angle = 45) : 
  (∃ n : ℕ, n = 360 / angle ∧ n ≠ 0) → n = 8 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_num_sides_l1630_163024


namespace NUMINAMATH_GPT_minimum_value_x_plus_y_l1630_163013

theorem minimum_value_x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 2 * y = x * y) :
  x + y = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_x_plus_y_l1630_163013


namespace NUMINAMATH_GPT_peanut_butter_candy_count_l1630_163080

-- Definitions derived from the conditions
def grape_candy (banana_candy : ℕ) := banana_candy + 5
def peanut_butter_candy (grape_candy : ℕ) := 4 * grape_candy

-- Given condition for the banana jar
def banana_candy := 43

-- The main theorem statement
theorem peanut_butter_candy_count : peanut_butter_candy (grape_candy banana_candy) = 192 :=
by
  sorry

end NUMINAMATH_GPT_peanut_butter_candy_count_l1630_163080


namespace NUMINAMATH_GPT_find_linear_function_and_unit_price_l1630_163048

def linear_function (k b x : ℝ) : ℝ := k * x + b

def profit (cost_price : ℝ) (selling_price : ℝ) (sales_volume : ℝ) : ℝ := 
  (selling_price - cost_price) * sales_volume

theorem find_linear_function_and_unit_price
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1 = 20) (h2 : y1 = 200)
  (h3 : x2 = 25) (h4 : y2 = 150)
  (h5 : x3 = 30) (h6 : y3 = 100)
  (cost_price := 10) (desired_profit := 2160) :
  ∃ k b x : ℝ, 
    (linear_function k b x1 = y1) ∧ 
    (linear_function k b x2 = y2) ∧ 
    (profit cost_price x (linear_function k b x) = desired_profit) ∧ 
    (linear_function k b x = -10 * x + 400) ∧ 
    (x = 22) :=
by
  sorry

end NUMINAMATH_GPT_find_linear_function_and_unit_price_l1630_163048


namespace NUMINAMATH_GPT_no_value_of_n_l1630_163053

noncomputable def t1 (n : ℕ) : ℚ :=
3 * n * (n + 2)

noncomputable def t2 (n : ℕ) : ℚ :=
(3 * n^2 + 19 * n) / 2

theorem no_value_of_n (n : ℕ) (h : n > 0) : t1 n ≠ t2 n :=
by {
  sorry
}

end NUMINAMATH_GPT_no_value_of_n_l1630_163053


namespace NUMINAMATH_GPT_smallest_r_l1630_163042

variables (p q r s : ℤ)

-- Define the conditions
def condition1 : Prop := p + 3 = q - 1
def condition2 : Prop := p + 3 = r + 5
def condition3 : Prop := p + 3 = s - 2

-- Prove that r is the smallest
theorem smallest_r (h1 : condition1 p q) (h2 : condition2 p r) (h3 : condition3 p s) : r < p ∧ r < q ∧ r < s :=
sorry

end NUMINAMATH_GPT_smallest_r_l1630_163042


namespace NUMINAMATH_GPT_total_payment_l1630_163045

-- Define the basic conditions
def hours_first_day : ℕ := 10
def hours_second_day : ℕ := 8
def hours_third_day : ℕ := 15
def hourly_wage : ℕ := 10
def number_of_workers : ℕ := 2

-- Define the proof problem
theorem total_payment : 
  (hours_first_day + hours_second_day + hours_third_day) * hourly_wage * number_of_workers = 660 := 
by
  sorry

end NUMINAMATH_GPT_total_payment_l1630_163045


namespace NUMINAMATH_GPT_locus_of_points_where_tangents_are_adjoint_lines_l1630_163014

theorem locus_of_points_where_tangents_are_adjoint_lines 
  (p : ℝ) (y x : ℝ)
  (h_parabola : y^2 = 2 * p * x) :
  y^2 = - (p / 2) * x :=
sorry

end NUMINAMATH_GPT_locus_of_points_where_tangents_are_adjoint_lines_l1630_163014


namespace NUMINAMATH_GPT_range_of_a_l1630_163027

theorem range_of_a (a : ℝ) 
  (h : ∀ x y, (a * x^2 - 3 * x + 2 = 0) ∧ (a * y^2 - 3 * y + 2 = 0) → x = y) :
  a = 0 ∨ a ≥ 9/8 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1630_163027


namespace NUMINAMATH_GPT_Terry_driving_speed_is_40_l1630_163065

-- Conditions
def distance_home_to_workplace : ℕ := 60
def total_time_driving : ℕ := 3

-- Computation for total distance
def total_distance := distance_home_to_workplace * 2

-- Desired speed computation
def driving_speed := total_distance / total_time_driving

-- Problem statement to prove
theorem Terry_driving_speed_is_40 : driving_speed = 40 :=
by 
  sorry -- proof not required as per instructions

end NUMINAMATH_GPT_Terry_driving_speed_is_40_l1630_163065


namespace NUMINAMATH_GPT_Jaymee_is_22_l1630_163068

-- Define Shara's age
def Shara_age : ℕ := 10

-- Define Jaymee's age according to the problem conditions
def Jaymee_age : ℕ := 2 + 2 * Shara_age

-- The proof statement to show that Jaymee's age is 22
theorem Jaymee_is_22 : Jaymee_age = 22 := by 
  -- The proof is omitted according to the instructions.
  sorry

end NUMINAMATH_GPT_Jaymee_is_22_l1630_163068
