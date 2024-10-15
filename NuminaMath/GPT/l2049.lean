import Mathlib

namespace NUMINAMATH_GPT_min_value_one_over_a_plus_two_over_b_l2049_204920

theorem min_value_one_over_a_plus_two_over_b :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 2 * b = 2) →
  ∃ (min_val : ℝ), min_val = (1 / a + 2 / b) ∧ min_val = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_one_over_a_plus_two_over_b_l2049_204920


namespace NUMINAMATH_GPT_abs_x_minus_one_sufficient_not_necessary_l2049_204931

variable (x : ℝ) -- x is a real number

theorem abs_x_minus_one_sufficient_not_necessary (h : |x - 1| > 2) :
  (x^2 > 1) ∧ (∃ (y : ℝ), x^2 > 1 ∧ |y - 1| ≤ 2) := by
  sorry

end NUMINAMATH_GPT_abs_x_minus_one_sufficient_not_necessary_l2049_204931


namespace NUMINAMATH_GPT_f_at_10_l2049_204988

variable (f : ℕ → ℝ)

-- Conditions
axiom f_1 : f 1 = 2
axiom f_relation : ∀ m n : ℕ, m ≥ n → f (m + n) + f (m - n) = (f (2 * m) + f (2 * n)) / 2 + 2 * n

-- Prove f(10) = 361
theorem f_at_10 : f 10 = 361 :=
by
  sorry

end NUMINAMATH_GPT_f_at_10_l2049_204988


namespace NUMINAMATH_GPT_steers_cows_unique_solution_l2049_204900

-- Definition of the problem
def steers_and_cows_problem (s c : ℕ) : Prop :=
  25 * s + 26 * c = 1000 ∧ s > 0 ∧ c > 0

-- The theorem statement to be proved
theorem steers_cows_unique_solution :
  ∃! (s c : ℕ), steers_and_cows_problem s c ∧ c > s :=
sorry

end NUMINAMATH_GPT_steers_cows_unique_solution_l2049_204900


namespace NUMINAMATH_GPT_simplify_expression_l2049_204916

theorem simplify_expression (x : ℝ) : 2 * x * (x - 4) - (2 * x - 3) * (x + 2) = -9 * x + 6 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2049_204916


namespace NUMINAMATH_GPT_probability_of_selecting_red_books_is_3_div_14_l2049_204918

-- Define the conditions
def total_books : ℕ := 8
def red_books : ℕ := 4
def blue_books : ℕ := 4
def books_selected : ℕ := 2

-- Define the calculation of the probability
def probability_red_books_selected : ℚ :=
  let total_outcomes := Nat.choose total_books books_selected
  let favorable_outcomes := Nat.choose red_books books_selected
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

-- State the theorem
theorem probability_of_selecting_red_books_is_3_div_14 :
  probability_red_books_selected = 3 / 14 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_selecting_red_books_is_3_div_14_l2049_204918


namespace NUMINAMATH_GPT_abc_positive_l2049_204987

theorem abc_positive (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) (h3 : abc > 0) : a > 0 ∧ b > 0 ∧ c > 0 :=
by
  sorry

end NUMINAMATH_GPT_abc_positive_l2049_204987


namespace NUMINAMATH_GPT_sums_ratio_l2049_204908

theorem sums_ratio (total_sums : ℕ) (sums_right : ℕ) (sums_wrong: ℕ) (h1 : total_sums = 24) (h2 : sums_right = 8) (h3 : sums_wrong = total_sums - sums_right) :
  sums_wrong / Nat.gcd sums_wrong sums_right = 2 ∧ sums_right / Nat.gcd sums_wrong sums_right = 1 := by
  sorry

end NUMINAMATH_GPT_sums_ratio_l2049_204908


namespace NUMINAMATH_GPT_sector_area_sexagesimal_l2049_204910

theorem sector_area_sexagesimal (r : ℝ) (n : ℝ) (α_sex : ℝ) (π : ℝ) (two_pi : ℝ):
  r = 4 →
  n = 6000 →
  α_sex = 625 →
  two_pi = 2 * π →
  (1/2 * (α_sex / n * two_pi) * r^2) = (5 * π) / 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sector_area_sexagesimal_l2049_204910


namespace NUMINAMATH_GPT_fill_tank_time_is_18_l2049_204936

def rate1 := 1 / 20
def rate2 := 1 / 30
def combined_rate := rate1 + rate2
def effective_rate := (2 / 3) * combined_rate
def T := 1 / effective_rate

theorem fill_tank_time_is_18 : T = 18 := by
  sorry

end NUMINAMATH_GPT_fill_tank_time_is_18_l2049_204936


namespace NUMINAMATH_GPT_area_triangle_ABC_l2049_204935

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let x1 := A.1
  let y1 := A.2
  let x2 := B.1
  let y2 := B.2
  let x3 := C.1
  let y3 := C.2
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem area_triangle_ABC :
  area_of_triangle (2, 4) (-1, 1) (1, -1) = 6 :=
by
  sorry

end NUMINAMATH_GPT_area_triangle_ABC_l2049_204935


namespace NUMINAMATH_GPT_find_guest_sets_l2049_204921

-- Definitions based on conditions
def cost_per_guest_set : ℝ := 32.0
def cost_per_master_set : ℝ := 40.0
def num_master_sets : ℕ := 4
def total_cost : ℝ := 224.0

-- The mathematical problem
theorem find_guest_sets (G : ℕ) (total_cost_eq : total_cost = cost_per_guest_set * G + cost_per_master_set * num_master_sets) : G = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_guest_sets_l2049_204921


namespace NUMINAMATH_GPT_circle_equation_l2049_204967

theorem circle_equation (x y : ℝ) :
    (x - 1) ^ 2 + (y - 1) ^ 2 = 1 ↔ (∃ (C : ℝ × ℝ), C = (1, 1) ∧ ∃ (r : ℝ), r = 1 ∧ (x - C.1) ^ 2 + (y - C.2) ^ 2 = r ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_l2049_204967


namespace NUMINAMATH_GPT_find_divisor_l2049_204989

-- Definitions of the conditions
def dividend : ℕ := 15968
def quotient : ℕ := 89
def remainder : ℕ := 37

-- The theorem stating the proof problem
theorem find_divisor (D : ℕ) (h : dividend = D * quotient + remainder) : D = 179 :=
sorry

end NUMINAMATH_GPT_find_divisor_l2049_204989


namespace NUMINAMATH_GPT_problem_statement_l2049_204968

theorem problem_statement (x : ℝ) : 
  (∀ (x : ℝ), (100 / x = 80 / (x - 4)) → true) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_problem_statement_l2049_204968


namespace NUMINAMATH_GPT_incorrect_statements_l2049_204923

-- Defining the first condition
def condition1 : Prop :=
  let a_sq := 169
  let b_sq := 144
  let c_sq := a_sq - b_sq
  let c_ := Real.sqrt c_sq
  let focal_points := [(0, c_), (0, -c_)]
  ¬((c_, 0) ∈ focal_points) ∧ ¬((-c_, 0) ∈ focal_points)

-- Defining the second condition
def condition2 : Prop :=
  let m := 1  -- Example choice since m is unspecified
  let a_sq := m^2 + 1
  let b_sq := m^2
  let c_sq := a_sq - b_sq
  let c_ := Real.sqrt c_sq
  let focal_points := [(0, c_), (0, -c_)]
  (0, 1) ∈ focal_points ∧ (0, -1) ∈ focal_points

-- Defining the third condition
def condition3 : Prop :=
  let a1_sq := 16
  let b1_sq := 7
  let c1_sq := a1_sq - b1_sq
  let c1_ := Real.sqrt c1_sq
  let focal_points1 := [(c1_, 0), (-c1_, 0)]
  
  let m := 10  -- Example choice since m > 0 is unspecified
  let a2_sq := m - 5
  let b2_sq := m + 4
  let c2_sq := a2_sq - b2_sq
  let focal_points2 := [(0, Real.sqrt c2_sq), (0, -Real.sqrt c2_sq)]
  
  ¬ (focal_points1 = focal_points2)

-- Defining the fourth condition
def condition4 : Prop :=
  let B := (-3, 0)
  let C := (3, 0)
  let BC := (C.1 - B.1, C.2 - B.2)
  let BC_dist := Real.sqrt (BC.1^2 + BC.2^2)
  let A_locus_eq := ∀ (x y : ℝ), x^2 / 36 + y^2 / 27 = 1
  2 * BC_dist = 12

-- Proof verification
theorem incorrect_statements : Prop :=
  condition1 ∧ condition3

end NUMINAMATH_GPT_incorrect_statements_l2049_204923


namespace NUMINAMATH_GPT_rectangle_dimensions_l2049_204957

-- Define the known shapes and their dimensions
def square (s : ℝ) : ℝ := s^2
def rectangle1 : ℝ := 10 * 24
def rectangle2 (a b : ℝ) : ℝ := a * b

-- The total area must match the area of a square of side length 24 cm
def total_area (s a b : ℝ) : ℝ := (2 * square s) + rectangle1 + rectangle2 a b

-- The problem statement
theorem rectangle_dimensions
  (s a b : ℝ)
  (h0 : a ∈ [2, 19, 34, 34, 14, 14, 24])
  (h1 : b ∈ [24, 17.68, 10, 44, 24, 17, 38])
  : (total_area s a b = 24^2) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_dimensions_l2049_204957


namespace NUMINAMATH_GPT_range_of_n_l2049_204941

noncomputable def parabola (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x + b

variable {a b n y1 y2 : ℝ}

theorem range_of_n (h_a : a > 0) 
  (hA : parabola a b (2*n + 3) = y1) 
  (hB : parabola a b (n - 1) = y2)
  (h_sym : y1 < y2) 
  (h_opposite_sides : (2*n + 3 - 1) * (n - 1 - 1) < 0) :
  -1 < n ∧ n < 0 :=
sorry

end NUMINAMATH_GPT_range_of_n_l2049_204941


namespace NUMINAMATH_GPT_number_of_children_correct_l2049_204985

def total_spectators : ℕ := 25000
def men_spectators : ℕ := 15320
def ratio_children_women : ℕ × ℕ := (7, 3)
def remaining_spectators : ℕ := total_spectators - men_spectators
def total_ratio_parts : ℕ := ratio_children_women.1 + ratio_children_women.2
def spectators_per_part : ℕ := remaining_spectators / total_ratio_parts

def children_spectators : ℕ := spectators_per_part * ratio_children_women.1

theorem number_of_children_correct : children_spectators = 6776 := by
  sorry

end NUMINAMATH_GPT_number_of_children_correct_l2049_204985


namespace NUMINAMATH_GPT_jims_speed_l2049_204970

variable (x : ℝ)

theorem jims_speed (bob_speed : ℝ) (bob_head_start : ℝ) (time : ℝ) (bob_distance : ℝ) :
  bob_speed = 6 →
  bob_head_start = 1 →
  time = 1 / 3 →
  bob_distance = bob_speed * time →
  (x * time = bob_distance + bob_head_start) →
  x = 9 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_jims_speed_l2049_204970


namespace NUMINAMATH_GPT_parabola_vertex_l2049_204906

theorem parabola_vertex:
  ∀ x: ℝ, ∀ y: ℝ, (y = (1 / 2) * x ^ 2 - 4 * x + 3) → (x = 4 ∧ y = -5) :=
sorry

end NUMINAMATH_GPT_parabola_vertex_l2049_204906


namespace NUMINAMATH_GPT_find_d_l2049_204915

theorem find_d (c : ℕ) (d : ℕ) : 
  (∀ n : ℕ, c = 3 ∧ ∀ k : ℕ, k ≠ 30 → ((1 : ℚ) * (29 / 30) * (28 / 30) = 203 / 225) → d = 203) := 
by
  intros
  sorry

end NUMINAMATH_GPT_find_d_l2049_204915


namespace NUMINAMATH_GPT_mr_lee_harvested_apples_l2049_204951

theorem mr_lee_harvested_apples :
  let number_of_baskets := 19
  let apples_per_basket := 25
  (number_of_baskets * apples_per_basket = 475) :=
by
  sorry

end NUMINAMATH_GPT_mr_lee_harvested_apples_l2049_204951


namespace NUMINAMATH_GPT_inequality_solution_l2049_204982

theorem inequality_solution (x : ℝ) : 
  (x - 3) / (x + 7) < 0 ↔ -7 < x ∧ x < 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2049_204982


namespace NUMINAMATH_GPT_box_contents_l2049_204929

-- Definitions for the boxes and balls
inductive Ball
| Black | White | Green

-- Define the labels on each box
def label_box1 := "white"
def label_box2 := "black"
def label_box3 := "white or green"

-- Conditions based on the problem
def box1_label := label_box1
def box2_label := label_box2
def box3_label := label_box3

-- Statement of the problem
theorem box_contents (b1 b2 b3 : Ball) 
  (h1 : b1 ≠ Ball.White) 
  (h2 : b2 ≠ Ball.Black) 
  (h3 : b3 = Ball.Black) 
  (h4 : ∀ (x y z : Ball), x ≠ y ∧ y ≠ z ∧ z ≠ x → 
        (x = b1 ∨ y = b1 ∨ z = b1) ∧
        (x = b2 ∨ y = b2 ∨ z = b2) ∧
        (x = b3 ∨ y = b3 ∨ z = b3)) : 
  b1 = Ball.Green ∧ b2 = Ball.White ∧ b3 = Ball.Black :=
sorry

end NUMINAMATH_GPT_box_contents_l2049_204929


namespace NUMINAMATH_GPT_original_number_of_men_l2049_204926

theorem original_number_of_men 
  (x : ℕ) 
  (H1 : x * 15 = (x - 8) * 18) : 
  x = 48 := 
sorry

end NUMINAMATH_GPT_original_number_of_men_l2049_204926


namespace NUMINAMATH_GPT_parallel_and_equidistant_line_l2049_204991

-- Define the two given lines
def line1 (x y : ℝ) : Prop := 3 * x + 2 * y - 6 = 0
def line2 (x y : ℝ) : Prop := 6 * x + 4 * y - 3 = 0

-- Define the desired property: a line parallel to line1 and line2, and equidistant from both
theorem parallel_and_equidistant_line :
  ∃ b : ℝ, ∀ x y : ℝ, (3 * x + 2 * y + b = 0) ∧
  (|-6 - b| / Real.sqrt (9 + 4) = |-3/2 - b| / Real.sqrt (9 + 4)) →
  (12 * x + 8 * y - 15 = 0) :=
by
  sorry

end NUMINAMATH_GPT_parallel_and_equidistant_line_l2049_204991


namespace NUMINAMATH_GPT_wizard_elixir_combinations_l2049_204907

theorem wizard_elixir_combinations :
  let herbs := 4
  let crystals := 6
  let invalid_combinations := 3
  herbs * crystals - invalid_combinations = 21 := 
by
  sorry

end NUMINAMATH_GPT_wizard_elixir_combinations_l2049_204907


namespace NUMINAMATH_GPT_workers_planted_33_walnut_trees_l2049_204930

def initial_walnut_trees : ℕ := 22
def total_walnut_trees_after_planting : ℕ := 55
def walnut_trees_planted (initial : ℕ) (total : ℕ) : ℕ := total - initial

theorem workers_planted_33_walnut_trees :
  walnut_trees_planted initial_walnut_trees total_walnut_trees_after_planting = 33 :=
by
  unfold walnut_trees_planted
  rfl

end NUMINAMATH_GPT_workers_planted_33_walnut_trees_l2049_204930


namespace NUMINAMATH_GPT_good_pair_exists_l2049_204927

theorem good_pair_exists (m : ℕ) : ∃ n : ℕ, n > m ∧ (∃ k1 k2 : ℕ, m * n = k1 * k1 ∧ (m + 1) * (n + 1) = k2 * k2) :=
by
  sorry

end NUMINAMATH_GPT_good_pair_exists_l2049_204927


namespace NUMINAMATH_GPT_multiplication_equivalence_l2049_204948

theorem multiplication_equivalence :
    44 * 22 = 88 * 11 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_equivalence_l2049_204948


namespace NUMINAMATH_GPT_average_minutes_heard_l2049_204949

theorem average_minutes_heard :
  let total_audience := 200
  let duration := 90
  let percent_entire := 0.15
  let percent_slept := 0.15
  let percent_half := 0.25
  let percent_one_fourth := 0.75
  let total_entire := total_audience * percent_entire
  let total_slept := total_audience * percent_slept
  let remaining := total_audience - total_entire - total_slept
  let total_half := remaining * percent_half
  let total_one_fourth := remaining * percent_one_fourth
  let minutes_entire := total_entire * duration
  let minutes_half := total_half * (duration / 2)
  let minutes_one_fourth := total_one_fourth * (duration / 4)
  let total_minutes_heard := minutes_entire + 0 + minutes_half + minutes_one_fourth
  let average_minutes := total_minutes_heard / total_audience
  average_minutes = 33 :=
by
  sorry

end NUMINAMATH_GPT_average_minutes_heard_l2049_204949


namespace NUMINAMATH_GPT_intersection_M_N_l2049_204975

noncomputable def M : Set ℝ := { x | -1 < x ∧ x < 3 }
noncomputable def N : Set ℝ := { x | ∃ y, y = Real.log (x - x^2) }
noncomputable def intersection (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∈ B }

theorem intersection_M_N : intersection M N = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2049_204975


namespace NUMINAMATH_GPT_soccer_uniform_probability_l2049_204911

-- Definitions for the conditions of the problem
def colorsSocks : List String := ["red", "blue"]
def colorsShirts : List String := ["red", "blue", "green"]

noncomputable def differentColorConfigurations : Nat :=
  let validConfigs := [("red", "blue"), ("red", "green"), ("blue", "red"), ("blue", "green")]
  validConfigs.length

noncomputable def totalConfigurations : Nat :=
  colorsSocks.length * colorsShirts.length

noncomputable def probabilityDifferentColors : ℚ :=
  (differentColorConfigurations : ℚ) / (totalConfigurations : ℚ)

-- The theorem to prove
theorem soccer_uniform_probability :
  probabilityDifferentColors = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_soccer_uniform_probability_l2049_204911


namespace NUMINAMATH_GPT_rectangle_sides_l2049_204905

def side_length_square : ℝ := 18
def num_rectangles : ℕ := 5

variable (a b : ℝ)
variables (h1 : 2 * (a + b) = side_length_square) (h2 : 3 * a = side_length_square)

theorem rectangle_sides : a = 6 ∧ b = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_rectangle_sides_l2049_204905


namespace NUMINAMATH_GPT_ratio_of_Jordyn_age_to_Zrinka_age_is_2_l2049_204976

variable (Mehki_age : ℕ) (Jordyn_age : ℕ) (Zrinka_age : ℕ)

-- Conditions
def Mehki_is_10_years_older_than_Jordyn := Mehki_age = Jordyn_age + 10
def Zrinka_age_is_6 := Zrinka_age = 6
def Mehki_age_is_22 := Mehki_age = 22

-- Theorem statement: the ratio of Jordyn's age to Zrinka's age is 2.
theorem ratio_of_Jordyn_age_to_Zrinka_age_is_2
  (h1 : Mehki_is_10_years_older_than_Jordyn Mehki_age Jordyn_age)
  (h2 : Zrinka_age_is_6 Zrinka_age)
  (h3 : Mehki_age_is_22 Mehki_age) : Jordyn_age / Zrinka_age = 2 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_ratio_of_Jordyn_age_to_Zrinka_age_is_2_l2049_204976


namespace NUMINAMATH_GPT_relationship_between_a_and_b_l2049_204953

theorem relationship_between_a_and_b {a b : ℝ} (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ x : ℝ, |(2 * x + 2)| < a → |(x + 1)| < b) : b ≥ a / 2 :=
by
  -- The proof steps will be inserted here
  sorry

end NUMINAMATH_GPT_relationship_between_a_and_b_l2049_204953


namespace NUMINAMATH_GPT_tetrahedron_edge_square_sum_l2049_204980

variable (A B C D : Point)
variable (AB AC AD BC BD CD : ℝ) -- Lengths of the edges
variable (m₁ m₂ m₃ : ℝ) -- Distances between the midpoints of the opposite edges

theorem tetrahedron_edge_square_sum:
  (AB ^ 2 + AC ^ 2 + AD ^ 2 + BC ^ 2 + BD ^ 2 + CD ^ 2) =
  4 * (m₁ ^ 2 + m₂ ^ 2 + m₃ ^ 2) :=
  sorry

end NUMINAMATH_GPT_tetrahedron_edge_square_sum_l2049_204980


namespace NUMINAMATH_GPT_three_integers_product_sum_l2049_204945

theorem three_integers_product_sum (a b c : ℤ) (h : a * b * c = -5) :
    a + b + c = 5 ∨ a + b + c = -3 ∨ a + b + c = -7 :=
sorry

end NUMINAMATH_GPT_three_integers_product_sum_l2049_204945


namespace NUMINAMATH_GPT_person6_number_l2049_204912

theorem person6_number (a : ℕ → ℕ) (x : ℕ → ℕ) 
  (mod12 : ∀ i, a (i % 12) = a i)
  (h5 : x 5 = 5)
  (h6 : x 6 = 8)
  (h7 : x 7 = 11) 
  (h_avg : ∀ i, x i = (a (i-1) + a (i+1)) / 2) : 
  a 6 = 6 := sorry

end NUMINAMATH_GPT_person6_number_l2049_204912


namespace NUMINAMATH_GPT_complement_of_16deg51min_is_73deg09min_l2049_204974

def complement_angle (A : ℝ) : ℝ := 90 - A

theorem complement_of_16deg51min_is_73deg09min :
  complement_angle 16.85 = 73.15 := by
  sorry

end NUMINAMATH_GPT_complement_of_16deg51min_is_73deg09min_l2049_204974


namespace NUMINAMATH_GPT_cube_sum_identity_l2049_204913

theorem cube_sum_identity (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
by
  sorry

end NUMINAMATH_GPT_cube_sum_identity_l2049_204913


namespace NUMINAMATH_GPT_largest_uncovered_squares_l2049_204904

theorem largest_uncovered_squares (board_size : ℕ) (total_squares : ℕ) (domino_size : ℕ) 
  (odd_property : ∀ (n : ℕ), n % 2 = 1 → (n - domino_size) % 2 = 1)
  (can_place_more : ∀ (placed_squares odd_squares : ℕ), placed_squares + domino_size ≤ total_squares → odd_squares - domino_size % 2 = 1 → odd_squares ≥ 0)
  : ∃ max_uncovered : ℕ, max_uncovered = 7 := by
  sorry

end NUMINAMATH_GPT_largest_uncovered_squares_l2049_204904


namespace NUMINAMATH_GPT_domain_sqrt_product_domain_log_fraction_l2049_204925

theorem domain_sqrt_product (x : ℝ) (h1 : x - 2 ≥ 0) (h2 : x + 2 ≥ 0) : 
  2 ≤ x :=
by sorry

theorem domain_log_fraction (x : ℝ) (h1 : x + 1 > 0) (h2 : -x^2 - 3 * x + 4 > 0) : 
  -1 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_GPT_domain_sqrt_product_domain_log_fraction_l2049_204925


namespace NUMINAMATH_GPT_weight_of_b_l2049_204964

variable {a b c : ℝ}

theorem weight_of_b (h1 : (a + b + c) / 3 = 45)
                    (h2 : (a + b) / 2 = 40)
                    (h3 : (b + c) / 2 = 43) :
                    b = 31 := by
  sorry

end NUMINAMATH_GPT_weight_of_b_l2049_204964


namespace NUMINAMATH_GPT_solid2_solid4_views_identical_l2049_204950

-- Define the solids and their orthographic views
structure Solid :=
  (top_view : String)
  (front_view : String)
  (side_view : String)

-- Given solids as provided by the problem
def solid1 : Solid := { top_view := "...", front_view := "...", side_view := "..." }
def solid2 : Solid := { top_view := "...", front_view := "...", side_view := "..." }
def solid3 : Solid := { top_view := "...", front_view := "...", side_view := "..." }
def solid4 : Solid := { top_view := "...", front_view := "...", side_view := "..." }

-- Function to compare two solids' views
def views_identical (s1 s2 : Solid) : Prop :=
  (s1.top_view = s2.top_view ∧ s1.front_view = s2.front_view) ∨
  (s1.top_view = s2.top_view ∧ s1.side_view = s2.side_view) ∨
  (s1.front_view = s2.front_view ∧ s1.side_view = s2.side_view)

-- Theorem statement
theorem solid2_solid4_views_identical : views_identical solid2 solid4 := 
sorry

end NUMINAMATH_GPT_solid2_solid4_views_identical_l2049_204950


namespace NUMINAMATH_GPT_geometric_seq_20th_term_l2049_204919

theorem geometric_seq_20th_term (a r : ℕ)
  (h1 : a * r ^ 4 = 5)
  (h2 : a * r ^ 11 = 1280) :
  a * r ^ 19 = 2621440 :=
sorry

end NUMINAMATH_GPT_geometric_seq_20th_term_l2049_204919


namespace NUMINAMATH_GPT_advanced_purchase_tickets_sold_l2049_204934

theorem advanced_purchase_tickets_sold (A D : ℕ) 
  (h1 : A + D = 140)
  (h2 : 8 * A + 14 * D = 1720) : 
  A = 40 :=
by
  sorry

end NUMINAMATH_GPT_advanced_purchase_tickets_sold_l2049_204934


namespace NUMINAMATH_GPT_one_eighth_percent_of_160_plus_half_l2049_204984

theorem one_eighth_percent_of_160_plus_half :
  ((1 / 8) / 100 * 160) + 0.5 = 0.7 :=
  sorry

end NUMINAMATH_GPT_one_eighth_percent_of_160_plus_half_l2049_204984


namespace NUMINAMATH_GPT_last_two_digits_of_7_pow_2015_l2049_204969

theorem last_two_digits_of_7_pow_2015 : ((7 ^ 2015) % 100) = 43 := 
by
  sorry

end NUMINAMATH_GPT_last_two_digits_of_7_pow_2015_l2049_204969


namespace NUMINAMATH_GPT_least_number_to_add_l2049_204986

theorem least_number_to_add (m n : ℕ) (h₁ : m = 1052) (h₂ : n = 23) : 
  ∃ k : ℕ, (m + k) % n = 0 ∧ k = 6 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_add_l2049_204986


namespace NUMINAMATH_GPT_speed_equivalence_l2049_204954

def convert_speed (speed_kmph : ℚ) : ℚ :=
  speed_kmph * 0.277778

theorem speed_equivalence : convert_speed 162 = 45 :=
by
  sorry

end NUMINAMATH_GPT_speed_equivalence_l2049_204954


namespace NUMINAMATH_GPT_parallelogram_coordinates_l2049_204933

/-- Given points A, B, and C, prove the coordinates of point D for the parallelogram -/
theorem parallelogram_coordinates (A B C: (ℝ × ℝ)) 
  (hA : A = (3, 7)) 
  (hB : B = (4, 6))
  (hC : C = (1, -2)) :
  D = (0, -1) ∨ D = (2, -3) ∨ D = (6, 15) :=
sorry

end NUMINAMATH_GPT_parallelogram_coordinates_l2049_204933


namespace NUMINAMATH_GPT_baker_sold_cakes_l2049_204971

theorem baker_sold_cakes (S : ℕ) (h1 : 154 = S + 63) : S = 91 :=
by
  sorry

end NUMINAMATH_GPT_baker_sold_cakes_l2049_204971


namespace NUMINAMATH_GPT_parallel_vectors_tan_l2049_204977

theorem parallel_vectors_tan (θ : ℝ)
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h₀ : a = (2, Real.sin θ))
  (h₁ : b = (1, Real.cos θ))
  (h_parallel : a.1 * b.2 = a.2 * b.1) :
  Real.tan θ = 2 := 
sorry

end NUMINAMATH_GPT_parallel_vectors_tan_l2049_204977


namespace NUMINAMATH_GPT_coconut_trees_per_sqm_l2049_204963

def farm_area : ℕ := 20
def harvests : ℕ := 2
def total_earnings : ℝ := 240
def coconut_price : ℝ := 0.50
def coconuts_per_tree : ℕ := 6

theorem coconut_trees_per_sqm : 
  let total_coconuts := total_earnings / coconut_price / harvests
  let total_trees := total_coconuts / coconuts_per_tree 
  let trees_per_sqm := total_trees / farm_area 
  trees_per_sqm = 2 :=
by
  sorry

end NUMINAMATH_GPT_coconut_trees_per_sqm_l2049_204963


namespace NUMINAMATH_GPT_triangle_C_squared_eq_b_a_plus_b_l2049_204961

variables {A B C a b : ℝ}

theorem triangle_C_squared_eq_b_a_plus_b
  (h1 : C = 2 * B)
  (h2 : A ≠ B) :
  C^2 = b * (a + b) :=
sorry

end NUMINAMATH_GPT_triangle_C_squared_eq_b_a_plus_b_l2049_204961


namespace NUMINAMATH_GPT_determine_h_l2049_204995

def h (x : ℝ) := -12 * x^4 - 4 * x^3 - 8 * x^2 + 3 * x - 1

theorem determine_h (x : ℝ) : 
  (12 * x^4 + 9 * x^3 - 3 * x + 1 + h x = 5 * x^3 - 8 * x^2 + 3) →
  h x = -12 * x^4 - 4 * x^3 - 8 * x^2 + 3 * x - 1 :=
by
  sorry

end NUMINAMATH_GPT_determine_h_l2049_204995


namespace NUMINAMATH_GPT_students_play_football_l2049_204990

theorem students_play_football 
  (total : ℕ) (C : ℕ) (B : ℕ) (Neither : ℕ) (F : ℕ) 
  (h_total : total = 420) 
  (h_C : C = 175) 
  (h_B : B = 130) 
  (h_Neither : Neither = 50) 
  (h_inclusion_exclusion : F + C - B = total - Neither) :
  F = 325 := 
sorry

end NUMINAMATH_GPT_students_play_football_l2049_204990


namespace NUMINAMATH_GPT_cos_8_minus_sin_8_l2049_204966

theorem cos_8_minus_sin_8 (α m : ℝ) (h : Real.cos (2 * α) = m) :
  Real.cos α ^ 8 - Real.sin α ^ 8 = m * (1 + m^2) / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_8_minus_sin_8_l2049_204966


namespace NUMINAMATH_GPT_rational_sum_zero_l2049_204983

theorem rational_sum_zero (x1 x2 x3 x4 : ℚ)
  (h1 : x1 = x2 + x3 + x4)
  (h2 : x2 = x1 + x3 + x4)
  (h3 : x3 = x1 + x2 + x4)
  (h4 : x4 = x1 + x2 + x3) : 
  x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 := 
sorry

end NUMINAMATH_GPT_rational_sum_zero_l2049_204983


namespace NUMINAMATH_GPT_olivia_time_spent_l2049_204943

theorem olivia_time_spent :
  ∀ (x : ℕ), 7 * x + 3 = 31 → x = 4 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_olivia_time_spent_l2049_204943


namespace NUMINAMATH_GPT_line_through_A1_slope_neg4_over_3_line_through_A2_l2049_204922

-- (1) The line passing through point (1, 3) with a slope -4/3
theorem line_through_A1_slope_neg4_over_3 : 
    ∃ (a b c : ℝ), a * 1 + b * 3 + c = 0 ∧ ∃ m : ℝ, m = -4 / 3 ∧ a * m + b = 0 ∧ b ≠ 0 ∧ c = -13 := by
sorry

-- (2) The line passing through point (-5, 2) with x-intercept twice the y-intercept
theorem line_through_A2 : 
    ∃ (a b c : ℝ), (a * -5 + b * 2 + c = 0) ∧ ((∃ m : ℝ, m = 2 ∧ a * m + b = 0 ∧ b = -a) ∨ ((b = -2 / 5 * a) ∧ (a * 2 + b = 0))) := by
sorry

end NUMINAMATH_GPT_line_through_A1_slope_neg4_over_3_line_through_A2_l2049_204922


namespace NUMINAMATH_GPT_cards_distribution_l2049_204955

theorem cards_distribution (total_cards : ℕ) (total_people : ℕ) (cards_per_person : ℕ) (extra_cards : ℕ) (people_with_extra_cards : ℕ) (people_with_fewer_cards : ℕ) :
  total_cards = 100 →
  total_people = 15 →
  total_cards / total_people = cards_per_person →
  total_cards % total_people = extra_cards →
  people_with_extra_cards = extra_cards →
  people_with_fewer_cards = total_people - people_with_extra_cards →
  people_with_fewer_cards = 5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_cards_distribution_l2049_204955


namespace NUMINAMATH_GPT_original_price_l2049_204917

theorem original_price (P : ℝ) (h : 0.684 * P = 6800) : P = 10000 :=
by
  sorry

end NUMINAMATH_GPT_original_price_l2049_204917


namespace NUMINAMATH_GPT_carousel_problem_l2049_204996

theorem carousel_problem (n : ℕ) : 
  (∃ (f : Fin n → Fin n), 
    (∀ i, f (f i) = i) ∧ 
    (∀ i j, i ≠ j → f i ≠ f j) ∧ 
    (∀ i, f i < n)) ↔ 
  (Even n) := 
sorry

end NUMINAMATH_GPT_carousel_problem_l2049_204996


namespace NUMINAMATH_GPT_statement_A_l2049_204940

theorem statement_A (x : ℝ) (h : x > 1) : x^2 > x := 
by
  sorry

end NUMINAMATH_GPT_statement_A_l2049_204940


namespace NUMINAMATH_GPT_distinct_flags_count_l2049_204901

theorem distinct_flags_count : 
  ∃ n, n = 36 ∧ (∀ c1 c2 c3 : Fin 4, c1 ≠ c2 ∧ c2 ≠ c3 → n = 4 * 3 * 3) := 
sorry

end NUMINAMATH_GPT_distinct_flags_count_l2049_204901


namespace NUMINAMATH_GPT_exactly_one_is_multiple_of_5_l2049_204965

theorem exactly_one_is_multiple_of_5 (a b : ℤ) (h: 24 * a^2 + 1 = b^2) : 
  (∃ k : ℤ, a = 5 * k) ∧ (∀ l : ℤ, b ≠ 5 * l) ∨ (∃ m : ℤ, b = 5 * m) ∧ (∀ n : ℤ, a ≠ 5 * n) :=
sorry

end NUMINAMATH_GPT_exactly_one_is_multiple_of_5_l2049_204965


namespace NUMINAMATH_GPT_simplify_expression_l2049_204997

theorem simplify_expression :
  (1 / (1 / (1 / 3 : ℝ)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3)) = (1 / 39 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2049_204997


namespace NUMINAMATH_GPT_two_n_minus_one_lt_n_plus_one_squared_l2049_204960

theorem two_n_minus_one_lt_n_plus_one_squared (n : ℕ) (h : n > 0) : 2 * n - 1 < (n + 1) ^ 2 := 
by
  sorry

end NUMINAMATH_GPT_two_n_minus_one_lt_n_plus_one_squared_l2049_204960


namespace NUMINAMATH_GPT_larger_square_side_length_l2049_204946

theorem larger_square_side_length :
  ∃ (a : ℕ), ∃ (b : ℕ), a^2 = b^2 + 2001 ∧ (a = 1001 ∨ a = 335 ∨ a = 55 ∨ a = 49) :=
by
  sorry

end NUMINAMATH_GPT_larger_square_side_length_l2049_204946


namespace NUMINAMATH_GPT_quadratic_roots_l2049_204903

theorem quadratic_roots (x : ℝ) : (x ^ 2 - 3 = 0) → (x = Real.sqrt 3 ∨ x = -Real.sqrt 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_roots_l2049_204903


namespace NUMINAMATH_GPT_initial_apples_l2049_204937

-- Define the initial conditions
def r : Nat := 14
def s : Nat := 2 * r
def remaining : Nat := 32
def total_removed : Nat := r + s

-- The proof problem: Prove that the initial number of apples is 74
theorem initial_apples : (total_removed + remaining = 74) :=
by
  sorry

end NUMINAMATH_GPT_initial_apples_l2049_204937


namespace NUMINAMATH_GPT_sine_of_smaller_angle_and_k_domain_l2049_204956

theorem sine_of_smaller_angle_and_k_domain (α : ℝ) (k : ℝ) (AD : ℝ) (h0 : 1 < k) 
  (h1 : CD = AD * Real.tan (2 * α)) (h2 : BD = AD * Real.tan α) 
  (h3 : k = CD / BD) :
  k > 2 ∧ Real.sin (Real.pi / 2 - 2 * α) = 1 / (k - 1) := by
  sorry

end NUMINAMATH_GPT_sine_of_smaller_angle_and_k_domain_l2049_204956


namespace NUMINAMATH_GPT_parallel_condition_coincide_condition_perpendicular_condition_l2049_204959

-- Define the equations of the lines
def l1 (m : ℝ) (x y : ℝ) : Prop := (m + 3) * x + 4 * y = 5 - 3 * m
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * x + (m + 5) * y = 8

-- Parallel lines condition
theorem parallel_condition (m : ℝ) : (l1 m = l2 m ↔ m = -7) →
  (∀ x y : ℝ, l1 m x y ∧ l2 m x y) → False := sorry

-- Coincidence condition
theorem coincide_condition (m : ℝ) : 
  (l1 (-1) = l2 (-1)) :=
sorry

-- Perpendicular lines condition
theorem perpendicular_condition (m : ℝ) : 
  (m = - 13 / 3 ↔ (2 * (m + 3) + 4 * (m + 5) = 0)) :=
sorry

end NUMINAMATH_GPT_parallel_condition_coincide_condition_perpendicular_condition_l2049_204959


namespace NUMINAMATH_GPT_highlights_part_to_whole_relation_l2049_204909

/-- A predicate representing different types of statistical graphs. -/
inductive StatGraphType where
  | BarGraph : StatGraphType
  | PieChart : StatGraphType
  | LineGraph : StatGraphType
  | FrequencyDistributionHistogram : StatGraphType

/-- A lemma specifying that the PieChart is the graph type that highlights the relationship between a part and the whole. -/
theorem highlights_part_to_whole_relation (t : StatGraphType) : t = StatGraphType.PieChart :=
  sorry

end NUMINAMATH_GPT_highlights_part_to_whole_relation_l2049_204909


namespace NUMINAMATH_GPT_sum_of_decimals_is_one_l2049_204952

-- Define digits for each decimal place
def digit_a : ℕ := 2
def digit_b : ℕ := 3
def digit_c : ℕ := 2
def digit_d : ℕ := 2

-- Define the decimal numbers with these digits
def decimal1 : Rat := (digit_b * 10 + digit_a) / 100
def decimal2 : Rat := (digit_d * 10 + digit_c) / 100
def decimal3 : Rat := (2 * 10 + 2) / 100
def decimal4 : Rat := (2 * 10 + 3) / 100

-- The main theorem that states the sum of these decimals is 1
theorem sum_of_decimals_is_one : decimal1 + decimal2 + decimal3 + decimal4 = 1 := by
  sorry

end NUMINAMATH_GPT_sum_of_decimals_is_one_l2049_204952


namespace NUMINAMATH_GPT_bags_total_on_next_day_l2049_204932

def bags_on_monday : ℕ := 7
def additional_bags : ℕ := 5
def bags_on_next_day : ℕ := bags_on_monday + additional_bags

theorem bags_total_on_next_day : bags_on_next_day = 12 := by
  unfold bags_on_next_day
  unfold bags_on_monday
  unfold additional_bags
  sorry

end NUMINAMATH_GPT_bags_total_on_next_day_l2049_204932


namespace NUMINAMATH_GPT_number_of_trees_is_eleven_l2049_204939

variables (N : ℕ)

-- Conditions
def Anya (N : ℕ) := N = 15
def Borya (N : ℕ) := 11 ∣ N
def Vera (N : ℕ) := N < 25
def Gena (N : ℕ) := 22 ∣ N

axiom OneBoyOneGirlTruth :
  (∃ (b : Prop) (g : Prop),
    (b ∨ ¬ b) ∧ (g ∨ ¬ g) ∧
    ((b = (Borya N ∨ Gena N)) ∧ (g = (Anya N ∨ Vera N)) ∧
     (b ↔ ¬g) ∧
     ((Anya N ∨ ¬Vera N) ∨ (¬Anya N ∨ Vera N)) ∧
     (Anya N = (N = 15)) ∧
     (Borya N = (11 ∣ N)) ∧
     (Vera N = (N < 25)) ∧
     (Gena N = (22 ∣ N))))

theorem number_of_trees_is_eleven: N = 11 :=
sorry

end NUMINAMATH_GPT_number_of_trees_is_eleven_l2049_204939


namespace NUMINAMATH_GPT_petya_second_race_finishes_first_l2049_204994

variable (t v_P v_V : ℝ)
variable (h1 : v_P * t = 100)
variable (h2 : v_V * t = 90)
variable (d : ℝ)

theorem petya_second_race_finishes_first :
  v_V = 0.9 * v_P ∧
  d * v_P = 10 + d * (0.9 * v_P) →
  ∃ t2 : ℝ, t2 = 100 / v_P ∧ (v_V * t2 = 90) →
  ∃ t3 : ℝ, t3 = t2 + d / 10 ∧ (d * v_P = 100) →
  v_P * d / 10 - v_V * d / 10 = 1 :=
by
  sorry

end NUMINAMATH_GPT_petya_second_race_finishes_first_l2049_204994


namespace NUMINAMATH_GPT_determine_k_l2049_204938

theorem determine_k 
  (k : ℝ) 
  (r s : ℝ) 
  (h1 : r + s = -k) 
  (h2 : r * s = 6) 
  (h3 : (r + 5) + (s + 5) = k) : 
  k = 5 := 
by 
  sorry

end NUMINAMATH_GPT_determine_k_l2049_204938


namespace NUMINAMATH_GPT_present_age_of_son_l2049_204942

variable (S M : ℕ)

-- Conditions
def condition1 := M = S + 28
def condition2 := M + 2 = 2 * (S + 2)

-- Theorem to be proven
theorem present_age_of_son : condition1 S M ∧ condition2 S M → S = 26 := by
  sorry

end NUMINAMATH_GPT_present_age_of_son_l2049_204942


namespace NUMINAMATH_GPT_number_of_smaller_cubes_in_larger_cube_l2049_204978

-- Defining the conditions
def volume_large_cube : ℝ := 125
def volume_small_cube : ℝ := 1
def surface_area_difference : ℝ := 600

-- Translating the question into a math proof problem
theorem number_of_smaller_cubes_in_larger_cube : 
  ∃ n : ℕ, n * 6 - 6 * (volume_large_cube^(1/3) ^ 2) = surface_area_difference :=
by
  sorry

end NUMINAMATH_GPT_number_of_smaller_cubes_in_larger_cube_l2049_204978


namespace NUMINAMATH_GPT_nat_condition_l2049_204947

theorem nat_condition (n : ℕ) (h : n ≥ 2) :
  (∀ i j : ℕ, 0 ≤ i → i ≤ j → j ≤ n → (i + j) % 2 = (Nat.choose n i + Nat.choose n j) % 2) ↔
  (∃ p : ℕ, n = 2^p - 2) :=
sorry

end NUMINAMATH_GPT_nat_condition_l2049_204947


namespace NUMINAMATH_GPT_negation_of_exists_gt_1_l2049_204992

theorem negation_of_exists_gt_1 :
  (∀ x : ℝ, x ≤ 1) ↔ ¬ (∃ x : ℝ, x > 1) :=
sorry

end NUMINAMATH_GPT_negation_of_exists_gt_1_l2049_204992


namespace NUMINAMATH_GPT_at_least_one_inequality_holds_l2049_204993

theorem at_least_one_inequality_holds
  (x y : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hxy : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_inequality_holds_l2049_204993


namespace NUMINAMATH_GPT_find_radius_l2049_204973

theorem find_radius (a : ℝ) :
  (∃ (x y : ℝ), (x + 2) ^ 2 + (y - 2) ^ 2 = a ∧ x + y + 2 = 0) ∧
  (∃ (l : ℝ), l = 6 ∧ 2 * Real.sqrt (a - 2) = l) →
  a = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_radius_l2049_204973


namespace NUMINAMATH_GPT_max_value_x_y2_z3_l2049_204962

theorem max_value_x_y2_z3 (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) : 
  x + y^2 + z^3 ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_max_value_x_y2_z3_l2049_204962


namespace NUMINAMATH_GPT_cell_survival_after_6_hours_l2049_204979

def cell_sequence (a : ℕ → ℕ) : Prop :=
  (a 0 = 2) ∧ (∀ n, a (n + 1) = 2 * a n - 1)

theorem cell_survival_after_6_hours :
  ∃ a : ℕ → ℕ, cell_sequence a ∧ a 6 = 65 :=
by
  sorry

end NUMINAMATH_GPT_cell_survival_after_6_hours_l2049_204979


namespace NUMINAMATH_GPT_relationship_between_a_b_c_l2049_204928

noncomputable def a : ℝ := 2^(4/3)
noncomputable def b : ℝ := 4^(2/5)
noncomputable def c : ℝ := 25^(1/3)

theorem relationship_between_a_b_c : c > a ∧ a > b := 
by
  have ha : a = 2^(4/3) := rfl
  have hb : b = 4^(2/5) := rfl
  have hc : c = 25^(1/3) := rfl

  sorry

end NUMINAMATH_GPT_relationship_between_a_b_c_l2049_204928


namespace NUMINAMATH_GPT_area_of_rectangle_l2049_204902

-- Define the conditions
def width : ℕ := 6
def perimeter : ℕ := 28

-- Define the theorem statement
theorem area_of_rectangle (w : ℕ) (p : ℕ) (h_width : w = width) (h_perimeter : p = perimeter) :
  ∃ l : ℕ, (2 * (l + w) = p) → (l * w = 48) :=
by
  use 8
  intro h
  simp only [h_width, h_perimeter] at h
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l2049_204902


namespace NUMINAMATH_GPT_relationship_between_f_x1_and_f_x2_l2049_204981

variable (f : ℝ → ℝ)
variable (x1 x2 : ℝ)

-- Conditions:
variable (h_even : ∀ x, f x = f (-x))          -- f is even
variable (h_increasing : ∀ a b, 0 < a → a < b → f a < f b)  -- f is increasing on (0, +∞)
variable (h_x1_neg : x1 < 0)                   -- x1 < 0
variable (h_x2_pos : 0 < x2)                   -- x2 > 0
variable (h_abs : |x1| > |x2|)                 -- |x1| > |x2|

-- Goal:
theorem relationship_between_f_x1_and_f_x2 : f x1 > f x2 :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_f_x1_and_f_x2_l2049_204981


namespace NUMINAMATH_GPT_total_students_at_competition_l2049_204999
-- Import necessary Lean libraries for arithmetic and logic

-- Define the conditions as variables and expressions
namespace ScienceFair

variables (K KKnowItAll KarenHigh NovelCoronaHigh : ℕ)
variables (hK : KKnowItAll = 50)
variables (hKH : KarenHigh = 3 * KKnowItAll / 5)
variables (hNCH : NovelCoronaHigh = 2 * (KKnowItAll + KarenHigh))

-- Define the proof problem
theorem total_students_at_competition (KKnowItAll KarenHigh NovelCoronaHigh : ℕ)
  (hK : KKnowItAll = 50)
  (hKH : KarenHigh = 3 * KKnowItAll / 5)
  (hNCH : NovelCoronaHigh = 2 * (KKnowItAll + KarenHigh)) :
  KKnowItAll + KarenHigh + NovelCoronaHigh = 240 := by
  sorry

end ScienceFair

end NUMINAMATH_GPT_total_students_at_competition_l2049_204999


namespace NUMINAMATH_GPT_other_group_less_garbage_l2049_204958

theorem other_group_less_garbage :
  387 + (735 - 387) = 735 :=
by
  sorry

end NUMINAMATH_GPT_other_group_less_garbage_l2049_204958


namespace NUMINAMATH_GPT_johns_elevation_after_travel_l2049_204998

-- Definitions based on conditions:
def initial_elevation : ℝ := 400
def downward_rate : ℝ := 10
def time_travelled : ℕ := 5

-- Proof statement:
theorem johns_elevation_after_travel:
  initial_elevation - (downward_rate * time_travelled) = 350 :=
by
  sorry

end NUMINAMATH_GPT_johns_elevation_after_travel_l2049_204998


namespace NUMINAMATH_GPT_harmonic_mean_pairs_l2049_204924

theorem harmonic_mean_pairs (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x < y) 
    (hmean : (2 * x * y) / (x + y) = 2^30) :
    (∃! n, n = 29) :=
by
  sorry

end NUMINAMATH_GPT_harmonic_mean_pairs_l2049_204924


namespace NUMINAMATH_GPT_president_vice_president_ways_l2049_204914

theorem president_vice_president_ways :
  let boys := 14
  let girls := 10
  let total_boys_ways := boys * (boys - 1)
  let total_girls_ways := girls * (girls - 1)
  total_boys_ways + total_girls_ways = 272 := 
by
  sorry

end NUMINAMATH_GPT_president_vice_president_ways_l2049_204914


namespace NUMINAMATH_GPT_solve_quadratic_eq_l2049_204944

theorem solve_quadratic_eq (x : ℝ) : x^2 + 8 * x = 9 ↔ x = -9 ∨ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l2049_204944


namespace NUMINAMATH_GPT_plastering_cost_correct_l2049_204972

def length : ℕ := 40
def width : ℕ := 18
def depth : ℕ := 10
def cost_per_sq_meter : ℚ := 1.25

def area_bottom (L W : ℕ) : ℕ := L * W
def perimeter_bottom (L W : ℕ) : ℕ := 2 * (L + W)
def area_walls (P D : ℕ) : ℕ := P * D
def total_area (A_bottom A_walls : ℕ) : ℕ := A_bottom + A_walls
def total_cost (A_total : ℕ) (cost_per_sq_meter : ℚ) : ℚ := A_total * cost_per_sq_meter

theorem plastering_cost_correct :
  total_cost (total_area (area_bottom length width)
                        (area_walls (perimeter_bottom length width) depth))
             cost_per_sq_meter = 2350 :=
by 
  sorry

end NUMINAMATH_GPT_plastering_cost_correct_l2049_204972
