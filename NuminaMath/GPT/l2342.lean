import Mathlib

namespace probability_not_red_is_two_thirds_l2342_234293

-- Given conditions as definitions
def number_of_orange_marbles : ℕ := 4
def number_of_purple_marbles : ℕ := 7
def number_of_red_marbles : ℕ := 8
def number_of_yellow_marbles : ℕ := 5

-- Define the total number of marbles
def total_marbles : ℕ :=
  number_of_orange_marbles + 
  number_of_purple_marbles + 
  number_of_red_marbles + 
  number_of_yellow_marbles

def number_of_non_red_marbles : ℕ :=
  number_of_orange_marbles + 
  number_of_purple_marbles + 
  number_of_yellow_marbles

-- Define the probability
def probability_not_red : ℚ :=
  number_of_non_red_marbles / total_marbles

-- The theorem that states the probability of not picking a red marble is 2/3
theorem probability_not_red_is_two_thirds :
  probability_not_red = 2 / 3 :=
by
  sorry

end probability_not_red_is_two_thirds_l2342_234293


namespace chord_square_l2342_234242

/-- 
Circles with radii 3 and 6 are externally tangent and are internally tangent to a circle with radius 9. 
The circle with radius 9 has a chord that is a common external tangent of the other two circles. Prove that 
the square of the length of this chord is 72.
-/
theorem chord_square (O₁ O₂ O₃ : Type) 
  (r₁ r₂ r₃ : ℝ) 
  (O₁_tangent_O₂ : r₁ + r₂ = 9) 
  (O₃_tangent_O₁ : r₃ - r₁ = 6) 
  (O₃_tangent_O₂ : r₃ - r₂ = 3) 
  (tangent_chord : ℝ) : 
  tangent_chord^2 = 72 :=
by sorry

end chord_square_l2342_234242


namespace distinct_integers_sum_l2342_234227

theorem distinct_integers_sum {a b c d : ℤ} (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_product : a * b * c * d = 25) : a + b + c + d = 0 := by
  sorry

end distinct_integers_sum_l2342_234227


namespace solution_set_of_inequality_l2342_234208

theorem solution_set_of_inequality (x : ℝ) :
  2 * |x - 1| - 1 < 0 ↔ (1 / 2) < x ∧ x < (3 / 2) :=
  sorry

end solution_set_of_inequality_l2342_234208


namespace unit_digit_4137_pow_754_l2342_234275

theorem unit_digit_4137_pow_754 : (4137 ^ 754) % 10 = 9 := by
  sorry

end unit_digit_4137_pow_754_l2342_234275


namespace liters_to_cubic_decimeters_eq_l2342_234203

-- Define the condition for unit conversion
def liter_to_cubic_decimeter : ℝ :=
  1 -- since 1 liter = 1 cubic decimeter

-- Prove the equality for the given quantities
theorem liters_to_cubic_decimeters_eq :
  1.5 = 1.5 * liter_to_cubic_decimeter :=
by
  -- Proof to be filled in
  sorry

end liters_to_cubic_decimeters_eq_l2342_234203


namespace total_pages_in_book_l2342_234244

theorem total_pages_in_book (pages_per_day : ℕ) (days : ℕ) (total_pages : ℕ) 
  (h1 : pages_per_day = 22) (h2 : days = 569) : total_pages = 12518 :=
by
  sorry

end total_pages_in_book_l2342_234244


namespace proof_problem_l2342_234237

-- Definitions of the function and conditions:
def f : ℝ → ℝ := sorry
axiom odd_f : ∀ x, f (-x) = -f x
axiom periodicity_f : ∀ x, f (x + 2) = -f x
axiom f_def_on_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2^x - 1

-- The theorem statement:
theorem proof_problem :
  f 6 < f (11 / 2) ∧ f (11 / 2) < f (-7) :=
by
  sorry

end proof_problem_l2342_234237


namespace negation_of_p_l2342_234253

namespace ProofProblem

variable (x : ℝ)

def p : Prop := ∃ x : ℝ, x^2 + x - 1 ≥ 0

def neg_p : Prop := ∀ x : ℝ, x^2 + x - 1 < 0

theorem negation_of_p : ¬p = neg_p := sorry

end ProofProblem

end negation_of_p_l2342_234253


namespace sum_S5_l2342_234296

-- Geometric sequence definitions and conditions
noncomputable def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^n

noncomputable def sum_of_geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

variables (a r : ℝ)

-- Given conditions translated into Lean:
-- a2 * a3 = 2 * a1
def condition1 := (geometric_sequence a r 1) * (geometric_sequence a r 2) = 2 * a

-- Arithmetic mean of a4 and 2 * a7 is 5/4
def condition2 := (geometric_sequence a r 3 + 2 * geometric_sequence a r 6) / 2 = 5 / 4

-- The final goal proving that S5 = 31
theorem sum_S5 (h1 : condition1 a r) (h2 : condition2 a r) : sum_of_geometric_sequence a r 5 = 31 := by
  apply sorry

end sum_S5_l2342_234296


namespace expression_equals_8_l2342_234273

-- Define the expression we are interested in.
def expression : ℚ :=
  (1 + 1 / 2) * (1 + 1 / 3) * (1 + 1 / 4) * (1 + 1 / 5) * (1 + 1 / 6) * (1 + 1 / 7)

-- Statement we need to prove
theorem expression_equals_8 : expression = 8 := by
  sorry

end expression_equals_8_l2342_234273


namespace chocolates_left_l2342_234216

-- Definitions based on the conditions
def initially_bought := 3
def gave_away := 2
def additionally_bought := 3

-- Theorem statement to prove
theorem chocolates_left : initially_bought - gave_away + additionally_bought = 4 := by
  -- Proof skipped
  sorry

end chocolates_left_l2342_234216


namespace LovelyCakeSlices_l2342_234278

/-- Lovely cuts her birthday cake into some equal pieces.
    One-fourth of the cake was eaten by her visitors.
    Nine slices of cake were kept, representing three-fourths of the total number of slices.
    Prove: Lovely cut her birthday cake into 12 equal pieces. -/
theorem LovelyCakeSlices (totalSlices : ℕ) 
  (h1 : (3 / 4 : ℚ) * totalSlices = 9) : totalSlices = 12 := by
  sorry

end LovelyCakeSlices_l2342_234278


namespace consecutive_integer_sum_l2342_234261

theorem consecutive_integer_sum (n : ℕ) (h1 : n * (n + 1) = 2720) : n + (n + 1) = 103 :=
sorry

end consecutive_integer_sum_l2342_234261


namespace total_area_of_forest_and_fields_l2342_234282

theorem total_area_of_forest_and_fields (r p k : ℝ) (h1 : k = 12) 
  (h2 : r^2 + 4 * p^2 + 45 = 12 * k) :
  (r^2 + 4 * p^2 + 12 * k = 135) :=
by
  -- Proof goes here
  sorry

end total_area_of_forest_and_fields_l2342_234282


namespace value_of_a_l2342_234229

theorem value_of_a 
  (x y a : ℝ)
  (h1 : 2 * x + y = 3 * a)
  (h2 : x - 2 * y = 9 * a)
  (h3 : x + 3 * y = 24) :
  a = -4 :=
sorry

end value_of_a_l2342_234229


namespace calc_expression_l2342_234291

theorem calc_expression :
  (12^4 + 375) * (24^4 + 375) * (36^4 + 375) * (48^4 + 375) * (60^4 + 375) /
  ((6^4 + 375) * (18^4 + 375) * (30^4 + 375) * (42^4 + 375) * (54^4 + 375)) = 159 :=
by
  sorry

end calc_expression_l2342_234291


namespace minimum_value_quot_l2342_234292

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem minimum_value_quot (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : f a = f b) :
  (a^2 + b^2) / (a - b) = 2 * Real.sqrt 2 :=
by
  sorry

end minimum_value_quot_l2342_234292


namespace average_of_last_four_numbers_l2342_234287

theorem average_of_last_four_numbers
  (seven_avg : ℝ)
  (first_three_avg : ℝ)
  (seven_avg_is_62 : seven_avg = 62)
  (first_three_avg_is_58 : first_three_avg = 58) :
  (7 * seven_avg - 3 * first_three_avg) / 4 = 65 :=
by
  rw [seven_avg_is_62, first_three_avg_is_58]
  sorry

end average_of_last_four_numbers_l2342_234287


namespace area_of_third_face_l2342_234236

-- Define the variables for the dimensions of the box: l, w, and h
variables (l w h: ℝ)

-- Given conditions
def face1_area := 120
def face2_area := 72
def volume := 720

-- The relationships between the dimensions and the given areas/volume
def face1_eq : Prop := l * w = face1_area
def face2_eq : Prop := w * h = face2_area
def volume_eq : Prop := l * w * h = volume

-- The statement we need to prove is that the area of the third face (l * h) is 60 cm² given the above equations
theorem area_of_third_face :
  face1_eq l w →
  face2_eq w h →
  volume_eq l w h →
  l * h = 60 :=
by
  intros h1 h2 h3
  sorry

end area_of_third_face_l2342_234236


namespace Jose_age_correct_l2342_234255

variable (Jose Zack Inez : ℕ)

-- Define the conditions
axiom Inez_age : Inez = 15
axiom Zack_age : Zack = Inez + 3
axiom Jose_age : Jose = Zack - 4

-- The proof statement
theorem Jose_age_correct : Jose = 14 :=
by
  -- Proof will be filled in later
  sorry

end Jose_age_correct_l2342_234255


namespace product_divisible_by_six_l2342_234234

theorem product_divisible_by_six (a : ℤ) : 6 ∣ a * (a + 1) * (2 * a + 1) := 
sorry

end product_divisible_by_six_l2342_234234


namespace place_mat_length_l2342_234277

theorem place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) (x : ℝ)
  (table_is_round : r = 3)
  (number_of_mats : n = 8)
  (mat_width : w = 1)
  (mat_length : ∀ (k: ℕ), 0 ≤ k ∧ k < n → (2 * r * Real.sin (Real.pi / n) = x)) :
  x = (3 * Real.sqrt 35) / 10 + 1 / 2 :=
sorry

end place_mat_length_l2342_234277


namespace Liked_Both_Proof_l2342_234225

section DessertProblem

variable (Total_Students Liked_Apple_Pie Liked_Chocolate_Cake Did_Not_Like_Either Liked_Both : ℕ)
variable (h1 : Total_Students = 50)
variable (h2 : Liked_Apple_Pie = 25)
variable (h3 : Liked_Chocolate_Cake = 20)
variable (h4 : Did_Not_Like_Either = 10)

theorem Liked_Both_Proof :
  Liked_Both = (Liked_Apple_Pie + Liked_Chocolate_Cake) - (Total_Students - Did_Not_Like_Either) :=
by
  sorry

end DessertProblem

end Liked_Both_Proof_l2342_234225


namespace optimal_fruit_combination_l2342_234257

structure FruitPrices :=
  (price_2_apples : ℕ)
  (price_6_apples : ℕ)
  (price_12_apples : ℕ)
  (price_2_oranges : ℕ)
  (price_6_oranges : ℕ)
  (price_12_oranges : ℕ)

def minCostFruits : ℕ :=
  sorry

theorem optimal_fruit_combination (fp : FruitPrices) (total_fruits : ℕ)
  (mult_2_or_3 : total_fruits = 15) :
  fp.price_2_apples = 48 →
  fp.price_6_apples = 126 →
  fp.price_12_apples = 224 →
  fp.price_2_oranges = 60 →
  fp.price_6_oranges = 164 →
  fp.price_12_oranges = 300 →
  minCostFruits = 314 :=
by
  sorry

end optimal_fruit_combination_l2342_234257


namespace find_Xe_minus_Ye_l2342_234263

theorem find_Xe_minus_Ye (e X Y : ℕ) (h1 : 8 < e) (h2 : e^2*X + e*Y + e*X + X + e^2*X + X = 243 * e^2):
  X - Y = (2 * e^2 + 4 * e - 726) / 3 :=
by
  sorry

end find_Xe_minus_Ye_l2342_234263


namespace greatest_positive_integer_N_l2342_234221

def condition (x : Int) (y : Int) : Prop :=
  (x^2 - x * y) % 1111 ≠ 0

theorem greatest_positive_integer_N :
  ∃ N : Nat, (∀ (x : Fin N) (y : Fin N), x ≠ y → condition x y) ∧ N = 1000 :=
by
  sorry

end greatest_positive_integer_N_l2342_234221


namespace find_speed_of_stream_l2342_234212

theorem find_speed_of_stream (x : ℝ) (h1 : ∃ x, 1 / (39 - x) = 2 * (1 / (39 + x))) : x = 13 :=
by
sorry

end find_speed_of_stream_l2342_234212


namespace triangle_side_length_sum_l2342_234281

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def distance_squared (p1 p2 : Point3D) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

structure Triangle where
  D : Point3D
  E : Point3D
  F : Point3D

noncomputable def centroid (t : Triangle) : Point3D :=
  let D := t.D
  let E := t.E
  let F := t.F
  { x := (D.x + E.x + F.x) / 3,
    y := (D.y + E.y + F.y) / 3,
    z := (D.z + E.z + F.z) / 3 }

noncomputable def sum_of_squares_centroid_distances (t : Triangle) : ℝ :=
  let G := centroid t
  distance_squared G t.D + distance_squared G t.E + distance_squared G t.F

noncomputable def sum_of_squares_side_lengths (t : Triangle) : ℝ :=
  distance_squared t.D t.E + distance_squared t.D t.F + distance_squared t.E t.F

theorem triangle_side_length_sum (t : Triangle) (h : sum_of_squares_centroid_distances t = 72) :
  sum_of_squares_side_lengths t = 216 :=
sorry

end triangle_side_length_sum_l2342_234281


namespace product_of_four_consecutive_naturals_is_square_l2342_234265

theorem product_of_four_consecutive_naturals_is_square (n : ℕ) : 
  (n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2) := 
by
  sorry

end product_of_four_consecutive_naturals_is_square_l2342_234265


namespace travel_times_either_24_or_72_l2342_234262

variable (A B C : String)
variable (travel_time : String → String → Float)
variable (current : Float)

-- Conditions:
-- 1. Travel times are 24 minutes or 72 minutes
-- 2. Traveling from dock B cannot be balanced with current constraints
-- 3. A 3 km travel with the current is 24 minutes
-- 4. A 3 km travel against the current is 72 minutes

theorem travel_times_either_24_or_72 :
  (∀ (P Q : String), P = A ∨ P = B ∨ P = C ∧ Q = A ∨ Q = B ∨ Q = C →
  (travel_time A C = 72 ∨ travel_time C A = 24)) :=
by
  intros
  sorry

end travel_times_either_24_or_72_l2342_234262


namespace compute_f_of_1_plus_g_of_3_l2342_234219

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x^2 + 1

theorem compute_f_of_1_plus_g_of_3 : f (1 + g 3) = 29 := by 
  sorry

end compute_f_of_1_plus_g_of_3_l2342_234219


namespace christen_peeled_20_potatoes_l2342_234213

-- Define the conditions and question
def homer_rate : ℕ := 3
def time_alone : ℕ := 4
def christen_rate : ℕ := 5
def total_potatoes : ℕ := 44

noncomputable def christen_potatoes : ℕ :=
  (total_potatoes - (homer_rate * time_alone)) / (homer_rate + christen_rate) * christen_rate

theorem christen_peeled_20_potatoes :
  christen_potatoes = 20 := by
  -- Proof steps would go here
  sorry

end christen_peeled_20_potatoes_l2342_234213


namespace largest_common_term_arith_seq_l2342_234299

theorem largest_common_term_arith_seq :
  ∃ a, a < 90 ∧ (∃ n : ℤ, a = 3 + 8 * n) ∧ (∃ m : ℤ, a = 5 + 9 * m) ∧ a = 59 :=
by
  sorry

end largest_common_term_arith_seq_l2342_234299


namespace prove_final_value_is_111_l2342_234222

theorem prove_final_value_is_111 :
  let initial_num := 16
  let doubled_num := initial_num * 2
  let added_five := doubled_num + 5
  let trebled_result := added_five * 3
  trebled_result = 111 :=
by
  sorry

end prove_final_value_is_111_l2342_234222


namespace candy_ratio_l2342_234288

theorem candy_ratio 
  (tabitha_candy : ℕ)
  (stan_candy : ℕ)
  (julie_candy : ℕ)
  (carlos_candy : ℕ)
  (total_candy : ℕ)
  (h1 : tabitha_candy = 22)
  (h2 : stan_candy = 13)
  (h3 : julie_candy = tabitha_candy / 2)
  (h4 : total_candy = 72)
  (h5 : tabitha_candy + stan_candy + julie_candy + carlos_candy = total_candy) :
  carlos_candy / stan_candy = 2 :=
by
  sorry

end candy_ratio_l2342_234288


namespace Gilda_marbles_left_l2342_234230

theorem Gilda_marbles_left (M : ℝ) (h1 : M > 0) :
  let remaining_after_pedro := M - 0.30 * M
  let remaining_after_ebony := remaining_after_pedro - 0.40 * remaining_after_pedro
  remaining_after_ebony / M * 100 = 42 :=
by
  sorry

end Gilda_marbles_left_l2342_234230


namespace multiply_and_simplify_l2342_234270
open Classical

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end multiply_and_simplify_l2342_234270


namespace arithmetic_computation_l2342_234226

theorem arithmetic_computation : 65 * 1515 - 25 * 1515 = 60600 := by
  sorry

end arithmetic_computation_l2342_234226


namespace depth_of_right_frustum_l2342_234204

-- Definitions
def volume_cm3 := 190000 -- Volume in cubic centimeters (190 liters)
def top_edge := 60 -- Length of the top edge in centimeters
def bottom_edge := 40 -- Length of the bottom edge in centimeters
def expected_depth := 75 -- Expected depth in centimeters

-- The following is the statement of the proof
theorem depth_of_right_frustum 
  (V : ℝ) (A1 A2 : ℝ) (h : ℝ)
  (hV : V = 190 * 1000)
  (hA1 : A1 = top_edge * top_edge)
  (hA2 : A2 = bottom_edge * bottom_edge)
  (h_avg : 2 * A1 / (top_edge + bottom_edge) = 2 * A2 / (top_edge + bottom_edge))
  : h = expected_depth := 
sorry

end depth_of_right_frustum_l2342_234204


namespace fraction_equality_l2342_234266

theorem fraction_equality :
  (2 - (1 / 2) * (1 - (1 / 4))) / (2 - (1 - (1 / 3))) = 39 / 32 := 
  sorry

end fraction_equality_l2342_234266


namespace y_intercept_of_line_eq_l2342_234254

theorem y_intercept_of_line_eq (x y : ℝ) (h : x + y - 1 = 0) : y = 1 :=
by
  sorry

end y_intercept_of_line_eq_l2342_234254


namespace Ahmad_eight_steps_l2342_234283

def reach_top (n : Nat) (holes : List Nat) : Nat := sorry

theorem Ahmad_eight_steps (h : reach_top 8 [6] = 8) : True := by 
  trivial

end Ahmad_eight_steps_l2342_234283


namespace odd_function_property_l2342_234232

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (4 - x^2)) / x

theorem odd_function_property (a : ℝ) (h_a : -2 ≤ a ∧ a ≤ 2) (h_fa : f a = -4) : f (-a) = 4 :=
by
  sorry

end odd_function_property_l2342_234232


namespace Kayla_total_items_l2342_234201

theorem Kayla_total_items (T_bars : ℕ) (T_cans : ℕ) (K_bars : ℕ) (K_cans : ℕ)
  (h1 : T_bars = 2 * K_bars) (h2 : T_cans = 2 * K_cans)
  (h3 : T_bars = 12) (h4 : T_cans = 18) : 
  K_bars + K_cans = 15 :=
by {
  -- In order to focus only on statement definition, we use sorry here
  sorry
}

end Kayla_total_items_l2342_234201


namespace michael_current_chickens_l2342_234272

-- Defining variables and constants
variable (initial_chickens final_chickens annual_increase : ℕ)

-- Given conditions
def chicken_increase_condition : Prop :=
  final_chickens = initial_chickens + annual_increase * 9

-- Question to answer
def current_chickens (final_chickens annual_increase : ℕ) : ℕ :=
  final_chickens - annual_increase * 9

-- Proof problem
theorem michael_current_chickens
  (initial_chickens : ℕ)
  (final_chickens : ℕ)
  (annual_increase : ℕ)
  (h1 : chicken_increase_condition final_chickens initial_chickens annual_increase) :
  initial_chickens = 550 :=
by
  -- Formal proof would go here.
  sorry

end michael_current_chickens_l2342_234272


namespace simplify_expression_l2342_234241

theorem simplify_expression (a1 a2 a3 a4 : ℝ) (h1 : 1 - a1 ≠ 0) (h2 : 1 - a2 ≠ 0) (h3 : 1 - a3 ≠ 0) (h4 : 1 - a4 ≠ 0) :
  1 + a1 / (1 - a1) + a2 / ((1 - a1) * (1 - a2)) + a3 / ((1 - a1) * (1 - a2) * (1 - a3)) + 
  (a4 - a1) / ((1 - a1) * (1 - a2) * (1 - a3) * (1 - a4)) = 
  1 / ((1 - a2) * (1 - a3) * (1 - a4)) :=
by
  sorry

end simplify_expression_l2342_234241


namespace representable_by_expression_l2342_234211

theorem representable_by_expression (n : ℕ) :
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (n = (x * y + y * z + z * x) / (x + y + z)) ↔ n ≠ 1 := by
  sorry

end representable_by_expression_l2342_234211


namespace value_of_b_l2342_234258

variable (a b c : ℕ)
variable (h_a_nonzero : a ≠ 0)
variable (h_a : a < 8)
variable (h_b : b < 8)
variable (h_c : c < 8)
variable (h_square : ∃ k, k^2 = a * 8^3 + 3 * 8^2 + b * 8 + c)

theorem value_of_b : b = 1 :=
by sorry

end value_of_b_l2342_234258


namespace intersection_complement_eq_l2342_234286

-- Definitions as per given conditions
def U : Set ℕ := { x | x > 0 ∧ x < 9 }
def A : Set ℕ := { 1, 2, 3, 4 }
def B : Set ℕ := { 3, 4, 5, 6 }

-- Complement of B with respect to U
def C_U_B : Set ℕ := U \ B

-- Statement of the theorem to be proved
theorem intersection_complement_eq : A ∩ C_U_B = { 1, 2 } :=
by
  sorry

end intersection_complement_eq_l2342_234286


namespace part1_part2_l2342_234267

def f (x m : ℝ) : ℝ := |x - 1| - |2 * x + m|

theorem part1 (x : ℝ) (m : ℝ) (h : m = -4) : 
    f x m < 0 ↔ x < 5 / 3 ∨ x > 3 := 
by 
  sorry

theorem part2 (x : ℝ) (h : 1 < x) (h' : ∀ x, 1 < x → f x m < 0) : 
    m ≥ -2 :=
by 
  sorry

end part1_part2_l2342_234267


namespace find_abc_l2342_234223

variables {a b c : ℕ}

theorem find_abc (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : abc ∣ ((a * b - 1) * (b * c - 1) * (c * a - 1))) : a = 2 ∧ b = 3 ∧ c = 5 :=
by {
    sorry
}

end find_abc_l2342_234223


namespace sum_of_terms_l2342_234252

noncomputable def u1 := 8
noncomputable def r := 2

def first_geometric (u2 u3 : ℝ) (u1 r : ℝ) : Prop := 
  u2 = r * u1 ∧ u3 = r^2 * u1

def last_arithmetic (u2 u3 u4 : ℝ) : Prop := 
  u3 - u2 = u4 - u3

def terms (u1 u2 u3 u4 : ℝ) (r : ℝ) : Prop :=
  first_geometric u2 u3 u1 r ∧
  last_arithmetic u2 u3 u4 ∧
  u4 = u1 + 40

theorem sum_of_terms (u1 u2 u3 u4 : ℝ)
  (h : terms u1 u2 u3 u4 r) : u1 + u2 + u3 + u4 = 104 :=
by {
  sorry
}

end sum_of_terms_l2342_234252


namespace production_growth_rate_eq_l2342_234205

theorem production_growth_rate_eq 
  (x : ℝ)
  (H : 100 + 100 * (1 + x) + 100 * (1 + x) ^ 2 = 364) : 
  100 + 100 * (1 + x) + 100 * (1 + x) ^ 2 = 364 :=
by {
  sorry
}

end production_growth_rate_eq_l2342_234205


namespace adjusted_area_difference_l2342_234206

noncomputable def largest_circle_area (d : ℝ) : ℝ :=
  let r := d / 2
  r^2 * Real.pi

noncomputable def middle_circle_area (r : ℝ) : ℝ :=
  r^2 * Real.pi

noncomputable def smaller_circle_area (r : ℝ) : ℝ :=
  r^2 * Real.pi

theorem adjusted_area_difference (d_large r_middle r_small : ℝ) 
  (h_large : d_large = 30) (h_middle : r_middle = 10) (h_small : r_small = 5) :
  largest_circle_area d_large - middle_circle_area r_middle - smaller_circle_area r_small = 100 * Real.pi :=
by
  sorry

end adjusted_area_difference_l2342_234206


namespace total_pies_l2342_234200

-- Define the number of each type of pie.
def apple_pies : Nat := 2
def pecan_pies : Nat := 4
def pumpkin_pies : Nat := 7

-- Prove the total number of pies.
theorem total_pies : apple_pies + pecan_pies + pumpkin_pies = 13 := by
  sorry

end total_pies_l2342_234200


namespace lolita_milk_per_week_l2342_234279

def weekday_milk : ℕ := 3
def saturday_milk : ℕ := 2 * weekday_milk
def sunday_milk : ℕ := 3 * weekday_milk
def total_milk_week : ℕ := 5 * weekday_milk + saturday_milk + sunday_milk

theorem lolita_milk_per_week : total_milk_week = 30 := 
by 
  sorry

end lolita_milk_per_week_l2342_234279


namespace find_smaller_root_l2342_234295

theorem find_smaller_root :
  ∀ x : ℝ, (x - 2 / 3) ^ 2 + (x - 2 / 3) * (x - 1 / 3) = 0 → x = 1 / 2 :=
by
  sorry

end find_smaller_root_l2342_234295


namespace cost_of_lamps_and_bulbs_l2342_234259

theorem cost_of_lamps_and_bulbs : 
    let lamp_cost := 7
    let bulb_cost := lamp_cost - 4
    let total_cost := (lamp_cost * 2) + (bulb_cost * 6)
    total_cost = 32 := by
  let lamp_cost := 7
  let bulb_cost := lamp_cost - 4
  let total_cost := (lamp_cost * 2) + (bulb_cost * 6)
  sorry

end cost_of_lamps_and_bulbs_l2342_234259


namespace min_value_of_parabola_in_interval_l2342_234280

theorem min_value_of_parabola_in_interval :
  ∀ x : ℝ, -10 ≤ x ∧ x ≤ 0 → (x^2 + 12 * x + 35) ≥ -1 := by
  sorry

end min_value_of_parabola_in_interval_l2342_234280


namespace arithmetic_sequence_problem_l2342_234202

theorem arithmetic_sequence_problem (a : ℕ → ℕ) (S : ℕ → ℕ) (k : ℕ) 
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : S 9 = 81)
  (h3 : a (k - 4) = 191)
  (h4 : S k = 10000) :
  k = 100 :=
by
  sorry

end arithmetic_sequence_problem_l2342_234202


namespace employee_price_l2342_234294

theorem employee_price (wholesale_cost retail_markup employee_discount : ℝ) 
    (h₁ : wholesale_cost = 200) 
    (h₂ : retail_markup = 0.20) 
    (h₃ : employee_discount = 0.25) : 
    (wholesale_cost * (1 + retail_markup)) * (1 - employee_discount) = 180 := 
by
  sorry

end employee_price_l2342_234294


namespace c_put_15_oxen_l2342_234250

theorem c_put_15_oxen (x : ℕ):
  (10 * 7 + 12 * 5 + 3 * x = 130 + 3 * x) →
  (175 * 3 * x / (130 + 3 * x) = 45) →
  x = 15 :=
by
  intros h1 h2
  sorry

end c_put_15_oxen_l2342_234250


namespace solve_for_x_l2342_234207

theorem solve_for_x (x : ℚ) :
  (x^2 - 4*x + 3) / (x^2 - 7*x + 6) = (x^2 - 3*x - 10) / (x^2 - 2*x - 15) →
  x = -3 / 4 :=
by
  intro h
  sorry

end solve_for_x_l2342_234207


namespace probability_at_least_one_female_is_five_sixths_l2342_234224

-- Declare the total number of male and female students
def total_male_students := 6
def total_female_students := 4
def total_students := total_male_students + total_female_students
def selected_students := 3

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total ways to select 3 students from 10 students
def total_ways_to_select_3 := binomial_coefficient total_students selected_students

-- Ways to select 3 male students from 6 male students
def ways_to_select_3_males := binomial_coefficient total_male_students selected_students

-- Probability of selecting at least one female student
def probability_of_at_least_one_female : ℚ := 1 - (ways_to_select_3_males / total_ways_to_select_3)

-- The theorem statement to be proved
theorem probability_at_least_one_female_is_five_sixths :
  probability_of_at_least_one_female = 5/6 := by
  sorry

end probability_at_least_one_female_is_five_sixths_l2342_234224


namespace sin_double_angle_l2342_234268

theorem sin_double_angle
  (α : ℝ) (h1 : Real.sin (3 * Real.pi / 2 - α) = 3 / 5) (h2 : α ∈ Set.Ioo Real.pi (3 * Real.pi / 2)) :
  Real.sin (2 * α) = 24 / 25 :=
sorry

end sin_double_angle_l2342_234268


namespace percentage_of_loss_is_25_l2342_234260

-- Definitions from conditions
def CP : ℝ := 2800
def SP : ℝ := 2100

-- Proof statement
theorem percentage_of_loss_is_25 : ((CP - SP) / CP) * 100 = 25 := by
  sorry

end percentage_of_loss_is_25_l2342_234260


namespace cos_alpha_beta_value_l2342_234276

theorem cos_alpha_beta_value
  (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : -π / 2 < β ∧ β < 0)
  (h3 : Real.cos (π / 4 + α) = 1 / 3)
  (h4 : Real.cos (π / 4 - β) = Real.sqrt 3 / 3) :
  Real.cos (α + β) = (5 * Real.sqrt 3) / 9 := 
by
  sorry

end cos_alpha_beta_value_l2342_234276


namespace pet_shop_dogs_l2342_234245

theorem pet_shop_dogs (D C B : ℕ) (x : ℕ) (h1 : D = 3 * x) (h2 : C = 5 * x) (h3 : B = 9 * x) (h4 : D + B = 204) : D = 51 := by
  -- omitted proof
  sorry

end pet_shop_dogs_l2342_234245


namespace divide_equally_l2342_234215

-- Define the input values based on the conditions.
def brother_strawberries := 3 * 15
def kimberly_strawberries := 8 * brother_strawberries
def parents_strawberries := kimberly_strawberries - 93
def total_strawberries := brother_strawberries + kimberly_strawberries + parents_strawberries
def family_members := 4

-- Define the theorem to prove the question.
theorem divide_equally : 
    (total_strawberries / family_members) = 168 :=
by
    -- (proof goes here)
    sorry

end divide_equally_l2342_234215


namespace carl_additional_hours_per_week_l2342_234209

def driving_hours_per_day : ℕ := 2

def days_per_week : ℕ := 7

def total_hours_two_weeks_after_promotion : ℕ := 40

def driving_hours_per_week_before_promotion : ℕ := driving_hours_per_day * days_per_week

def driving_hours_per_week_after_promotion : ℕ := total_hours_two_weeks_after_promotion / 2

def additional_hours_per_week : ℕ := driving_hours_per_week_after_promotion - driving_hours_per_week_before_promotion

theorem carl_additional_hours_per_week : 
  additional_hours_per_week = 6 :=
by
  -- Using plain arithmetic based on given definitions
  sorry

end carl_additional_hours_per_week_l2342_234209


namespace amount_spent_on_milk_l2342_234214

-- Define conditions
def monthly_salary (S : ℝ) := 0.10 * S = 1800
def rent := 5000
def groceries := 4500
def education := 2500
def petrol := 2000
def miscellaneous := 700
def total_expenses (S : ℝ) := S - 1800
def known_expenses := rent + groceries + education + petrol + miscellaneous

-- Define the proof problem
theorem amount_spent_on_milk (S : ℝ) (milk : ℝ) :
  monthly_salary S →
  total_expenses S = known_expenses + milk →
  milk = 1500 :=
by
  sorry

end amount_spent_on_milk_l2342_234214


namespace wages_of_one_man_l2342_234249

variable (R : Type) [DivisionRing R] [DecidableEq R]
variable (money : R)
variable (num_men : ℕ := 5)
variable (num_women : ℕ := 8)
variable (total_wages : R := 180)
variable (wages_men : R := 36)

axiom equal_women : num_men = num_women
axiom total_earnings (wages : ℕ → R) :
  (wages num_men) + (wages num_women) + (wages 8) = total_wages

theorem wages_of_one_man :
  wages_men = total_wages / num_men := by
  sorry

end wages_of_one_man_l2342_234249


namespace sum_of_cubes_eq_twice_product_of_roots_l2342_234290

theorem sum_of_cubes_eq_twice_product_of_roots (m : ℝ) :
  (∃ a b : ℝ, (3*a^2 + 6*a + m = 0) ∧ (3*b^2 + 6*b + m = 0) ∧ (a ≠ b)) → 
  (a^3 + b^3 = 2 * a * b) → 
  m = 6 :=
by
  intros h_exists sum_eq_twice_product
  sorry

end sum_of_cubes_eq_twice_product_of_roots_l2342_234290


namespace absolute_difference_rectangle_l2342_234239

theorem absolute_difference_rectangle 
  (x y r k : ℝ)
  (h1 : 2 * x + 2 * y = 4 * r)
  (h2 : (x^2 + y^2) = (k * x)^2) :
  |x - y| = k * x :=
by
  sorry

end absolute_difference_rectangle_l2342_234239


namespace prob_one_mistake_eq_l2342_234218

-- Define the probability of making a mistake on a single question
def prob_mistake : ℝ := 0.1

-- Define the probability of answering correctly on a single question
def prob_correct : ℝ := 1 - prob_mistake

-- Define the probability of answering all three questions correctly
def three_correct : ℝ := prob_correct ^ 3

-- Define the probability of making at least one mistake in three questions
def prob_at_least_one_mistake := 1 - three_correct

-- The theorem states that the above probability is equal to 1 - 0.9^3
theorem prob_one_mistake_eq :
  prob_at_least_one_mistake = 1 - (0.9 ^ 3) :=
by
  sorry

end prob_one_mistake_eq_l2342_234218


namespace value_of_bc_l2342_234233

theorem value_of_bc (a b c d : ℝ) (h1 : a + b = 14) (h2 : c + d = 3) (h3 : a + d = 8) : b + c = 9 :=
sorry

end value_of_bc_l2342_234233


namespace Shyam_money_l2342_234238

theorem Shyam_money (r g k s : ℕ) 
  (h1 : 7 * g = 17 * r) 
  (h2 : 7 * k = 17 * g)
  (h3 : 11 * s = 13 * k)
  (hr : r = 735) : 
  s = 2119 := 
by
  sorry

end Shyam_money_l2342_234238


namespace boys_count_l2342_234220

def total_pupils : ℕ := 485
def number_of_girls : ℕ := 232
def number_of_boys : ℕ := total_pupils - number_of_girls

theorem boys_count : number_of_boys = 253 := by
  -- The proof is omitted according to instruction
  sorry

end boys_count_l2342_234220


namespace range_of_a_l2342_234243

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
if h : x < 1 then a * x^2 - 6 * x + a^2 + 1 else x^(5 - 2 * a)

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (5/2 < a ∧ a ≤ 3) :=
sorry

end range_of_a_l2342_234243


namespace monotonicity_range_of_a_l2342_234251

noncomputable def f (x a : ℝ) : ℝ := Real.log x + a * (1 - x)
noncomputable def f' (x a : ℝ) : ℝ := 1 / x - a

-- 1. Monotonicity discussion
theorem monotonicity (a x : ℝ) (h : 0 < x) : 
  (a ≤ 0 → ∀ x, 0 < x → f' x a > 0) ∧
  (a > 0 → (∀ x, 0 < x ∧ x < 1 / a → f' x a > 0) ∧ (∀ x, x > 1 / a → f' x a < 0)) :=
sorry

-- 2. Range of a for maximum value condition
noncomputable def g (a : ℝ) : ℝ := Real.log a + a - 1

theorem range_of_a (a : ℝ) : 
  (0 < a) ∧ (a < 1) ↔ g a < 0 :=
sorry

end monotonicity_range_of_a_l2342_234251


namespace tony_water_trips_calculation_l2342_234235

noncomputable def tony_drinks_water_after_every_n_trips (bucket_capacity_sand : ℤ) 
                                                        (sandbox_depth : ℤ) (sandbox_width : ℤ) 
                                                        (sandbox_length : ℤ) (sand_weight_cubic_foot : ℤ) 
                                                        (water_consumption : ℤ) (water_bottle_ounces : ℤ) 
                                                        (water_bottle_cost : ℤ) (money_with_tony : ℤ) 
                                                        (expected_change : ℤ) : ℤ :=
  let volume_sandbox := sandbox_depth * sandbox_width * sandbox_length
  let total_sand_weight := volume_sandbox * sand_weight_cubic_foot
  let trips_needed := total_sand_weight / bucket_capacity_sand
  let money_spent_on_water := money_with_tony - expected_change
  let water_bottles_bought := money_spent_on_water / water_bottle_cost
  let total_water_ounces := water_bottles_bought * water_bottle_ounces
  let drinking_sessions := total_water_ounces / water_consumption
  trips_needed / drinking_sessions

theorem tony_water_trips_calculation : 
  tony_drinks_water_after_every_n_trips 2 2 4 5 3 3 15 2 10 4 = 4 := 
by 
  sorry

end tony_water_trips_calculation_l2342_234235


namespace bottles_produced_by_twenty_machines_l2342_234240

-- Definitions corresponding to conditions
def bottles_per_machine_per_minute (total_machines : ℕ) (total_bottles : ℕ) : ℕ :=
  total_bottles / total_machines

def bottles_produced (machines : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  machines * rate * time

-- Given conditions
axiom six_machines_rate : ∀ (machines total_bottles : ℕ), machines = 6 → total_bottles = 270 →
  bottles_per_machine_per_minute machines total_bottles = 45

-- Prove the question == answer given conditions
theorem bottles_produced_by_twenty_machines :
  bottles_produced 20 45 4 = 3600 :=
by sorry

end bottles_produced_by_twenty_machines_l2342_234240


namespace range_of_m_l2342_234246

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) (h_even : ∀ x, f x = f (-x)) 
 (h_decreasing : ∀ {x y}, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x)
 (h_condition : ∀ x, 1 ≤ x → x ≤ 3 → f (2 * m * x - Real.log x - 3) ≥ 2 * f 3 - f (Real.log x + 3 - 2 * m * x)) :
  m ∈ Set.Icc (1 / (2 * Real.exp 1)) ((Real.log 3 + 6) / 6) :=
sorry

end range_of_m_l2342_234246


namespace find_divisors_of_10_pow_10_sum_157_l2342_234271

theorem find_divisors_of_10_pow_10_sum_157 
  (x y : ℕ) 
  (hx₁ : 0 < x) 
  (hy₁ : 0 < y) 
  (hx₂ : x ∣ 10^10) 
  (hy₂ : y ∣ 10^10) 
  (hxy₁ : x ≠ y) 
  (hxy₂ : x + y = 157) : 
  (x = 32 ∧ y = 125) ∨ (x = 125 ∧ y = 32) := 
by
  sorry

end find_divisors_of_10_pow_10_sum_157_l2342_234271


namespace total_apples_l2342_234247

-- Definitions based on the problem conditions
def marin_apples : ℕ := 8
def david_apples : ℕ := (3 * marin_apples) / 4
def amanda_apples : ℕ := (3 * david_apples) / 2 + 2

-- The statement that we need to prove
theorem total_apples : marin_apples + david_apples + amanda_apples = 25 := by
  -- The proof steps will go here
  sorry

end total_apples_l2342_234247


namespace initial_value_amount_l2342_234289

theorem initial_value_amount (P : ℝ) 
  (h1 : ∀ t, t ≥ 0 → t = P * (1 + (1/8)) ^ t) 
  (h2 : P * (1 + (1/8)) ^ 2 = 105300) : 
  P = 83200 := 
sorry

end initial_value_amount_l2342_234289


namespace find_mistaken_number_l2342_234248

theorem find_mistaken_number : 
  ∃! x : ℕ, (x ∈ {n : ℕ | n ≥ 10 ∧ n < 100 ∧ (n % 10 = 5 ∨ n % 10 = 0)} ∧ 
  (10 + 15 + 20 + 25 + 30 + 35 + 40 + 45 + 50 + 55 + 60 + 65 + 70 + 75 + 80 + 85 + 90 + 95) + 2 * x = 1035) :=
sorry

end find_mistaken_number_l2342_234248


namespace conic_section_focus_l2342_234297

theorem conic_section_focus {m : ℝ} (h_non_zero : m ≠ 0) (h_non_five : m ≠ 5)
  (h_focus : ∃ (x_focus y_focus : ℝ), (x_focus, y_focus) = (2, 0) 
  ∧ (x_focus = c ∧ x_focus^2 / 4 = 5 * (1 - c^2 / m))) : m = 9 := 
by
  sorry

end conic_section_focus_l2342_234297


namespace larger_number_is_21_l2342_234231

theorem larger_number_is_21 (x y : ℤ) (h1 : x + y = 35) (h2 : x - y = 7) : x = 21 := 
by 
  sorry

end larger_number_is_21_l2342_234231


namespace teresa_age_when_michiko_born_l2342_234284

def conditions (T M Michiko K Yuki : ℕ) : Prop := 
  T = 59 ∧ 
  M = 71 ∧ 
  M - Michiko = 38 ∧ 
  K = Michiko - 4 ∧ 
  Yuki = K - 3 ∧ 
  (Yuki + 3) - (26 - 25) = 25

theorem teresa_age_when_michiko_born :
  ∃ T M Michiko K Yuki, conditions T M Michiko K Yuki → T - Michiko = 26 :=
  by
  sorry

end teresa_age_when_michiko_born_l2342_234284


namespace number_equation_form_l2342_234264

variable (a : ℝ)

theorem number_equation_form :
  3 * a + 5 = 4 * a := 
sorry

end number_equation_form_l2342_234264


namespace zoey_finishes_on_wednesday_l2342_234217

noncomputable def day_zoey_finishes (n : ℕ) : String :=
  let total_days := (n * (n + 1)) / 2
  match total_days % 7 with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | 6 => "Saturday"
  | _ => "Error"

theorem zoey_finishes_on_wednesday : day_zoey_finishes 18 = "Wednesday" :=
by
  -- Calculate that Zoey takes 171 days to read 18 books
  -- Recall that 171 mod 7 = 3, so she finishes on "Wednesday"
  sorry

end zoey_finishes_on_wednesday_l2342_234217


namespace direction_vector_of_line_l2342_234274

noncomputable def direction_vector_of_line_eq : Prop :=
  ∃ u v, ∀ x y, (x / 4) + (y / 2) = 1 → (u, v) = (-2, 1)

theorem direction_vector_of_line :
  direction_vector_of_line_eq := sorry

end direction_vector_of_line_l2342_234274


namespace max_x_plus_2y_l2342_234256

theorem max_x_plus_2y (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 9) 
  (h2 : 3 * x + 5 * y ≤ 15) : 
  x + 2 * y ≤ 6 :=
sorry

end max_x_plus_2y_l2342_234256


namespace angle_terminal_side_equivalence_l2342_234228

theorem angle_terminal_side_equivalence (k : ℤ) : 
    ∃ k : ℤ, 405 = k * 360 + 45 :=
by
  sorry

end angle_terminal_side_equivalence_l2342_234228


namespace correct_calculation_is_7_88_l2342_234298

theorem correct_calculation_is_7_88 (x : ℝ) (h : x * 8 = 56) : (x / 8) + 7 = 7.88 :=
by
  have hx : x = 7 := by
    linarith [h]
  rw [hx]
  norm_num
  sorry

end correct_calculation_is_7_88_l2342_234298


namespace polynomial_arithmetic_sequence_roots_l2342_234210

theorem polynomial_arithmetic_sequence_roots (p q : ℝ) (h : ∃ a b c d : ℝ, 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  a + 3*(b - a) = b ∧ b + 3*(c - b) = c ∧ c + 3*(d - c) = d ∧ 
  (a^4 + p * a^2 + q = 0) ∧ (b^4 + p * b^2 + q = 0) ∧ 
  (c^4 + p * c^2 + q = 0) ∧ (d^4 + p * d^2 + q = 0)) :
  p ≤ 0 ∧ q = 0.09 * p^2 := 
sorry

end polynomial_arithmetic_sequence_roots_l2342_234210


namespace combined_jail_time_in_weeks_l2342_234269

-- Definitions based on conditions
def days_of_protest : ℕ := 30
def number_of_cities : ℕ := 21
def daily_arrests_per_city : ℕ := 10
def days_in_jail_pre_trial : ℕ := 4
def sentence_weeks : ℕ := 2
def jail_fraction_of_sentence : ℕ := 1 / 2

-- Calculate the combined weeks of jail time
theorem combined_jail_time_in_weeks : 
  (days_of_protest * daily_arrests_per_city * number_of_cities) * 
  (days_in_jail_pre_trial + (sentence_weeks * 7 * jail_fraction_of_sentence)) / 
  7 = 9900 := 
by sorry

end combined_jail_time_in_weeks_l2342_234269


namespace pizzas_ordered_l2342_234285

def number_of_people : ℝ := 8.0
def slices_per_person : ℝ := 2.625
def slices_per_pizza : ℝ := 8.0

theorem pizzas_ordered : ⌈number_of_people * slices_per_person / slices_per_pizza⌉ = 3 := 
by
  sorry

end pizzas_ordered_l2342_234285
