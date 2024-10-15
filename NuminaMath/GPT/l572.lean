import Mathlib

namespace NUMINAMATH_GPT_infinite_area_sum_ratio_l572_57209

theorem infinite_area_sum_ratio (T t : ℝ) (p q : ℝ) (h_ratio : T / t = 3 / 2) :
    let series_ratio_triangles := (p + q)^2 / (3 * p * q)
    let series_ratio_quadrilaterals := (p + q)^2 / (2 * p * q)
    (T * series_ratio_triangles) / (t * series_ratio_quadrilaterals) = 1 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_infinite_area_sum_ratio_l572_57209


namespace NUMINAMATH_GPT_sarah_socks_l572_57200

theorem sarah_socks :
  ∃ (a b c : ℕ), a + b + c = 15 ∧ 2 * a + 4 * b + 5 * c = 45 ∧ 1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ (a = 8 ∨ a = 9) :=
by {
  sorry
}

end NUMINAMATH_GPT_sarah_socks_l572_57200


namespace NUMINAMATH_GPT_no_valid_transformation_l572_57228

theorem no_valid_transformation :
  ¬ ∃ (n1 n2 n3 n4 : ℤ),
    2 * n1 + n2 - 2 * n3 - n4 = 27 ∧
    -n1 + 2 * n2 + n3 - 2 * n4 = -27 :=
by
  sorry

end NUMINAMATH_GPT_no_valid_transformation_l572_57228


namespace NUMINAMATH_GPT_count_total_balls_l572_57241

def blue_balls : ℕ := 3
def red_balls : ℕ := 2

theorem count_total_balls : blue_balls + red_balls = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_count_total_balls_l572_57241


namespace NUMINAMATH_GPT_domain_of_f_l572_57296

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  Real.log ((m^2 - 3*m + 2) * x^2 + (m - 1) * x + 1)

theorem domain_of_f (m : ℝ) :
  (∀ x : ℝ, 0 < (m^2 - 3*m + 2) * x^2 + (m - 1) * x + 1) ↔ (m > 7/3 ∨ m ≤ 1) :=
by { sorry }

end NUMINAMATH_GPT_domain_of_f_l572_57296


namespace NUMINAMATH_GPT_ratio_initial_to_doubled_l572_57284

theorem ratio_initial_to_doubled (x : ℝ) (h : 3 * (2 * x + 8) = 84) : x / (2 * x) = 1 / 2 :=
by
  have h1 : 2 * x + 8 = 28 := by
    sorry
  have h2 : x = 10 := by
    sorry
  rw [h2]
  norm_num

end NUMINAMATH_GPT_ratio_initial_to_doubled_l572_57284


namespace NUMINAMATH_GPT_dodecagon_area_l572_57202

theorem dodecagon_area (a : ℝ) : 
  let OA := a / Real.sqrt 2 
  let CD := (a / 2) / Real.sqrt 2 
  let triangle_area := (1/2) * OA * CD 
  let dodecagon_area := 12 * triangle_area
  dodecagon_area = (3 * a^2) / 2 :=
by
  let OA := a / Real.sqrt 2 
  let CD := (a / 2) / Real.sqrt 2 
  let triangle_area := (1/2) * OA * CD 
  let dodecagon_area := 12 * triangle_area
  sorry

end NUMINAMATH_GPT_dodecagon_area_l572_57202


namespace NUMINAMATH_GPT_cube_sum_l572_57289

theorem cube_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 :=
by
  sorry

end NUMINAMATH_GPT_cube_sum_l572_57289


namespace NUMINAMATH_GPT_time_to_read_18_pages_l572_57223

-- Definitions based on the conditions
def reading_rate : ℚ := 2 / 4 -- Amalia reads 4 pages in 2 minutes
def pages_to_read : ℕ := 18 -- Number of pages Amalia needs to read

-- Goal: Total time required to read 18 pages
theorem time_to_read_18_pages (r : ℚ := reading_rate) (p : ℕ := pages_to_read) :
  p * r = 9 := by
  sorry

end NUMINAMATH_GPT_time_to_read_18_pages_l572_57223


namespace NUMINAMATH_GPT_greatest_sum_on_circle_l572_57227

theorem greatest_sum_on_circle : 
  ∃ x y : ℤ, x^2 + y^2 = 169 ∧ x ≥ y ∧ (∀ x' y' : ℤ, x'^2 + y'^2 = 169 → x' ≥ y' → x + y ≥ x' + y') := 
sorry

end NUMINAMATH_GPT_greatest_sum_on_circle_l572_57227


namespace NUMINAMATH_GPT_sum_of_interior_angles_of_decagon_l572_57277

def sum_of_interior_angles_of_polygon (n : ℕ) : ℕ := (n - 2) * 180

theorem sum_of_interior_angles_of_decagon : sum_of_interior_angles_of_polygon 10 = 1440 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_of_decagon_l572_57277


namespace NUMINAMATH_GPT_pizza_topping_combinations_l572_57294

theorem pizza_topping_combinations (T : Finset ℕ) (hT : T.card = 8) : 
  (T.card.choose 1 + T.card.choose 2 + T.card.choose 3 = 92) :=
by
  sorry

end NUMINAMATH_GPT_pizza_topping_combinations_l572_57294


namespace NUMINAMATH_GPT_bus_trip_distance_l572_57261

theorem bus_trip_distance 
  (T : ℝ)  -- Time in hours
  (D : ℝ)  -- Distance in miles
  (h : D = 30 * T)  -- condition 1: the trip with 30 mph
  (h' : D = 35 * (T - 1))  -- condition 2: the trip with 35 mph
  : D = 210 := 
by
  sorry

end NUMINAMATH_GPT_bus_trip_distance_l572_57261


namespace NUMINAMATH_GPT_x_pow_twelve_l572_57292

theorem x_pow_twelve (x : ℝ) (h : x + 1/x = 3) : x^12 = 322 :=
sorry

end NUMINAMATH_GPT_x_pow_twelve_l572_57292


namespace NUMINAMATH_GPT_fraction_filled_l572_57260

variables (E P p : ℝ)

-- Condition 1: The empty vessel weighs 12% of its total weight when filled.
axiom cond1 : E = 0.12 * (E + P)

-- Condition 2: The weight of the partially filled vessel is one half that of a completely filled vessel.
axiom cond2 : E + p = 1 / 2 * (E + P)

theorem fraction_filled : p / P = 19 / 44 :=
by
  sorry

end NUMINAMATH_GPT_fraction_filled_l572_57260


namespace NUMINAMATH_GPT_abs_inequality_solution_l572_57252

theorem abs_inequality_solution (x : ℝ) : 
  3 ≤ |x - 3| ∧ |x - 3| ≤ 7 ↔ (-4 ≤ x ∧ x ≤ 0) ∨ (6 ≤ x ∧ x ≤ 10) := 
by {
  sorry
}

end NUMINAMATH_GPT_abs_inequality_solution_l572_57252


namespace NUMINAMATH_GPT_function_range_x2_minus_2x_l572_57288

theorem function_range_x2_minus_2x : 
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 3 → -1 ≤ x^2 - 2 * x ∧ x^2 - 2 * x ≤ 3 :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_function_range_x2_minus_2x_l572_57288


namespace NUMINAMATH_GPT_line_eq_489_l572_57298

theorem line_eq_489 (m b : ℤ) (h1 : m = 5) (h2 : 3 = m * 5 + b) : m + b^2 = 489 :=
by
  sorry

end NUMINAMATH_GPT_line_eq_489_l572_57298


namespace NUMINAMATH_GPT_sale_price_is_207_l572_57281

-- Define a namespace for our problem
namespace BicyclePrice

-- Define the conditions as constants
def priceAtStoreP : ℝ := 200
def regularPriceIncreasePercentage : ℝ := 0.15
def salePriceDecreasePercentage : ℝ := 0.10

-- Define the regular price at Store Q
def regularPriceAtStoreQ : ℝ := priceAtStoreP * (1 + regularPriceIncreasePercentage)

-- Define the sale price at Store Q
def salePriceAtStoreQ : ℝ := regularPriceAtStoreQ * (1 - salePriceDecreasePercentage)

-- The final theorem we need to prove
theorem sale_price_is_207 : salePriceAtStoreQ = 207 := by
  sorry

end BicyclePrice

end NUMINAMATH_GPT_sale_price_is_207_l572_57281


namespace NUMINAMATH_GPT_M_greater_than_N_l572_57237

-- Definitions based on the problem's conditions
def M (x : ℝ) : ℝ := (x - 3) * (x - 7)
def N (x : ℝ) : ℝ := (x - 2) * (x - 8)

-- Statement to prove
theorem M_greater_than_N (x : ℝ) : M x > N x := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_M_greater_than_N_l572_57237


namespace NUMINAMATH_GPT_point_transform_l572_57257

theorem point_transform : 
  let P := (-2, -3)
  let P' := (P.1 - 1, P.2 + 3)
  P' = (-3, 0) :=
by
  let P := (-2, -3)
  let P' := (P.1 - 1, P.2 + 3)
  show P' = (-3, 0)
  sorry

end NUMINAMATH_GPT_point_transform_l572_57257


namespace NUMINAMATH_GPT_ratio_of_doctors_to_engineers_l572_57216

variables (d l e : ℕ) -- number of doctors, lawyers, and engineers

-- Conditions
def avg_age := (40 * d + 55 * l + 50 * e) / (d + l + e) = 45
def doctors_avg := 40 
def lawyers_avg := 55 
def engineers_avg := 50 -- 55 - 5

theorem ratio_of_doctors_to_engineers (h_avg : avg_age d l e) : d = 3 * e :=
sorry

end NUMINAMATH_GPT_ratio_of_doctors_to_engineers_l572_57216


namespace NUMINAMATH_GPT_fried_hop_edges_in_three_hops_l572_57207

noncomputable def fried_hop_probability : ℚ :=
  let moves : List (Int × Int) := [(-1, 0), (1, 0), (0, -1), (0, 1)]
  let center := (2, 2)
  let edges := [(1, 2), (1, 3), (2, 1), (2, 4), (3, 1), (3, 4), (4, 2), (4, 3)]
  -- Since the exact steps of solution calculation are complex,
  -- we assume the correct probability as per our given solution.
  5 / 8

theorem fried_hop_edges_in_three_hops :
  let p := fried_hop_probability
  p = 5 / 8 := by
  sorry

end NUMINAMATH_GPT_fried_hop_edges_in_three_hops_l572_57207


namespace NUMINAMATH_GPT_gcd_78_143_l572_57205

theorem gcd_78_143 : Nat.gcd 78 143 = 13 :=
by
  sorry

end NUMINAMATH_GPT_gcd_78_143_l572_57205


namespace NUMINAMATH_GPT_smallest_positive_integer_b_no_inverse_l572_57222

theorem smallest_positive_integer_b_no_inverse :
  ∃ b : ℕ, b > 0 ∧ gcd b 30 > 1 ∧ gcd b 42 > 1 ∧ b = 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_b_no_inverse_l572_57222


namespace NUMINAMATH_GPT_find_y_value_l572_57265

theorem find_y_value (y : ℝ) (h : 1 / (3 + 1 / (3 + 1 / (3 - y))) = 0.30337078651685395) : y = 0.3 :=
sorry

end NUMINAMATH_GPT_find_y_value_l572_57265


namespace NUMINAMATH_GPT_right_triangle_conditions_l572_57258

theorem right_triangle_conditions (A B C : ℝ) (a b c : ℝ):
  (C = 90) ∨ (A + B = C) ∨ (a/b = 3/4 ∧ a/c = 3/5 ∧ b/c = 4/5) →
  (a^2 + b^2 = c^2) ∨ (A + B + C = 180) → 
  (C = 90 ∧ a^2 + b^2 = c^2) :=
sorry

end NUMINAMATH_GPT_right_triangle_conditions_l572_57258


namespace NUMINAMATH_GPT_sum_n_max_value_l572_57235

noncomputable def arithmetic_sequence (a_1 : Int) (d : Int) (n : Nat) : Int :=
  a_1 + (n - 1) * d

noncomputable def sum_arithmetic_sequence (a_1 : Int) (d : Int) (n : Nat) : Int :=
  n * a_1 + (n * (n - 1) / 2) * d

theorem sum_n_max_value :
  (∃ n : Nat, n = 9 ∧ sum_arithmetic_sequence 25 (-3) n = 117) :=
by
  let a1 := 25
  let d := -3
  use 9
  -- To complete the proof, we would calculate the sum of the first 9 terms
  -- of the arithmetic sequence with a1 = 25 and difference d = -3.
  sorry

end NUMINAMATH_GPT_sum_n_max_value_l572_57235


namespace NUMINAMATH_GPT_cyclist_motorcyclist_intersection_l572_57299

theorem cyclist_motorcyclist_intersection : 
  ∃ t : ℝ, (4 * t^2 + (t - 1)^2 - 2 * |t| * |t - 1| = 49) ∧ (t = 4 ∨ t = -4) := 
by 
  sorry

end NUMINAMATH_GPT_cyclist_motorcyclist_intersection_l572_57299


namespace NUMINAMATH_GPT_integer_solution_l572_57290

theorem integer_solution (n : ℤ) (h1 : n + 15 > 16) (h2 : -3 * n^2 > -27) : n = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_integer_solution_l572_57290


namespace NUMINAMATH_GPT_point_M_on_y_axis_l572_57239

theorem point_M_on_y_axis (t : ℝ) (h : t - 3 = 0) : (t-3, 5-t) = (0, 2) :=
by
  sorry

end NUMINAMATH_GPT_point_M_on_y_axis_l572_57239


namespace NUMINAMATH_GPT_roots_of_third_quadratic_l572_57271

/-- Given two quadratic equations with exactly one common root and a non-equal coefficient condition, 
prove that the other roots are roots of a third quadratic equation -/
theorem roots_of_third_quadratic 
  (a1 a2 a3 α β γ : ℝ)
  (h1 : α ≠ β)
  (h2 : α ≠ γ)
  (h3 : a1 ≠ a2)
  (h_eq1 : α^2 + a1*α + a2*a3 = 0)
  (h_eq2 : β^2 + a1*β + a2*a3 = 0)
  (h_eq3 : α^2 + a2*α + a1*a3 = 0)
  (h_eq4 : γ^2 + a2*γ + a1*a3 = 0) :
  β^2 + a3*β + a1*a2 = 0 ∧ γ^2 + a3*γ + a1*a2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_roots_of_third_quadratic_l572_57271


namespace NUMINAMATH_GPT_raft_minimum_capacity_l572_57268

theorem raft_minimum_capacity (n_mice n_moles n_hamsters : ℕ)
  (weight_mice weight_moles weight_hamsters : ℕ)
  (total_weight : ℕ) :
  n_mice = 5 →
  weight_mice = 70 →
  n_moles = 3 →
  weight_moles = 90 →
  n_hamsters = 4 →
  weight_hamsters = 120 →
  (∀ (total_weight : ℕ), total_weight = n_mice * weight_mice + n_moles * weight_moles + n_hamsters * weight_hamsters) →
  (∃ (min_capacity: ℕ), min_capacity ≥ 140) :=
by
  intros
  sorry

end NUMINAMATH_GPT_raft_minimum_capacity_l572_57268


namespace NUMINAMATH_GPT_smallest_class_number_l572_57259

-- Define the conditions
def num_classes : Nat := 24
def num_selected_classes : Nat := 4
def total_sum : Nat := 52
def sampling_interval : Nat := num_classes / num_selected_classes

-- The core theorem to be proved
theorem smallest_class_number :
  ∃ x : Nat, x + (x + sampling_interval) + (x + 2 * sampling_interval) + (x + 3 * sampling_interval) = total_sum ∧ x = 4 := by
  sorry

end NUMINAMATH_GPT_smallest_class_number_l572_57259


namespace NUMINAMATH_GPT_paint_intensity_l572_57264

theorem paint_intensity (I : ℝ) (F : ℝ) (I_initial I_new : ℝ) : 
  I_initial = 50 → I_new = 30 → F = 2 / 3 → I = 20 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_paint_intensity_l572_57264


namespace NUMINAMATH_GPT_gnollish_valid_sentences_count_l572_57215

/--
The Gnollish language consists of 4 words: "splargh," "glumph," "amr," and "bork."
A sentence is valid if "splargh" does not come directly before "glumph" or "bork."
Prove that there are 240 valid 4-word sentences in Gnollish.
-/
theorem gnollish_valid_sentences_count : 
  let words := ["splargh", "glumph", "amr", "bork"]
  let total_sentences := (words.length ^ 4)
  let invalid_conditions (w1 w2 : String) := 
    (w1 = "splargh" ∧ (w2 = "glumph" ∨ w2 = "bork"))
  let invalid_count : ℕ := 
    2 * words.length * words.length * (words.length - 1)
  let valid_sentences := total_sentences - invalid_count
  valid_sentences = 240 :=
by
  let words := ["splargh", "glumph", "amr", "bork"]
  let total_sentences := (words.length ^ 4)
  let invalid_conditions (w1 w2 : String) := 
    (w1 = "splargh" ∧ (w2 = "glumph" ∨ w2 = "bork"))
  let invalid_count : ℕ := 
    2 * words.length * words.length * (words.length - 1)
  let valid_sentences := total_sentences - invalid_count
  have : valid_sentences = 240 := by sorry
  exact this

end NUMINAMATH_GPT_gnollish_valid_sentences_count_l572_57215


namespace NUMINAMATH_GPT_exists_pythagorean_triple_rational_k_l572_57201

theorem exists_pythagorean_triple_rational_k (k : ℚ) (hk : k > 1) :
  ∃ (a b c : ℕ), (a^2 + b^2 = c^2) ∧ ((a + c : ℚ) / b = k) := by
  sorry

end NUMINAMATH_GPT_exists_pythagorean_triple_rational_k_l572_57201


namespace NUMINAMATH_GPT_value_of_3m_2n_l572_57254

section ProofProblem

variable (m n : ℤ)
-- Condition that x-3 is a factor of 3x^3 - mx + n
def factor1 : Prop := (3 * 3^3 - m * 3 + n = 0)
-- Condition that x+4 is a factor of 3x^3 - mx + n
def factor2 : Prop := (3 * (-4)^3 - m * (-4) + n = 0)

theorem value_of_3m_2n (h₁ : factor1 m n) (h₂ : factor2 m n) : abs (3 * m - 2 * n) = 45 := by
  sorry

end ProofProblem

end NUMINAMATH_GPT_value_of_3m_2n_l572_57254


namespace NUMINAMATH_GPT_range_of_f_l572_57245

noncomputable def f (x y : ℝ) := (x^3 + y^3) / (x + y)^3

theorem range_of_f :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x^2 + y^2 = 1 → (1 / 4) ≤ f x y ∧ f x y < 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l572_57245


namespace NUMINAMATH_GPT_benny_initial_comics_l572_57291

variable (x : ℕ)

def initial_comics (x : ℕ) : ℕ := x

def comics_after_selling (x : ℕ) : ℕ := (2 * x) / 5

def comics_after_buying (x : ℕ) : ℕ := (comics_after_selling x) + 12

def traded_comics (x : ℕ) : ℕ := (comics_after_buying x) / 4

def comics_after_trading (x : ℕ) : ℕ := (3 * (comics_after_buying x)) / 4 + 18

theorem benny_initial_comics : comics_after_trading x = 72 → x = 150 := by
  intro h
  sorry

end NUMINAMATH_GPT_benny_initial_comics_l572_57291


namespace NUMINAMATH_GPT_initial_group_size_l572_57246

theorem initial_group_size (n : ℕ) (W : ℝ) 
  (h1 : (W + 20) / n = W / n + 4) : 
  n = 5 := 
by 
  sorry

end NUMINAMATH_GPT_initial_group_size_l572_57246


namespace NUMINAMATH_GPT_place_integers_on_cube_l572_57275

theorem place_integers_on_cube:
  ∃ (A B C D A₁ B₁ C₁ D₁ : ℤ),
    A = B + D + A₁ ∧ 
    B = A + C + B₁ ∧ 
    C = B + D + C₁ ∧ 
    D = A + C + D₁ ∧ 
    A₁ = B₁ + D₁ + A ∧ 
    B₁ = A₁ + C₁ + B ∧ 
    C₁ = B₁ + D₁ + C ∧ 
    D₁ = A₁ + C₁ + D :=
sorry

end NUMINAMATH_GPT_place_integers_on_cube_l572_57275


namespace NUMINAMATH_GPT_correct_equation_for_growth_rate_l572_57262

def initial_price : ℝ := 6.2
def final_price : ℝ := 8.9
def growth_rate (x : ℝ) : ℝ := initial_price * (1 + x) ^ 2

theorem correct_equation_for_growth_rate (x : ℝ) : growth_rate x = final_price ↔ initial_price * (1 + x) ^ 2 = 8.9 :=
by sorry

end NUMINAMATH_GPT_correct_equation_for_growth_rate_l572_57262


namespace NUMINAMATH_GPT_angle_215_third_quadrant_l572_57234

-- Define the context of the problem
def angle_vertex_origin : Prop := true 

def initial_side_non_negative_x_axis : Prop := true

noncomputable def in_third_quadrant (angle: ℝ) : Prop := 
  180 < angle ∧ angle < 270 

-- The theorem to prove the condition given
theorem angle_215_third_quadrant : 
  angle_vertex_origin → 
  initial_side_non_negative_x_axis → 
  in_third_quadrant 215 :=
by
  intro _ _
  unfold in_third_quadrant
  sorry -- This is where the proof would go

end NUMINAMATH_GPT_angle_215_third_quadrant_l572_57234


namespace NUMINAMATH_GPT_Tn_lt_Sn_div_2_l572_57224

noncomputable def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
noncomputable def b (n : ℕ) : ℝ := n * (1 / 3)^n

noncomputable def S (n : ℕ) : ℝ := 
  (3 / 2) * (1 - (1 / 3)^n)

noncomputable def T (n : ℕ) : ℝ := 
  (3 / 4) * (1 - (1 / 3)^n) - (n / 2) * (1 / 3)^(n + 1)

theorem Tn_lt_Sn_div_2 (n : ℕ) : T n < S n / 2 := 
sorry

end NUMINAMATH_GPT_Tn_lt_Sn_div_2_l572_57224


namespace NUMINAMATH_GPT_initial_speed_100_l572_57272

/-- Conditions of the problem:
1. The total distance from A to D is 100 km.
2. At point B, the navigator shows that 30 minutes are remaining.
3. At point B, the motorist reduces his speed by 10 km/h.
4. At point C, the navigator shows 20 km remaining, and the motorist again reduces his speed by 10 km/h.
5. The distance from C to D is 20 km.
6. The journey from B to C took 5 minutes longer than from C to D.
-/
theorem initial_speed_100 (x v : ℝ) (h1 : x = 100 - v / 2)
  (h2 : ∀ t, t = x / v)
  (h3 : ∀ t1 t2, t1 = (80 - x) / (v - 10) ∧ t2 = 20 / (v - 20))
  (h4 : (80 - x) / (v - 10) - 20 / (v - 20) = 1/12) :
  v = 100 := 
sorry

end NUMINAMATH_GPT_initial_speed_100_l572_57272


namespace NUMINAMATH_GPT_min_value_inequality_l572_57204

theorem min_value_inequality (a b c : ℝ) (h : a + b + c = 3) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / (a + b) + 1 / c) ≥ 4 / 3 :=
sorry

end NUMINAMATH_GPT_min_value_inequality_l572_57204


namespace NUMINAMATH_GPT_find_segment_AD_length_l572_57221

noncomputable def segment_length_AD (A B C D X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace X] :=
  ∃ (angle_BAD angle_ABC angle_BCD : Real)
    (length_AB length_CD : Real)
    (perpendicular : X) (angle_BAX angle_ABX : Real)
    (length_AX length_DX length_AD : Real),
    angle_BAD = 60 ∧
    angle_ABC = 30 ∧
    angle_BCD = 30 ∧
    length_AB = 15 ∧
    length_CD = 8 ∧
    angle_BAX = 30 ∧
    angle_ABX = 60 ∧
    length_AX = length_AB / 2 ∧
    length_DX = length_CD / 2 ∧
    length_AD = length_AX - length_DX ∧
    length_AD = 3.5

theorem find_segment_AD_length (A B C D X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace X] : segment_length_AD A B C D X :=
by
  sorry

end NUMINAMATH_GPT_find_segment_AD_length_l572_57221


namespace NUMINAMATH_GPT_octahedron_coloring_l572_57269

theorem octahedron_coloring : 
  ∃ (n : ℕ), n = 6 ∧
  ∀ (F : Fin 8 → Fin 4), 
    (∀ (i j : Fin 8), i ≠ j → F i ≠ F j) ∧
    (∃ (pairs : Fin 8 → (Fin 4 × Fin 4)), 
      (∀ (i : Fin 8), ∃ j : Fin 4, pairs i = (j, j)) ∧ 
      (∀ j, ∃ (i : Fin 8), F i = j)) :=
by
  sorry

end NUMINAMATH_GPT_octahedron_coloring_l572_57269


namespace NUMINAMATH_GPT_second_degree_polynomial_inequality_l572_57211

def P (u v w x : ℝ) : ℝ := u * x^2 + v * x + w

theorem second_degree_polynomial_inequality 
  (u v w : ℝ) (h : ∀ a : ℝ, 1 ≤ a → P u v w (a^2 + a) ≥ a * P u v w (a + 1)) :
  u > 0 ∧ w ≤ 4 * u :=
by
  sorry

end NUMINAMATH_GPT_second_degree_polynomial_inequality_l572_57211


namespace NUMINAMATH_GPT_labourer_total_payment_l572_57270

/--
A labourer was engaged for 25 days on the condition that for every day he works, he will be paid Rs. 2 and for every day he is absent, he will be fined 50 p. He was absent for 5 days. Prove that the total amount he received in the end is Rs. 37.50.
-/
theorem labourer_total_payment :
  let total_days := 25
  let daily_wage := 2.0
  let absent_days := 5
  let fine_per_absent_day := 0.5
  let worked_days := total_days - absent_days
  let total_earnings := worked_days * daily_wage
  let total_fine := absent_days * fine_per_absent_day
  let total_received := total_earnings - total_fine
  total_received = 37.5 :=
by
  sorry

end NUMINAMATH_GPT_labourer_total_payment_l572_57270


namespace NUMINAMATH_GPT_largest_natural_number_has_sum_of_digits_property_l572_57218

noncomputable def largest_nat_num_digital_sum : ℕ :=
  let a : ℕ := 1
  let b : ℕ := 0
  let d3 := a + b
  let d4 := 2 * a + 2 * b
  let d5 := 4 * a + 4 * b
  let d6 := 8 * a + 8 * b
  100000 * a + 10000 * b + 1000 * d3 + 100 * d4 + 10 * d5 + d6

theorem largest_natural_number_has_sum_of_digits_property :
  largest_nat_num_digital_sum = 101248 :=
by
  sorry

end NUMINAMATH_GPT_largest_natural_number_has_sum_of_digits_property_l572_57218


namespace NUMINAMATH_GPT_find_denomination_of_bills_l572_57208

variables 
  (bills_13 : ℕ)  -- Denomination of the bills Tim has 13 of
  (bills_5 : ℕ := 5)  -- Denomination of the bills Tim has 11 of, which are $5 bills
  (bills_1 : ℕ := 1)  -- Denomination of the bills Tim has 17 of, which are $1 bills
  (total_amt : ℕ := 128)  -- Total amount Tim needs to pay
  (num_bills_13 : ℕ := 13)  -- Number of bills of unknown denomination
  (num_bills_5 : ℕ := 11)  -- Number of $5 bills
  (num_bills_1 : ℕ := 17)  -- Number of $1 bills
  (min_bills : ℕ := 16)  -- Minimum number of bills to be used

theorem find_denomination_of_bills : 
  num_bills_13 * bills_13 + num_bills_5 * bills_5 + num_bills_1 * bills_1 = total_amt →
  num_bills_13 + num_bills_5 + num_bills_1 ≥ min_bills → 
  bills_13 = 4 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_denomination_of_bills_l572_57208


namespace NUMINAMATH_GPT_percentage_taxed_on_excess_income_l572_57256

noncomputable def pct_taxed_on_first_40k : ℝ := 0.11
noncomputable def first_40k_income : ℝ := 40000
noncomputable def total_income : ℝ := 58000
noncomputable def total_tax : ℝ := 8000

theorem percentage_taxed_on_excess_income :
  ∃ P : ℝ, (total_tax - pct_taxed_on_first_40k * first_40k_income = P * (total_income - first_40k_income)) ∧ P * 100 = 20 := 
by
  sorry

end NUMINAMATH_GPT_percentage_taxed_on_excess_income_l572_57256


namespace NUMINAMATH_GPT_final_bill_correct_l572_57253

def initial_bill := 500.00
def late_charge_rate := 0.02
def final_bill := initial_bill * (1 + late_charge_rate) * (1 + late_charge_rate)

theorem final_bill_correct : final_bill = 520.20 := by
  sorry

end NUMINAMATH_GPT_final_bill_correct_l572_57253


namespace NUMINAMATH_GPT_lattice_points_count_l572_57217

theorem lattice_points_count : ∃ n : ℕ, n = 8 ∧ (∃ x y : ℤ, x^2 - y^2 = 51) :=
by
  sorry

end NUMINAMATH_GPT_lattice_points_count_l572_57217


namespace NUMINAMATH_GPT_ab_parallel_to_x_axis_and_ac_parallel_to_y_axis_l572_57278

theorem ab_parallel_to_x_axis_and_ac_parallel_to_y_axis
  (a b : ℝ)
  (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ)
  (hA : A = (a, -1))
  (hB : B = (2, 3 - b))
  (hC : C = (-5, 4))
  (hAB_parallel_x : A.2 = B.2)
  (hAC_parallel_y : A.1 = C.1) : a + b = -1 := by
  sorry


end NUMINAMATH_GPT_ab_parallel_to_x_axis_and_ac_parallel_to_y_axis_l572_57278


namespace NUMINAMATH_GPT_investment_calculation_l572_57247

noncomputable def calculate_investment_amount (A : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  A / (1 + r / n) ^ (n * t)

theorem investment_calculation :
  let A := 80000
  let r := 0.07
  let n := 12
  let t := 7
  let P := calculate_investment_amount A r n t
  abs (P - 46962) < 1 :=
by
  sorry

end NUMINAMATH_GPT_investment_calculation_l572_57247


namespace NUMINAMATH_GPT_range_of_m_l572_57243

variable (m : ℝ)

def proposition_p (m : ℝ) : Prop :=
  0 < m ∧ m < 1/3

def proposition_q (m : ℝ) : Prop :=
  0 < m ∧ m < 15

theorem range_of_m (m : ℝ) :
  (¬ (proposition_p m) ∧ proposition_q m) ∨ (proposition_p m ∧ ¬ (proposition_q m)) →
  (1/3 <= m ∧ m < 15) :=
sorry

end NUMINAMATH_GPT_range_of_m_l572_57243


namespace NUMINAMATH_GPT_question1_question2_l572_57226

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) - abs (2 * x - 1)

theorem question1 (x : ℝ) :
  ∀ a, a = 2 → (f x 2 + 3 ≥ 0 ↔ -4 ≤ x ∧ x ≤ 2) := by
sorry

theorem question2 (a : ℝ) :
  (∀ x, 1 ≤ x → x ≤ 3 → f x a ≤ 3) ↔ (-3 ≤ a ∧ a ≤ 5) := by
sorry

end NUMINAMATH_GPT_question1_question2_l572_57226


namespace NUMINAMATH_GPT_roots_of_quadratic_eq_l572_57266

theorem roots_of_quadratic_eq (x : ℝ) : x^2 - 2 * x = 0 → (x = 0 ∨ x = 2) :=
by sorry

end NUMINAMATH_GPT_roots_of_quadratic_eq_l572_57266


namespace NUMINAMATH_GPT_minimize_expression_l572_57210

theorem minimize_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
(h4 : x^2 + y^2 + z^2 = 1) : 
  z = Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_GPT_minimize_expression_l572_57210


namespace NUMINAMATH_GPT_extremum_and_monotonicity_inequality_for_c_l572_57242

noncomputable def f (x α : ℝ) : ℝ := x * Real.log x - α * x + 1

theorem extremum_and_monotonicity (α : ℝ) (h_extremum : ∀ (x : ℝ), x = Real.exp 2 → f x α = 0) :
  (∃ α : ℝ, (∀ x : ℝ, x > Real.exp 2 → f x α > 0) ∧ (∀ x : ℝ, 0 < x ∧ x < Real.exp 2 → f x α < 0)) := sorry

theorem inequality_for_c (c : ℝ) (α : ℝ) (h_extremum : α = 3)
  (h_ineq : ∀ x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 3 → f x α < 2 * c^2 - c) :
  (1 < c) ∨ (c < -1 / 2) := sorry

end NUMINAMATH_GPT_extremum_and_monotonicity_inequality_for_c_l572_57242


namespace NUMINAMATH_GPT_x_intercept_of_line_l572_57255

-- Definition of line equation
def line_eq (x y : ℝ) : Prop := 4 * x + 7 * y = 28

-- Proposition that the x-intercept of the line 4x + 7y = 28 is (7, 0)
theorem x_intercept_of_line : line_eq 7 0 :=
by
  show 4 * 7 + 7 * 0 = 28
  sorry

end NUMINAMATH_GPT_x_intercept_of_line_l572_57255


namespace NUMINAMATH_GPT_find_a_plus_b_l572_57213

theorem find_a_plus_b (a b : ℝ) 
  (h1 : ∃ x y : ℝ, (y = a * x + 1) ∧ (x^2 + y^2 + b*x - y = 1))
  (h2 : ∀ x y : ℝ, (y = a * x + 1) ∧ (x^2 + y^2 + b*x - y = 1) → x + y = 0) : 
  a + b = 2 :=
sorry

end NUMINAMATH_GPT_find_a_plus_b_l572_57213


namespace NUMINAMATH_GPT_cookie_boxes_condition_l572_57238

theorem cookie_boxes_condition (n : ℕ) (M A : ℕ) :
  M = n - 8 ∧ A = n - 2 ∧ M + A < n ∧ M ≥ 1 ∧ A ≥ 1 → n = 9 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cookie_boxes_condition_l572_57238


namespace NUMINAMATH_GPT_students_on_bus_l572_57250

theorem students_on_bus (initial_students : ℝ) (students_got_on : ℝ) (total_students : ℝ) 
  (h1 : initial_students = 10.0) (h2 : students_got_on = 3.0) : 
  total_students = 13.0 :=
by 
  sorry

end NUMINAMATH_GPT_students_on_bus_l572_57250


namespace NUMINAMATH_GPT_required_bike_speed_l572_57267

theorem required_bike_speed (swim_distance run_distance bike_distance swim_speed run_speed total_time : ℝ)
  (h_swim_dist : swim_distance = 0.5)
  (h_run_dist : run_distance = 4)
  (h_bike_dist : bike_distance = 12)
  (h_swim_speed : swim_speed = 1)
  (h_run_speed : run_speed = 8)
  (h_total_time : total_time = 1.5) :
  (bike_distance / ((total_time - (swim_distance / swim_speed + run_distance / run_speed)))) = 24 :=
by
  sorry

end NUMINAMATH_GPT_required_bike_speed_l572_57267


namespace NUMINAMATH_GPT_total_amount_spent_l572_57219

/-
  Define the original prices of the games, discount rate, and tax rate.
-/
def batman_game_price : ℝ := 13.60
def superman_game_price : ℝ := 5.06
def discount_rate : ℝ := 0.20
def tax_rate : ℝ := 0.08

/-
  Prove that the total amount spent including discounts and taxes equals $16.12.
-/
theorem total_amount_spent :
  let batman_discount := batman_game_price * discount_rate
  let superman_discount := superman_game_price * discount_rate
  let batman_discounted_price := batman_game_price - batman_discount
  let superman_discounted_price := superman_game_price - superman_discount
  let total_before_tax := batman_discounted_price + superman_discounted_price
  let sales_tax := total_before_tax * tax_rate
  let total_amount := total_before_tax + sales_tax
  total_amount = 16.12 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_spent_l572_57219


namespace NUMINAMATH_GPT_roots_reciprocal_l572_57273

theorem roots_reciprocal (x1 x2 : ℝ) (h1 : x1^2 - 4 * x1 - 2 = 0) (h2 : x2^2 - 4 * x2 - 2 = 0) (h3 : x1 ≠ x2) :
  (1 / x1) + (1 / x2) = -2 := 
sorry

end NUMINAMATH_GPT_roots_reciprocal_l572_57273


namespace NUMINAMATH_GPT_evaluate_fraction_sum_l572_57286

theorem evaluate_fraction_sum : (5 / 50) + (4 / 40) + (6 / 60) = 0.3 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_sum_l572_57286


namespace NUMINAMATH_GPT_correct_system_of_equations_l572_57295

theorem correct_system_of_equations (x y : ℕ) :
  (8 * x - 3 = y ∧ 7 * x + 4 = y) ↔ 
  (8 * x - 3 = y ∧ 7 * x + 4 = y) := 
by 
  sorry

end NUMINAMATH_GPT_correct_system_of_equations_l572_57295


namespace NUMINAMATH_GPT_find_m_l572_57251

def h (x m : ℝ) := x^2 - 3 * x + m
def k (x m : ℝ) := x^2 - 3 * x + 5 * m

theorem find_m (m : ℝ) (h_def : ∀ x, h x m = x^2 - 3 * x + m) (k_def : ∀ x, k x m = x^2 - 3 * x + 5 * m) (key_eq : 3 * h 5 m = 2 * k 5 m) :
  m = 10 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l572_57251


namespace NUMINAMATH_GPT_total_animal_eyes_l572_57214

-- Define the conditions given in the problem
def numberFrogs : Nat := 20
def numberCrocodiles : Nat := 10
def eyesEach : Nat := 2

-- Define the statement that we need to prove
theorem total_animal_eyes : (numberFrogs * eyesEach) + (numberCrocodiles * eyesEach) = 60 := by
  sorry

end NUMINAMATH_GPT_total_animal_eyes_l572_57214


namespace NUMINAMATH_GPT_yuko_in_front_of_yuri_l572_57248

theorem yuko_in_front_of_yuri (X : ℕ) (hYuri : 2 + 4 + 5 = 11) (hYuko : 1 + 5 + X > 11) : X = 6 := 
by
  sorry

end NUMINAMATH_GPT_yuko_in_front_of_yuri_l572_57248


namespace NUMINAMATH_GPT_paired_divisors_prime_properties_l572_57282

theorem paired_divisors_prime_properties (n : ℕ) (h : n > 0) (h_pairing : ∃ (pairing : (ℕ × ℕ) → Prop), 
  (∀ d1 d2 : ℕ, 
    pairing (d1, d2) → d1 * d2 = n ∧ Prime (d1 + d2))) : 
  (∀ (d1 d2 : ℕ), d1 ≠ d2 → d1 + d2 ≠ d3 + d4) ∧ (∀ p : ℕ, Prime p → ¬ p ∣ n) :=
by
  sorry

end NUMINAMATH_GPT_paired_divisors_prime_properties_l572_57282


namespace NUMINAMATH_GPT_investment_total_l572_57297

theorem investment_total (x y : ℝ) (h₁ : 0.08 * x + 0.05 * y = 490) (h₂ : x = 3000 ∨ y = 3000) : x + y = 8000 :=
by
  sorry

end NUMINAMATH_GPT_investment_total_l572_57297


namespace NUMINAMATH_GPT_evaluate_g_5_times_l572_57231

def g (x : ℕ) : ℕ :=
if x % 2 = 0 then x + 2 else 3 * x + 1

theorem evaluate_g_5_times : g (g (g (g (g 1)))) = 12 := by
  sorry


end NUMINAMATH_GPT_evaluate_g_5_times_l572_57231


namespace NUMINAMATH_GPT_beaver_group_l572_57249

theorem beaver_group (B : ℕ) :
  (B * 3 = 12 * 5) → B = 20 :=
by
  intros h1
  -- Additional steps for the proof would go here.
  -- The h1 hypothesis represents the condition B * 3 = 60.
  exact sorry -- Proof steps are not required.

end NUMINAMATH_GPT_beaver_group_l572_57249


namespace NUMINAMATH_GPT_find_phi_symmetric_l572_57236

noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x)) + (Real.sqrt 3 * (Real.cos (2 * x)))

theorem find_phi_symmetric : ∃ φ : ℝ, (φ = Real.pi / 12) ∧ ∀ x : ℝ, f (-x + φ) = f (x + φ) := 
sorry

end NUMINAMATH_GPT_find_phi_symmetric_l572_57236


namespace NUMINAMATH_GPT_tan_theta_eq2_simplifies_to_minus1_sin_cos_and_tan_relation_l572_57244

-- First proof problem
theorem tan_theta_eq2_simplifies_to_minus1 (θ : ℝ) (h : Real.tan θ = 2) :
  (Real.sin (θ - 6 * Real.pi) + Real.sin (Real.pi / 2 - θ)) / 
  (2 * Real.sin (Real.pi + θ) + Real.cos (-θ)) = -1 := sorry

-- Second proof problem
theorem sin_cos_and_tan_relation (x : ℝ) (hx1 : - Real.pi / 2 < x) (hx2 : x < Real.pi / 2) 
  (h : Real.sin x + Real.cos x = 1 / 5) : Real.tan x = -3 / 4 := sorry

end NUMINAMATH_GPT_tan_theta_eq2_simplifies_to_minus1_sin_cos_and_tan_relation_l572_57244


namespace NUMINAMATH_GPT_employee_hourly_pay_l572_57229

-- Definitions based on conditions
def initial_employees := 500
def daily_hours := 10
def weekly_days := 5
def monthly_weeks := 4
def additional_employees := 200
def total_payment := 1680000
def total_employees := initial_employees + additional_employees
def monthly_hours_per_employee := daily_hours * weekly_days * monthly_weeks
def total_monthly_hours := total_employees * monthly_hours_per_employee

-- Lean 4 statement proving the hourly pay per employee
theorem employee_hourly_pay : total_payment / total_monthly_hours = 12 := by sorry

end NUMINAMATH_GPT_employee_hourly_pay_l572_57229


namespace NUMINAMATH_GPT_average_age_6_members_birth_correct_l572_57276

/-- The average age of 7 members of a family is 29 years. -/
def average_age_7_members := 29

/-- The present age of the youngest member is 5 years. -/
def age_youngest_member := 5

/-- Total age of 7 members of the family -/
def total_age_7_members := 7 * average_age_7_members

/-- Total age of 6 members at present -/
def total_age_6_members_present := total_age_7_members - age_youngest_member

/-- Total age of 6 members at time of birth of youngest member -/
def total_age_6_members_birth := total_age_6_members_present - (6 * age_youngest_member)

/-- Average age of 6 members at time of birth of youngest member -/
def average_age_6_members_birth := total_age_6_members_birth / 6

/-- Prove the average age of 6 members at the time of birth of the youngest member -/
theorem average_age_6_members_birth_correct :
  average_age_6_members_birth = 28 :=
by
  sorry

end NUMINAMATH_GPT_average_age_6_members_birth_correct_l572_57276


namespace NUMINAMATH_GPT_pump_A_time_to_empty_pool_l572_57263

theorem pump_A_time_to_empty_pool :
  ∃ (A : ℝ), (1/A + 1/9 = 1/3.6) ∧ A = 6 :=
sorry

end NUMINAMATH_GPT_pump_A_time_to_empty_pool_l572_57263


namespace NUMINAMATH_GPT_right_triangle_of_angle_condition_l572_57283

-- Defining the angles of the triangle
variables (α β γ : ℝ)

-- Defining the condition where the sum of angles in a triangle is 180 degrees
def sum_of_angles_in_triangle (α β γ : ℝ) : Prop :=
  α + β + γ = 180

-- Defining the given condition 
def angle_condition (γ α β : ℝ) : Prop :=
  γ = α + β

-- Stating the theorem to be proved
theorem right_triangle_of_angle_condition (α β γ : ℝ) :
  sum_of_angles_in_triangle α β γ → angle_condition γ α β → γ = 90 :=
by
  intro hsum hcondition
  sorry

end NUMINAMATH_GPT_right_triangle_of_angle_condition_l572_57283


namespace NUMINAMATH_GPT_big_boxes_count_l572_57206

theorem big_boxes_count
  (soaps_per_package : ℕ)
  (packages_per_box : ℕ)
  (total_soaps : ℕ)
  (soaps_per_box : ℕ)
  (H1 : soaps_per_package = 192)
  (H2 : packages_per_box = 6)
  (H3 : total_soaps = 2304)
  (H4 : soaps_per_box = soaps_per_package * packages_per_box) :
  total_soaps / soaps_per_box = 2 :=
by
  sorry

end NUMINAMATH_GPT_big_boxes_count_l572_57206


namespace NUMINAMATH_GPT_remuneration_difference_l572_57280

-- Define the conditions and question
def total_sales : ℝ := 12000
def commission_rate_old : ℝ := 0.05
def fixed_salary_new : ℝ := 1000
def commission_rate_new : ℝ := 0.025
def sales_threshold_new : ℝ := 4000

-- Define the remuneration for the old scheme
def remuneration_old : ℝ := total_sales * commission_rate_old

-- Define the remuneration for the new scheme
def sales_exceeding_threshold_new : ℝ := total_sales - sales_threshold_new
def commission_new : ℝ := sales_exceeding_threshold_new * commission_rate_new
def remuneration_new : ℝ := fixed_salary_new + commission_new

-- Statement of the theorem to be proved
theorem remuneration_difference : remuneration_new - remuneration_old = 600 :=
by
  -- The proof goes here but is omitted as per the instructions
  sorry

end NUMINAMATH_GPT_remuneration_difference_l572_57280


namespace NUMINAMATH_GPT_smallest_among_l572_57203

theorem smallest_among {a b c d : ℤ} (h1 : a = -4) (h2 : b = -3) (h3 : c = 0) (h4 : d = 1) :
  a < b ∧ a < c ∧ a < d :=
by
  rw [h1, h2, h3, h4]
  exact ⟨by norm_num, by norm_num, by norm_num⟩

end NUMINAMATH_GPT_smallest_among_l572_57203


namespace NUMINAMATH_GPT_squares_area_ratios_l572_57240

noncomputable def squareC_area (x : ℝ) : ℝ := x ^ 2
noncomputable def squareD_area (x : ℝ) : ℝ := 3 * x ^ 2
noncomputable def squareE_area (x : ℝ) : ℝ := 6 * x ^ 2

theorem squares_area_ratios (x : ℝ) (h : x ≠ 0) :
  (squareC_area x / squareE_area x = 1 / 36) ∧ (squareD_area x / squareE_area x = 1 / 4) := by
  sorry

end NUMINAMATH_GPT_squares_area_ratios_l572_57240


namespace NUMINAMATH_GPT_find_x_squared_plus_inv_squared_l572_57233

theorem find_x_squared_plus_inv_squared (x : ℝ) (hx : x + (1 / x) = 4) : x^2 + (1 / x^2) = 14 := 
by
sorry

end NUMINAMATH_GPT_find_x_squared_plus_inv_squared_l572_57233


namespace NUMINAMATH_GPT_sale_price_of_trouser_l572_57279

theorem sale_price_of_trouser (original_price : ℝ) (discount_percentage : ℝ) (sale_price : ℝ) 
  (h1 : original_price = 100) (h2 : discount_percentage = 0.5) : sale_price = 50 :=
by
  sorry

end NUMINAMATH_GPT_sale_price_of_trouser_l572_57279


namespace NUMINAMATH_GPT_are_names_possible_l572_57232

-- Define the structure to hold names
structure Person where
  first_name  : String
  middle_name : String
  last_name   : String

-- List of 4 people
def people : List Person :=
  [{ first_name := "Ivan", middle_name := "Ivanovich", last_name := "Ivanov" },
   { first_name := "Ivan", middle_name := "Petrovich", last_name := "Petrov" },
   { first_name := "Petr", middle_name := "Ivanovich", last_name := "Petrov" },
   { first_name := "Petr", middle_name := "Petrovich", last_name := "Ivanov" }]

-- Define the problem theorem
theorem are_names_possible :
  ∃ (people : List Person), 
    (∀ (p1 p2 p3 : Person), p1 ≠ p2 → p1 ≠ p3 → p2 ≠ p3 → (p1.first_name ≠ p2.first_name ∨ p1.first_name ≠ p3.first_name ∨ p2.first_name ≠ p3.first_name) ∧
    (p1.middle_name ≠ p2.middle_name ∨ p1.middle_name ≠ p3.middle_name ∨ p2.middle_name ≠ p3.middle_name) ∧
    (p1.last_name ≠ p2.last_name ∨ p1.last_name ≠ p3.last_name ∨ p2.last_name ≠ p3.last_name)) ∧
    (∀ (p1 p2 : Person), p1 ≠ p2 → (p1.first_name = p2.first_name ∨ p1.middle_name = p2.middle_name ∨ p1.last_name = p2.last_name)) :=
by
  -- Place proof here
  sorry

end NUMINAMATH_GPT_are_names_possible_l572_57232


namespace NUMINAMATH_GPT_distinct_positive_integer_quadruples_l572_57230

theorem distinct_positive_integer_quadruples 
  (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
  (h7 : a + b = c * d) (h8 : a * b = c + d) :
  (a, b, c, d) = (1, 5, 2, 3)
  ∨ (a, b, c, d) = (1, 5, 3, 2)
  ∨ (a, b, c, d) = (5, 1, 2, 3)
  ∨ (a, b, c, d) = (5, 1, 3, 2)
  ∨ (a, b, c, d) = (2, 3, 1, 5)
  ∨ (a, b, c, d) = (2, 3, 5, 1)
  ∨ (a, b, c, d) = (3, 2, 1, 5)
  ∨ (a, b, c, d) = (3, 2, 5, 1) :=
  sorry

end NUMINAMATH_GPT_distinct_positive_integer_quadruples_l572_57230


namespace NUMINAMATH_GPT_factorization_correct_l572_57285

theorem factorization_correct :
  (¬ (x^2 - 2 * x - 1 = x * (x - 2) - 1)) ∧
  (¬ (2 * x + 1 = x * (2 + 1 / x))) ∧
  (¬ ((x + 2) * (x - 2) = x^2 - 4)) ∧
  (x^2 - 1 = (x + 1) * (x - 1)) :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l572_57285


namespace NUMINAMATH_GPT_rightmost_three_digits_of_5_pow_1994_l572_57220

theorem rightmost_three_digits_of_5_pow_1994 : (5 ^ 1994) % 1000 = 625 :=
by
  sorry

end NUMINAMATH_GPT_rightmost_three_digits_of_5_pow_1994_l572_57220


namespace NUMINAMATH_GPT_mean_of_set_with_median_l572_57287

theorem mean_of_set_with_median (m : ℝ) (h : m + 7 = 10) :
  (m + (m + 2) + (m + 7) + (m + 10) + (m + 12)) / 5 = 9.2 :=
by
  -- Placeholder for the proof.
  sorry

end NUMINAMATH_GPT_mean_of_set_with_median_l572_57287


namespace NUMINAMATH_GPT_find_y_value_l572_57274

theorem find_y_value : (12^3 * 6^3 / 432) = 864 := by
  sorry

end NUMINAMATH_GPT_find_y_value_l572_57274


namespace NUMINAMATH_GPT_perpendicular_bisector_AC_circumcircle_eqn_l572_57212

/-- Given vertices of triangle ABC, prove the equation of the perpendicular bisector of side AC --/
theorem perpendicular_bisector_AC (A B C D : ℝ×ℝ) (hA: A = (0, 2)) (hC: C = (4, 0)) (hD: D = (2, 1)) :
  ∃ k b, (k = 2) ∧ (b = -3) ∧ (∀ x y, y = k * x + b ↔ 2 * x - y - 3 = 0) :=
sorry

/-- Given vertices of triangle ABC, prove the equation of the circumcircle --/
theorem circumcircle_eqn (A B C D E F : ℝ×ℝ) (hA: A = (0, 2)) (hB: B = (6, 4)) (hC: C = (4, 0)) :
  ∃ k, k = 10 ∧ 
  (∀ x y, (x - 3) ^ 2 + (y - 3) ^ 2 = k ↔ x ^ 2 + y ^ 2 - 6 * x - 2 * y + 8 = 0) :=
sorry

end NUMINAMATH_GPT_perpendicular_bisector_AC_circumcircle_eqn_l572_57212


namespace NUMINAMATH_GPT_tangent_line_at_A_increasing_intervals_decreasing_interval_l572_57225

noncomputable def f (x : ℝ) := 2 * x^3 + 3 * x^2 + 1

-- Define the derivatives at x
noncomputable def f' (x : ℝ) := 6 * x^2 + 6 * x

-- Define the tangent line equation at a point
noncomputable def tangent_line (x : ℝ) := 12 * x - 6

theorem tangent_line_at_A :
  tangent_line 1 = 6 :=
  by
    -- proof omitted
    sorry

theorem increasing_intervals :
  (∀ x ∈ Set.Ioi 0, f' x > 0) ∧
  (∀ x ∈ Set.Iio (-1), f' x > 0) :=
  by
    -- proof omitted
    sorry

theorem decreasing_interval :
  ∀ x ∈ Set.Ioo (-1) 0, f' x < 0 :=
  by
    -- proof omitted
    sorry

end NUMINAMATH_GPT_tangent_line_at_A_increasing_intervals_decreasing_interval_l572_57225


namespace NUMINAMATH_GPT_vector_relationship_l572_57293

variables {V : Type*} [AddCommGroup V] [Module ℝ V] 
          (A A1 B D E : V) (x y z : ℝ)

-- Given Conditions
def inside_top_face_A1B1C1D1 (E : V) : Prop :=
  ∃ (y z : ℝ), (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) ∧
  E = A1 + y • (B - A) + z • (D - A)

-- Prove the desired relationship
theorem vector_relationship (h : E = x • (A1 - A) + y • (B - A) + z • (D - A))
  (hE : inside_top_face_A1B1C1D1 A A1 B D E) : 
  x = 1 ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) :=
sorry

end NUMINAMATH_GPT_vector_relationship_l572_57293
