import Mathlib

namespace triangle_cannot_have_two_right_angles_l1774_177418

theorem triangle_cannot_have_two_right_angles (A B C : ℝ) (h : A + B + C = 180) : 
  ¬ (A = 90 ∧ B = 90) :=
by {
  sorry
}

end triangle_cannot_have_two_right_angles_l1774_177418


namespace ages_total_l1774_177451

theorem ages_total (P Q : ℕ) (h1 : P - 8 = (1 / 2) * (Q - 8)) (h2 : P / Q = 3 / 4) : P + Q = 28 :=
by
  sorry

end ages_total_l1774_177451


namespace salary_of_A_l1774_177492

theorem salary_of_A (A B : ℝ) (h1 : A + B = 7000) (h2 : 0.05 * A = 0.15 * B) : A = 5250 := 
by 
  sorry

end salary_of_A_l1774_177492


namespace compare_powers_l1774_177407

theorem compare_powers (a b c : ℝ) (h1 : a = 2^555) (h2 : b = 3^444) (h3 : c = 6^222) : a < c ∧ c < b :=
by
  sorry

end compare_powers_l1774_177407


namespace part_a_part_b_part_c_l1774_177466

-- Given conditions and questions
variable (x y : ℝ)
variable (h : (x - y)^2 - 2 * (x + y) + 1 = 0)

-- Part (a): Prove neither x nor y can be negative
theorem part_a (h : (x - y)^2 - 2 * (x + y) + 1 = 0) : x ≥ 0 ∧ y ≥ 0 := 
sorry

-- Part (b): Prove if x > 1 and y < x, then sqrt{x} - sqrt{y} = 1
theorem part_b (h : (x - y)^2 - 2 * (x + y) + 1 = 0) (hx : x > 1) (hy : y < x) : 
  Real.sqrt x - Real.sqrt y = 1 := 
sorry

-- Part (c): Prove if x < 1 and y < 1, then sqrt{x} + sqrt{y} = 1
theorem part_c (h : (x - y)^2 - 2 * (x + y) + 1 = 0) (hx : x < 1) (hy : y < 1) : 
  Real.sqrt x + Real.sqrt y = 1 := 
sorry

end part_a_part_b_part_c_l1774_177466


namespace selling_price_correct_l1774_177438

def initial_cost : ℕ := 800
def repair_cost : ℕ := 200
def gain_percent : ℕ := 40
def total_cost := initial_cost + repair_cost
def gain := (gain_percent * total_cost) / 100
def selling_price := total_cost + gain

theorem selling_price_correct : selling_price = 1400 := 
by
  sorry

end selling_price_correct_l1774_177438


namespace mrs_doe_inheritance_l1774_177433

noncomputable def calculateInheritance (totalTaxes : ℝ) : ℝ :=
  totalTaxes / 0.3625

theorem mrs_doe_inheritance (h : 0.3625 * calculateInheritance 15000 = 15000) :
  calculateInheritance 15000 = 41379 :=
by
  unfold calculateInheritance
  field_simp
  norm_cast
  sorry

end mrs_doe_inheritance_l1774_177433


namespace largest_value_l1774_177428

def value (word : List Char) : Nat :=
  word.foldr (fun c acc =>
    acc + match c with
      | 'A' => 1
      | 'B' => 2
      | 'C' => 3
      | 'D' => 4
      | 'E' => 5
      | _ => 0
    ) 0

theorem largest_value :
  value ['B', 'E', 'E'] > value ['D', 'A', 'D'] ∧
  value ['B', 'E', 'E'] > value ['B', 'A', 'D'] ∧
  value ['B', 'E', 'E'] > value ['C', 'A', 'B'] ∧
  value ['B', 'E', 'E'] > value ['B', 'E', 'D'] :=
by sorry

end largest_value_l1774_177428


namespace car_speed_l1774_177496

theorem car_speed (v : ℝ) (h : (1 / v) = (1 / 100 + 2 / 3600)) : v = 3600 / 38 := 
by
  sorry

end car_speed_l1774_177496


namespace find_m_l1774_177454

noncomputable def g (n : ℤ) : ℤ :=
if n % 2 ≠ 0 then 2 * n + 3
else if n % 3 = 0 then n / 3
else n - 1

theorem find_m :
  ∃ m : ℤ, m % 2 ≠ 0 ∧ g (g (g m)) = 36 ∧ m = 54 :=
by
  sorry

end find_m_l1774_177454


namespace compare_logarithmic_values_l1774_177459

theorem compare_logarithmic_values :
  let a := Real.log 3.4 / Real.log 2
  let b := Real.log 3.6 / Real.log 4
  let c := Real.log 0.3 / Real.log 3
  c < b ∧ b < a :=
by
  sorry

end compare_logarithmic_values_l1774_177459


namespace find_g7_l1774_177443

noncomputable def g (x : ℝ) (a b c d : ℝ) : ℝ := a * x ^ 7 + b * x ^ 3 + d * x ^ 2 + c * x - 8

theorem find_g7 (a b c d : ℝ) (h : g (-7) a b c d = 3) (h_d : d = 0) : g 7 a b c d = -19 :=
by
  simp [g, h, h_d]
  sorry

end find_g7_l1774_177443


namespace determine_signs_l1774_177421

theorem determine_signs (a b c : ℝ) (h1 : a != 0 ∧ b != 0 ∧ c == 0)
  (h2 : a > 0 ∨ (b + c) > 0) : a > 0 ∧ b < 0 ∧ c = 0 :=
by
  sorry

end determine_signs_l1774_177421


namespace complex_number_quadrant_l1774_177490

def i := Complex.I
def z := i * (1 + i)

theorem complex_number_quadrant 
  : z.re < 0 ∧ z.im > 0 := 
by
  sorry

end complex_number_quadrant_l1774_177490


namespace square_simplify_l1774_177469

   variable (y : ℝ)

   theorem square_simplify :
     (7 - Real.sqrt (y^2 - 49)) ^ 2 = y^2 - 14 * Real.sqrt (y^2 - 49) :=
   sorry
   
end square_simplify_l1774_177469


namespace prime_square_plus_two_is_prime_iff_l1774_177417

theorem prime_square_plus_two_is_prime_iff (p : ℕ) (hp : Prime p) : Prime (p^2 + 2) ↔ p = 3 :=
sorry

end prime_square_plus_two_is_prime_iff_l1774_177417


namespace min_area_rectangle_l1774_177462

theorem min_area_rectangle (P : ℕ) (hP : P = 60) :
  ∃ (l w : ℕ), 2 * l + 2 * w = P ∧ l * w = 29 :=
by
  sorry

end min_area_rectangle_l1774_177462


namespace molecular_weight_l1774_177420

theorem molecular_weight (w8 : ℝ) (n : ℝ) (w1 : ℝ) (h1 : w8 = 2376) (h2 : n = 8) : w1 = 297 :=
by
  sorry

end molecular_weight_l1774_177420


namespace polynomial_roots_distinct_and_expression_is_integer_l1774_177452

-- Defining the conditions and the main theorem
theorem polynomial_roots_distinct_and_expression_is_integer (a b c : ℂ) :
  (a^3 - a^2 - a - 1 = 0) → (b^3 - b^2 - b - 1 = 0) → (c^3 - c^2 - c - 1 = 0) → 
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  ∃ k : ℤ, ((a^(1982) - b^(1982)) / (a - b) + (b^(1982) - c^(1982)) / (b - c) + (c^(1982) - a^(1982)) / (c - a) = k) :=
by
  intros h1 h2 h3
  -- Proof omitted
  sorry

end polynomial_roots_distinct_and_expression_is_integer_l1774_177452


namespace initial_number_of_men_l1774_177424

theorem initial_number_of_men (M : ℕ) 
  (h1 : M * 8 * 40 = (M + 30) * 6 * 50) 
  : M = 450 :=
by 
  sorry

end initial_number_of_men_l1774_177424


namespace evaluate_g_at_8_l1774_177476

def g (n : ℕ) : ℕ := n^2 - 3 * n + 29

theorem evaluate_g_at_8 : g 8 = 69 := by
  unfold g
  calc
    8^2 - 3 * 8 + 29 = 64 - 24 + 29 := by simp
                      _ = 69 := by norm_num

end evaluate_g_at_8_l1774_177476


namespace shifted_parabola_eq_l1774_177431

-- Definitions
def original_parabola (x y : ℝ) : Prop := y = 3 * x^2

def shifted_origin (x' y' x y : ℝ) : Prop :=
  (x' = x + 1) ∧ (y' = y + 1)

-- Target statement
theorem shifted_parabola_eq : ∀ (x y x' y' : ℝ),
  original_parabola x y →
  shifted_origin x' y' x y →
  y' = 3*(x' - 1)*(x' - 1) + 1 → 
  y = 3*(x + 1)*(x + 1) - 1 :=
by
  intros x y x' y' h_orig h_shifted h_new_eq
  sorry

end shifted_parabola_eq_l1774_177431


namespace multiples_of_4_between_50_and_300_l1774_177445

theorem multiples_of_4_between_50_and_300 : 
  (∃ n : ℕ, 50 < n ∧ n < 300 ∧ n % 4 = 0) ∧ 
  (∃ k : ℕ, k = 62) :=
by
  sorry

end multiples_of_4_between_50_and_300_l1774_177445


namespace tan_sum_identity_sin_2alpha_l1774_177471

theorem tan_sum_identity_sin_2alpha (α : ℝ) (h : Real.tan (π/4 + α) = 2) : Real.sin (2*α) = 3/5 :=
by
  sorry

end tan_sum_identity_sin_2alpha_l1774_177471


namespace sin_cos_identity_l1774_177444

variable (α : Real)

theorem sin_cos_identity (h : Real.sin α - Real.cos α = -5/4) : Real.sin α * Real.cos α = -9/32 :=
by
  sorry

end sin_cos_identity_l1774_177444


namespace find_y_coordinate_l1774_177429

noncomputable def y_coordinate_of_point_on_line : ℝ :=
  let x1 := 10
  let y1 := 3
  let x2 := 4
  let y2 := 0
  let x := -2
  let m := (y1 - y2) / (x1 - x2)
  let b := y1 - m * x1
  m * x + b

theorem find_y_coordinate :
  (y_coordinate_of_point_on_line = -3) :=
by
  sorry

end find_y_coordinate_l1774_177429


namespace terminating_decimal_expansion_of_17_div_200_l1774_177423

theorem terminating_decimal_expansion_of_17_div_200 :
  (17 / 200 : ℚ) = 34 / 10000 := sorry

end terminating_decimal_expansion_of_17_div_200_l1774_177423


namespace race_runners_l1774_177477

theorem race_runners (k : ℕ) (h1 : 2*(k - 1) = k - 1) (h2 : 2*(2*(k + 9) - 12) = k + 9) : 3*k - 2 = 31 :=
by
  sorry

end race_runners_l1774_177477


namespace multiple_of_larger_number_l1774_177425

variables (S L M : ℝ)

-- Conditions
def small_num := S = 10.0
def sum_eq := S + L = 24
def multiplication_relation := 7 * S = M * L

-- Theorem statement
theorem multiple_of_larger_number (S L M : ℝ) 
  (h1 : small_num S) 
  (h2 : sum_eq S L) 
  (h3 : multiplication_relation S L M) : 
  M = 5 := by
  sorry

end multiple_of_larger_number_l1774_177425


namespace sector_area_l1774_177453

theorem sector_area (θ : ℝ) (r : ℝ) (hθ : θ = 2 * Real.pi / 5) (hr : r = 20) :
  1 / 2 * r^2 * θ = 80 * Real.pi := by
  sorry

end sector_area_l1774_177453


namespace flea_jump_no_lava_l1774_177416

theorem flea_jump_no_lava
  (A B F : ℕ)
  (n : ℕ) 
  (h_posA : 0 < A)
  (h_posB : 0 < B)
  (h_AB : A < B)
  (h_2A : B < 2 * A)
  (h_ineq1 : A * (n + 1) ≤ B - A * n)
  (h_ineq2 : B - A < A * n) :
  ∃ (F : ℕ), F = (n - 1) * A + B := sorry

end flea_jump_no_lava_l1774_177416


namespace mike_books_before_yard_sale_l1774_177434

-- Problem definitions based on conditions
def books_bought_at_yard_sale : ℕ := 21
def books_now_in_library : ℕ := 56
def books_before_yard_sale := books_now_in_library - books_bought_at_yard_sale

-- Theorem to prove the equivalent proof problem
theorem mike_books_before_yard_sale : books_before_yard_sale = 35 := by
  sorry

end mike_books_before_yard_sale_l1774_177434


namespace find_b_minus_a_l1774_177403

theorem find_b_minus_a (a b : ℤ) (h1 : a * b = 2 * (a + b) + 11) (h2 : b = 7) : b - a = 2 :=
by sorry

end find_b_minus_a_l1774_177403


namespace train_length_is_135_l1774_177474

noncomputable def speed_km_per_hr : ℝ := 54
noncomputable def time_seconds : ℝ := 9
noncomputable def speed_m_per_s : ℝ := speed_km_per_hr * (1000 / 3600)
noncomputable def length_of_train : ℝ := speed_m_per_s * time_seconds

theorem train_length_is_135 : length_of_train = 135 := by
  sorry

end train_length_is_135_l1774_177474


namespace intersection_is_solution_l1774_177470

theorem intersection_is_solution (a b : ℝ) :
  (b = 3 * a + 6 ∧ b = 2 * a - 4) ↔ (3 * a - b = -6 ∧ 2 * a - b = 4) := 
by sorry

end intersection_is_solution_l1774_177470


namespace boxes_in_attic_l1774_177458

theorem boxes_in_attic (B : ℕ)
  (h1 : 6 ≤ B)
  (h2 : ∀ T : ℕ, T = (B - 6) / 2 ∧ T = 10)
  (h3 : ∀ O : ℕ, O = 180 + 2 * T ∧ O = 20 * T) :
  B = 26 :=
by
  sorry

end boxes_in_attic_l1774_177458


namespace HCF_of_two_numbers_l1774_177404

theorem HCF_of_two_numbers (H L : ℕ) (product : ℕ) (h1 : product = 2560) (h2 : L = 128)
  (h3 : H * L = product) : H = 20 := by {
  -- The proof goes here.
  sorry
}

end HCF_of_two_numbers_l1774_177404


namespace cafeteria_apples_count_l1774_177460

def initial_apples : ℕ := 17
def used_monday : ℕ := 2
def bought_monday : ℕ := 23
def used_tuesday : ℕ := 4
def bought_tuesday : ℕ := 15
def used_wednesday : ℕ := 3

def final_apples (initial_apples used_monday bought_monday used_tuesday bought_tuesday used_wednesday : ℕ) : ℕ :=
  initial_apples - used_monday + bought_monday - used_tuesday + bought_tuesday - used_wednesday

theorem cafeteria_apples_count :
  final_apples initial_apples used_monday bought_monday used_tuesday bought_tuesday used_wednesday = 46 :=
by
  sorry

end cafeteria_apples_count_l1774_177460


namespace faye_pencils_l1774_177498

theorem faye_pencils (rows : ℕ) (pencils_per_row : ℕ) (h_rows : rows = 30) (h_pencils_per_row : pencils_per_row = 24) :
  rows * pencils_per_row = 720 :=
by
  sorry

end faye_pencils_l1774_177498


namespace complement_P_eq_Ioo_l1774_177497

def U : Set ℝ := Set.univ
def P : Set ℝ := { x | x^2 - 5 * x - 6 ≥ 0 }
def complement_of_P_in_U : Set ℝ := Set.Ioo (-1) 6

theorem complement_P_eq_Ioo :
  (U \ P) = complement_of_P_in_U :=
by sorry

end complement_P_eq_Ioo_l1774_177497


namespace binary_to_decimal_110_eq_6_l1774_177494

theorem binary_to_decimal_110_eq_6 : (1 * 2^2 + 1 * 2^1 + 0 * 2^0 = 6) :=
by
  sorry

end binary_to_decimal_110_eq_6_l1774_177494


namespace area_of_parallelogram_l1774_177447

-- Define the vectors
def v : ℝ × ℝ := (7, -5)
def w : ℝ × ℝ := (14, -4)

-- Prove the area of the parallelogram
theorem area_of_parallelogram : 
  abs (v.1 * w.2 - v.2 * w.1) = 42 :=
by
  sorry

end area_of_parallelogram_l1774_177447


namespace scientific_notation_1742000_l1774_177484

theorem scientific_notation_1742000 : 1742000 = 1.742 * 10^6 := 
by
  sorry

end scientific_notation_1742000_l1774_177484


namespace equal_share_of_candles_l1774_177456

-- Define conditions
def ambika_candles : ℕ := 4
def aniyah_candles : ℕ := 6 * ambika_candles
def bree_candles : ℕ := 2 * aniyah_candles
def caleb_candles : ℕ := bree_candles + (bree_candles / 2)

-- Define the total candles and the equal share
def total_candles : ℕ := ambika_candles + aniyah_candles + bree_candles + caleb_candles
def each_share : ℕ := total_candles / 4

-- State the problem
theorem equal_share_of_candles : each_share = 37 := by
  sorry

end equal_share_of_candles_l1774_177456


namespace cubic_has_one_real_root_iff_l1774_177411

theorem cubic_has_one_real_root_iff (a : ℝ) :
  (∃! x : ℝ, x^3 + (1 - a) * x^2 - 2 * a * x + a^2 = 0) ↔ a < -1/4 := by
  sorry

end cubic_has_one_real_root_iff_l1774_177411


namespace find_x_l1774_177439

def vector := (ℝ × ℝ)

-- Define the vectors a and b
def a (x : ℝ) : vector := (x, 3)
def b : vector := (3, 1)

-- Define the perpendicular condition
def perpendicular (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Prove that under the given conditions, x = -1
theorem find_x (x : ℝ) (h : perpendicular (a x) b) : x = -1 :=
  sorry

end find_x_l1774_177439


namespace sum_of_first_3n_terms_l1774_177499

variable {S : ℕ → ℝ}
variable {n : ℕ}
variable {a b : ℝ}

def arithmetic_sum (S : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ m : ℕ, S (m + 1) = S m + (d * (m + 1))

theorem sum_of_first_3n_terms (h1 : S n = a) (h2 : S (2 * n) = b) 
  (h3 : arithmetic_sum S) : S (3 * n) = 3 * b - 2 * a :=
by
  sorry

end sum_of_first_3n_terms_l1774_177499


namespace range_of_z_minus_x_z_minus_y_l1774_177422

theorem range_of_z_minus_x_z_minus_y (x y z : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_z : 0 ≤ z) (h_sum : x + y + z = 1) :
  -1 / 8 ≤ (z - x) * (z - y) ∧ (z - x) * (z - y) ≤ 1 := by
  sorry

end range_of_z_minus_x_z_minus_y_l1774_177422


namespace total_birds_is_1300_l1774_177442

def initial_birds : ℕ := 300
def birds_doubled (b : ℕ) : ℕ := 2 * b
def birds_reduced (b : ℕ) : ℕ := b - 200
def total_birds_three_days : ℕ := initial_birds + birds_doubled initial_birds + birds_reduced (birds_doubled initial_birds)

theorem total_birds_is_1300 : total_birds_three_days = 1300 :=
by
  unfold total_birds_three_days initial_birds birds_doubled birds_reduced
  simp
  done

end total_birds_is_1300_l1774_177442


namespace maximum_food_per_guest_l1774_177436

theorem maximum_food_per_guest (total_food : ℕ) (min_guests : ℕ) (total_food_eq : total_food = 337) (min_guests_eq : min_guests = 169) :
  ∃ max_food_per_guest, max_food_per_guest = total_food / min_guests ∧ max_food_per_guest = 2 := 
by
  sorry

end maximum_food_per_guest_l1774_177436


namespace ratio_of_sector_CPD_l1774_177475

-- Define the given angles
def angle_AOC : ℝ := 40
def angle_DOB : ℝ := 60
def angle_COP : ℝ := 110

-- Calculate the angle CPD
def angle_CPD : ℝ := angle_COP - angle_AOC - angle_DOB

-- State the theorem to prove the ratio
theorem ratio_of_sector_CPD (hAOC : angle_AOC = 40) (hDOB : angle_DOB = 60)
(hCOP : angle_COP = 110) : 
  angle_CPD / 360 = 1 / 36 := by
  -- Proof will go here
  sorry

end ratio_of_sector_CPD_l1774_177475


namespace intersection_is_correct_l1774_177491

noncomputable def A := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }
noncomputable def B := { x : ℝ | 0 < x ∧ x ≤ 3 }

theorem intersection_is_correct : 
  (A ∩ B) = { x : ℝ | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_is_correct_l1774_177491


namespace product_of_fractions_is_eight_l1774_177478

theorem product_of_fractions_is_eight :
  (8 / 4) * (14 / 7) * (20 / 10) * (25 / 50) * (9 / 18) * (12 / 6) * (21 / 42) * (16 / 8) = 8 :=
by
  sorry

end product_of_fractions_is_eight_l1774_177478


namespace aarti_work_multiple_l1774_177468

-- Aarti can do a piece of work in 5 days
def days_per_unit_work := 5

-- It takes her 15 days to complete the certain multiple of work
def days_for_multiple_work := 15

-- Prove the ratio of the days for multiple work to the days per unit work equals 3
theorem aarti_work_multiple :
  days_for_multiple_work / days_per_unit_work = 3 :=
sorry

end aarti_work_multiple_l1774_177468


namespace product_of_g_at_roots_l1774_177489

noncomputable def f (x : ℝ) : ℝ := x^5 + x^2 + 1
noncomputable def g (x : ℝ) : ℝ := x^2 - 2
noncomputable def roots : List ℝ := sorry -- To indicate the list of roots x_1, x_2, x_3, x_4, x_5 of the polynomial f(x)

theorem product_of_g_at_roots :
  (roots.map g).prod = -23 := sorry

end product_of_g_at_roots_l1774_177489


namespace boys_without_calculators_l1774_177419

/-- In Mrs. Robinson's math class, there are 20 boys, and 30 of her students bring their calculators to class. 
    If 18 of the students who brought calculators are girls, then the number of boys who didn't bring their calculators is 8. -/
theorem boys_without_calculators (num_boys : ℕ) (num_students_with_calculators : ℕ) (num_girls_with_calculators : ℕ)
  (h1 : num_boys = 20)
  (h2 : num_students_with_calculators = 30)
  (h3 : num_girls_with_calculators = 18) :
  num_boys - (num_students_with_calculators - num_girls_with_calculators) = 8 :=
by 
  -- proof goes here
  sorry

end boys_without_calculators_l1774_177419


namespace inequality_always_holds_l1774_177432

theorem inequality_always_holds (a b c : ℝ) (h1 : a > b) (h2 : a * b ≠ 0) : a + c > b + c :=
sorry

end inequality_always_holds_l1774_177432


namespace unique_not_in_range_of_g_l1774_177409

noncomputable def g (m n p q : ℝ) (x : ℝ) : ℝ := (m * x + n) / (p * x + q)

theorem unique_not_in_range_of_g (m n p q : ℝ) (hne1 : m ≠ 0) (hne2 : n ≠ 0) (hne3 : p ≠ 0) (hne4 : q ≠ 0)
  (h₁ : g m n p q 23 = 23) (h₂ : g m n p q 53 = 53) (h₃ : ∀ (x : ℝ), x ≠ -q / p → g m n p q (g m n p q x) = x) :
  ∃! x : ℝ, ¬ ∃ y : ℝ, g m n p q y = x ∧ x = -38 :=
sorry

end unique_not_in_range_of_g_l1774_177409


namespace product_of_integers_abs_val_not_less_than_1_and_less_than_3_l1774_177457

theorem product_of_integers_abs_val_not_less_than_1_and_less_than_3 :
  (-2) * (-1) * 1 * 2 = 4 :=
by
  sorry

end product_of_integers_abs_val_not_less_than_1_and_less_than_3_l1774_177457


namespace sqrt_of_4_l1774_177463

theorem sqrt_of_4 : {x : ℤ | x^2 = 4} = {2, -2} :=
by
  sorry

end sqrt_of_4_l1774_177463


namespace parallel_vectors_condition_l1774_177479

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ b = (k * a.1, k * a.2)

theorem parallel_vectors_condition (m : ℝ) :
  vectors_parallel (1, m + 1) (m, 2) ↔ m = -2 ∨ m = 1 := by
  sorry

end parallel_vectors_condition_l1774_177479


namespace relatively_prime_bound_l1774_177414

theorem relatively_prime_bound {m n : ℕ} {a : ℕ → ℕ} (h1 : 1 < m) (h2 : 1 < n) (h3 : m ≥ n)
  (h4 : ∀ i j, i ≠ j → a i = a j → False) (h5 : ∀ i, a i ≤ m) (h6 : ∀ i j, i ≠ j → a i ∣ a j → a i = 1) 
  (x : ℝ) : ∃ i, dist (a i * x) (round (a i * x)) ≥ 2 / (m * (m + 1)) * dist x (round x) :=
sorry

end relatively_prime_bound_l1774_177414


namespace part1_part2_l1774_177485

def first_order_ratio_increasing (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), 0 < x → x < y → (f x) / x < (f y) / y

def second_order_ratio_increasing (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), 0 < x → x < y → (f x) / x^2 < (f y) / y^2

noncomputable def f (h : ℝ) (x : ℝ) : ℝ :=
  x^3 - 2 * h * x^2 - h * x

theorem part1 (h : ℝ) (h1 : first_order_ratio_increasing (f h)) (h2 : ¬ second_order_ratio_increasing (f h)) :
  h < 0 :=
sorry

theorem part2 (f : ℝ → ℝ) (h : second_order_ratio_increasing f) (h2 : ∃ k > 0, ∀ x > 0, f x < k) :
  ∃ k, k = 0 ∧ ∀ x > 0, f x < k :=
sorry

end part1_part2_l1774_177485


namespace tan_alpha_solution_l1774_177464

theorem tan_alpha_solution (α : ℝ) (h1 : 0 < α) (h2 : α < (Real.pi / 2))
  (h3 : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α)) :
  Real.tan α = (Real.sqrt 15) / 15 := 
sorry

end tan_alpha_solution_l1774_177464


namespace candy_bars_to_buy_l1774_177473

variable (x : ℕ)

theorem candy_bars_to_buy (h1 : 25 * x + 2 * 75 + 50 = 11 * 25) : x = 3 :=
by
  sorry

end candy_bars_to_buy_l1774_177473


namespace boar_sausages_left_l1774_177448

def boar_sausages_final_count(sausages_initial : ℕ) : ℕ :=
  let after_monday := sausages_initial - (2 / 5 * sausages_initial)
  let after_tuesday := after_monday - (1 / 2 * after_monday)
  let after_wednesday := after_tuesday - (1 / 4 * after_tuesday)
  let after_thursday := after_wednesday - (1 / 3 * after_wednesday)
  let after_sharing := after_thursday - (1 / 5 * after_thursday)
  let after_eating := after_sharing - (3 / 5 * after_sharing)
  after_eating

theorem boar_sausages_left : boar_sausages_final_count 1200 = 58 := 
  sorry

end boar_sausages_left_l1774_177448


namespace pure_imaginary_m_eq_zero_l1774_177426

noncomputable def z (m : ℝ) : ℂ := (m * (m - 1) : ℂ) + (m - 1) * Complex.I

theorem pure_imaginary_m_eq_zero (m : ℝ) (h : z m = (m - 1) * Complex.I) : m = 0 :=
by
  sorry

end pure_imaginary_m_eq_zero_l1774_177426


namespace length_segment_l1774_177449

/--
Given a cylinder with a radius of 5 units capped with hemispheres at each end and having a total volume of 900π,
prove that the length of the line segment AB is 88/3 units.
-/
theorem length_segment (r : ℝ) (V : ℝ) (h : ℝ) : r = 5 ∧ V = 900 * Real.pi → h = 88 / 3 := by
  sorry

end length_segment_l1774_177449


namespace find_price_of_each_part_l1774_177493

def original_price (total_cost : ℝ) (num_parts : ℕ) (price_per_part : ℝ) :=
  num_parts * price_per_part = total_cost

theorem find_price_of_each_part :
  original_price 439 7 62.71 :=
by
  sorry

end find_price_of_each_part_l1774_177493


namespace red_flower_ratio_l1774_177406

theorem red_flower_ratio
  (total : ℕ)
  (O : ℕ)
  (P Pu : ℕ)
  (R Y : ℕ)
  (h_total : total = 105)
  (h_orange : O = 10)
  (h_pink_purple : P + Pu = 30)
  (h_equal_pink_purple : P = Pu)
  (h_yellow : Y = R - 5)
  (h_sum : R + Y + O + P + Pu = total) :
  (R / O) = 7 / 2 :=
by
  sorry

end red_flower_ratio_l1774_177406


namespace bisection_min_calculations_l1774_177446

theorem bisection_min_calculations 
  (a b : ℝ)
  (h_interval : a = 1.4 ∧ b = 1.5)
  (delta : ℝ)
  (h_delta : delta = 0.001) :
  ∃ n : ℕ, 0.1 / (2 ^ n) ≤ delta ∧ n = 7 :=
sorry

end bisection_min_calculations_l1774_177446


namespace books_read_in_eight_hours_l1774_177487

noncomputable def pages_per_hour : ℕ := 120
noncomputable def pages_per_book : ℕ := 360
noncomputable def total_reading_time : ℕ := 8

theorem books_read_in_eight_hours (h1 : pages_per_hour = 120) 
                                  (h2 : pages_per_book = 360) 
                                  (h3 : total_reading_time = 8) : 
                                  total_reading_time * pages_per_hour / pages_per_book = 2 := 
by sorry

end books_read_in_eight_hours_l1774_177487


namespace find_fraction_l1774_177482

variable (x : ℝ) (f : ℝ)
axiom thirty_percent_of_x : 0.30 * x = 63.0000000000001
axiom fraction_condition : f = 0.40 * x + 12

theorem find_fraction : f = 96 := by
  sorry

end find_fraction_l1774_177482


namespace range_of_m_l1774_177488

theorem range_of_m (m : ℝ) : 0 < m ∧ m < 2 ↔ (2 - m > 0 ∧ - (1 / 2) * m < 0) := by
  sorry

end range_of_m_l1774_177488


namespace binary_to_decimal_l1774_177400

theorem binary_to_decimal : (11010 : ℕ) = 26 := by
  sorry

end binary_to_decimal_l1774_177400


namespace aria_cookies_per_day_l1774_177435

theorem aria_cookies_per_day 
  (cost_per_cookie : ℕ)
  (total_amount_spent : ℕ)
  (days_in_march : ℕ)
  (h_cost : cost_per_cookie = 19)
  (h_spent : total_amount_spent = 2356)
  (h_days : days_in_march = 31) : 
  (total_amount_spent / cost_per_cookie) / days_in_march = 4 :=
by
  sorry

end aria_cookies_per_day_l1774_177435


namespace product_of_equal_numbers_l1774_177441

theorem product_of_equal_numbers (a b : ℕ) (mean : ℕ) (sum : ℕ)
  (h1 : mean = 20)
  (h2 : a = 22)
  (h3 : b = 34)
  (h4 : sum = 4 * mean)
  (h5 : sum - a - b = 2 * x)
  (h6 : sum = 80)
  (h7 : x = 12) 
  : x * x = 144 :=
by
  sorry

end product_of_equal_numbers_l1774_177441


namespace pure_imaginary_sol_l1774_177401

theorem pure_imaginary_sol (m : ℝ) (h : (m^2 - m - 2) = 0 ∧ (m + 1) ≠ 0) : m = 2 :=
sorry

end pure_imaginary_sol_l1774_177401


namespace volume_of_cuboid_l1774_177455

theorem volume_of_cuboid (a b c : ℕ) (h_a : a = 2) (h_b : b = 5) (h_c : c = 8) : 
  a * b * c = 80 := 
by 
  sorry

end volume_of_cuboid_l1774_177455


namespace Charles_has_13_whistles_l1774_177450

-- Conditions
def Sean_whistles : ℕ := 45
def more_whistles_than_Charles : ℕ := 32

-- Let C be the number of whistles Charles has
def C : ℕ := Sean_whistles - more_whistles_than_Charles

-- Theorem to be proven
theorem Charles_has_13_whistles : C = 13 := by
  -- skipping proof
  sorry

end Charles_has_13_whistles_l1774_177450


namespace total_number_of_seats_l1774_177465

def number_of_trains : ℕ := 3
def cars_per_train : ℕ := 12
def seats_per_car : ℕ := 24

theorem total_number_of_seats :
  number_of_trains * cars_per_train * seats_per_car = 864 := by
  sorry

end total_number_of_seats_l1774_177465


namespace remainder_of_2_pow_2017_mod_11_l1774_177472

theorem remainder_of_2_pow_2017_mod_11 : (2 ^ 2017) % 11 = 7 := by
  sorry

end remainder_of_2_pow_2017_mod_11_l1774_177472


namespace seq_10_is_4_l1774_177440

-- Define the sequence with given properties
def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 3
  | 1 => 4
  | (n + 2) => if n % 2 = 0 then 4 else 3

-- Theorem statement: The 10th term of the sequence is 4
theorem seq_10_is_4 : seq 9 = 4 :=
by sorry

end seq_10_is_4_l1774_177440


namespace number_of_people_l1774_177413

theorem number_of_people (x : ℕ) (h1 : 175 = 175) (h2: 2 = 2) (h3 : ∀ (p : ℕ), p * x = 175 + p * 10) : x = 7 :=
sorry

end number_of_people_l1774_177413


namespace solve_for_a_l1774_177481

noncomputable def a_value (a x : ℝ) : Prop :=
  (3 / 10) * a + (2 * x + 4) / 2 = 4 * (x - 1)

theorem solve_for_a (a : ℝ) : a_value a 3 → a = 10 :=
by
  sorry

end solve_for_a_l1774_177481


namespace tram_speed_l1774_177412

variables (V : ℝ)

theorem tram_speed (h : (V + 5) / (V - 5) = 600 / 225) : V = 11 :=
sorry

end tram_speed_l1774_177412


namespace g_f_neg3_eq_1741_l1774_177405

def f (x : ℤ) : ℤ := x^3 - 3
def g (x : ℤ) : ℤ := 2*x^2 + 2*x + 1

theorem g_f_neg3_eq_1741 : g (f (-3)) = 1741 := 
by 
  sorry

end g_f_neg3_eq_1741_l1774_177405


namespace total_students_correct_l1774_177461

def num_boys : ℕ := 272
def num_girls : ℕ := num_boys + 106
def total_students : ℕ := num_boys + num_girls

theorem total_students_correct : total_students = 650 :=
by
  sorry

end total_students_correct_l1774_177461


namespace problem_solution_l1774_177415

theorem problem_solution (a b : ℕ) (x : ℝ) (h1 : x^2 + 14 * x = 24) (h2 : x = Real.sqrt a - b) (h3 : a > 0) (h4 : b > 0) :
  a + b = 80 := 
sorry

end problem_solution_l1774_177415


namespace no_solution_l1774_177486

theorem no_solution (x : ℝ) : ¬ (x / -4 ≥ 3 + x ∧ |2*x - 1| < 4 + 2*x) := 
by sorry

end no_solution_l1774_177486


namespace least_number_subtracted_l1774_177437

theorem least_number_subtracted (n m1 m2 m3 r : ℕ) (h_n : n = 642) (h_m1 : m1 = 11) (h_m2 : m2 = 13) (h_m3 : m3 = 17) (h_r : r = 4) :
  ∃ x : ℕ, (n - x) % m1 = r ∧ (n - x) % m2 = r ∧ (n - x) % m3 = r ∧ n - x = 638 :=
sorry

end least_number_subtracted_l1774_177437


namespace direct_proportion_inequality_l1774_177467

theorem direct_proportion_inequality (k x1 x2 y1 y2 : ℝ) (h_k : k < 0) (h_y1 : y1 = k * x1) (h_y2 : y2 = k * x2) (h_x : x1 < x2) : y1 > y2 :=
by
  -- The proof will be written here, currently leaving it as sorry
  sorry

end direct_proportion_inequality_l1774_177467


namespace one_plus_i_squared_eq_two_i_l1774_177430

theorem one_plus_i_squared_eq_two_i (i : ℂ) (h : i^2 = -1) : (1 + i)^2 = 2 * i :=
by
  sorry

end one_plus_i_squared_eq_two_i_l1774_177430


namespace smallest_integer_x_l1774_177427

theorem smallest_integer_x (x : ℤ) : (x^2 - 11 * x + 24 < 0) → x ≥ 4 ∧ x < 8 :=
by
sorry

end smallest_integer_x_l1774_177427


namespace factor_81_minus_36x4_l1774_177495

theorem factor_81_minus_36x4 (x : ℝ) : 
    81 - 36 * x^4 = 9 * (Real.sqrt 3 - Real.sqrt 2 * x) * (Real.sqrt 3 + Real.sqrt 2 * x) * (3 + 2 * x^2) :=
sorry

end factor_81_minus_36x4_l1774_177495


namespace students_ages_average_l1774_177402

variables (a b c : ℕ)

theorem students_ages_average (h1 : (14 * a + 13 * b + 12 * c) = 13 * (a + b + c)) : a = c :=
by
  sorry

end students_ages_average_l1774_177402


namespace b_investment_l1774_177408

theorem b_investment (x : ℝ) (total_profit A_investment B_investment C_investment A_profit: ℝ)
  (h1 : A_investment = 6300)
  (h2 : B_investment = x)
  (h3 : C_investment = 10500)
  (h4 : total_profit = 12600)
  (h5 : A_profit = 3780)
  (ratio_eq : (A_investment / (A_investment + B_investment + C_investment)) = (A_profit / total_profit)) :
  B_investment = 13700 :=
  sorry

end b_investment_l1774_177408


namespace estimate_students_correct_l1774_177410

noncomputable def estimate_students_below_85 
  (total_students : ℕ)
  (mean_score : ℝ)
  (variance : ℝ)
  (prob_90_to_95 : ℝ) : ℕ :=
if total_students = 50 ∧ mean_score = 90 ∧ prob_90_to_95 = 0.3 then 10 else 0

theorem estimate_students_correct 
  (total_students : ℕ)
  (mean_score : ℝ)
  (variance : ℝ)
  (prob_90_to_95 : ℝ)
  (h1 : total_students = 50) 
  (h2 : mean_score = 90)
  (h3 : prob_90_to_95 = 0.3) : 
  estimate_students_below_85 total_students mean_score variance prob_90_to_95 = 10 :=
by
  sorry

end estimate_students_correct_l1774_177410


namespace baseball_attendance_difference_l1774_177480

theorem baseball_attendance_difference:
  ∃ C D: ℝ, 
    (59500 ≤ C ∧ C ≤ 80500 ∧ 69565 ≤ D ∧ D ≤ 94118) ∧ 
    (max (D - C) (C - D) = 35000 ∧ min (D - C) (C - D) = 11000) := by
  sorry

end baseball_attendance_difference_l1774_177480


namespace freken_bok_weight_l1774_177483

variables (K F M : ℕ)

theorem freken_bok_weight 
  (h1 : K + F = M + 75) 
  (h2 : F + M = K + 45) : 
  F = 60 :=
sorry

end freken_bok_weight_l1774_177483
