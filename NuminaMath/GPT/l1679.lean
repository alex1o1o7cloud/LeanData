import Mathlib

namespace sequence_divisibility_condition_l1679_167955

theorem sequence_divisibility_condition (t a b x1 : ℕ) (x : ℕ → ℕ)
  (h1 : a = 1) (h2 : b = t) (h3 : x1 = t) (h4 : x 1 = x1)
  (h5 : ∀ n, n ≥ 2 → x n = a * x (n - 1) + b) :
  (∀ m n, m ∣ n → x m ∣ x n) ↔ (a = 1 ∧ b = t ∧ x1 = t) := sorry

end sequence_divisibility_condition_l1679_167955


namespace sqrt_neg9_squared_l1679_167999

theorem sqrt_neg9_squared : Real.sqrt ((-9: ℝ)^2) = 9 := by
  sorry

end sqrt_neg9_squared_l1679_167999


namespace geometric_sequence_properties_l1679_167946

theorem geometric_sequence_properties (a b c : ℝ) (r : ℝ)
    (h1 : a = -2 * r)
    (h2 : b = a * r)
    (h3 : c = b * r)
    (h4 : -8 = c * r) :
    b = -4 ∧ a * c = 16 :=
by
  sorry

end geometric_sequence_properties_l1679_167946


namespace problem_statement_l1679_167984

def f (x : ℤ) : ℤ := x^2 + 3
def g (x : ℤ) : ℤ := 2 * x + 5

theorem problem_statement : f (g 4) - g (f 4) = 129 := by
  sorry

end problem_statement_l1679_167984


namespace not_prime_sum_l1679_167914

theorem not_prime_sum (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_eq : a * b = c * d) : ¬ Nat.Prime (a + b + c + d) :=
sorry

end not_prime_sum_l1679_167914


namespace problem_condition_l1679_167926

theorem problem_condition (m : ℝ) :
  (∀ x : ℝ, x ≤ -1 → (m^2 - m) * 4^x - 2^x < 0) → -1 < m ∧ m < 2 :=
sorry

end problem_condition_l1679_167926


namespace min_abs_x1_x2_l1679_167947

noncomputable def f (a x : ℝ) : ℝ := a * Real.sin x - 2 * Real.sqrt 3 * Real.cos x

theorem min_abs_x1_x2 
  (a x1 x2 : ℝ)
  (h_symmetry : ∃ c : ℝ, c = -Real.pi / 6 ∧ (∀ x, f a (x - c) = f a x))
  (h_product : f a x1 * f a x2 = -16) :
  ∃ m : ℝ, m = abs (x1 + x2) ∧ m = 2 * Real.pi / 3 :=
by sorry

end min_abs_x1_x2_l1679_167947


namespace value_of_k_l1679_167995

theorem value_of_k (k : ℤ) : 
  (∃ a b : ℤ, x^2 + k * x + 81 = a^2 * x^2 + 2 * a * b * x + b^2) → (k = 18 ∨ k = -18) :=
by
  sorry

end value_of_k_l1679_167995


namespace tim_kittens_l1679_167978

theorem tim_kittens (initial_kittens : ℕ) (given_to_jessica_fraction : ℕ) (saras_kittens : ℕ) (adopted_fraction : ℕ) 
  (h_initial : initial_kittens = 12)
  (h_fraction_to_jessica : given_to_jessica_fraction = 3)
  (h_saras_kittens : saras_kittens = 14)
  (h_adopted_fraction : adopted_fraction = 2) :
  let kittens_after_jessica := initial_kittens - initial_kittens / given_to_jessica_fraction
  let total_kittens_after_sara := kittens_after_jessica + saras_kittens
  let adopted_kittens := saras_kittens / adopted_fraction
  let final_kittens := total_kittens_after_sara - adopted_kittens
  final_kittens = 15 :=
by {
  sorry
}

end tim_kittens_l1679_167978


namespace line_passes_through_quadrants_l1679_167934

theorem line_passes_through_quadrants (a b c : ℝ) (hab : a * b < 0) (hbc : b * c < 0) : 
  ∀ (x y : ℝ), (a * x + b * y + c = 0) → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) :=
by {
  sorry
}

end line_passes_through_quadrants_l1679_167934


namespace minimum_mn_l1679_167918

noncomputable def f (x : ℝ) (n m : ℝ) : ℝ := Real.log x - n * x + Real.log m + 1

noncomputable def f' (x : ℝ) (n : ℝ) : ℝ := 1/x - n

theorem minimum_mn (m n x_0 : ℝ) (h_m : m > 1) (h_tangent : 2*x_0 - (f x_0 n m) + 1 = 0) :
  mn = e * ((1/x_0 - 1) ^ 2 - 1) :=
sorry

end minimum_mn_l1679_167918


namespace passengers_on_ship_l1679_167921

theorem passengers_on_ship : 
  ∀ (P : ℕ), 
    P / 20 + P / 15 + P / 10 + P / 12 + P / 30 + 60 = P → 
    P = 90 :=
by 
  intros P h
  sorry

end passengers_on_ship_l1679_167921


namespace sum_of_first_15_terms_l1679_167976

theorem sum_of_first_15_terms (S : ℕ → ℕ) (h1 : S 5 = 48) (h2 : S 10 = 60) : S 15 = 72 :=
sorry

end sum_of_first_15_terms_l1679_167976


namespace probability_not_below_x_axis_half_l1679_167904

-- Define the vertices of the parallelogram
def P : (ℝ × ℝ) := (4, 4)
def Q : (ℝ × ℝ) := (-2, -2)
def R : (ℝ × ℝ) := (-8, -2)
def S : (ℝ × ℝ) := (-2, 4)

-- Define a predicate for points within the parallelogram
def in_parallelogram (A B C D : ℝ × ℝ) (p : ℝ × ℝ) : Prop := sorry

-- Define the area function
def area_of_parallelogram (A B C D : ℝ × ℝ) : ℝ := sorry

noncomputable def probability_not_below_x_axis (A B C D : ℝ × ℝ) : ℝ :=
  let total_area := area_of_parallelogram A B C D
  let area_above_x_axis := area_of_parallelogram (0, 0) D A (0, 0) / 2
  area_above_x_axis / total_area

theorem probability_not_below_x_axis_half :
  probability_not_below_x_axis P Q R S = 1 / 2 :=
sorry

end probability_not_below_x_axis_half_l1679_167904


namespace books_before_addition_l1679_167961

-- Let b be the initial number of books on the shelf
variable (b : ℕ)

theorem books_before_addition (h : b + 10 = 19) : b = 9 := by
  sorry

end books_before_addition_l1679_167961


namespace radius_I_l1679_167906

noncomputable def radius_O1 : ℝ := 3
noncomputable def radius_O2 : ℝ := 3
noncomputable def radius_O3 : ℝ := 3

axiom O1_O2_tangent : ∀ (O1 O2 : ℝ), O1 + O2 = radius_O1 + radius_O2
axiom O2_O3_tangent : ∀ (O2 O3 : ℝ), O2 + O3 = radius_O2 + radius_O3
axiom O3_O1_tangent : ∀ (O3 O1 : ℝ), O3 + O1 = radius_O3 + radius_O1

axiom I_O1_tangent : ∀ (I O1 : ℝ), I + O1 = radius_O1 + I
axiom I_O2_tangent : ∀ (I O2 : ℝ), I + O2 = radius_O2 + I
axiom I_O3_tangent : ∀ (I O3 : ℝ), I + O3 = radius_O3 + I

theorem radius_I : ∀ (I : ℝ), I = radius_O1 :=
by
  sorry

end radius_I_l1679_167906


namespace fraction_of_income_from_tips_l1679_167917

variable (S T I : ℝ)

-- Conditions
def tips_are_fraction_of_salary : Prop := T = (3/4) * S
def total_income_is_sum_of_salary_and_tips : Prop := I = S + T

-- Statement to prove
theorem fraction_of_income_from_tips (h1 : tips_are_fraction_of_salary S T) (h2 : total_income_is_sum_of_salary_and_tips S T I) :
  T / I = 3 / 7 := by
  sorry

end fraction_of_income_from_tips_l1679_167917


namespace find_m_l1679_167936

theorem find_m : ∃ m : ℝ, 
  (∀ x : ℝ, 0 < x → (m^2 - m - 5) * x^(m - 1) = (m^2 - m - 5) * x^(m - 1) ∧ 
  (m^2 - m - 5) * (m - 1) * x^(m - 2) > 0) → m = 3 :=
by
  sorry

end find_m_l1679_167936


namespace max_value_sqrt_abcd_l1679_167905

theorem max_value_sqrt_abcd (a b c d : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) (h_sum : a + b + c + d = 1) :
  Real.sqrt (abcd) ^ (1 / 4) + Real.sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ^ (1 / 4) ≤ 1 := 
sorry

end max_value_sqrt_abcd_l1679_167905


namespace find_strawberry_jelly_amount_l1679_167973

noncomputable def strawberry_jelly (t b : ℕ) : ℕ := t - b

theorem find_strawberry_jelly_amount (h₁ : 6310 = 4518 + s) : s = 1792 := by
  sorry

end find_strawberry_jelly_amount_l1679_167973


namespace smallest_whole_number_larger_than_perimeter_l1679_167932

theorem smallest_whole_number_larger_than_perimeter (s : ℝ) (h1 : 7 + 23 > s) (h2 : 7 + s > 23) (h3 : 23 + s > 7) : 
  60 = Int.ceil (7 + 23 + s - 1) :=
by
  sorry

end smallest_whole_number_larger_than_perimeter_l1679_167932


namespace triangle_classification_l1679_167916

def is_obtuse_triangle (a b c : ℕ) : Prop :=
c^2 > a^2 + b^2 ∧ a < b ∧ b < c

def is_right_triangle (a b c : ℕ) : Prop :=
c^2 = a^2 + b^2 ∧ a < b ∧ b < c

def is_acute_triangle (a b c : ℕ) : Prop :=
c^2 < a^2 + b^2 ∧ a < b ∧ b < c

theorem triangle_classification :
    is_acute_triangle 10 12 14 ∧ 
    is_right_triangle 10 24 26 ∧ 
    is_obtuse_triangle 4 6 8 :=
by 
  sorry

end triangle_classification_l1679_167916


namespace clean_house_time_l1679_167974

theorem clean_house_time (B A: ℝ) (h1: A = 1/12) (h2: B + A = 1/4):
  (B + 2 * A) = 1/3 → 1 / (B + 2 * A) = 3 :=
by
  -- Proof omitted.
  sorry

end clean_house_time_l1679_167974


namespace remainder_when_sum_divided_by_29_l1679_167909

theorem remainder_when_sum_divided_by_29 (c d : ℤ) (k j : ℤ) 
  (hc : c = 52 * k + 48) 
  (hd : d = 87 * j + 82) : 
  (c + d) % 29 = 22 := 
by 
  sorry

end remainder_when_sum_divided_by_29_l1679_167909


namespace range_of_y_for_x_gt_2_l1679_167928

theorem range_of_y_for_x_gt_2 :
  (∀ x : ℝ, x > 2 → 0 < 2 / x ∧ 2 / x < 1) :=
by 
  -- Proof is omitted
  sorry

end range_of_y_for_x_gt_2_l1679_167928


namespace ab_value_l1679_167988

-- Defining the conditions as Lean assumptions
theorem ab_value (a b c : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) (h3 : a + b + c = 10) : a * b = 9 :=
by
  sorry

end ab_value_l1679_167988


namespace crabapple_recipients_sequence_count_l1679_167989

/-- Mrs. Crabapple teaches a class of 15 students and her advanced literature class meets three times a week.
    She picks a new student each period to receive a crabapple, ensuring no student receives more than one
    crabapple in a week. Prove that the number of different sequences of crabapple recipients is 2730. -/
theorem crabapple_recipients_sequence_count :
  ∃ sequence_count : ℕ, sequence_count = 15 * 14 * 13 ∧ sequence_count = 2730 :=
by
  sorry

end crabapple_recipients_sequence_count_l1679_167989


namespace right_triangle_other_angle_l1679_167931

theorem right_triangle_other_angle (a b c : ℝ) 
  (h_triangle_sum : a + b + c = 180) 
  (h_right_angle : a = 90) 
  (h_acute_angle : b = 60) : 
  c = 30 :=
by
  sorry

end right_triangle_other_angle_l1679_167931


namespace mohan_cookies_l1679_167960

theorem mohan_cookies :
  ∃ a : ℕ, 
    a % 4 = 3 ∧
    a % 5 = 2 ∧
    a % 7 = 4 ∧
    a = 67 :=
by
  -- The proof will be written here.
  sorry

end mohan_cookies_l1679_167960


namespace rachel_milk_correct_l1679_167937

-- Define the initial amount of milk Don has
def don_milk : ℚ := 1 / 5

-- Define the fraction of milk Rachel drinks
def rachel_drinks_fraction : ℚ := 2 / 3

-- Define the total amount of milk Rachel drinks
def rachel_milk : ℚ := rachel_drinks_fraction * don_milk

-- The goal is to prove that Rachel drinks a specific amount of milk
theorem rachel_milk_correct : rachel_milk = 2 / 15 :=
by
  -- The proof would be here
  sorry

end rachel_milk_correct_l1679_167937


namespace part1_part2_1_part2_2_l1679_167933

-- Define the operation
def mul_op (x y : ℚ) : ℚ := x ^ 2 - 3 * y + 3

-- Part 1: Prove (-4) * 2 = 13 given the operation definition
theorem part1 : mul_op (-4) 2 = 13 := sorry

-- Part 2.1: Simplify (a - b) * (a - b)^2
theorem part2_1 (a b : ℚ) : mul_op (a - b) ((a - b) ^ 2) = -2 * a ^ 2 - 2 * b ^ 2 + 4 * a * b + 3 := sorry

-- Part 2.2: Find the value of the expression when a = -2 and b = 1/2
theorem part2_2 : mul_op (-2 - 1/2) ((-2 - 1/2) ^ 2) = -13 / 2 := sorry

end part1_part2_1_part2_2_l1679_167933


namespace inner_prod_sum_real_inner_prod_modulus_l1679_167952

open Complex

-- Define the given mathematical expressions
noncomputable def pair (α β : ℂ) : ℝ := (1 / 4) * (norm (α + β) ^ 2 - norm (α - β) ^ 2)

noncomputable def inner_prod (α β : ℂ) : ℂ := pair α β + Complex.I * pair α (Complex.I * β)

-- Prove the given mathematical statements

-- 1. Prove that ⟨α, β⟩ + ⟨β, α⟩ is a real number
theorem inner_prod_sum_real (α β : ℂ) : (inner_prod α β + inner_prod β α).im = 0 := sorry

-- 2. Prove that |⟨α, β⟩| = |α| * |β|
theorem inner_prod_modulus (α β : ℂ) : Complex.abs (inner_prod α β) = Complex.abs α * Complex.abs β := sorry

end inner_prod_sum_real_inner_prod_modulus_l1679_167952


namespace distance_to_center_square_l1679_167969

theorem distance_to_center_square (x y : ℝ) (h : x*x + y*y = 72) (h1 : x*x + (y + 8)*(y + 8) = 72) (h2 : (x + 4)*(x + 4) + y*y = 72) :
  x*x + y*y = 9 ∨ x*x + y*y = 185 :=
by
  sorry

end distance_to_center_square_l1679_167969


namespace det_A_zero_l1679_167957

theorem det_A_zero
  (x1 x2 x3 y1 y2 y3 : ℝ)
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ)
  (h1 : a11 = Real.sin (x1 - y1)) (h2 : a12 = Real.sin (x1 - y2)) (h3 : a13 = Real.sin (x1 - y3))
  (h4 : a21 = Real.sin (x2 - y1)) (h5 : a22 = Real.sin (x2 - y2)) (h6 : a23 = Real.sin (x2 - y3))
  (h7 : a31 = Real.sin (x3 - y1)) (h8 : a32 = Real.sin (x3 - y2)) (h9 : a33 = Real.sin (x3 - y3)) :
  (Matrix.det ![![a11, a12, a13], ![a21, a22, a23], ![a31, a32, a33]]) = 0 := sorry

end det_A_zero_l1679_167957


namespace plane_perpendicular_l1679_167943

-- Define types for lines and planes
axiom Line : Type
axiom Plane : Type

-- Define the relationships between lines and planes
axiom Parallel (l : Line) (p : Plane) : Prop
axiom Perpendicular (l : Line) (p : Plane) : Prop
axiom PlanePerpendicular (p1 p2 : Plane) : Prop

-- The setting conditions
variables (c : Line) (α β : Plane)

-- The given conditions
axiom c_perpendicular_β : Perpendicular c β
axiom c_parallel_α : Parallel c α

-- The proof goal (without the proof body)
theorem plane_perpendicular : PlanePerpendicular α β :=
by
  sorry

end plane_perpendicular_l1679_167943


namespace simplify_and_evaluate_expr_l1679_167991

namespace SimplificationProof

variable (x : ℝ)

theorem simplify_and_evaluate_expr (h : x = Real.sqrt 5 - 1) :
  ((x / (x - 1) - 1) / ((x ^ 2 - 1) / (x ^ 2 - 2 * x + 1))) = Real.sqrt 5 / 5 :=
by
  sorry

end SimplificationProof

end simplify_and_evaluate_expr_l1679_167991


namespace base_area_of_cuboid_eq_seven_l1679_167908

-- Definitions of the conditions
def volume_of_cuboid : ℝ := 28 -- Volume is 28 cm³
def height_of_cuboid : ℝ := 4  -- Height is 4 cm

-- The theorem statement for the problem
theorem base_area_of_cuboid_eq_seven
  (Volume : ℝ)
  (Height : ℝ)
  (h1 : Volume = 28)
  (h2 : Height = 4) :
  Volume / Height = 7 := by
  sorry

end base_area_of_cuboid_eq_seven_l1679_167908


namespace find_area_of_oblique_triangle_l1679_167951

noncomputable def area_triangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1 / 2 * a * b * Real.sin C

theorem find_area_of_oblique_triangle
  (A B C a b c : ℝ)
  (h1 : c = Real.sqrt 21)
  (h2 : c * Real.sin A = Real.sqrt 3 * a * Real.cos C)
  (h3 : Real.sin C + Real.sin (B - A) = 5 * Real.sin (2 * A))
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum_ABC : A + B + C = Real.pi)
  (habc_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (tri_angle_pos : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) :
  area_triangle a b c A B C = 5 * Real.sqrt 3 / 4 := 
sorry

end find_area_of_oblique_triangle_l1679_167951


namespace find_a12_l1679_167902

variable (a : ℕ → ℤ)
variable (H1 : a 1 = 1) 
variable (H2 : ∀ m n : ℕ, a (m + n) = a m + a n + m * n)

theorem find_a12 : a 12 = 78 := 
by
  sorry

end find_a12_l1679_167902


namespace opposite_of_neg3_squared_l1679_167920

theorem opposite_of_neg3_squared : -(-3^2) = 9 :=
by
  sorry

end opposite_of_neg3_squared_l1679_167920


namespace less_than_reciprocal_l1679_167993

theorem less_than_reciprocal (n : ℚ) : 
  n = -3 ∨ n = 3/4 ↔ (n = -1/2 → n >= 1/(-1/2)) ∧
                           (n = -3 → n < 1/(-3)) ∧
                           (n = 3/4 → n < 1/(3/4)) ∧
                           (n = 3 → n > 1/3) ∧
                           (n = 0 → false) := sorry

end less_than_reciprocal_l1679_167993


namespace side_length_of_regular_pentagon_l1679_167998

theorem side_length_of_regular_pentagon (perimeter : ℝ) (number_of_sides : ℕ) (h1 : perimeter = 23.4) (h2 : number_of_sides = 5) : 
  perimeter / number_of_sides = 4.68 :=
by
  sorry

end side_length_of_regular_pentagon_l1679_167998


namespace probability_same_color_l1679_167964

/-- Define the number of green plates. -/
def green_plates : ℕ := 7

/-- Define the number of yellow plates. -/
def yellow_plates : ℕ := 5

/-- Define the total number of plates. -/
def total_plates : ℕ := green_plates + yellow_plates

/-- Calculate the binomial coefficient for choosing k items from a set of n items. -/
def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Prove that the probability of selecting three plates of the same color is 9/44. -/
theorem probability_same_color :
  (binomial_coeff green_plates 3 + binomial_coeff yellow_plates 3) / binomial_coeff total_plates 3 = 9 / 44 :=
by
  sorry

end probability_same_color_l1679_167964


namespace solution_set_characterization_l1679_167979

noncomputable def satisfies_inequality (x : ℝ) : Bool :=
  (3 / (x + 2) + 4 / (x + 6)) > 1

theorem solution_set_characterization :
  ∀ x : ℝ, (satisfies_inequality x) ↔ (x < -7 ∨ (-6 < x ∧ x < -2) ∨ x > 2) :=
by
  intro x
  unfold satisfies_inequality
  -- here we would provide the proof
  sorry

end solution_set_characterization_l1679_167979


namespace first_half_day_wednesday_l1679_167907

theorem first_half_day_wednesday (h1 : ¬(1 : ℕ) = (4 % 7) ∨ 1 % 7 != 0)
  (h2 : ∀ d : ℕ, d ≤ 31 → d % 7 = ((d + 3) % 7)) : 
  ∃ d : ℕ, d = 25 ∧ ∃ W : ℕ → Prop, W d := sorry

end first_half_day_wednesday_l1679_167907


namespace tan_2x_abs_properties_l1679_167922

open Real

theorem tan_2x_abs_properties :
  (∀ x : ℝ, |tan (2 * x)| = |tan (2 * (-x))|) ∧ (∀ x : ℝ, |tan (2 * x)| = |tan (2 * (x + π / 2))|) :=
by
  sorry

end tan_2x_abs_properties_l1679_167922


namespace f_f_0_eq_zero_number_of_zeros_l1679_167950

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x > 0 then 1 - 1/x else (a - 1) * x + 1

theorem f_f_0_eq_zero (a : ℝ) : f a (f a 0) = 0 := by
  sorry

theorem number_of_zeros (a : ℝ) : 
  if a = 1 then ∃! x, f a x = 0 else
  if a > 1 then ∃! x1, ∃! x2, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0 else
  ∃! x, f a x = 0 := by sorry

end f_f_0_eq_zero_number_of_zeros_l1679_167950


namespace shopkeeper_discount_and_selling_price_l1679_167981

theorem shopkeeper_discount_and_selling_price :
  let CP := 100
  let MP := CP + 0.5 * CP
  let SP := CP + 0.15 * CP
  let Discount := (MP - SP) / MP * 100
  Discount = 23.33 ∧ SP = 115 :=
by
  sorry

end shopkeeper_discount_and_selling_price_l1679_167981


namespace find_a_l1679_167901

theorem find_a (a : ℝ) :
  (∀ x : ℝ, ((x^2 - 4 * x + a) + |x - 3| ≤ 5) → x ≤ 3) →
  (∃ x : ℝ, x = 3 ∧ ((x^2 - 4 * x + a) + |x - 3| ≤ 5)) →
  a = 2 := 
by
  sorry

end find_a_l1679_167901


namespace major_premise_is_false_l1679_167987

-- Define the major premise
def major_premise (a : ℝ) : Prop := a^2 > 0

-- Define the minor premise
def minor_premise (a : ℝ) := true

-- Define the conclusion based on the premises
def conclusion (a : ℝ) : Prop := a^2 > 0

-- Show that the major premise is false by finding a counterexample
theorem major_premise_is_false : ¬ ∀ a : ℝ, major_premise a := by
  sorry

end major_premise_is_false_l1679_167987


namespace num_of_original_numbers_l1679_167966

theorem num_of_original_numbers
    (n : ℕ) 
    (S : ℤ) 
    (incorrect_avg correct_avg : ℤ)
    (incorrect_num correct_num : ℤ)
    (h1 : incorrect_avg = 46)
    (h2 : correct_avg = 51)
    (h3 : incorrect_num = 25)
    (h4 : correct_num = 75)
    (h5 : S + correct_num = correct_avg * n)
    (h6 : S + incorrect_num = incorrect_avg * n) :
  n = 10 := by
  sorry

end num_of_original_numbers_l1679_167966


namespace final_amounts_total_l1679_167942

variable {Ben_initial Tom_initial Max_initial: ℕ}
variable {Ben_final Tom_final Max_final: ℕ}

theorem final_amounts_total (h1: Ben_initial = 48) 
                           (h2: Max_initial = 48) 
                           (h3: Ben_final = ((Ben_initial - Tom_initial - Max_initial) * 3 / 2))
                           (h4: Max_final = ((Max_initial * 3 / 2))) 
                           (h5: Tom_final = (Tom_initial * 2 - ((Ben_initial - Tom_initial - Max_initial) / 2) - 48))
                           (h6: Max_final = 48) :
  Ben_final + Tom_final + Max_final = 144 := 
by 
  sorry

end final_amounts_total_l1679_167942


namespace same_heads_probability_l1679_167959

theorem same_heads_probability
  (fair_coin : Real := 1/2)
  (biased_coin : Real := 5/8)
  (prob_Jackie_eq_Phil : Real := 77/225) :
  let m := 77
  let n := 225
  (m : ℕ) + (n : ℕ) = 302 := 
by {
  -- The proof would involve constructing the generating functions,
  -- calculating the sum of corresponding coefficients and showing that the
  -- resulting probability reduces to 77/225
  sorry
}

end same_heads_probability_l1679_167959


namespace min_abs_sum_l1679_167949

theorem min_abs_sum (x y : ℝ) : |x - 1| + |x| + |y - 1| + |y + 1| ≥ 3 :=
by sorry

end min_abs_sum_l1679_167949


namespace graph_fixed_point_l1679_167971

theorem graph_fixed_point {a : ℝ} (h₁ : a > 0) (h₂ : a ≠ 1) : 
  ∃ A : ℝ × ℝ, A = (-2, -1) ∧ ∀ x : ℝ, y = a^(x + 2) - 2 ↔ (x, y) = A := 
by 
  sorry

end graph_fixed_point_l1679_167971


namespace mrs_hilt_current_rocks_l1679_167975

-- Definitions based on conditions
def total_rocks_needed : ℕ := 125
def more_rocks_needed : ℕ := 61

-- Lean statement proving the required amount of currently held rocks
theorem mrs_hilt_current_rocks : (total_rocks_needed - more_rocks_needed) = 64 :=
by
  -- proof will be here
  sorry

end mrs_hilt_current_rocks_l1679_167975


namespace cone_central_angle_l1679_167923

/-- Proof Problem Statement: Given the radius of the base circle of a cone (r) and the slant height of the cone (l),
    prove that the central angle (θ) of the unfolded diagram of the lateral surface of this cone is 120 degrees. -/
theorem cone_central_angle (r l : ℝ) (h_r : r = 10) (h_l : l = 30) : (360 * r) / l = 120 :=
by
  -- The proof steps are omitted
  sorry

end cone_central_angle_l1679_167923


namespace problem_proof_l1679_167965

variable (P Q M N : ℝ)

axiom hp1 : M = 0.40 * Q
axiom hp2 : Q = 0.30 * P
axiom hp3 : N = 1.20 * P

theorem problem_proof : (M / N) = (1 / 10) := by
  sorry

end problem_proof_l1679_167965


namespace german_team_goals_l1679_167980

theorem german_team_goals :
  ∃ (x : ℕ), 10 < x ∧ x < 17 ∧ 11 < x ∧ x < 18 ∧ x % 2 = 1 ∧ 
             ((10 < x ∧ x < 17) ∧ (11 < x ∧ x < 18) ↔ (x % 2 = 0)) :=
sorry

end german_team_goals_l1679_167980


namespace max_x_value_l1679_167930

theorem max_x_value (x y z : ℝ) (h1 : x + y + z = 5) (h2 : xy + xz + yz = 8) : 
  x ≤ 7 / 3 :=
sorry

end max_x_value_l1679_167930


namespace positive_roots_of_x_pow_x_eq_one_over_sqrt_two_l1679_167944

theorem positive_roots_of_x_pow_x_eq_one_over_sqrt_two (x : ℝ) (h : x > 0) : 
  (x^x = 1 / Real.sqrt 2) ↔ (x = 1 / 2 ∨ x = 1 / 4) := by
  sorry

end positive_roots_of_x_pow_x_eq_one_over_sqrt_two_l1679_167944


namespace work_rate_l1679_167938

theorem work_rate (R_B : ℚ) (R_A : ℚ) (R_total : ℚ) (days : ℚ)
  (h1 : R_A = (1/2) * R_B)
  (h2 : R_B = 1 / 22.5)
  (h3 : R_total = R_A + R_B)
  (h4 : days = 1 / R_total) : 
  days = 15 := 
sorry

end work_rate_l1679_167938


namespace length_after_haircut_l1679_167925

-- Definitions
def original_length : ℕ := 18
def cut_length : ℕ := 9

-- Target statement to prove
theorem length_after_haircut : original_length - cut_length = 9 :=
by
  -- Simplification and proof
  sorry

end length_after_haircut_l1679_167925


namespace Janka_bottle_caps_l1679_167945

theorem Janka_bottle_caps (n : ℕ) :
  (∃ k1 : ℕ, n = 3 * k1) ∧ (∃ k2 : ℕ, n = 4 * k2) ↔ n = 12 ∨ n = 24 :=
by
  sorry

end Janka_bottle_caps_l1679_167945


namespace quadratic_vertex_l1679_167977

noncomputable def quadratic_vertex_max (c d : ℝ) (h : -x^2 + c * x + d ≤ 0) : (ℝ × ℝ) :=
sorry

theorem quadratic_vertex 
  (c d : ℝ)
  (h : -x^2 + c * x + d ≤ 0)
  (root1 root2 : ℝ)
  (h_roots : root1 = -5 ∧ root2 = 3) :
  quadratic_vertex_max c d h = (4, 1) ∧ (∀ x: ℝ, (x - 4)^2 ≤ 1) :=
sorry

end quadratic_vertex_l1679_167977


namespace expand_expression_l1679_167919

variable {R : Type _} [CommRing R] (x : R)

theorem expand_expression :
  (3*x^2 + 7*x + 4) * (5*x - 2) = 15*x^3 + 29*x^2 + 6*x - 8 :=
by
  sorry

end expand_expression_l1679_167919


namespace share_of_C_l1679_167968

variable (A B C x : ℝ)

theorem share_of_C (hA : A = (2/3) * B) 
(hB : B = (1/4) * C) 
(hTotal : A + B + C = 595) 
(hC : C = x) : x = 420 :=
by
  -- Proof will follow here
  sorry

end share_of_C_l1679_167968


namespace sufficient_but_not_necessary_l1679_167992

def p (x : ℝ) : Prop := x > 0
def q (x : ℝ) : Prop := |x| > 0

theorem sufficient_but_not_necessary (x : ℝ) : 
  (p x → q x) ∧ (¬(q x → p x)) :=
by
  sorry

end sufficient_but_not_necessary_l1679_167992


namespace integral_f_equals_neg_third_l1679_167972

def f (x : ℝ) (c : ℝ) : ℝ := x^2 + 2 * c

theorem integral_f_equals_neg_third :
  (∫ x in (0 : ℝ)..(1 : ℝ), f x (∫ t in (0 : ℝ)..(1 : ℝ), f t (∫ t in (0 : ℝ)..(1 : ℝ), f t 0))) = -1/3 :=
by
  sorry

end integral_f_equals_neg_third_l1679_167972


namespace no_nat_n_divisible_by_169_l1679_167983

theorem no_nat_n_divisible_by_169 (n : ℕ) : ¬ ∃ k : ℕ, n^2 + 5 * n + 16 = 169 * k :=
sorry

end no_nat_n_divisible_by_169_l1679_167983


namespace students_voted_for_meat_l1679_167903

theorem students_voted_for_meat (total_votes veggies_votes : ℕ) (h_total: total_votes = 672) (h_veggies: veggies_votes = 337) :
  total_votes - veggies_votes = 335 := 
by
  -- Proof steps go here
  sorry

end students_voted_for_meat_l1679_167903


namespace nested_sqrt_eq_two_l1679_167985

theorem nested_sqrt_eq_two (y : ℝ) (h : y = Real.sqrt (2 + y)) : y = 2 :=
by {
    -- Proof skipped
    sorry
}

end nested_sqrt_eq_two_l1679_167985


namespace trim_area_dodecagon_pie_l1679_167935

theorem trim_area_dodecagon_pie :
  let d := 8 -- diameter of the pie
  let r := d / 2 -- radius of the pie
  let A_circle := π * r^2 -- area of the circle
  let A_dodecagon := 3 * r^2 -- area of the dodecagon
  let A_trimmed := A_circle - A_dodecagon -- area to be trimmed
  let a := 16 -- coefficient of π in A_trimmed
  let b := 48 -- constant term in A_trimmed
  a + b = 64 := 
by 
  sorry

end trim_area_dodecagon_pie_l1679_167935


namespace algebraic_expression_value_l1679_167967

variable {R : Type} [CommRing R]

theorem algebraic_expression_value (m n : R) (h1 : m - n = -2) (h2 : m * n = 3) :
  -m^3 * n + 2 * m^2 * n^2 - m * n^3 = -12 :=
sorry

end algebraic_expression_value_l1679_167967


namespace cost_to_open_store_l1679_167994

-- Define the conditions as constants
def revenue_per_month : ℕ := 4000
def expenses_per_month : ℕ := 1500
def months_to_payback : ℕ := 10

-- Theorem stating the cost to open the store
theorem cost_to_open_store : (revenue_per_month - expenses_per_month) * months_to_payback = 25000 :=
by
  sorry

end cost_to_open_store_l1679_167994


namespace find_certain_number_l1679_167958

def certain_number (x : ℚ) : Prop := 5 * 1.6 - (1.4 * x) / 1.3 = 4

theorem find_certain_number : certain_number (-(26/7)) :=
by 
  simp [certain_number]
  sorry

end find_certain_number_l1679_167958


namespace tan_inverse_least_positive_l1679_167910

variables (a b x : ℝ)

-- Condition 1: tan(x) = a / (2*b)
def condition1 : Prop := Real.tan x = a / (2 * b)

-- Condition 2: tan(2*x) = 2*b / (a + 2*b)
def condition2 : Prop := Real.tan (2 * x) = (2 * b) / (a + 2 * b)

-- The theorem stating the least positive value of x is arctan(0)
theorem tan_inverse_least_positive (h1 : condition1 a b x) (h2 : condition2 a b x) : ∃ k : ℝ, Real.arctan k = 0 :=
by
  sorry

end tan_inverse_least_positive_l1679_167910


namespace max_value_l1679_167996

-- Define the weights and values of gemstones
def weight_sapphire : ℕ := 6
def value_sapphire : ℕ := 15
def weight_ruby : ℕ := 3
def value_ruby : ℕ := 9
def weight_diamond : ℕ := 2
def value_diamond : ℕ := 5

-- Define the weight capacity
def max_weight : ℕ := 24

-- Define the availability constraint
def min_availability : ℕ := 10

-- The goal is to prove that the maximum value is 72
theorem max_value : ∃ (num_sapphire num_ruby num_diamond : ℕ),
  num_sapphire >= min_availability ∧
  num_ruby >= min_availability ∧
  num_diamond >= min_availability ∧
  num_sapphire * weight_sapphire + num_ruby * weight_ruby + num_diamond * weight_diamond ≤ max_weight ∧
  num_sapphire * value_sapphire + num_ruby * value_ruby + num_diamond * value_diamond = 72 :=
by sorry

end max_value_l1679_167996


namespace convert_to_scientific_notation_l1679_167929

-- Problem statement: convert 120 million to scientific notation and validate the format.
theorem convert_to_scientific_notation :
  120000000 = 1.2 * 10^7 :=
sorry

end convert_to_scientific_notation_l1679_167929


namespace range_of_a_plus_abs_b_l1679_167982

theorem range_of_a_plus_abs_b (a b : ℝ)
  (h1 : -1 ≤ a) (h2 : a ≤ 3)
  (h3 : -5 < b) (h4 : b < 3) :
  -1 ≤ a + |b| ∧ a + |b| < 8 := by
sorry

end range_of_a_plus_abs_b_l1679_167982


namespace percent_of_y_l1679_167970

theorem percent_of_y (y : ℝ) (hy : y > 0) : (6 * y / 20) + (3 * y / 10) = 0.6 * y :=
by
  sorry

end percent_of_y_l1679_167970


namespace mod_multiplication_l1679_167900

theorem mod_multiplication :
  (176 * 929) % 50 = 4 :=
by
  sorry

end mod_multiplication_l1679_167900


namespace largest_7_10_triple_l1679_167924

theorem largest_7_10_triple :
  ∃ M : ℕ, (3 * M = Nat.ofDigits 10 (Nat.digits 7 M))
  ∧ (∀ N : ℕ, (3 * N = Nat.ofDigits 10 (Nat.digits 7 N)) → N ≤ M)
  ∧ M = 335 :=
sorry

end largest_7_10_triple_l1679_167924


namespace solve_system_eqs_l1679_167990
noncomputable section

theorem solve_system_eqs (x y z : ℝ) :
  (x * y = 5 * (x + y) ∧ x * z = 4 * (x + z) ∧ y * z = 2 * (y + z))
  ↔ (x = 0 ∧ y = 0 ∧ z = 0)
  ∨ (x = -40 ∧ y = 40 / 9 ∧ z = 40 / 11) := sorry

end solve_system_eqs_l1679_167990


namespace hyperbola_problem_l1679_167997

theorem hyperbola_problem (s : ℝ) :
    (∃ b > 0, ∀ (x y : ℝ), (x, y) = (-4, 5) → (x^2 / 9) - (y^2 / b^2) = 1)
    ∧ (∀ (x y : ℝ), (x, y) = (-3, 0) → (x^2 / 9) - (y^2 / b^2) = 1)
    ∧ (∀ b > 0, (x, y) = (s, 3) → (x^2 / 9) - (7 * y^2 / 225) = 1)
    → s^2 = (288 / 25) :=
by
  sorry

end hyperbola_problem_l1679_167997


namespace alice_and_bob_pies_l1679_167948

theorem alice_and_bob_pies (T : ℝ) : (T / 5 = T / 6 + 2) → T = 60 := by
  sorry

end alice_and_bob_pies_l1679_167948


namespace average_age_add_person_l1679_167911

theorem average_age_add_person (n : ℕ) (h1 : (∀ T, T = n * 14 → (T + 34) / (n + 1) = 16)) : n = 9 :=
by
  sorry

end average_age_add_person_l1679_167911


namespace exponent_equation_l1679_167963

theorem exponent_equation (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by sorry

end exponent_equation_l1679_167963


namespace calculate_fraction_square_mul_l1679_167953

theorem calculate_fraction_square_mul :
  ((8 / 9) ^ 2) * ((1 / 3) ^ 2) = 64 / 729 :=
by
  sorry

end calculate_fraction_square_mul_l1679_167953


namespace vector_sum_l1679_167912

def a : ℝ × ℝ := (-2, 3)
def b : ℝ × ℝ := (1, -2)

theorem vector_sum:
  2 • a + b = (-3, 4) :=
by 
  sorry

end vector_sum_l1679_167912


namespace cylinder_volume_l1679_167962

theorem cylinder_volume (r h : ℝ) (hr : r = 5) (hh : h = 10) :
    π * r^2 * h = 250 * π := by
  -- We leave the actual proof as sorry for now
  sorry

end cylinder_volume_l1679_167962


namespace willie_bananas_remain_same_l1679_167954

variable (Willie_bananas Charles_bananas Charles_loses : ℕ)

theorem willie_bananas_remain_same (h_willie : Willie_bananas = 48) (h_charles_initial : Charles_bananas = 14) (h_charles_loses : Charles_loses = 35) :
  Willie_bananas = 48 :=
by
  sorry

end willie_bananas_remain_same_l1679_167954


namespace function_symmetry_origin_l1679_167927

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - x

theorem function_symmetry_origin : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end function_symmetry_origin_l1679_167927


namespace poultry_count_correct_l1679_167913

noncomputable def total_poultry : ℝ :=
  let hens_total := 40
  let ducks_total := 20
  let geese_total := 10
  let pigeons_total := 30

  -- Calculate males and females
  let hens_males := (2/9) * hens_total
  let hens_females := hens_total - hens_males

  let ducks_males := (1/4) * ducks_total
  let ducks_females := ducks_total - ducks_males

  let geese_males := (3/11) * geese_total
  let geese_females := geese_total - geese_males

  let pigeons_males := (1/2) * pigeons_total
  let pigeons_females := pigeons_total - pigeons_males

  -- Offspring calculations using breeding success rates
  let hens_offspring := (0.85 * hens_females) * 7
  let ducks_offspring := (0.75 * ducks_females) * 9
  let geese_offspring := (0.9 * geese_females) * 5
  let pigeons_pairs := 0.8 * (pigeons_females / 2)
  let pigeons_offspring := pigeons_pairs * 2 * 0.8

  -- Total poultry count
  (hens_total + ducks_total + geese_total + pigeons_total) + (hens_offspring + ducks_offspring + geese_offspring + pigeons_offspring)

theorem poultry_count_correct : total_poultry = 442 := by
  sorry

end poultry_count_correct_l1679_167913


namespace weight_labels_correct_l1679_167915

-- Noncomputable because we're dealing with theoretical weight comparisons
noncomputable section

-- Defining the weights and their properties
variables {x1 x2 x3 x4 x5 x6 : ℕ}

-- Given conditions as stated
axiom h1 : x1 + x2 + x3 = 6
axiom h2 : x6 = 6
axiom h3 : x1 + x6 < x3 + x5

theorem weight_labels_correct :
  x1 = 1 ∧ x2 = 2 ∧ x3 = 3 ∧ x4 = 4 ∧ x5 = 5 ∧ x6 = 6 :=
sorry

end weight_labels_correct_l1679_167915


namespace sum_of_15_terms_l1679_167986

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem sum_of_15_terms 
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_sum1 : a 1 + a 2 + a 3 = 1)
  (h_sum2 : a 4 + a 5 + a 6 = -2) :
  (a 1 + a 2 + a 3) + (a 4 + a 5 + a 6) + (a 7 + a 8 + a 9) +
  (a 10 + a 11 + a 12) + (a 13 + a 14 + a 15) = 11 :=
sorry

end sum_of_15_terms_l1679_167986


namespace people_in_club_M_l1679_167939

theorem people_in_club_M (m s z n : ℕ) (h1 : s = 18) (h2 : z = 11) (h3 : m + s + z + n = 60) (h4 : n ≤ 26) : m = 5 :=
sorry

end people_in_club_M_l1679_167939


namespace line_parallel_to_plane_line_perpendicular_to_plane_l1679_167956

theorem line_parallel_to_plane (A B C D x1 y1 z1 m n p : ℝ) :
  A * m + B * n + C * p = 0 ↔ 
  ∀ x y z, ((A * x + B * y + C * z + D = 0) → 
  (∃ t, x = x1 + m * t ∧ y = y1 + n * t ∧ z = z1 + p * t)) :=
sorry

theorem line_perpendicular_to_plane (A B C D x1 y1 z1 m n p : ℝ) :
  (A / m = B / n ∧ B / n = C / p) ↔ 
  ∀ x y z, ((A * x + B * y + C * z + D = 0) → 
  (∃ t, x = x1 + m * t ∧ y = y1 + n * t ∧ z = z1 + p * t)) :=
sorry

end line_parallel_to_plane_line_perpendicular_to_plane_l1679_167956


namespace equivalent_multipliers_l1679_167940

variable (a b c : ℝ)

theorem equivalent_multipliers :
  (a - 0.07 * a + 0.05 * b) / c = (0.93 * a + 0.05 * b) / c :=
sorry

end equivalent_multipliers_l1679_167940


namespace zs_share_in_profit_l1679_167941

noncomputable def calculateProfitShare (x_investment y_investment z_investment z_months total_profit : ℚ) : ℚ :=
  let x_invest_months := x_investment * 12
  let y_invest_months := y_investment * 12
  let z_invest_months := z_investment * z_months
  let total_invest_months := x_invest_months + y_invest_months + z_invest_months
  let z_share := z_invest_months / total_invest_months
  total_profit * z_share

theorem zs_share_in_profit :
  calculateProfitShare 36000 42000 48000 8 14190 = 2580 :=
by
  sorry

end zs_share_in_profit_l1679_167941
