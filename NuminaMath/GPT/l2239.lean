import Mathlib

namespace number_of_smallest_squares_l2239_223995

-- Conditions
def length_cm : ℝ := 28
def width_cm : ℝ := 48
def total_lines_cm : ℝ := 6493.6

-- The main question is about the number of smallest squares
theorem number_of_smallest_squares (d : ℝ) (h_d : d = 0.4) :
  ∃ n : ℕ, n = (length_cm / d - 2) * (width_cm / d - 2) ∧ n = 8024 :=
by
  sorry

end number_of_smallest_squares_l2239_223995


namespace pre_bought_ticket_price_l2239_223963

variable (P : ℕ)

theorem pre_bought_ticket_price :
  (20 * P = 6000 - 2900) → P = 155 :=
by
  intro h
  sorry

end pre_bought_ticket_price_l2239_223963


namespace arrangement_count1_arrangement_count2_arrangement_count3_arrangement_count4_l2239_223929

-- Define the entities in the problem
inductive Participant
| Teacher
| Boy (id : Nat)
| Girl (id : Nat)

-- Define the conditions as properties or predicates
def girlsNextToEachOther (arrangement : List Participant) : Prop :=
  -- assuming the arrangement is a list of Participant
  sorry -- insert the actual condition as needed

def boysNotNextToEachOther (arrangement : List Participant) : Prop :=
  sorry -- insert the actual condition as needed

def boysInDecreasingOrder (arrangement : List Participant) : Prop :=
  sorry -- insert the actual condition as needed

def teacherNotInMiddle (arrangement : List Participant) : Prop :=
  sorry -- insert the actual condition as needed

def girlsNotAtEnds (arrangement : List Participant) : Prop :=
  sorry -- insert the actual condition as needed

-- Problem 1: Two girls must stand next to each other
theorem arrangement_count1 : ∃ arrangements, 1440 = List.length arrangements ∧ 
  ∀ a ∈ arrangements, girlsNextToEachOther a := sorry

-- Problem 2: Boys must not stand next to each other
theorem arrangement_count2 : ∃ arrangements, 144 = List.length arrangements ∧ 
  ∀ a ∈ arrangements, boysNotNextToEachOther a := sorry

-- Problem 3: Boys must stand in decreasing order of height
theorem arrangement_count3 : ∃ arrangements, 210 = List.length arrangements ∧ 
  ∀ a ∈ arrangements, boysInDecreasingOrder a := sorry

-- Problem 4: Teacher not in middle, girls not at the ends
theorem arrangement_count4 : ∃ arrangements, 2112 = List.length arrangements ∧ 
  ∀ a ∈ arrangements, teacherNotInMiddle a ∧ girlsNotAtEnds a := sorry

end arrangement_count1_arrangement_count2_arrangement_count3_arrangement_count4_l2239_223929


namespace merchants_tea_cups_l2239_223958

theorem merchants_tea_cups (a b c : ℕ) 
  (h1 : a + b = 11)
  (h2 : b + c = 15)
  (h3 : a + c = 14) : 
  a + b + c = 20 :=
by
  sorry

end merchants_tea_cups_l2239_223958


namespace store_profit_l2239_223961

variable (C : ℝ)  -- Cost price of a turtleneck sweater

noncomputable def initial_marked_price : ℝ := 1.20 * C
noncomputable def new_year_marked_price : ℝ := 1.25 * initial_marked_price C
noncomputable def discount_amount : ℝ := 0.08 * new_year_marked_price C
noncomputable def final_selling_price : ℝ := new_year_marked_price C - discount_amount C
noncomputable def profit : ℝ := final_selling_price C - C

theorem store_profit (C : ℝ) : profit C = 0.38 * C :=
by
  -- The detailed steps are omitted, as required by the instructions.
  sorry

end store_profit_l2239_223961


namespace no_real_sol_l2239_223959

open Complex

theorem no_real_sol (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (↑(x.re) ≠ x ∨ ↑(y.re) ≠ y) → (x + y) / y ≠ x / (y + x) := by
  sorry

end no_real_sol_l2239_223959


namespace fraction_furniture_spent_l2239_223920

theorem fraction_furniture_spent (S T : ℕ) (hS : S = 600) (hT : T = 300) : (S - T) / S = 1 / 2 :=
by
  sorry

end fraction_furniture_spent_l2239_223920


namespace find_pairs_l2239_223989

theorem find_pairs (m n: ℕ) (h: m > 0 ∧ n > 0 ∧ m + n - (3 * m * n) / (m + n) = 2011 / 3) : (m = 1144 ∧ n = 377) ∨ (m = 377 ∧ n = 1144) :=
by sorry

end find_pairs_l2239_223989


namespace yuan_representation_l2239_223919

-- Define the essential conditions and numeric values
def receiving (amount : Int) : Int := amount
def spending (amount : Int) : Int := -amount

-- The main theorem statement
theorem yuan_representation :
  receiving 80 = 80 ∧ spending 50 = -50 → receiving (-50) = spending 50 :=
by
  intros h
  sorry

end yuan_representation_l2239_223919


namespace factorize_a3_minus_4a_l2239_223923

theorem factorize_a3_minus_4a (a : ℝ) : a^3 - 4 * a = a * (a + 2) * (a - 2) := 
by
  sorry

end factorize_a3_minus_4a_l2239_223923


namespace sequence_third_term_l2239_223966

theorem sequence_third_term (n : ℕ) (a : ℕ → ℤ) (h : ∀ n, a n = 3 * n - 5) : a 3 = 4 := by
  sorry

end sequence_third_term_l2239_223966


namespace rationalize_denominator_l2239_223990

theorem rationalize_denominator :
  let a := (5 : ℝ)^(1/3)
  let b := (4 : ℝ)^(1/3)
  let A := 25
  let B := 20
  let C := 16
  let D := 1
  (1 / (a - b)) = ((A : ℝ)^(1/3) + (B : ℝ)^(1/3) + (C : ℝ)^(1/3)) / D ∧ (A + B + C + D = 62) := by
  sorry

end rationalize_denominator_l2239_223990


namespace intersection_A_B_union_A_B_subset_C_B_l2239_223992

open Set

noncomputable def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
noncomputable def B : Set ℝ := {x | 2 < x ∧ x < 9}
noncomputable def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem intersection_A_B : A ∩ B = {x | 3 ≤ x ∧ x < 6} :=
by
  sorry

theorem union_A_B : A ∪ B = {x | 2 < x ∧ x < 9} :=
by
  sorry

theorem subset_C_B (a : ℝ) : C a ⊆ B → 2 ≤ a ∧ a ≤ 8 :=
by
  sorry

end intersection_A_B_union_A_B_subset_C_B_l2239_223992


namespace smallest_five_digit_neg_int_congruent_to_one_mod_17_l2239_223981

theorem smallest_five_digit_neg_int_congruent_to_one_mod_17 :
  ∃ (x : ℤ), x < -9999 ∧ x % 17 = 1 ∧ x = -10011 := by
  -- The proof would go here
  sorry

end smallest_five_digit_neg_int_congruent_to_one_mod_17_l2239_223981


namespace students_taking_german_l2239_223982

theorem students_taking_german
  (total_students : ℕ)
  (french_students : ℕ)
  (both_courses_students : ℕ)
  (no_course_students : ℕ)
  (h1 : total_students = 87)
  (h2 : french_students = 41)
  (h3 : both_courses_students = 9)
  (h4 : no_course_students = 33)
  : ∃ german_students : ℕ, german_students = 22 := 
by
  -- proof can be filled in here
  sorry

end students_taking_german_l2239_223982


namespace ratio_of_length_to_width_of_field_is_two_to_one_l2239_223950

-- Definitions based on conditions
def lengthOfField : ℕ := 80
def widthOfField (field_area pond_area : ℕ) : ℕ := field_area / lengthOfField
def pondSideLength : ℕ := 8
def pondArea : ℕ := pondSideLength * pondSideLength
def fieldArea : ℕ := pondArea * 50
def lengthMultipleOfWidth (length width : ℕ) := ∃ k : ℕ, length = k * width

-- Main statement to prove the ratio of length to width is 2:1
theorem ratio_of_length_to_width_of_field_is_two_to_one :
  lengthMultipleOfWidth lengthOfField (widthOfField fieldArea pondArea) →
  lengthOfField = 2 * (widthOfField fieldArea pondArea) :=
by
  -- Conditions
  have h1 : pondSideLength = 8 := rfl
  have h2 : pondArea = pondSideLength * pondSideLength := rfl
  have h3 : fieldArea = pondArea * 50 := rfl
  have h4 : lengthOfField = 80 := rfl
  sorry

end ratio_of_length_to_width_of_field_is_two_to_one_l2239_223950


namespace total_cups_of_liquid_drunk_l2239_223934

-- Definitions for the problem conditions
def elijah_pints : ℝ := 8.5
def emilio_pints : ℝ := 9.5
def cups_per_pint : ℝ := 2
def elijah_cups : ℝ := elijah_pints * cups_per_pint
def emilio_cups : ℝ := emilio_pints * cups_per_pint
def total_cups : ℝ := elijah_cups + emilio_cups

-- Theorem to prove the required equality
theorem total_cups_of_liquid_drunk : total_cups = 36 :=
by
  sorry

end total_cups_of_liquid_drunk_l2239_223934


namespace tan_45_degree_is_one_l2239_223960

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end tan_45_degree_is_one_l2239_223960


namespace math_problem_l2239_223974

theorem math_problem 
  (num := 1 * 2 * 3 * 4 * 5 * 6 * 7)
  (den := 1 + 2 + 3 + 4 + 5 + 6 + 7) :
  (num / den) = 180 :=
by
  sorry

end math_problem_l2239_223974


namespace meals_neither_vegan_kosher_nor_gluten_free_l2239_223970

def total_clients : ℕ := 50
def n_vegan : ℕ := 10
def n_kosher : ℕ := 12
def n_gluten_free : ℕ := 6
def n_both_vegan_kosher : ℕ := 3
def n_both_vegan_gluten_free : ℕ := 4
def n_both_kosher_gluten_free : ℕ := 2
def n_all_three : ℕ := 1

/-- The number of clients who need a meal that is neither vegan, kosher, nor gluten-free. --/
theorem meals_neither_vegan_kosher_nor_gluten_free :
  total_clients - (n_vegan + n_kosher + n_gluten_free - n_both_vegan_kosher - n_both_vegan_gluten_free - n_both_kosher_gluten_free + n_all_three) = 30 :=
by
  sorry

end meals_neither_vegan_kosher_nor_gluten_free_l2239_223970


namespace greatest_value_of_x_l2239_223964

theorem greatest_value_of_x (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
sorry

end greatest_value_of_x_l2239_223964


namespace largest_n_for_factorization_l2239_223945

theorem largest_n_for_factorization :
  ∃ (n : ℤ), (∀ (A B : ℤ), AB = 96 → n = 4 * B + A) ∧ (n = 385) := by
  sorry

end largest_n_for_factorization_l2239_223945


namespace min_combined_horses_and_ponies_l2239_223993

theorem min_combined_horses_and_ponies : 
  ∀ (P : ℕ), 
  (∃ (P' : ℕ), 
    (P = P' ∧ (∃ (x : ℕ), x = 3 * P' / 10 ∧ x = 3 * P' / 16) ∧
     (∃ (y : ℕ), y = 5 * x / 8) ∧ 
      ∀ (H : ℕ), (H = 3 + P')) → 
  P + (3 + P) = 35) := 
sorry

end min_combined_horses_and_ponies_l2239_223993


namespace f_neg_a_l2239_223909

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 := by
  sorry

end f_neg_a_l2239_223909


namespace hyperbola_eq_l2239_223918

theorem hyperbola_eq (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : -b / a = -1/2) (h4 : a^2 + b^2 = 5^2) :
  ∃ (a b : ℝ), (a = 2 * Real.sqrt 5 ∧ b = Real.sqrt 5 ∧
  (∀ x y : ℝ, (x^2 / 20 - y^2 / 5 = 1) ↔ (x, y) ∈ {p : ℝ × ℝ | (x^2 / a^2 - y^2 / b^2 = 1)})) := sorry

end hyperbola_eq_l2239_223918


namespace find_sides_of_isosceles_triangle_l2239_223907

noncomputable def isosceles_triangle_sides (b a : ℝ) : Prop :=
  ∃ (AI IL₁ : ℝ), AI = 5 ∧ IL₁ = 3 ∧
  b = 10 ∧ a = 12 ∧
  a = (6 / 5) * b ∧
  (b^2 = 8^2 + (3/5 * b)^2)

-- Proof problem statement
theorem find_sides_of_isosceles_triangle :
  ∀ (b a : ℝ), isosceles_triangle_sides b a → b = 10 ∧ a = 12 :=
by
  intros b a h
  sorry

end find_sides_of_isosceles_triangle_l2239_223907


namespace vacuum_pump_operations_l2239_223932

theorem vacuum_pump_operations (n : ℕ) (h : n ≥ 10) : 
  ∀ a : ℝ, 
  a > 0 → 
  (0.5 ^ n) * a < 0.001 * a :=
by
  intros a h_a
  sorry

end vacuum_pump_operations_l2239_223932


namespace triangle_perimeter_l2239_223986

-- Define the conditions of the problem
def a := 4
def b := 8
def quadratic_eq (x : ℝ) : Prop := x^2 - 14 * x + 40 = 0

-- Define the perimeter calculation, ensuring triangle inequality and correct side length
def valid_triangle (x : ℝ) : Prop :=
  x ≠ a ∧ x ≠ b ∧ quadratic_eq x ∧ (a + b > x) ∧ (a + x > b) ∧ (b + x > a)

-- Define the problem statement as a theorem
theorem triangle_perimeter : ∃ x : ℝ, valid_triangle x ∧ (a + b + x = 22) :=
by {
  -- Placeholder for the proof
  sorry
}

end triangle_perimeter_l2239_223986


namespace sum_A_B_equals_1_l2239_223948

-- Definitions for the digits and the properties defined in conditions
variables (A B C D : ℕ)
variable (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
variable (h_digit_bounds : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10)
noncomputable def ABCD := 1000 * A + 100 * B + 10 * C + D
axiom h_mult : ABCD * 2 = ABCD * 10

theorem sum_A_B_equals_1 : A + B = 1 :=
by
  sorry

end sum_A_B_equals_1_l2239_223948


namespace average_of_remaining_two_numbers_l2239_223933

theorem average_of_remaining_two_numbers (A B C D E : ℝ) 
  (h1 : A + B + C + D + E = 50) 
  (h2 : A + B + C = 12) : 
  (D + E) / 2 = 19 :=
by
  sorry

end average_of_remaining_two_numbers_l2239_223933


namespace sum_of_fifth_terms_arithmetic_sequences_l2239_223913

theorem sum_of_fifth_terms_arithmetic_sequences (a b : ℕ → ℝ) (d₁ d₂ : ℝ) 
  (h₁ : ∀ n, a (n + 1) = a n + d₁)
  (h₂ : ∀ n, b (n + 1) = b n + d₂)
  (h₃ : a 1 + b 1 = 7)
  (h₄ : a 3 + b 3 = 21) :
  a 5 + b 5 = 35 :=
sorry

end sum_of_fifth_terms_arithmetic_sequences_l2239_223913


namespace solve_for_y_l2239_223957

theorem solve_for_y (x y : ℝ) (h1 : x = 8) (h2 : x^(3 * y) = 8) : y = 1 / 3 := 
by
  sorry

end solve_for_y_l2239_223957


namespace initial_time_is_11_55_l2239_223973

-- Definitions for the conditions
variable (X : ℕ) (Y : ℕ)

def initial_time_shown_by_clock (X Y : ℕ) : Prop :=
  (5 * (18 - X) = 35) ∧ (Y = 60 - 5)

theorem initial_time_is_11_55 (h : initial_time_shown_by_clock X Y) : (X = 11) ∧ (Y = 55) :=
sorry

end initial_time_is_11_55_l2239_223973


namespace not_or_implies_both_false_l2239_223998

-- The statement of the problem in Lean
theorem not_or_implies_both_false {p q : Prop} (h : ¬(p ∨ q)) : ¬p ∧ ¬q :=
sorry

end not_or_implies_both_false_l2239_223998


namespace no_solution_exists_l2239_223979

   theorem no_solution_exists (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
     ¬ (3 / a + 4 / b = 12 / (a + b)) := 
   sorry
   
end no_solution_exists_l2239_223979


namespace maximum_alpha_l2239_223962

noncomputable def is_in_F (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, f (3 * x) ≥ f (f (2 * x)) + x

theorem maximum_alpha :
  (∀ f : ℝ → ℝ, is_in_F f → ∀ x > 0, f x ≥ (1 / 2) * x) := 
by
  sorry

end maximum_alpha_l2239_223962


namespace gcd_repeated_five_digit_number_l2239_223914

theorem gcd_repeated_five_digit_number :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 →
  ∀ m : ℕ, 10000 ≤ m ∧ m < 100000 →
  (10000100001 : ℕ) ∣ ((10^10 + 10^5 + 1) * n) ∧
  (10000100001 : ℕ) ∣ ((10^10 + 10^5 + 1) * m) →
  gcd ((10^10 + 10^5 + 1) * n) ((10^10 + 10^5 + 1) * m) = 10000100001 :=
sorry

end gcd_repeated_five_digit_number_l2239_223914


namespace sum_of_perimeters_l2239_223955

theorem sum_of_perimeters (x y z : ℝ) 
    (h_large_triangle_perimeter : 3 * 20 = 60)
    (h_hexagon_perimeter : 60 - (x + y + z) = 40) :
    3 * (x + y + z) = 60 := by
  sorry

end sum_of_perimeters_l2239_223955


namespace sqrt_nested_l2239_223953

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_nested_l2239_223953


namespace simplify_composite_product_fraction_l2239_223954

def first_four_composite_product : ℤ := 4 * 6 * 8 * 9
def next_four_composite_product : ℤ := 10 * 12 * 14 * 15
def expected_fraction_num : ℤ := 12
def expected_fraction_den : ℤ := 175

theorem simplify_composite_product_fraction :
  (first_four_composite_product / next_four_composite_product : ℚ) = (expected_fraction_num / expected_fraction_den) :=
by
  rw [first_four_composite_product, next_four_composite_product]
  norm_num
  sorry

end simplify_composite_product_fraction_l2239_223954


namespace eval_frac_equal_two_l2239_223951

noncomputable def eval_frac (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 - a*b + b^2 = 0) : ℂ :=
  (a^8 + b^8) / (a^2 + b^2)^4

theorem eval_frac_equal_two (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 - a*b + b^2 = 0) : eval_frac a b h1 h2 h3 = 2 :=
by {
  sorry
}

end eval_frac_equal_two_l2239_223951


namespace coeff_x3y2z5_in_expansion_l2239_223952

def binomialCoeff (n k : ℕ) : ℕ := Nat.choose n k

theorem coeff_x3y2z5_in_expansion :
  let x := 1
  let y := 1
  let z := 1
  let x_term := 2 * x
  let y_term := y
  let z_term := z
  let target_term := x_term ^ 3 * y_term ^ 2 * z_term ^ 5
  let coeff := 2^3 * binomialCoeff 10 3 * binomialCoeff 7 2 * binomialCoeff 5 5
  coeff = 20160 :=
by
  sorry

end coeff_x3y2z5_in_expansion_l2239_223952


namespace smallest_value_is_A_l2239_223912

def A : ℤ := -(-3 - 2)^2
def B : ℤ := (-3) * (-2)
def C : ℚ := ((-3)^2 : ℚ) / (-2)^2
def D : ℚ := ((-3)^2 : ℚ) / (-2)

theorem smallest_value_is_A : A < B ∧ A < C ∧ A < D :=
by
  sorry

end smallest_value_is_A_l2239_223912


namespace mail_distribution_l2239_223967

-- Define the number of houses
def num_houses : ℕ := 10

-- Define the pieces of junk mail per house
def mail_per_house : ℕ := 35

-- Define total pieces of junk mail delivered
def total_pieces_of_junk_mail : ℕ := num_houses * mail_per_house

-- Main theorem statement
theorem mail_distribution : total_pieces_of_junk_mail = 350 := by
  sorry

end mail_distribution_l2239_223967


namespace angle_D_measure_l2239_223944

theorem angle_D_measure (A B C D : ℝ) (hA : A = 50) (hB : B = 35) (hC : C = 35) :
  D = 120 :=
  sorry

end angle_D_measure_l2239_223944


namespace range_of_x_l2239_223975

noncomputable def function_y (x : ℝ) : ℝ := 2 / (Real.sqrt (x + 4))

theorem range_of_x : ∀ x : ℝ, (∃ y : ℝ, y = function_y x) → x > -4 :=
by
  intro x h
  sorry

end range_of_x_l2239_223975


namespace area_of_inner_square_l2239_223978

theorem area_of_inner_square (s₁ s₂ : ℝ) (side_length_WXYZ : ℝ) (WI : ℝ) (area_IJKL : ℝ) 
  (h1 : s₁ = 10) 
  (h2 : s₂ = 10 - 2 * Real.sqrt 2)
  (h3 : side_length_WXYZ = 10)
  (h4 : WI = 2)
  (h5 : area_IJKL = (s₂)^2): 
  area_IJKL = 102 - 20 * Real.sqrt 2 :=
by
  sorry

end area_of_inner_square_l2239_223978


namespace correct_judgment_is_C_l2239_223994

-- Definitions based on conditions
def three_points_determine_a_plane (p1 p2 p3 : Point) : Prop :=
  -- This would use some axiom or definition of a plane determined by three points
  sorry

def line_and_point_determine_a_plane (l : Line) (p : Point) : Prop :=
  -- This would use some axiom or definition of a plane determined by a line and a point not on the line
  sorry

def two_parallel_lines_and_intersecting_line_same_plane (l1 l2 l3 : Line) : Prop :=
  -- Axiom 3 and its corollary stating that two parallel lines intersected by the same line are in the same plane
  sorry

def three_lines_intersect_pairwise_same_plane (l1 l2 l3 : Line) : Prop :=
  -- Definition stating that three lines intersecting pairwise might be co-planar or not
  sorry

-- Statement of the problem in Lean
theorem correct_judgment_is_C :
    ¬ (three_points_determine_a_plane p1 p2 p3)
  ∧ ¬ (line_and_point_determine_a_plane l p)
  ∧ (two_parallel_lines_and_intersecting_line_same_plane l1 l2 l3)
  ∧ ¬ (three_lines_intersect_pairwise_same_plane l1 l2 l3) :=
  sorry

end correct_judgment_is_C_l2239_223994


namespace problem1_problem2_l2239_223908

-- Problem 1
theorem problem1 (x : ℤ) : (x - 2) ^ 2 - (x - 3) * (x + 3) = -4 * x + 13 := by
  sorry

-- Problem 2
theorem problem2 (x : ℤ) (h₁ : x ≠ 1) : 
  (x^2 + 2 * x) / (x^2 - 1) / (x + 1 + (2 * x + 1) / (x - 1)) = 1 / (x + 1) := by 
  sorry

end problem1_problem2_l2239_223908


namespace purchase_options_l2239_223903

def item_cost (a : Nat) : Nat := 100 * a + 99

def total_cost : Nat := 20083

theorem purchase_options (a : Nat) (n : Nat) (h : n * item_cost a = total_cost) :
  n = 17 ∨ n = 117 :=
by
  sorry

end purchase_options_l2239_223903


namespace cost_of_450_candies_l2239_223902

theorem cost_of_450_candies :
  let cost_per_box := 8
  let candies_per_box := 30
  let num_candies := 450
  cost_per_box * (num_candies / candies_per_box) = 120 := 
by 
  sorry

end cost_of_450_candies_l2239_223902


namespace Taimour_painting_time_l2239_223905

theorem Taimour_painting_time (T : ℝ) 
  (h1 : ∀ (T : ℝ), Jamshid_time = 0.5 * T) 
  (h2 : (1 / T + 2 / T) * 7 = 1) : 
    T = 21 :=
by
  sorry

end Taimour_painting_time_l2239_223905


namespace max_subsequences_2001_l2239_223931

theorem max_subsequences_2001 (seq : List ℕ) (h_len : seq.length = 2001) : 
  ∃ n : ℕ, n = 667^3 :=
sorry

end max_subsequences_2001_l2239_223931


namespace student_ticket_count_l2239_223926

theorem student_ticket_count (S N : ℕ) (h1 : S + N = 821) (h2 : 2 * S + 3 * N = 1933) : S = 530 :=
sorry

end student_ticket_count_l2239_223926


namespace survey_respondents_l2239_223900

theorem survey_respondents (X Y : ℕ) (hX : X = 150) (hRatio : X / Y = 5) : X + Y = 180 :=
by
  sorry

end survey_respondents_l2239_223900


namespace combined_error_percentage_l2239_223924

theorem combined_error_percentage 
  (S : ℝ) 
  (error_side : ℝ) 
  (error_area : ℝ) 
  (h1 : error_side = 0.20) 
  (h2 : error_area = 0.04) :
  (1.04 * ((1 + error_side) * S) ^ 2 - S ^ 2) / S ^ 2 * 100 = 49.76 := 
by
  sorry

end combined_error_percentage_l2239_223924


namespace triangle_angle_contradiction_l2239_223972

theorem triangle_angle_contradiction (α β γ : ℝ)
  (h_sum : α + β + γ = 180)
  (h1 : α > 60) (h2 : β > 60) (h3 : γ > 60) :
  false :=
by
  have h : α + β + γ > 180 := by
  { linarith }
  linarith

end triangle_angle_contradiction_l2239_223972


namespace regular_15gon_symmetry_l2239_223980

theorem regular_15gon_symmetry :
  ∀ (L R : ℕ),
  (L = 15) →
  (R = 24) →
  L + R = 39 :=
by
  intros L R hL hR
  exact sorry

end regular_15gon_symmetry_l2239_223980


namespace john_needs_packs_l2239_223985

-- Definitions based on conditions
def utensils_per_pack : Nat := 30
def utensils_types : Nat := 3
def spoons_per_pack : Nat := utensils_per_pack / utensils_types
def spoons_needed : Nat := 50

-- Statement to prove
theorem john_needs_packs : (50 / spoons_per_pack) = 5 :=
by
  -- To complete the proof
  sorry

end john_needs_packs_l2239_223985


namespace alice_twice_bob_in_some_years_l2239_223946

def alice_age (B : ℕ) : ℕ := B + 10
def future_age_condition (A : ℕ) : Prop := A + 5 = 19
def twice_as_old_condition (A B x : ℕ) : Prop := A + x = 2 * (B + x)

theorem alice_twice_bob_in_some_years :
  ∃ x, ∀ A B,
  alice_age B = A →
  future_age_condition A →
  twice_as_old_condition A B x := by
  sorry

end alice_twice_bob_in_some_years_l2239_223946


namespace modulo_remainder_l2239_223910

theorem modulo_remainder :
  (7 * 10^24 + 2^24) % 13 = 8 := 
by
  sorry

end modulo_remainder_l2239_223910


namespace pie_difference_l2239_223996

theorem pie_difference (s1 s3 : ℚ) (h1 : s1 = 7/8) (h3 : s3 = 3/4) :
  s1 - s3 = 1/8 :=
by
  sorry

end pie_difference_l2239_223996


namespace degrees_to_radians_neg_210_l2239_223941

theorem degrees_to_radians_neg_210 :
  -210 * (Real.pi / 180) = - (7 / 6) * Real.pi :=
by
  sorry

end degrees_to_radians_neg_210_l2239_223941


namespace tablecloth_radius_l2239_223997

theorem tablecloth_radius (diameter : ℝ) (h : diameter = 10) : diameter / 2 = 5 :=
by {
  -- Outline the proof structure to ensure the statement is correct
  sorry
}

end tablecloth_radius_l2239_223997


namespace distance_between_cityA_and_cityB_l2239_223956

noncomputable def distanceBetweenCities (time_to_cityB time_from_cityB saved_time round_trip_speed: ℝ) : ℝ :=
  let total_distance := 90 * (time_to_cityB + saved_time + time_from_cityB + saved_time) / 2
  total_distance / 2

theorem distance_between_cityA_and_cityB 
  (time_to_cityB : ℝ)
  (time_from_cityB : ℝ)
  (saved_time : ℝ)
  (round_trip_speed : ℝ)
  (distance : ℝ)
  (h1 : time_to_cityB = 6)
  (h2 : time_from_cityB = 4.5)
  (h3 : saved_time = 0.5)
  (h4 : round_trip_speed = 90)
  (h5 : distanceBetweenCities time_to_cityB time_from_cityB saved_time round_trip_speed = distance)
: distance = 427.5 := by
  sorry

end distance_between_cityA_and_cityB_l2239_223956


namespace common_divisor_greater_than_1_l2239_223927
open Nat

theorem common_divisor_greater_than_1 (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_ab : (a + b) ∣ (a * b)) (h_bc : (b + c) ∣ (b * c)) (h_ca : (c + a) ∣ (c * a)) :
    ∃ k : ℕ, k > 1 ∧ k ∣ a ∧ k ∣ b ∧ k ∣ c := 
by
  sorry

end common_divisor_greater_than_1_l2239_223927


namespace ratio_of_populations_l2239_223901

theorem ratio_of_populations (ne_pop : ℕ) (combined_pop : ℕ) (ny_pop : ℕ) (h1 : ne_pop = 2100000) 
                            (h2 : combined_pop = 3500000) (h3 : ny_pop = combined_pop - ne_pop) :
                            (ny_pop * 3 = ne_pop * 2) :=
by
  sorry

end ratio_of_populations_l2239_223901


namespace rectangular_plot_breadth_l2239_223965

theorem rectangular_plot_breadth (b : ℝ) 
    (h1 : ∃ l : ℝ, l = 3 * b)
    (h2 : 432 = 3 * b * b) : b = 12 :=
by
  sorry

end rectangular_plot_breadth_l2239_223965


namespace average_weight_of_three_l2239_223935

theorem average_weight_of_three (Ishmael Ponce Jalen : ℕ) 
  (h1 : Jalen = 160) 
  (h2 : Ponce = Jalen - 10) 
  (h3 : Ishmael = Ponce + 20) : 
  (Ishmael + Ponce + Jalen) / 3 = 160 := 
sorry

end average_weight_of_three_l2239_223935


namespace gain_percent_l2239_223938

theorem gain_percent (CP SP : ℕ) (h1 : CP = 20) (h2 : SP = 25) : 
  (SP - CP) * 100 / CP = 25 := by
  sorry

end gain_percent_l2239_223938


namespace cody_initial_money_l2239_223976

variable (x : ℤ)

theorem cody_initial_money :
  (x + 9 - 19 = 35) → (x = 45) :=
by
  intro h
  sorry

end cody_initial_money_l2239_223976


namespace negation_of_prop_equiv_l2239_223925

-- Define the proposition
def prop (x : ℝ) : Prop := x^2 + 1 > 0

-- State the theorem that negation of proposition forall x, prop x is equivalent to exists x, ¬ prop x
theorem negation_of_prop_equiv :
  ¬ (∀ x : ℝ, prop x) ↔ ∃ x : ℝ, ¬ prop x :=
by
  sorry

end negation_of_prop_equiv_l2239_223925


namespace thirty_percent_less_eq_one_fourth_more_l2239_223939

theorem thirty_percent_less_eq_one_fourth_more (x : ℝ) (hx1 : 0.7 * 90 = 63) (hx2 : (5 / 4) * x = 63) : x = 50 :=
sorry

end thirty_percent_less_eq_one_fourth_more_l2239_223939


namespace find_car_costs_optimize_purchasing_plan_minimum_cost_l2239_223916

theorem find_car_costs (x y : ℝ) (h1 : 3 * x + y = 85) (h2 : 2 * x + 4 * y = 140) :
    x = 20 ∧ y = 25 :=
by
  sorry

theorem optimize_purchasing_plan (m : ℕ) (h_total : m + (15 - m) = 15) (h_constraint : m ≤ 2 * (15 - m)) :
    m = 10 :=
by
  sorry

theorem minimum_cost (w : ℝ) (h_cost_expr : ∀ (m : ℕ), w = 20 * m + 25 * (15 - m)) (m := 10) :
    w = 325 :=
by
  sorry

end find_car_costs_optimize_purchasing_plan_minimum_cost_l2239_223916


namespace percentage_of_profit_without_discount_l2239_223942

-- Definitions for the conditions
def cost_price : ℝ := 100
def discount_rate : ℝ := 0.04
def profit_rate : ℝ := 0.32

-- The statement to prove
theorem percentage_of_profit_without_discount :
  let selling_price := cost_price + (profit_rate * cost_price)
  (selling_price - cost_price) / cost_price * 100 = 32 := by
  let selling_price := cost_price + (profit_rate * cost_price)
  sorry

end percentage_of_profit_without_discount_l2239_223942


namespace contrapositive_statement_l2239_223904

theorem contrapositive_statement (x y : ℤ) : ¬ (x + y) % 2 = 1 → ¬ (x % 2 = 1 ∧ y % 2 = 1) :=
sorry

end contrapositive_statement_l2239_223904


namespace rhombus_diagonal_length_l2239_223969

theorem rhombus_diagonal_length (d1 : ℝ) : 
  (d1 * 12) / 2 = 60 → d1 = 10 := 
by 
  sorry

end rhombus_diagonal_length_l2239_223969


namespace sum_of_first_15_terms_of_geometric_sequence_l2239_223936

theorem sum_of_first_15_terms_of_geometric_sequence (a r : ℝ) 
  (h₁ : (a * (1 - r^5)) / (1 - r) = 10) 
  (h₂ : (a * (1 - r^10)) / (1 - r) = 50) : 
  (a * (1 - r^15)) / (1 - r) = 210 := 
by 
  sorry

end sum_of_first_15_terms_of_geometric_sequence_l2239_223936


namespace magnet_cost_times_sticker_l2239_223977

theorem magnet_cost_times_sticker
  (M S A : ℝ)
  (hM : M = 3)
  (hA : A = 6)
  (hMagnetCost : M = (1/4) * 2 * A) :
  M = 4 * S :=
by
  -- Placeholder, the actual proof would go here
  sorry

end magnet_cost_times_sticker_l2239_223977


namespace power_neg8_equality_l2239_223922

theorem power_neg8_equality :
  (1 / ((-8 : ℤ) ^ 2)^3) * (-8 : ℤ)^7 = 8 :=
by
  sorry

end power_neg8_equality_l2239_223922


namespace slower_time_l2239_223949

-- Definitions for the problem conditions
def num_stories : ℕ := 50
def lola_time_per_story : ℕ := 12
def tara_time_per_story : ℕ := 10
def tara_stop_time : ℕ := 4
def tara_num_stops : ℕ := num_stories - 2 -- Stops on each floor except the first and last

-- Calculations based on the conditions
def lola_total_time : ℕ := num_stories * lola_time_per_story
def tara_total_time : ℕ := num_stories * tara_time_per_story + tara_num_stops * tara_stop_time

-- Target statement to be proven
theorem slower_time : tara_total_time = 692 := by
  sorry  -- Proof goes here (excluded as per instructions)

end slower_time_l2239_223949


namespace integer_a_for_factoring_l2239_223984

theorem integer_a_for_factoring (a : ℤ) :
  (∃ c d : ℤ, (x - a) * (x - 10) + 1 = (x + c) * (x + d)) → (a = 8 ∨ a = 12) :=
by
  sorry

end integer_a_for_factoring_l2239_223984


namespace find_values_of_symbols_l2239_223987

theorem find_values_of_symbols (a b : ℕ) (h1 : a + b + b = 55) (h2 : a + b = 40) : b = 15 ∧ a = 25 :=
  by
    sorry

end find_values_of_symbols_l2239_223987


namespace range_f_sum_l2239_223906

noncomputable def f (x : ℝ) : ℝ := 3 / (1 + 3 * x ^ 2)

theorem range_f_sum {a b : ℝ} (h₁ : Set.Ioo a b = (Set.Ioo (0 : ℝ) (3 : ℝ))) :
  a + b = 3 :=
sorry

end range_f_sum_l2239_223906


namespace transfer_balls_l2239_223915

theorem transfer_balls (X Y q p b : ℕ) (h : p + b = q) :
  b = q - p :=
by
  sorry

end transfer_balls_l2239_223915


namespace investment_time_p_l2239_223988

theorem investment_time_p (p_investment q_investment p_profit q_profit : ℝ) (p_invest_time : ℝ) (investment_ratio_pq : p_investment / q_investment = 7 / 5.00001) (profit_ratio_pq : p_profit / q_profit = 7.00001 / 10) (q_invest_time : q_invest_time = 9.999965714374696) : p_invest_time = 50 :=
sorry

end investment_time_p_l2239_223988


namespace exists_fraction_expression_l2239_223947

theorem exists_fraction_expression (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_three : p > 3) :
  ∃ (m : ℕ) (h₀ : 3 ≤ m) (h₁ : m ≤ p - 2) (x y : ℕ), (m : ℚ) / (p^2 : ℚ) = 1 / (x : ℚ) + 1 / (y : ℚ) :=
sorry

end exists_fraction_expression_l2239_223947


namespace squares_in_region_l2239_223911

theorem squares_in_region :
  let bounded_region (x y : ℤ) := y ≤ 2 * x ∧ y ≥ -1 ∧ x ≤ 6
  ∃ n : ℕ, ∀ (a b : ℤ), bounded_region a b → n = 118
:= 
  sorry

end squares_in_region_l2239_223911


namespace find_a_n_l2239_223930

-- Definitions from the conditions
def seq (a : ℕ → ℤ) : Prop :=
  ∀ n, (3 - a (n + 1)) * (6 + a n) = 18

-- The Lean statement of the problem
theorem find_a_n (a : ℕ → ℤ) (h_a0 : a 0 ≠ 3) (h_seq : seq a) :
  ∀ n, a n = 2 ^ (n + 2) - n - 3 :=
by
  sorry

end find_a_n_l2239_223930


namespace convert_binary_to_decimal_l2239_223971

theorem convert_binary_to_decimal : (1 * 2^2 + 1 * 2^1 + 1 * 2^0) = 7 := by
  sorry

end convert_binary_to_decimal_l2239_223971


namespace integral_sin_pi_half_to_three_pi_half_l2239_223991

theorem integral_sin_pi_half_to_three_pi_half :
  ∫ x in (Set.Icc (Real.pi / 2) (3 * Real.pi / 2)), Real.sin x = 0 :=
by
  sorry

end integral_sin_pi_half_to_three_pi_half_l2239_223991


namespace boxes_left_for_Sonny_l2239_223999

def initial_boxes : ℕ := 45
def boxes_given_to_brother : ℕ := 12
def boxes_given_to_sister : ℕ := 9
def boxes_given_to_cousin : ℕ := 7

def total_given_away : ℕ := boxes_given_to_brother + boxes_given_to_sister + boxes_given_to_cousin

def remaining_boxes : ℕ := initial_boxes - total_given_away

theorem boxes_left_for_Sonny : remaining_boxes = 17 := by
  sorry

end boxes_left_for_Sonny_l2239_223999


namespace drainage_capacity_per_day_l2239_223983

theorem drainage_capacity_per_day
  (capacity : ℝ)
  (rain_1 : ℝ)
  (rain_2 : ℝ)
  (rain_3 : ℝ)
  (rain_4_min : ℝ)
  (total_days : ℕ) 
  (days_to_drain : ℕ)
  (feet_to_inches : ℝ := 12)
  (required_rain_capacity : ℝ) 
  (drain_capacity_per_day : ℝ)

  (h1: capacity = 6 * feet_to_inches)
  (h2: rain_1 = 10)
  (h3: rain_2 = 2 * rain_1)
  (h4: rain_3 = 1.5 * rain_2)
  (h5: rain_4_min = 21)
  (h6: total_days = 4)
  (h7: days_to_drain = 3)
  (h8: required_rain_capacity = capacity - (rain_1 + rain_2 + rain_3))

  : drain_capacity_per_day = (rain_1 + rain_2 + rain_3 - required_rain_capacity + rain_4_min) / days_to_drain :=
sorry

end drainage_capacity_per_day_l2239_223983


namespace find_value_l2239_223921

theorem find_value (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = 108) : a^2 * b + a * b^2 = 108 :=
by
  sorry

end find_value_l2239_223921


namespace x_coordinate_of_first_point_l2239_223943

theorem x_coordinate_of_first_point (m n : ℝ) :
  (m = 2 * n + 3) ↔ (∃ (p1 p2 : ℝ × ℝ), p1 = (m, n) ∧ p2 = (m + 2, n + 1) ∧ 
    (p1.1 = 2 * p1.2 + 3) ∧ (p2.1 = 2 * p2.2 + 3)) :=
by
  sorry

end x_coordinate_of_first_point_l2239_223943


namespace solve_for_C_l2239_223937

theorem solve_for_C : 
  ∃ C : ℝ, 80 - (5 - (6 + 2 * (7 - C - 5))) = 89 ∧ C = -2 :=
by
  sorry

end solve_for_C_l2239_223937


namespace max_chocolate_bars_l2239_223917

-- Definitions
def john_money := 2450
def chocolate_bar_cost := 220

-- Theorem statement
theorem max_chocolate_bars : ∃ (x : ℕ), x = 11 ∧ chocolate_bar_cost * x ≤ john_money ∧ (chocolate_bar_cost * (x + 1) > john_money) := 
by 
  -- This is to indicate we're acknowledging that the proof is left as an exercise
  sorry

end max_chocolate_bars_l2239_223917


namespace trajectory_of_M_l2239_223940

theorem trajectory_of_M {x y x₀ y₀ : ℝ} (P_on_parabola : x₀^2 = 2 * y₀)
(line_PQ_perpendicular : ∀ Q : ℝ, true)
(vector_PM_PQ_relation : x₀ = x ∧ y₀ = 2 * y) :
  x^2 = 4 * y := by
  sorry

end trajectory_of_M_l2239_223940


namespace abs_less_than_zero_impossible_l2239_223968

theorem abs_less_than_zero_impossible (x : ℝ) : |x| < 0 → false :=
by
  sorry

end abs_less_than_zero_impossible_l2239_223968


namespace total_people_present_l2239_223928

def parents : ℕ := 105
def pupils : ℕ := 698
def total_people (parents pupils : ℕ) : ℕ := parents + pupils

theorem total_people_present : total_people parents pupils = 803 :=
by
  sorry

end total_people_present_l2239_223928
