import Mathlib

namespace rods_in_one_mile_l1087_108779

/-- Definitions based on given conditions -/
def miles_to_furlongs := 8
def furlongs_to_rods := 40

/-- The theorem stating the number of rods in one mile -/
theorem rods_in_one_mile : (miles_to_furlongs * furlongs_to_rods) = 320 := 
  sorry

end rods_in_one_mile_l1087_108779


namespace find_alpha_l1087_108770

-- Declare the conditions
variables (α : ℝ) (h₀ : 0 < α) (h₁ : α < 90) (h₂ : Real.sin (α - 10 * Real.pi / 180) = Real.sqrt 3 / 2)

theorem find_alpha : α = 70 * Real.pi / 180 :=
sorry

end find_alpha_l1087_108770


namespace range_of_a_for_decreasing_f_l1087_108706

theorem range_of_a_for_decreasing_f :
  (∀ x : ℝ, (-3) * x^2 + 2 * a * x - 1 ≤ 0) ↔ (-Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3) :=
by
  -- The proof goes here
  sorry

end range_of_a_for_decreasing_f_l1087_108706


namespace gcd_sub_12_eq_36_l1087_108778

theorem gcd_sub_12_eq_36 :
  Nat.gcd 7344 48 - 12 = 36 := 
by 
  sorry

end gcd_sub_12_eq_36_l1087_108778


namespace geometric_sequence_a3_equals_4_l1087_108761

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ i, a (i+1) = a i * r

theorem geometric_sequence_a3_equals_4 
    (a_seq : is_geometric_sequence a) 
    (a_6_eq : a 6 = 6)
    (a_9_eq : a 9 = 9) : 
    a 3 = 4 := 
sorry

end geometric_sequence_a3_equals_4_l1087_108761


namespace B_share_is_102_l1087_108701

variables (A B C : ℝ)
variables (total : ℝ)
variables (rA_B : ℝ) (rB_C : ℝ)

-- Conditions
def conditions : Prop :=
  (total = 578) ∧
  (rA_B = 2 / 3) ∧
  (rB_C = 1 / 4) ∧
  (A = rA_B * B) ∧
  (B = rB_C * C) ∧
  (A + B + C = total)

-- Theorem to prove B's share
theorem B_share_is_102 (h : conditions A B C total rA_B rB_C) : B = 102 :=
by sorry

end B_share_is_102_l1087_108701


namespace ticket_prices_count_l1087_108788

theorem ticket_prices_count :
  let y := 30
  let divisors := [1, 2, 3, 5, 6, 10, 15, 30]
  ∀ (k : ℕ), (k ∈ divisors) ↔ (60 % k = 0 ∧ 90 % k = 0) → 
  (∃ n : ℕ, n = 8) :=
by
  sorry

end ticket_prices_count_l1087_108788


namespace sum_of_digits_is_32_l1087_108748

/-- 
Prove that the sum of digits \( A, B, C, D, E \) is 32 given the constraints
1. \( A, B, C, D, E \) are single digits.
2. The sum of the units column 3E results in 1 (units place of 2011).
3. The sum of the hundreds column 3A and carry equals 20 (hundreds place of 2011).
-/
theorem sum_of_digits_is_32
  (A B C D E : ℕ)
  (h1 : A < 10)
  (h2 : B < 10)
  (h3 : C < 10)
  (h4 : D < 10)
  (h5 : E < 10)
  (units_condition : 3 * E % 10 = 1)
  (hundreds_condition : ∃ carry: ℕ, carry < 10 ∧ 3 * A + carry = 20) :
  A + B + C + D + E = 32 := 
sorry

end sum_of_digits_is_32_l1087_108748


namespace circumscribed_triangle_area_relationship_l1087_108795

theorem circumscribed_triangle_area_relationship (X Y Z : ℝ) :
  let a := 15
  let b := 20
  let c := 25
  let triangle_area := (1/2) * a * b
  let diameter := c
  let radius := diameter / 2
  let circle_area := Real.pi * radius^2
  let Z := circle_area / 2
  (X + Y + triangle_area = Z) :=
sorry

end circumscribed_triangle_area_relationship_l1087_108795


namespace quadrilaterals_property_A_false_l1087_108726

theorem quadrilaterals_property_A_false (Q A : Type → Prop) 
  (h : ¬ ∃ x, Q x ∧ A x) : ¬ ∀ x, Q x → A x :=
by
  sorry

end quadrilaterals_property_A_false_l1087_108726


namespace frac_sum_equals_seven_eights_l1087_108725

theorem frac_sum_equals_seven_eights (p q r u v w : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p * u + q * v + r * w = 56) :
  (p + q + r) / (u + v + w) = 7 / 8 := 
  sorry

end frac_sum_equals_seven_eights_l1087_108725


namespace ratio_nephews_l1087_108740

variable (N : ℕ) -- The number of nephews Alden has now.
variable (Alden_had_50 : Prop := 50 = 50)
variable (Vihaan_more_60 : Prop := Vihaan = N + 60)
variable (Together_260 : Prop := N + (N + 60) = 260)

theorem ratio_nephews (N : ℕ) 
  (H1 : Alden_had_50)
  (H2 : Vihaan_more_60)
  (H3 : Together_260) :
  50 / N = 1 / 2 :=
by
  sorry

end ratio_nephews_l1087_108740


namespace no_solution_exists_l1087_108716

theorem no_solution_exists :
  ∀ a b : ℕ, a - b = 5 ∨ b - a = 5 → a * b = 132 → false :=
by
  sorry

end no_solution_exists_l1087_108716


namespace minimize_quadratic_l1087_108742

theorem minimize_quadratic (y : ℝ) : 
  ∃ m, m = 3 * y ^ 2 - 18 * y + 11 ∧ 
       (∀ z : ℝ, 3 * z ^ 2 - 18 * z + 11 ≥ m) ∧ 
       m = -16 := 
sorry

end minimize_quadratic_l1087_108742


namespace identify_wrong_operator_l1087_108758

def original_expr (x y z w u v p q : Int) : Int := x + y - z + w - u + v - p + q
def wrong_expr (x y z w u v p q : Int) : Int := x + y - z - w - u + v - p + q

theorem identify_wrong_operator :
  original_expr 3 5 7 9 11 13 15 17 ≠ -4 →
  wrong_expr 3 5 7 9 11 13 15 17 = -4 :=
by
  sorry

end identify_wrong_operator_l1087_108758


namespace combined_area_is_256_l1087_108777

-- Define the conditions
def side_length : ℝ := 16
def area_square : ℝ := side_length ^ 2

-- Define the property of the sides r and s
def r_s_property (r s : ℝ) : Prop :=
  (r + s)^2 + (r - s)^2 = side_length^2

-- The combined area of the four triangles
def combined_area_of_triangles (r s : ℝ) : ℝ :=
  2 * (r ^ 2 + s ^ 2)

-- Prove the final statement
theorem combined_area_is_256 (r s : ℝ) (h : r_s_property r s) :
  combined_area_of_triangles r s = 256 := by
  sorry

end combined_area_is_256_l1087_108777


namespace joao_chocolates_l1087_108711

theorem joao_chocolates (n : ℕ) (hn1 : 30 < n) (hn2 : n < 100) (h1 : n % 7 = 1) (h2 : n % 10 = 2) : n = 92 :=
sorry

end joao_chocolates_l1087_108711


namespace largest_multiple_of_15_less_than_500_l1087_108750

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l1087_108750


namespace number_of_democrats_in_senate_l1087_108769

/-
This Lean statement captures the essence of the problem: proving the number of Democrats in the Senate (S_D) is 55,
under given conditions involving the House's and Senate's number of Democrats and Republicans.
-/

theorem number_of_democrats_in_senate
  (D R S_D S_R : ℕ)
  (h1 : D + R = 434)
  (h2 : R = D + 30)
  (h3 : S_D + S_R = 100)
  (h4 : S_D * 4 = S_R * 5) :
  S_D = 55 := by
  sorry

end number_of_democrats_in_senate_l1087_108769


namespace solution_is_unique_l1087_108722

noncomputable def solution (f : ℝ → ℝ) (α : ℝ) :=
  ∀ x y : ℝ, f (f (x + y) * f (x - y)) = x^2 + α * y * f y

theorem solution_is_unique (f : ℝ → ℝ) (α : ℝ)
  (h : solution f α) :
  f = id ∧ α = -1 :=
sorry

end solution_is_unique_l1087_108722


namespace minimum_value_expression_l1087_108776

theorem minimum_value_expression (x : ℝ) : ∃ y : ℝ, (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2050 = y ∧ ∀ z : ℝ, ((x + 1) * (x + 3) * (x + 5) * (x + 7) + 2050 ≥ z) ↔ (z = 2034) :=
by
  sorry

end minimum_value_expression_l1087_108776


namespace find_phi_l1087_108747

open Real

noncomputable def f (x φ : ℝ) : ℝ := cos (2 * x + φ)
noncomputable def g (x φ : ℝ) : ℝ := cos (2 * x - π/2 + φ)

theorem find_phi 
  (h1 : 0 < φ) 
  (h2 : φ < π) 
  (symmetry_condition : ∀ x, g (π/2 - x) φ = g (π/2 + x) φ) 
  : φ = π / 2 
:= by 
  sorry

end find_phi_l1087_108747


namespace moles_of_nacl_formed_l1087_108730

noncomputable def reaction (nh4cl: ℕ) (naoh: ℕ) : ℕ :=
  if nh4cl = naoh then nh4cl else min nh4cl naoh

theorem moles_of_nacl_formed (nh4cl: ℕ) (naoh: ℕ) (h_nh4cl: nh4cl = 2) (h_naoh: naoh = 2) :
  reaction nh4cl naoh = 2 :=
by
  rw [h_nh4cl, h_naoh]
  sorry

end moles_of_nacl_formed_l1087_108730


namespace binomial_expansion_l1087_108738

theorem binomial_expansion : 
  (102: ℕ)^4 - 4 * (102: ℕ)^3 + 6 * (102: ℕ)^2 - 4 * (102: ℕ) + 1 = (101: ℕ)^4 :=
by sorry

end binomial_expansion_l1087_108738


namespace parts_of_second_liquid_l1087_108752

theorem parts_of_second_liquid (x : ℝ) :
    (0.10 * 5 + 0.15 * x) / (5 + x) = 11.42857142857143 / 100 ↔ x = 2 :=
by
  sorry

end parts_of_second_liquid_l1087_108752


namespace train_speed_second_part_l1087_108791

variables (x v : ℝ)

theorem train_speed_second_part
  (h1 : ∀ t1 : ℝ, t1 = x / 30)
  (h2 : ∀ t2 : ℝ, t2 = 2 * x / v)
  (h3 : ∀ t : ℝ, t = 3 * x / 22.5) :
  (x / 30) + (2 * x / v) = (3 * x / 22.5) → v = 20 :=
by
  intros h4
  sorry

end train_speed_second_part_l1087_108791


namespace find_k_l1087_108756

noncomputable def f (a b c : ℤ) (x : ℤ) := a * x^2 + b * x + c

theorem find_k (a b c k : ℤ) 
  (h1 : f a b c 1 = 0) 
  (h2 : 50 < f a b c 7) (h2' : f a b c 7 < 60) 
  (h3 : 70 < f a b c 8) (h3' : f a b c 8 < 80) 
  (h4 : 5000 * k < f a b c 100) (h4' : f a b c 100 < 5000 * (k + 1)) : 
  k = 3 := 
sorry

end find_k_l1087_108756


namespace arithmetic_sequence_a7_l1087_108762

theorem arithmetic_sequence_a7 :
  ∀ (a : ℕ → ℕ) (d : ℕ),
  (∀ n, a (n + 1) = a n + d) →
  a 1 = 2 →
  a 3 + a 5 = 10 →
  a 7 = 8 :=
by
  intros a d h_seq h_a1 h_sum
  sorry

end arithmetic_sequence_a7_l1087_108762


namespace number_of_digits_in_N_l1087_108760

noncomputable def N : ℕ := 2^12 * 5^8

theorem number_of_digits_in_N : (Nat.digits 10 N).length = 10 := by
  sorry

end number_of_digits_in_N_l1087_108760


namespace triangle_third_side_length_l1087_108739

theorem triangle_third_side_length (A B C : Type) 
  (AB : ℝ) (AC : ℝ) 
  (angle_ABC angle_ACB : ℝ) 
  (BC : ℝ) 
  (h1 : AB = 7) 
  (h2 : AC = 21) 
  (h3 : angle_ABC = 3 * angle_ACB) 
  : 
  BC = (some_correct_value ) := 
sorry

end triangle_third_side_length_l1087_108739


namespace johns_subtraction_l1087_108766

theorem johns_subtraction 
  (a : ℕ) 
  (h₁ : (51 : ℕ)^2 = (50 : ℕ)^2 + 101) 
  (h₂ : (49 : ℕ)^2 = (50 : ℕ)^2 - b) 
  : b = 99 := 
by 
  sorry

end johns_subtraction_l1087_108766


namespace sum_of_divisors_117_l1087_108799

-- Defining the conditions in Lean
def n : ℕ := 117
def is_factorization : n = 3^2 * 13 := by rfl

-- The sum-of-divisors function can be defined based on the problem
def sum_of_divisors (n : ℕ) : ℕ :=
  (1 + 3 + 3^2) * (1 + 13)

-- Assertion of the correct answer
theorem sum_of_divisors_117 : sum_of_divisors n = 182 := by
  sorry

end sum_of_divisors_117_l1087_108799


namespace nested_sqrt_eq_two_l1087_108757

theorem nested_sqrt_eq_two (y : ℝ) (h : y = Real.sqrt (2 + y)) : y = 2 :=
by
  sorry

end nested_sqrt_eq_two_l1087_108757


namespace difference_is_1365_l1087_108780

-- Define the conditions as hypotheses
def difference_between_numbers (L S : ℕ) : Prop :=
  L = 1637 ∧ L = 6 * S + 5

-- State the theorem to prove the difference is 1365
theorem difference_is_1365 {L S : ℕ} (h₁ : L = 1637) (h₂ : L = 6 * S + 5) :
  L - S = 1365 :=
by
  sorry

end difference_is_1365_l1087_108780


namespace min_value_z_l1087_108703

theorem min_value_z : ∀ (x y : ℝ), ∃ z, z = 3 * x^2 + y^2 + 12 * x - 6 * y + 40 ∧ z = 19 :=
by
  intro x y
  use 3 * x^2 + y^2 + 12 * x - 6 * y + 40 -- Define z
  sorry -- Proof is skipped for now

end min_value_z_l1087_108703


namespace arithmetic_seq_slope_l1087_108792

theorem arithmetic_seq_slope {a : ℕ → ℤ} (h : a 2 - a 4 = 2) : ∃ a1 : ℤ, ∀ n : ℕ, a n = -n + (a 1) + 1 := 
by {
  sorry
}

end arithmetic_seq_slope_l1087_108792


namespace expression_positive_l1087_108753

theorem expression_positive (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) : 5 * a ^ 2 - 6 * a * b + 5 * b ^ 2 > 0 :=
by
  sorry

end expression_positive_l1087_108753


namespace total_min_waiting_time_total_max_waiting_time_total_expected_waiting_time_l1087_108759

variables (a b: ℕ) (n m: ℕ)

def C (x y : ℕ) : ℕ := x.choose y

def T_min (a n m : ℕ) : ℕ :=
  a * C n 2 + a * m * n + b * C m 2

def T_max (a n m : ℕ) : ℕ :=
  a * C n 2 + b * m * n + b * C m 2

def E_T (a b n m : ℕ) : ℕ :=
  C (n + m) 2 * ((b * m + a * n) / (m + n))

theorem total_min_waiting_time (a b : ℕ) : T_min 1 5 3 = 40 :=
  by sorry

theorem total_max_waiting_time (a b : ℕ) : T_max 1 5 3 = 100 :=
  by sorry

theorem total_expected_waiting_time (a b : ℕ) : E_T 1 5 5 3 = 70 :=
  by sorry

end total_min_waiting_time_total_max_waiting_time_total_expected_waiting_time_l1087_108759


namespace squared_product_l1087_108797

theorem squared_product (a b : ℝ) : (- (1 / 2) * a^2 * b)^2 = (1 / 4) * a^4 * b^2 := by 
  sorry

end squared_product_l1087_108797


namespace robert_ate_more_l1087_108768

variable (robert_chocolates : ℕ) (nickel_chocolates : ℕ)
variable (robert_ate_9 : robert_chocolates = 9) (nickel_ate_2 : nickel_chocolates = 2)

theorem robert_ate_more : robert_chocolates - nickel_chocolates = 7 :=
  by
    sorry

end robert_ate_more_l1087_108768


namespace option_D_is_negative_l1087_108715

theorem option_D_is_negative :
  let A := abs (-4)
  let B := -(-4)
  let C := (-4) ^ 2
  let D := -(4 ^ 2)
  D < 0 := by
{
  -- Place sorry here since we are not required to provide the proof
  sorry
}

end option_D_is_negative_l1087_108715


namespace areas_of_triangle_and_parallelogram_are_equal_l1087_108764

theorem areas_of_triangle_and_parallelogram_are_equal (b : ℝ) :
  let parallelogram_height := 100
  let triangle_height := 200
  let area_parallelogram := b * parallelogram_height
  let area_triangle := (1/2) * b * triangle_height
  area_parallelogram = area_triangle :=
by
  -- conditions
  let parallelogram_height := 100
  let triangle_height := 200
  let area_parallelogram := b * parallelogram_height
  let area_triangle := (1 / 2) * b * triangle_height
  -- relationship
  show area_parallelogram = area_triangle
  sorry

end areas_of_triangle_and_parallelogram_are_equal_l1087_108764


namespace students_at_1544_l1087_108717

noncomputable def students_in_lab : Nat := 44

theorem students_at_1544 :
  let initial_students := 20
  let enter_interval := 3
  let enter_students := 4
  let leave_interval := 10
  let leave_students := 8

  ∃ students : Nat,
    students = initial_students
    + (34 / enter_interval) * enter_students
    - (34 / leave_interval) * leave_students
    ∧ students = students_in_lab :=
by
  let initial_students := 20
  let enter_interval := 3
  let enter_students := 4
  let leave_interval := 10
  let leave_students := 8
  use 20 + (34 / 3) * 4 - (34 / 10) * 8
  sorry

end students_at_1544_l1087_108717


namespace paula_bracelets_count_l1087_108713

-- Defining the given conditions
def cost_bracelet := 4
def cost_keychain := 5
def cost_coloring_book := 3
def total_spent := 20

-- Defining the cost for Paula's items
def cost_paula (B : ℕ) := B * cost_bracelet + cost_keychain

-- Defining the cost for Olive's items
def cost_olive := cost_coloring_book + cost_bracelet

-- Defining the main problem
theorem paula_bracelets_count (B : ℕ) (h : cost_paula B + cost_olive = total_spent) : B = 2 := by
  sorry

end paula_bracelets_count_l1087_108713


namespace arcade_spending_fraction_l1087_108794

theorem arcade_spending_fraction (allowance remaining_after_arcade remaining_after_toystore: ℝ) (f: ℝ) : 
  allowance = 3.75 ∧
  remaining_after_arcade = (1 - f) * allowance ∧
  remaining_after_toystore = remaining_after_arcade - (1 / 3) * remaining_after_arcade ∧
  remaining_after_toystore = 1 →
  f = 3 / 5 :=
by
  sorry

end arcade_spending_fraction_l1087_108794


namespace faye_rows_l1087_108719

theorem faye_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h_total_pencils : total_pencils = 720)
  (h_pencils_per_row : pencils_per_row = 24) : 
  total_pencils / pencils_per_row = 30 := by 
  sorry

end faye_rows_l1087_108719


namespace quadrilateral_inscribed_circumscribed_l1087_108710

theorem quadrilateral_inscribed_circumscribed 
  (r R d : ℝ) --Given variables with their types
  (K O : Type) (radius_K : K → ℝ) (radius_O : O → ℝ) (dist : (K × O) → ℝ)  -- Defining circles properties
  (K_inside_O : ∀ p : K × O, radius_K p.fst < radius_O p.snd) 
  (dist_centers : ∀ p : K × O, dist p = d) -- Distance between the centers
  : 
  (1 / (R + d)^2) + (1 / (R - d)^2) = (1 / r^2) := 
by 
  sorry

end quadrilateral_inscribed_circumscribed_l1087_108710


namespace circle_circumference_difference_l1087_108754

theorem circle_circumference_difference (d_inner : ℝ) (h_inner : d_inner = 100) 
  (d_outer : ℝ) (h_outer : d_outer = d_inner + 30) :
  ((π * d_outer) - (π * d_inner)) = 30 * π :=
by 
  sorry

end circle_circumference_difference_l1087_108754


namespace fourth_vertex_l1087_108735

-- Define the given vertices
def vertex1 := (2, 1)
def vertex2 := (4, 1)
def vertex3 := (2, 5)

-- Define what it means to be a rectangle in this context
def is_vertical_segment (p1 p2 : ℕ × ℕ) : Prop :=
  p1.1 = p2.1

def is_horizontal_segment (p1 p2 : ℕ × ℕ) : Prop :=
  p1.2 = p2.2

def is_rectangle (v1 v2 v3 v4: (ℕ × ℕ)) : Prop :=
  is_vertical_segment v1 v3 ∧
  is_horizontal_segment v1 v2 ∧
  is_vertical_segment v2 v4 ∧
  is_horizontal_segment v3 v4 ∧
  is_vertical_segment v1 v4 ∧ -- additional condition to ensure opposite sides are equal
  is_horizontal_segment v2 v3

-- Prove the coordinates of the fourth vertex of the rectangle
theorem fourth_vertex (v4 : ℕ × ℕ) : 
  is_rectangle vertex1 vertex2 vertex3 v4 → v4 = (4, 5) := 
by
  intro h_rect
  sorry

end fourth_vertex_l1087_108735


namespace max_value_frac_sqrt_eq_sqrt_35_l1087_108734

theorem max_value_frac_sqrt_eq_sqrt_35 :
  ∀ x y : ℝ, 
  (x + 3 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 35 
  ∧ (∃ x y : ℝ, x = 2 / 5 ∧ y = 6 / 5 ∧ (x + 3 * y + 5) / Real.sqrt (x^2 + y^2 + 4) = Real.sqrt 35) :=
by {
  sorry
}

end max_value_frac_sqrt_eq_sqrt_35_l1087_108734


namespace sin_from_tan_l1087_108705

theorem sin_from_tan (A : ℝ) (h : Real.tan A = Real.sqrt 2 / 3) : 
  Real.sin A = Real.sqrt 22 / 11 := 
by 
  sorry

end sin_from_tan_l1087_108705


namespace simplify_and_evaluate_l1087_108702

noncomputable def expr (x : ℝ) : ℝ :=
  ((x^2 + x - 2) / (x - 2) - x - 2) / ((x^2 + 4 * x + 4) / x)

theorem simplify_and_evaluate : expr 1 = -1 / 3 :=
by
  sorry

end simplify_and_evaluate_l1087_108702


namespace math_problem_l1087_108737

theorem math_problem (x y n : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100)
  (hy_reverse : ∃ a b, x = 10 * a + b ∧ y = 10 * b + a) 
  (h_xy_square_sum : x^2 + y^2 = n^2) : x + y + n = 132 :=
sorry

end math_problem_l1087_108737


namespace lucas_purchase_l1087_108733

-- Define the variables and assumptions.
variables (a b c : ℕ)
variables (h1 : a + b + c = 50) (h2 : 50 * a + 400 * b + 500 * c = 10000)

-- Goal: Prove that the number of 50-cent items (a) is 30.
theorem lucas_purchase : a = 30 :=
by sorry

end lucas_purchase_l1087_108733


namespace new_remainder_when_scaled_l1087_108729

theorem new_remainder_when_scaled (a b c : ℕ) (h : a = b * c + 7) : (10 * a) % (10 * b) = 70 := by
  sorry

end new_remainder_when_scaled_l1087_108729


namespace bernoulli_inequality_l1087_108790

theorem bernoulli_inequality (x : ℝ) (n : ℕ) (hx : x > -1) (hn : n > 0) : 
  (1 + x) ^ n ≥ 1 + n * x := 
sorry

end bernoulli_inequality_l1087_108790


namespace combined_weight_of_parcels_l1087_108775

variable (x y z : ℕ)

theorem combined_weight_of_parcels : 
  (x + y = 132) ∧ (y + z = 135) ∧ (z + x = 140) → x + y + z = 204 :=
by 
  intros
  sorry

end combined_weight_of_parcels_l1087_108775


namespace power_mod_remainder_l1087_108763

theorem power_mod_remainder (a : ℕ) (n : ℕ) (h1 : 3^5 % 11 = 1) (h2 : 221 % 5 = 1) : 3^221 % 11 = 3 :=
by
  sorry

end power_mod_remainder_l1087_108763


namespace range_of_a_l1087_108789

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, e^x + 1/e^x > a) ∧ (∃ x : ℝ, x^2 + 8*x + a^2 = 0) ↔ (-4 ≤ a ∧ a < 2) :=
by
  sorry

end range_of_a_l1087_108789


namespace rhombus_area_is_160_l1087_108781

-- Define the values of the diagonals
def d1 : ℝ := 16
def d2 : ℝ := 20

-- Define the formula for the area of the rhombus
noncomputable def area_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

-- State the theorem to be proved
theorem rhombus_area_is_160 :
  area_rhombus d1 d2 = 160 :=
by
  sorry

end rhombus_area_is_160_l1087_108781


namespace inequality_proof_l1087_108709

theorem inequality_proof
  (a b c d e f : ℝ)
  (h : b^2 ≥ a^2 + c^2) :
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 := 
sorry

end inequality_proof_l1087_108709


namespace octal_rep_square_l1087_108784

theorem octal_rep_square (a b c : ℕ) (n : ℕ) (h : n^2 = 8^3 * a + 8^2 * b + 8 * 3 + c) (h₀ : a ≠ 0) : c = 1 :=
sorry

end octal_rep_square_l1087_108784


namespace inequality_count_l1087_108774

theorem inequality_count
  (x y a b : ℝ)
  (hx_pos : 0 < x)
  (hy_pos : 0 < y)
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (hx_lt_one : x < 1)
  (hy_lt_one : y < 1)
  (hx_lt_a : x < a)
  (hy_lt_b : y < b)
  (h_sum : x + y = a - b) :
  ({(x + y < a + b), (x - y < a - b), (x * y < a * b)}:Finset Prop).card = 3 :=
by
  sorry

end inequality_count_l1087_108774


namespace max_product_distance_l1087_108755

-- Definitions for the conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 4) = 1
def is_focus (F : ℝ × ℝ) : Prop := F = (3, 0) ∨ F = (-3, 0)

-- The theorem statement
theorem max_product_distance (M : ℝ × ℝ) (F1 F2 : ℝ × ℝ) 
  (h1 : ellipse M.1 M.2) 
  (h2 : is_focus F1) 
  (h3 : is_focus F2) : 
  (∃ x y, M = (x, y) ∧ ellipse x y) → 
  |(M.1 - F1.1)^2 + (M.2 - F1.2)^2| * |(M.1 - F2.1)^2 + (M.2 - F2.2)^2| ≤ 81 := 
sorry

end max_product_distance_l1087_108755


namespace sam_quarters_l1087_108707

theorem sam_quarters (pennies : ℕ) (total : ℝ) (value_penny : ℝ) (value_quarter : ℝ) (quarters : ℕ) :
  pennies = 9 →
  total = 1.84 →
  value_penny = 0.01 →
  value_quarter = 0.25 →
  quarters = (total - pennies * value_penny) / value_quarter →
  quarters = 7 :=
by
  intros
  sorry

end sam_quarters_l1087_108707


namespace total_amount_paid_l1087_108749

theorem total_amount_paid (g_p g_q m_p m_q : ℝ) (g_d g_t m_d m_t : ℝ) : 
    g_p = 70 -> g_q = 8 -> g_d = 0.05 -> g_t = 0.08 -> 
    m_p = 55 -> m_q = 9 -> m_d = 0.07 -> m_t = 0.11 -> 
    (g_p * g_q * (1 - g_d) * (1 + g_t) + m_p * m_q * (1 - m_d) * (1 + m_t)) = 1085.55 := by 
    sorry

end total_amount_paid_l1087_108749


namespace find_a7_l1087_108785

def arithmetic_seq (a₁ d : ℤ) (n : ℤ) : ℤ := a₁ + (n-1) * d

theorem find_a7 (a₁ d : ℤ)
  (h₁ : arithmetic_seq a₁ d 3 + arithmetic_seq a₁ d 7 - arithmetic_seq a₁ d 10 = -1)
  (h₂ : arithmetic_seq a₁ d 11 - arithmetic_seq a₁ d 4 = 21) :
  arithmetic_seq a₁ d 7 = 20 :=
by
  sorry

end find_a7_l1087_108785


namespace find_possible_values_l1087_108720

theorem find_possible_values (a b c k : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_zero : a + b + c = 0) :
  (k * a^2 * b^2 + k * a^2 * c^2 + k * b^2 * c^2) / 
  ((a^2 - b * c) * (b^2 - a * c) + 
   (a^2 - b * c) * (c^2 - a * b) + 
   (b^2 - a * c) * (c^2 - a * b)) 
  = k / 3 :=
by 
  sorry

end find_possible_values_l1087_108720


namespace kevin_hopped_distance_after_four_hops_l1087_108773

noncomputable def kevin_total_hopped_distance : ℚ :=
  let hop1 := 1
  let hop2 := 1 / 2
  let hop3 := 1 / 4
  let hop4 := 1 / 8
  hop1 + hop2 + hop3 + hop4

theorem kevin_hopped_distance_after_four_hops :
  kevin_total_hopped_distance = 15 / 8 :=
by
  sorry

end kevin_hopped_distance_after_four_hops_l1087_108773


namespace solve_arcsin_eq_l1087_108732

noncomputable def arcsin (x : ℝ) : ℝ := Real.arcsin x
noncomputable def pi : ℝ := Real.pi

theorem solve_arcsin_eq :
  ∃ x : ℝ, arcsin x + arcsin (3 * x) = pi / 4 ∧ x = 1 / Real.sqrt 19 :=
sorry

end solve_arcsin_eq_l1087_108732


namespace perpendicular_lines_a_value_l1087_108783

theorem perpendicular_lines_a_value (a : ℝ) :
  (∀ x y : ℝ, (a-2)*x + a*y = 1 ↔ 2*x + 3*y = 5) → a = 4/5 := by
sorry

end perpendicular_lines_a_value_l1087_108783


namespace Tom_completes_wall_l1087_108712

theorem Tom_completes_wall :
  let avery_rate_per_hour := (1:ℝ)/3
  let tom_rate_per_hour := (1:ℝ)/2
  let combined_rate_per_hour := avery_rate_per_hour + tom_rate_per_hour
  let portion_completed_together := combined_rate_per_hour * 1 
  let remaining_wall := 1 - portion_completed_together
  let time_for_tom := remaining_wall / tom_rate_per_hour
  time_for_tom = (1:ℝ)/3 := 
by 
  sorry

end Tom_completes_wall_l1087_108712


namespace sum_of_remainders_and_smallest_n_l1087_108782

theorem sum_of_remainders_and_smallest_n (n : ℕ) (h : n % 20 = 11) :
    (n % 4 + n % 5 = 4) ∧ (∃ (k : ℕ), k > 2 ∧ n = 20 * k + 11 ∧ n > 50) := by
  sorry

end sum_of_remainders_and_smallest_n_l1087_108782


namespace largest_decimal_of_four_digit_binary_l1087_108772

theorem largest_decimal_of_four_digit_binary : ∀ n : ℕ, (n < 16) → n ≤ 15 :=
by {
  -- conditions: a four-digit binary number implies \( n \) must be less than \( 2^4 = 16 \)
  sorry
}

end largest_decimal_of_four_digit_binary_l1087_108772


namespace systematic_sampling_l1087_108728

theorem systematic_sampling (E P: ℕ) (a b: ℕ) (g: ℕ) 
  (hE: E = 840)
  (hP: P = 42)
  (ha: a = 61)
  (hb: b = 140)
  (hg: g = E / P)
  (hEpos: 0 < E)
  (hPpos: 0 < P)
  (hgpos: 0 < g):
  (b - a + 1) / g = 4 := 
by
  sorry

end systematic_sampling_l1087_108728


namespace sequence_value_2016_l1087_108731

theorem sequence_value_2016 (a : ℕ → ℕ) (h₁ : a 1 = 0) (h₂ : ∀ n, a (n + 1) = a n + 2 * n) : a 2016 = 2016 * 2015 :=
by 
  sorry

end sequence_value_2016_l1087_108731


namespace difference_of_numbers_l1087_108786

theorem difference_of_numbers (a b : ℕ) (h1 : a = 2 * b) (h2 : (a + 4) / (b + 4) = 5 / 7) : a - b = 8 := 
by
  sorry

end difference_of_numbers_l1087_108786


namespace second_movie_duration_proof_l1087_108796

-- initial duration for the first movie (in minutes)
def first_movie_duration_minutes : ℕ := 1 * 60 + 48

-- additional duration for the second movie (in minutes)
def additional_duration_minutes : ℕ := 25

-- total duration for the second movie (in minutes)
def second_movie_duration_minutes : ℕ := first_movie_duration_minutes + additional_duration_minutes

-- convert total minutes to hours and minutes
def duration_in_hours_and_minutes (total_minutes : ℕ) : ℕ × ℕ :=
  (total_minutes / 60, total_minutes % 60)

theorem second_movie_duration_proof :
  duration_in_hours_and_minutes second_movie_duration_minutes = (2, 13) :=
by
  -- proof would go here
  sorry

end second_movie_duration_proof_l1087_108796


namespace geometric_sequence_arithmetic_condition_l1087_108798

noncomputable def geometric_sequence_ratio (q : ℝ) : Prop :=
  q > 0

def arithmetic_sequence (a₁ a₂ a₃ : ℝ) : Prop :=
  2 * a₃ = a₁ + 2 * a₂

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a 0 * q ^ n

theorem geometric_sequence_arithmetic_condition
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (q : ℝ)
  (hq : geometric_sequence_ratio q)
  (h_arith : arithmetic_sequence (a 0) (geometric_sequence a q 1) (geometric_sequence a q 2)) :
  (geometric_sequence a q 9 + geometric_sequence a q 10) / 
  (geometric_sequence a q 7 + geometric_sequence a q 8) = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequence_arithmetic_condition_l1087_108798


namespace monotonicity_of_f_range_of_k_for_three_zeros_l1087_108744

noncomputable def f (x k : ℝ) : ℝ := x^3 - k * x + k^2

def f_derivative (x k : ℝ) : ℝ := 3 * x^2 - k

theorem monotonicity_of_f (k : ℝ) : 
  (∀ x : ℝ, 0 <= f_derivative x k) ↔ k <= 0 :=
by sorry

theorem range_of_k_for_three_zeros : 
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 k = 0 ∧ f x2 k = 0 ∧ f x3 k = 0) ↔ (0 < k ∧ k < 4 / 27) :=
by sorry

end monotonicity_of_f_range_of_k_for_three_zeros_l1087_108744


namespace point_on_y_axis_l1087_108745

theorem point_on_y_axis (m n : ℝ) (h : (m, n).1 = 0) : m = 0 :=
by
  sorry

end point_on_y_axis_l1087_108745


namespace yulgi_allowance_l1087_108727

theorem yulgi_allowance (Y G : ℕ) (h₁ : Y + G = 6000) (h₂ : (Y + G) - (Y - G) = 4800) (h₃ : Y > G) : Y = 3600 :=
sorry

end yulgi_allowance_l1087_108727


namespace radius_increase_l1087_108704

-- Definitions and conditions
def initial_circumference : ℝ := 24
def final_circumference : ℝ := 30
def circumference_radius_relation (C : ℝ) (r : ℝ) : Prop := C = 2 * Real.pi * r

-- Required proof statement
theorem radius_increase (r1 r2 Δr : ℝ)
  (h1 : circumference_radius_relation initial_circumference r1)
  (h2 : circumference_radius_relation final_circumference r2)
  (h3 : Δr = r2 - r1) :
  Δr = 3 / Real.pi :=
by
  sorry

end radius_increase_l1087_108704


namespace smallest_possible_x2_plus_y2_l1087_108771

theorem smallest_possible_x2_plus_y2 (x y : ℝ) (h : (x + 3) * (y - 3) = 0) : x^2 + y^2 = 18 :=
sorry

end smallest_possible_x2_plus_y2_l1087_108771


namespace root_quadratic_eq_l1087_108721

theorem root_quadratic_eq (n m : ℝ) (h : n ≠ 0) (root_condition : n^2 + m * n + 3 * n = 0) : m + n = -3 :=
  sorry

end root_quadratic_eq_l1087_108721


namespace simplify_complex_expr_l1087_108743

theorem simplify_complex_expr : ∀ (i : ℂ), (4 - 2 * i) - (7 - 2 * i) + (6 - 3 * i) = 3 - 3 * i := by
  intro i
  sorry

end simplify_complex_expr_l1087_108743


namespace unoccupied_volume_correct_l1087_108751

-- Define the conditions given in the problem
def tank_length := 12 -- inches
def tank_width := 8 -- inches
def tank_height := 10 -- inches
def water_fraction := 1 / 3
def ice_cube_side := 1 -- inches
def num_ice_cubes := 12

-- Calculate the occupied volume
noncomputable def tank_volume : ℝ := tank_length * tank_width * tank_height
noncomputable def water_volume : ℝ := tank_volume * water_fraction
noncomputable def ice_cube_volume : ℝ := ice_cube_side^3
noncomputable def total_ice_volume : ℝ := ice_cube_volume * num_ice_cubes
noncomputable def total_occupied_volume : ℝ := water_volume + total_ice_volume

-- Calculate the unoccupied volume
noncomputable def unoccupied_volume : ℝ := tank_volume - total_occupied_volume

-- State the problem
theorem unoccupied_volume_correct : unoccupied_volume = 628 := by
  sorry

end unoccupied_volume_correct_l1087_108751


namespace inequality_proof_l1087_108708

theorem inequality_proof (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) : 
  1 + a^2 + b^2 > 3 * a * b := 
sorry

end inequality_proof_l1087_108708


namespace initial_water_amount_l1087_108767

theorem initial_water_amount (W : ℝ) 
  (evap_per_day : ℝ := 0.0008) 
  (days : ℤ := 50) 
  (percentage_evap : ℝ := 0.004) 
  (evap_total : ℝ := evap_per_day * days) 
  (evap_eq : evap_total = percentage_evap * W) : 
  W = 10 := 
by
  sorry

end initial_water_amount_l1087_108767


namespace age_of_oldest_child_l1087_108723

def average_age_of_children (a b c d : ℕ) : ℕ := (a + b + c + d) / 4

theorem age_of_oldest_child :
  ∀ (a b c d : ℕ), a = 6 → b = 9 → c = 12 → average_age_of_children a b c d = 9 → d = 9 :=
by
  intros a b c d h_a h_b h_c h_avg
  sorry

end age_of_oldest_child_l1087_108723


namespace larger_number_l1087_108793

theorem larger_number (HCF A B : ℕ) (factor1 factor2 : ℕ) (h_HCF : HCF = 23) (h_factor1 : factor1 = 14) (h_factor2 : factor2 = 15) (h_LCM : HCF * factor1 * factor2 = A * B) (h_A : A = HCF * factor2) (h_B : B = HCF * factor1) : A = 345 :=
by
  sorry

end larger_number_l1087_108793


namespace percent_absent_math_dept_l1087_108714

theorem percent_absent_math_dept (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
  (male_absent_fraction : ℚ) (female_absent_fraction : ℚ)
  (h1 : total_students = 160) 
  (h2 : male_students = 90) 
  (h3 : female_students = 70) 
  (h4 : male_absent_fraction = 1 / 5) 
  (h5 : female_absent_fraction = 2 / 7) :
  ((male_absent_fraction * male_students + female_absent_fraction * female_students) / total_students) * 100 = 23.75 :=
by
  sorry

end percent_absent_math_dept_l1087_108714


namespace tom_bought_new_books_l1087_108765

def original_books : ℕ := 5
def sold_books : ℕ := 4
def current_books : ℕ := 39

def new_books (original_books sold_books current_books : ℕ) : ℕ :=
  current_books - (original_books - sold_books)

theorem tom_bought_new_books :
  new_books original_books sold_books current_books = 38 :=
by
  sorry

end tom_bought_new_books_l1087_108765


namespace evaluate_expression_l1087_108741

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 :=
by sorry

end evaluate_expression_l1087_108741


namespace next_four_customers_cases_l1087_108724

theorem next_four_customers_cases (total_people : ℕ) (first_eight_cases : ℕ) (last_eight_cases : ℕ) (total_cases : ℕ) :
    total_people = 20 →
    first_eight_cases = 24 →
    last_eight_cases = 8 →
    total_cases = 40 →
    (total_cases - (first_eight_cases + last_eight_cases)) / 4 = 2 :=
by
  intro h1 h2 h3 h4
  -- Fill in the proof steps using h1, h2, h3, and h4
  sorry

end next_four_customers_cases_l1087_108724


namespace sum_boundary_values_of_range_l1087_108718

noncomputable def f (x : ℝ) : ℝ := 3 / (3 + 3 * x^2 + 6 * x)

theorem sum_boundary_values_of_range : 
  let c := 0
  let d := 1
  c + d = 1 :=
by
  sorry

end sum_boundary_values_of_range_l1087_108718


namespace sector_area_l1087_108736

-- Define the given parameters
def central_angle : ℝ := 2
def radius : ℝ := 3

-- Define the statement about the area of the sector
theorem sector_area (α r : ℝ) (hα : α = 2) (hr : r = 3) :
  let l := α * r
  let A := 0.5 * l * r
  A = 9 :=
by
  -- The proof is not required
  sorry

end sector_area_l1087_108736


namespace base9_minus_base6_to_decimal_l1087_108700

theorem base9_minus_base6_to_decimal :
  let b9 := 3 * 9^2 + 2 * 9^1 + 1 * 9^0
  let b6 := 2 * 6^2 + 5 * 6^1 + 4 * 6^0
  b9 - b6 = 156 := by
sorry

end base9_minus_base6_to_decimal_l1087_108700


namespace candy_distribution_l1087_108787

theorem candy_distribution (n : ℕ) (h : n ≥ 2) :
  (∀ i : ℕ, i < n → ∃ k : ℕ, ((k * (k + 1)) / 2) % n = i) ↔ ∃ k : ℕ, n = 2 ^ k :=
by
  sorry

end candy_distribution_l1087_108787


namespace no_14_consecutive_divisible_by_2_to_11_l1087_108746

theorem no_14_consecutive_divisible_by_2_to_11 :
  ¬ ∃ (a : ℕ), ∀ i, i < 14 → ∃ p, Nat.Prime p ∧ 2 ≤ p ∧ p ≤ 11 ∧ (a + i) % p = 0 :=
by sorry

end no_14_consecutive_divisible_by_2_to_11_l1087_108746
