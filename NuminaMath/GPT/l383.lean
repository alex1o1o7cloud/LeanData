import Mathlib

namespace september_first_wednesday_l383_383318

/-- Mathematical problem:
On which day of the week did September 1 fall that year?
Conditions:
1. Every Monday - 1 computer science lesson.
2. Every Tuesday - 2 computer science lessons.
3. Every Wednesday - 3 computer science lessons.
4. Every Thursday - 4 computer science lessons.
5. Every Friday - 5 computer science lessons.
6. No lessons on Saturdays and Sundays.
7. No lessons on September 1.
8. A total of 64 computer science lessons in September.
9. September has 30 days.
Conclusion: September 1 was a Wednesday.
--/
theorem september_first_wednesday :
  (∀ n : ℕ, (n >= 1 ∧ n <= 30) →
    let lessons :=
      (if (n % 7 = 1 ∧ n ≠ 1) then 1 else
      if (n % 7 = 2 ∧ n ≠ 1) then 2 else
      if (n % 7 = 3 ∧ n ≠ 1) then 3 else
      if (n % 7 = 4 ∧ n ≠ 1) then 4 else
      if (n % 7 = 5 ∧ n ≠ 1) then 5 else 0)
    in
    ∑ k in finset.range 30, lessons k = 64) →
  (if ((1 % 7) = 3) then true else false) :=
begin
  sorry -- Proof is omitted as per instructions
end

end september_first_wednesday_l383_383318


namespace find_value_of_m_l383_383195

open Real

theorem find_value_of_m (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : m = sqrt 10 := by
  sorry

end find_value_of_m_l383_383195


namespace diet_soda_count_l383_383096

theorem diet_soda_count (D : ℕ) (h1 : 81 = D + 21) : D = 60 := by
  sorry

end diet_soda_count_l383_383096


namespace angle_between_b_c_l383_383654

-- Define the conditions
variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

-- The given assumptions
axiom non_zero_a : a ≠ 0
axiom non_zero_b : b ≠ 0
axiom non_zero_c : c ≠ 0
axiom sum_eq_zero : a + 2 • b + 3 • c = 0
axiom dot_eq : ⟪a, b⟫ = ⟪b, c⟫ ∧ ⟪b, c⟫ = ⟪c, a⟫

-- The goal is to prove the angle θ between b and c is 3π/4
theorem angle_between_b_c : real.angle (b : V) (c : V) = real.pi * 3 / 4 :=
sorry

end angle_between_b_c_l383_383654


namespace multiple_of_x_remainder_l383_383422

theorem multiple_of_x_remainder
  (x : ℕ) (hx : x % 9 = 5) : ∃ k : ℕ, k * x % 9 = 2 :=
by
  use 4
  have hx' : x = 9 * (x / 9) + 5,
  {
    have h := Nat.div_add_mod x 9,
    rw [hx] at h,
    simpa [hx],
  }
  have h : (4 * x) % 9 = ((4 * (9 * (x / 9) + 5)) % 9),
  {
    rw [hx'],
    exact Nat.mod_eq_of_lt (Nat.zero_le _) (by linarith),
  }
  rw [mul_add, Nat.mul_mod, add_mod, Nat.mod_mul_right, add_zero] at h,
  show 4 * 5 % 9 = 2,
  {
    norm_num,
  }
  sorry

end multiple_of_x_remainder_l383_383422


namespace no_valid_placement_of_prisms_l383_383283

-- Definitions: Rectangular prism with edges parallel to OX, OY, and OZ axes.
structure RectPrism :=
  (x_interval : Set ℝ)
  (y_interval : Set ℝ)
  (z_interval : Set ℝ)

-- Function to determine if two rectangular prisms intersect.
def intersects (P Q : RectPrism) : Prop :=
  ¬ Disjoint P.x_interval Q.x_interval ∧
  ¬ Disjoint P.y_interval Q.y_interval ∧
  ¬ Disjoint P.z_interval Q.z_interval

-- Definition of the 12 rectangular prisms
def prisms := Fin 12 → RectPrism

-- Conditions for intersection:
def intersection_condition (prisms : prisms) : Prop :=
  ∀ i : Fin 12, ∀ j : Fin 12,
    (j = (i + 1) % 12) ∨ (j = (i - 1 + 12) % 12) ∨ intersects (prisms i) (prisms j)

theorem no_valid_placement_of_prisms :
  ¬ ∃ (prisms : prisms), intersection_condition prisms :=
sorry

end no_valid_placement_of_prisms_l383_383283


namespace total_visitors_is_correct_l383_383926

noncomputable def october_visitors := 100

noncomputable def november_first_half_visitors := october_visitors * 1.15

noncomputable def november_second_half_visitors := november_first_half_visitors * 0.9

noncomputable def november_weighted_average := (november_first_half_visitors + november_second_half_visitors) / 2

noncomputable def december_first_half_visitors := november_weighted_average * 0.9

noncomputable def december_second_half_visitors := december_first_half_visitors + 20

noncomputable def december_weighted_average := (december_first_half_visitors + december_second_half_visitors) / 2

noncomputable def total_visitors := october_visitors + november_weighted_average + december_weighted_average

theorem total_visitors_is_correct : total_visitors = 318 := by
  sorry

end total_visitors_is_correct_l383_383926


namespace product_of_odd_integers_lt_500_l383_383020

theorem product_of_odd_integers_lt_500 :
  (∏ i in Finset.range(500) | Odd i, (i:ℕ)) = 500! / (2^250 * 250!) :=
sorry

end product_of_odd_integers_lt_500_l383_383020


namespace max_s_family_size_6_elements_l383_383789

-- Definitions based on conditions:
def is_s_family (R : Set ℕ) (F : Set (Set ℕ)) : Prop :=
  (∀ X ∈ F, ∀ Y ∈ F, X ≠ Y → ¬ (X ⊆ Y)) ∧
  (∀ X ∈ F, ∀ Y ∈ F, ∀ Z ∈ F, X ∪ Y ∪ Z ≠ R) ∧
  (⋃₀ F = R)

-- Proof statement:
theorem max_s_family_size_6_elements {R : Set ℕ} (hR : R.card = 6) :
  ∃ F : Set (Set ℕ), is_s_family R F ∧ F.card = 11 :=
sorry

end max_s_family_size_6_elements_l383_383789


namespace Sanji_received_86_coins_l383_383634

noncomputable def total_coins := 280

def Jack_coins (x : ℕ) := x
def Jimmy_coins (x : ℕ) := x + 11
def Tom_coins (x : ℕ) := x - 15
def Sanji_coins (x : ℕ) := x + 20

theorem Sanji_received_86_coins (x : ℕ) (hx : Jack_coins x + Jimmy_coins x + Tom_coins x + Sanji_coins x = total_coins) : Sanji_coins x = 86 :=
sorry

end Sanji_received_86_coins_l383_383634


namespace hypotenuse_length_l383_383500

theorem hypotenuse_length (a b : ℤ) (h₀ : a = 15) (h₁ : b = 36) : 
  ∃ c : ℤ, c^2 = a^2 + b^2 ∧ c = 39 := 
by {
  sorry
}

end hypotenuse_length_l383_383500


namespace problem_a_problem_b_l383_383315

-- Define the board and domino properties
structure Board (m n : ℕ) := 
  (isFree : ℕ × ℕ → Prop) -- Indicates whether a cell is free
  (dominoOccupied : (ℕ × ℕ) → (ℕ × ℕ) → Prop) -- Indicates two adjacent cells occupied by a domino

def validDominoPlacement (b : Board) : Prop := 
  ∀ (p1 p2 : ℕ × ℕ), b.dominoOccupied p1 p2 → (isAdjacent p1 p2 ∧ b.isFree p1 = false ∧ b.isFree p2 = false)

def isAdjacent (p1 p2 : ℕ × ℕ) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p1.2 + 1 = p2.2)) ∨ ((p1.1 + 1 = p2.1 ∨ p1.1 = p2.1 + 1) ∧ p1.2 = p2.2)

def canGoFromBottomLeftToTopRight (m n : ℕ) (b : Board m n) : Prop := 
  ∃ (path : list (ℕ × ℕ)), (path.head = (0, 0) ∧ path.last = (m - 1, n - 1)) ∧
    (∀ i, i < path.length - 1 → isAdjacent (path.nth_le i sorry) (path.nth_le (i+1) sorry)) ∧
    (∀ p, p ∈ path → b.isFree p)

-- Problem statements
theorem problem_a (m : ℕ := 1001) (n : ℕ := 101) (b : Board m n) (h : validDominoPlacement b) : ¬ canGoFromBottomLeftToTopRight m n b :=
sorry

theorem problem_b (m : ℕ := 1001) (n : ℕ := 100) (b : Board m n) (h : validDominoPlacement b) : canGoFromBottomLeftToTopRight m n b :=
sorry

end problem_a_problem_b_l383_383315


namespace inscribed_square_ratio_l383_383564

-- Define the problem context:
variables {x y : ℝ}

-- Conditions on the triangles and squares:
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ a > 0 ∧ b > 0 ∧ c > 0

def inscribed_square_first_triangle (a b c x : ℝ) : Prop :=
  is_right_triangle a b c ∧ a = 5 ∧ b = 12 ∧ c = 13 ∧
  x = 60 / 17

def inscribed_square_second_triangle (d e f y : ℝ) : Prop :=
  is_right_triangle d e f ∧ d = 6 ∧ e = 8 ∧ f = 10 ∧
  y = 25 / 8

-- Lean theorem to be proven with given conditions:
theorem inscribed_square_ratio :
  inscribed_square_first_triangle 5 12 13 x →
  inscribed_square_second_triangle 6 8 10 y →
  x / y = 96 / 85 := by
  sorry

end inscribed_square_ratio_l383_383564


namespace boys_in_order_l383_383606

open Fin

def solution (order : Fin 8 → Type) : Prop :=
  -- Dima's number is three times Ilya's number
  (∃ d i, order d = "Dima" ∧ order i = "Ilya" ∧ 3 * i.val.succ = d.val.succ) ∧

  -- Fedya stood somewhere after the third boy but before Kolya
  (∃ f k, order f = "Fedya" ∧ order k = "Kolya" ∧ 3 < f.val.succ ∧ f.val.succ < k.val.succ) ∧

  -- Vasya's number is half of Petya's number
  (∃ v p, order v = "Vasya" ∧ order p = "Petya" ∧ 2 * v.val.succ = p.val.succ) ∧

  -- The fourth boy is immediately after Tema and somewhere before Petya
  (∃ t f p, order f = "4th" ∧ order t = "Tema" ∧ order p = "Petya"
            ∧ f.val.succ = t.val.succ + 1 ∧ f.val.succ < p.val.succ) ∧

  -- The order is as determined in the solution
  (order 0 = "Egor" ∧ order 1 = "Ilya" ∧ order 2 = "Tema" ∧ order 3 = "Vasya"
   ∧ order 4 = "Fedya" ∧ order 5 = "Dima" ∧ order 6 = "Kolya" ∧ order 7 = "Petya")

theorem boys_in_order : ∃ order : Fin 8 → Type, solution order :=
by
  sorry

end boys_in_order_l383_383606


namespace salary_spent_each_week_l383_383881

theorem salary_spent_each_week 
  (S : ℝ)
  (hsalary_nonneg: 0 ≤ S)
  (h_spent_first_week : S / 4 )
  (h_unspent_end_month : 0.15 * S ) :
  ∃ (x : ℝ), x = 0.20 * S ∧ 0.85 * S - S / 4 = 3 * x :=
by
  sorry

end salary_spent_each_week_l383_383881


namespace vector_properties_l383_383231

/-- The vectors a, b, and c used in the problem. --/
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-4, 2)
def c : ℝ × ℝ := (1, 2)

theorem vector_properties :
  ((∃ k : ℝ, b = k • a) ∧ (b.1 * c.1 + b.2 * c.2 = 0) ∧ (a.1*a.1 + a.2*a.2 = c.1*c.1 + c.2*c.2)) :=
  by sorry

end vector_properties_l383_383231


namespace hyperbola_standard_equation_l383_383201

theorem hyperbola_standard_equation :
  ∃ (k : ℝ), 
    let equation := (λ x y : ℝ, y^2 / 4 - x^2 / 9 = k) in
        (
            equation 3 (2 * Real.sqrt 2)
            ∧
            (∀ x y : ℝ, equation x y → (x / y = 2 / 3 ∨ x / y = -2 / 3))
            ∧
            (k = -1)
        )
  → equation = (λ x y : ℝ, y^2 / 4 - x^2 / 9 = 1) :=
sorry

end hyperbola_standard_equation_l383_383201


namespace count_yellow_balls_l383_383445

theorem count_yellow_balls (total white green yellow red purple : ℕ) (prob : ℚ)
  (h_total : total = 100)
  (h_white : white = 50)
  (h_green : green = 30)
  (h_red : red = 9)
  (h_purple : purple = 3)
  (h_prob : prob = 0.88) :
  yellow = 8 :=
by
  -- The proof will be here
  sorry

end count_yellow_balls_l383_383445


namespace coefficient_of_x3_l383_383276

def polynomial_expansion (f : ℚ[X]) : Prop :=
  f = (X - X^2 + 1)^20

theorem coefficient_of_x3 (f : ℚ[X]) (h : polynomial_expansion f) :
  coeff f 3 = 760 :=
sorry

end coefficient_of_x3_l383_383276


namespace last_digit_of_N_l383_383572

def sum_of_first_n_natural_numbers (N : ℕ) : ℕ :=
  N * (N + 1) / 2

theorem last_digit_of_N (N : ℕ) (h : sum_of_first_n_natural_numbers N = 3080) :
  N % 10 = 8 :=
by {
  sorry
}

end last_digit_of_N_l383_383572


namespace max_area_square_l383_383330

theorem max_area_square (P : ℝ) : 
  ∀ x y : ℝ, 2 * x + 2 * y = P → (x * y ≤ (P / 4) ^ 2) :=
by
  sorry

end max_area_square_l383_383330


namespace janet_hourly_wage_l383_383285

theorem janet_hourly_wage : 
  ∃ x : ℝ, 
    (20 * x + (5 * 20 + 7 * 20) = 1640) ∧ 
    x = 70 :=
by
  use 70
  sorry

end janet_hourly_wage_l383_383285


namespace calculate_revolutions_l383_383587

def wheel_diameter : ℝ := 8
def distance_traveled_miles : ℝ := 0.5
def feet_per_mile : ℝ := 5280
def distance_traveled_feet : ℝ := distance_traveled_miles * feet_per_mile

theorem calculate_revolutions :
  let radius : ℝ := wheel_diameter / 2
  let circumference : ℝ := 2 * Real.pi * radius
  let revolutions : ℝ := distance_traveled_feet / circumference
  revolutions = 330 / Real.pi := by
  sorry

end calculate_revolutions_l383_383587


namespace eight_non_overlapping_squares_l383_383325

theorem eight_non_overlapping_squares (A B C D A₁ B₁ C₁ D₁ : Type)
  (h1 : centered A B C D A₁ B₁ C₁ D₁)
  (h2 : twice_as_long A B C D A₁ B₁ C₁ D₁)
  (h3 : non_overlapping_squares A B C D A₁ B₁ C₁ D₁) :
  at_most_eight_squares A B C D A₁ B₁ C₁ D₁ :=
sorry

end eight_non_overlapping_squares_l383_383325


namespace matrix_invertible_iff_square_free_l383_383980

def is_square_free (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 1 → k^2 ∣ n → False

def matrix_M_entry (i j : ℕ) : ℕ :=
  if j ∣ (i + 1) then 1 else 0

def matrix_M (n : ℕ) : Matrix (Fin n) (Fin n) ℕ :=
  λ i j => matrix_M_entry i j

theorem matrix_invertible_iff_square_free (n : ℕ) :
  (∃ M_inv : Matrix (Fin n) (Fin n) ℕ, matrix_M n ⬝ M_inv = 1) ↔ is_square_free (n + 1) :=
by
  sorry

end matrix_invertible_iff_square_free_l383_383980


namespace hypotenuse_length_l383_383497

theorem hypotenuse_length (a b : ℤ) (h₀ : a = 15) (h₁ : b = 36) : 
  ∃ c : ℤ, c^2 = a^2 + b^2 ∧ c = 39 := 
by {
  sorry
}

end hypotenuse_length_l383_383497


namespace sum_of_cubes_modulo_l383_383629

def d (n : ℕ) : ℕ :=
  (nat.binary_digits n).sum

def S : ℕ :=
  (finset.range 2020).sum (λ k, (-1) ^ (d k) * k ^ 3)

theorem sum_of_cubes_modulo :
  S % 2020 = 100 :=
sorry

end sum_of_cubes_modulo_l383_383629


namespace maximize_garden_area_l383_383467

-- Conditions
def garden_side_parallel_to_house_max_area (total_fence_cost : ℝ) (fence_cost_per_foot : ℝ) (length_of_house_side : ℝ)
    (side_perpendicular_to_house : ℝ) : ℝ :=
  let total_fence_length := total_fence_cost / fence_cost_per_foot
  let side_parallel_to_house := total_fence_length - 2 * side_perpendicular_to_house
  side_parallel_to_house

theorem maximize_garden_area :
  ∃ side_parallel_to_house_max : ℝ,
    side_parallel_to_house_max = garden_side_parallel_to_house_max_area 2000 10 500 50 → side_parallel_to_house_max = 100 :=
begin
  sorry
end

end maximize_garden_area_l383_383467


namespace solution_set_f_x_minus_1_lt_0_l383_383712

noncomputable def f (x : ℝ) : ℝ :=
if h : x ≥ 0 then x - 1 else -x - 1

theorem solution_set_f_x_minus_1_lt_0 :
  {x : ℝ | f (x - 1) < 0} = {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

end solution_set_f_x_minus_1_lt_0_l383_383712


namespace first_three_digits_of_exp_l383_383932

def firstThreeDigits (x : Real) : Nat :=
  let decimalPart := x - Real.floor x
  (Real.floor (decimalPart * 1000)).natAbs

theorem first_three_digits_of_exp:
  firstThreeDigits ((10 ^ 1001 + 1) ^ (14 / 3)) = 40 := 
  sorry

end first_three_digits_of_exp_l383_383932


namespace floor_euler_number_l383_383966

theorem floor_euler_number : ⌊Real.e⌋ = 2 := by
  sorry

end floor_euler_number_l383_383966


namespace number_of_valid_numbers_l383_383004

def is_valid_digit (d : ℕ) : Prop := d < 10

def is_valid_number (A B C : ℕ) : Prop :=
  100 ≤ A * 100 + B * 10 + C ∧ A % 2 = 0 ∧ is_valid_digit B ∧ is_valid_digit C ∧ B = 2 * C - A

theorem number_of_valid_numbers : ∑ A in {2, 4, 6, 8}, ∑ C in (0:ℕ)..10, if is_valid_number A (2*C - A) C then 1 else 0 = 21 :=
sorry

end number_of_valid_numbers_l383_383004


namespace toothpicks_needed_l383_383385

-- Definitions of the conditions
def height := 15
def width := 8

def total_horizontal_toothpicks := (height + 1) * width
def total_vertical_toothpicks := (width + 1) * height
def total_diagonal_toothpicks := height * width

def total_toothpicks := total_horizontal_toothpicks + total_vertical_toothpicks + total_diagonal_toothpicks

theorem toothpicks_needed : total_toothpicks = 383 :=
by
  sorry

end toothpicks_needed_l383_383385


namespace card_star_PQ_l383_383774

def P : Set ℕ := {3, 4, 5}
def Q : Set ℕ := {4, 5, 6, 7}

def star (A B : Set ℕ) : Set (ℕ × ℕ) := {p | p.1 ∈ A ∧ p.2 ∈ B}

theorem card_star_PQ : (star P Q).card = 12 := by
  have hP : P.card = 3 := by sorry
  have hQ : Q.card = 4 := by sorry
  calc
    (star P Q).card = P.card * Q.card := by sorry
    ... = 3 * 4 := by sorry
    ... = 12 := by sorry

end card_star_PQ_l383_383774


namespace borrowed_money_rate_is_4_percent_l383_383105

noncomputable def borrowed_rate (P L T G: ℝ) (R_loan R_lend: ℝ) :=
  let Interest_lend := P * (R_lend / 100) * T
  let Total_gain := G * T
  let Interest_borrow := Interest_lend - Total_gain
  let R_loan := (Interest_borrow * 100) / (P * T)
  R_loan

theorem borrowed_money_rate_is_4_percent :
  ∀ (P L T G R_lend: ℝ),
    P = 5000 →
    T = 2 →
    R_lend = 6 →
    G = 100 →
    borrowed_rate P L T G 4 R_lend = 4 := 
by
  intros P L T G R_lend h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  unfold borrowed_rate
  norm_num
  sorry

end borrowed_money_rate_is_4_percent_l383_383105


namespace even_perfect_squares_count_l383_383950

theorem even_perfect_squares_count (b : ℕ) : 
  ∃ k : ℕ, (k < 64 ∧ (∃ a : ℕ, even a ∧ a^2 < 16000 ∧ a^2 = 4 * (b + 1))) := 
sorry

end even_perfect_squares_count_l383_383950


namespace paul_initial_tickets_l383_383135

theorem paul_initial_tickets (tickets_spent : ℕ) (tickets_left : ℕ) (h1 : tickets_spent = 3) (h2 : tickets_left = 8) : tickets_spent + tickets_left = 11 :=
by
  rw [h1, h2]
  exact rfl

end paul_initial_tickets_l383_383135


namespace value_of_inverse_squared_l383_383243

noncomputable def f (x : ℝ) : ℝ := 25 / (7 + 4 * x)
noncomputable def f_inv (y : ℝ) : ℝ := (25 / y - 7) / 4

theorem value_of_inverse_squared : (f_inv 3)⁻² = 9 := by
  sorry

end value_of_inverse_squared_l383_383243


namespace circles_in_parabola_sum_of_areas_l383_383149

noncomputable def sum_of_areas_of_circles (a : ℝ) (n : ℕ) : ℝ :=
  (π / (6 * a^2)) * n * (n + 1) * (2 * n + 1)

theorem circles_in_parabola_sum_of_areas (a b c : ℝ) (n : ℕ) :
  -- Given sequence of circles inscribed in the parabola and externally tangent.
  -- Each C_{k} is externally tangent to C_{k-1}.
  -- C_{1} is inscribed in the parabola.
  ∃ (C : ℕ → ℝ) (C_radius : ℕ → ℝ),
    (∀ k : ℕ, 1 ≤ k → 
      (let r_k := C_radius k 
      in ∃ (x y : ℝ), (x + b / (2 * a))^2 + (y - k * r_k)^2 = r_k^2 
      ∧ (ax^2 + bx + c = y)) ∧ 
      C k = r_k 
      ∧ (k ≥ 2 → r_k + C_radius (k - 1) = |k * r_k - (k - 1) * (C_radius (k - 1)))) →
  -- The sum of areas of the first n circles C_1 through C_n:
  ∑ k in finset.range n, π * (C_radius (k + 1))^2 = sum_of_areas_of_circles a n :=
begin
  sorry
end

end circles_in_parabola_sum_of_areas_l383_383149


namespace right_triangle_hypotenuse_length_l383_383524

theorem right_triangle_hypotenuse_length :
  ∀ (a b h : ℕ), a = 15 → b = 36 → h^2 = a^2 + b^2 → h = 39 :=
by
  intros a b h ha hb hyp
  -- In the proof, we would use ha, hb, and hyp to show h = 39
  sorry

end right_triangle_hypotenuse_length_l383_383524


namespace income_exceeds_repayment_after_9_years_cumulative_payment_up_to_year_8_l383_383122

-- Define the conditions
def annual_income (year : ℕ) : ℝ := 0.0124 * (1 + 0.2) ^ (year - 1)
def annual_repayment : ℝ := 0.05

-- Proof Problem 1: Show that the subway's annual operating income exceeds the annual repayment at year 9
theorem income_exceeds_repayment_after_9_years :
  ∀ n ≥ 9, annual_income n > annual_repayment :=
by
  sorry

-- Define the cumulative payment function for the municipal government
def cumulative_payment (years : ℕ) : ℝ :=
  (annual_repayment * years) - (List.sum (List.map annual_income (List.range years)))

-- Proof Problem 2: Show the cumulative payment by the municipal government up to year 8 is 19,541,135 RMB
theorem cumulative_payment_up_to_year_8 :
  cumulative_payment 8 = 0.1954113485 :=
by
  sorry

end income_exceeds_repayment_after_9_years_cumulative_payment_up_to_year_8_l383_383122


namespace hyperbola_eq_from_focus_and_eccentricity_l383_383213

theorem hyperbola_eq_from_focus_and_eccentricity
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : a = 3)
  (h4 : a^2 + b^2 = 10)
  (h5 : ∃ (e : ℝ), e = (c : ℝ) / a ∧ c = sqrt 10 ∧ e = sqrt 10 / 3) :
  ∃ (f : ℝ → ℝ → Prop), f x y ↔ (x^2 / 9 - y^2 = 1) :=
by
  sorry

end hyperbola_eq_from_focus_and_eccentricity_l383_383213


namespace simplify_expression_l383_383332

theorem simplify_expression (x : ℝ) : 
  (x^3 * x^2 * x + (x^3)^2 + (-2 * x^2)^3) = -6 * x^6 := 
by 
  sorry

end simplify_expression_l383_383332


namespace part1_l383_383675

def f (x : ℝ) : ℝ := abs (x + 2) + abs (x - 3)

theorem part1 (x : ℝ) : f(x) ≥ 2 * x ↔ x ∈ Set.Iic (5 / 2) := sorry

end part1_l383_383675


namespace find_smallest_N_l383_383187

-- Define the sum of digits functions as described
def sum_of_digits_base (n : ℕ) (b : ℕ) : ℕ :=
  n.digits b |>.sum

-- Define f(n) which is the sum of digits in base-five representation of n
def f (n : ℕ) : ℕ :=
  sum_of_digits_base n 5

-- Define g(n) which is the sum of digits in base-seven representation of f(n)
def g (n : ℕ) : ℕ :=
  sum_of_digits_base (f n) 7

-- The statement of the problem: find the smallest N such that 
-- g(N) in base-sixteen cannot be represented using only digits 0 to 9
theorem find_smallest_N : ∃ N : ℕ, (g N ≥ 10) ∧ (N % 1000 = 610) :=
by
  sorry

end find_smallest_N_l383_383187


namespace hypotenuse_right_triangle_l383_383549

theorem hypotenuse_right_triangle (a b : ℕ) (h1 : a = 15) (h2 : b = 36) :
  ∃ c, c ^ 2 = a ^ 2 + b ^ 2 ∧ c = 39 :=
by
  sorry

end hypotenuse_right_triangle_l383_383549


namespace smallest_sum_of_five_consecutive_primes_divisible_by_three_l383_383626

-- Definition of the conditions
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (a b c d e : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧
  (b = a + 1 ∨ b = a + 2) ∧ (c = b + 1 ∨ c = b + 2) ∧
  (d = c + 1 ∨ d = c + 2) ∧ (e = d + 1 ∨ e = d + 2)

theorem smallest_sum_of_five_consecutive_primes_divisible_by_three :
  ∃ a b c d e, consecutive_primes a b c d e ∧ a + b + c + d + e = 39 ∧ 39 % 3 = 0 :=
sorry

end smallest_sum_of_five_consecutive_primes_divisible_by_three_l383_383626


namespace general_term_of_arithmetic_sequence_sum_of_first_n_terms_l383_383995

variable {a_n : ℕ → ℕ}

noncomputable def arithmetic_sequence (a_1 : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a_1 + (n - 1) * d

variable (a1 d : ℕ)
variable (a : ℕ → ℕ := arithmetic_sequence a1 d)

theorem general_term_of_arithmetic_sequence
  (h1 : a1 = 1)
  (h_common_difference_nonzero : d ≠ 0)
  (h_geometric_sequence : (a 1) * (a 9) = (a 3) ^ 2) :
  ∀ n, a n = n :=
by
  sorry

noncomputable def geometric_sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in Finset.range n+1, 2 ^ (a i)

theorem sum_of_first_n_terms
  (h_arithmetic : ∀ n, a n = n ) :
  ∀ n, geometric_sequence_sum a n = 2^(n + 1) - 2 :=
by
  sorry

end general_term_of_arithmetic_sequence_sum_of_first_n_terms_l383_383995


namespace unique_positions_l383_383458

-- Define a structure representing the position of the flea on the Cartesian plane.
structure Position where
  x : ℤ
  y : ℤ

-- Initialize the initial position.
def initialPosition : Position := ⟨0, 0⟩

-- Define a function representing the length of each jump.
def jumpLength (i : ℕ) : ℤ := 2 ^ (i - 1)

-- Define the main theorem statement.
theorem unique_positions (n : ℕ) (finalPosition : Position) :
  ∃! positions : Fin n → Position, 
    positions 0 = initialPosition ∧
    all_steps_valid positions finalPosition := sorry

-- Define a function to check if all step transitions are valid.
def all_steps_valid (positions : Fin n → Position) (finalPosition : Position) : Prop :=
  ∀ i, 
    let d_x := positions (i + 1).x - positions i.x in
    let d_y := positions (i + 1).y - positions i.y in
    ((d_x = 0 ∨ d_x = 2^i ∨ d_x = -2^i) ∧ (d_y = 0 ∨ d_y = 2^i ∨ d_y = -2^i)) ∧
    (positions n = finalPosition)


end unique_positions_l383_383458


namespace distance_between_trucks_l383_383388

-- Define the speeds of the trucks in kilometers per hour
def speedA_kmh : ℝ := 54
def speedB_kmh : ℝ := 72

-- Convert speeds to meters per second
def kmh_to_mps (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600

-- Speeds in meters per second
def speedA_mps : ℝ := kmh_to_mps speedA_kmh
def speedB_mps : ℝ := kmh_to_mps speedB_kmh

-- Time elapsed in seconds
def time_seconds : ℝ := 30

-- Distances traveled by each truck
def distanceA : ℝ := speedA_mps * time_seconds
def distanceB : ℝ := speedB_mps * time_seconds

-- The total distance between the two trucks
def total_distance : ℝ := distanceA + distanceB

-- Theorem to prove the total distance is 1050 meters
theorem distance_between_trucks : total_distance = 1050 := by
  sorry

end distance_between_trucks_l383_383388


namespace fourth_point_of_intersection_l383_383752

variable (a b r : ℝ)

def is_on_curve (p : ℝ × ℝ) : Prop := p.1 * p.2 = 2

def is_on_circle (p : ℝ × ℝ) : Prop := (p.1 - a) ^ 2 + (p.2 - b) ^ 2 = r ^ 2

theorem fourth_point_of_intersection (h1 : is_on_circle (4, 1/2))
                                     (h2 : is_on_circle (-2, -1))
                                     (h3 : is_on_circle (2/3, 3))
                                     (h_curve1 : is_on_curve (4, 1/2))
                                     (h_curve2 : is_on_curve (-2, -1))
                                     (h_curve3 : is_on_curve (2/3, 3)) :
  ∃ p : ℝ × ℝ, is_on_curve p ∧ is_on_circle p ∧ p ≠ (4, 1/2) ∧ p ≠ (-2, -1) ∧ p ≠ (2/3, 3) ∧ p = (-3/4, -8/3).

end fourth_point_of_intersection_l383_383752


namespace smallest_value_not_f1_l383_383304

-- Define the function and its properties
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Given condition
def symmetric_about_2 (a b c : ℝ) := ∀ t : ℝ, f(a, b, c, 2 + t) = f(a, b, c, 2 - t)

-- Prove that the smallest value among f(-1), f(1), f(2), f(5) cannot be f(1).
theorem smallest_value_not_f1 (a b c : ℝ) (h : symmetric_about_2 a b c) :
  ¬ (min (min (min (f a b c (-1)) (f a b c 1)) (f a b c 2)) (f a b c 5) = f a b c 1) :=
sorry

end smallest_value_not_f1_l383_383304


namespace factorial_lcm_product_l383_383817

open Nat

theorem factorial_lcm_product (n : ℕ) (h : 0 < n) :
  factorial n = ∏ i in range (n + 1), lcm (Finset.range (i + 1)).gcd :=
sorry

end factorial_lcm_product_l383_383817


namespace maximize_expression_l383_383277

theorem maximize_expression :
  ∀ (a b c d e : ℕ),
    a ≠ b → a ≠ c → a ≠ d → a ≠ e → b ≠ c → b ≠ d → b ≠ e → c ≠ d → c ≠ e → d ≠ e →
    (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 6) → 
    (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 ∨ b = 6) →
    (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 6) →
    (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6) →
    (e = 1 ∨ e = 2 ∨ e = 3 ∨ e = 4 ∨ e = 6) →
    ((a : ℚ) / 2 + (d : ℚ) / e * (c / b)) ≤ 9 :=
by
  sorry

end maximize_expression_l383_383277


namespace clock_meets_22_times_per_day_l383_383101

open Real

-- Define the clock hand movements
def hour_hand_degrees_per_minute : ℝ := 0.5
def minute_hand_degrees_per_minute : ℝ := 6
def total_degrees_in_circle : ℝ := 360

-- Define the equation for the hands meeting every x minutes
def hands_meeting_equation (x : ℝ) : Prop :=
  hour_hand_degrees_per_minute * x + total_degrees_in_circle = minute_hand_degrees_per_minute * x

-- Define the number of minutes in a day
def minutes_in_a_day : ℝ := 24 * 60

-- The main theorem statement
theorem clock_meets_22_times_per_day :
  let x := 720 / 11 in
  (minutes_in_a_day / x).floor = 22 :=
by
  -- This is where the proof would go, but we'll add sorry to skip it.
  sorry

end clock_meets_22_times_per_day_l383_383101


namespace ellipse_equivalence_l383_383627

open Real

-- Defining the given ellipse
def given_ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 4) = 1

-- Defining the constants from the given ellipse
def a_sq : ℝ := 9
def b_sq : ℝ := 4
def c_sq : ℝ := a_sq - b_sq
def c : ℝ := sqrt c_sq

-- Given eccentricity
def eccentricity : ℝ := sqrt 5 / 5

-- New ellipse parameters A and B
def a : ℝ := 5
def b_sq_new : ℝ := a^2 - c^2

-- Required ellipse equation
def required_ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b_sq_new) = 1

theorem ellipse_equivalence :
  ∀ (x y : ℝ),
    given_ellipse x y → required_ellipse x y :=
by
  sorry

end ellipse_equivalence_l383_383627


namespace spheres_do_not_protrude_l383_383457

-- Define the basic parameters
variables (R r : ℝ) (h_cylinder : ℝ) (h_cone : ℝ)
-- Assume conditions
axiom cylinder_height_diameter : h_cylinder = 2 * R
axiom cone_dimensions : h_cone = h_cylinder ∧ h_cone = R

-- The given radius relationship
axiom radius_relation : R = 3 * r

-- Prove the spheres do not protrude from the container
theorem spheres_do_not_protrude (R r h_cylinder h_cone : ℝ)
  (cylinder_height_diameter : h_cylinder = 2 * R)
  (cone_dimensions : h_cone = h_cylinder ∧ h_cone = R)
  (radius_relation : R = 3 * r) : r ≤ R / 2 :=
sorry

end spheres_do_not_protrude_l383_383457


namespace lines_intersect_l383_383463

noncomputable def line1 (t : ℚ) : ℚ × ℚ :=
(1 + 2 * t, 2 - 3 * t)

noncomputable def line2 (u : ℚ) : ℚ × ℚ :=
(-1 + 3 * u, 4 + u)

theorem lines_intersect :
  ∃ t u : ℚ, line1 t = (-5 / 11, 46 / 11) ∧ line2 u = (-5 / 11, 46 / 11) :=
sorry

end lines_intersect_l383_383463


namespace max_acute_in_convex_polygon_l383_383268

-- Defining a predicate to capture the concept of a convex polygon and its properties
def is_convex_polygon (n : ℕ) : Prop :=
  n >= 3

-- The main theorem stating the problem
theorem max_acute_in_convex_polygon (n : ℕ) (h : is_convex_polygon n) : 
  max_acute_angles n = 3 :=
sorry -- Proof omitted

end max_acute_in_convex_polygon_l383_383268


namespace curve_not_parabola_l383_383713

theorem curve_not_parabola (k : ℝ) : ¬ ∃ (x y : ℝ), (x^2 + k * y^2 = 1) ↔ (k = -y / x) :=
by
  sorry

end curve_not_parabola_l383_383713


namespace solve_log_eq_l383_383819

theorem solve_log_eq : 
    ∀ x : ℝ, (log 2 (4^x + 4) = x + log 2 (2^(x+1) - 3)) ↔ x = 2 :=
by
  intro x
  sorry

end solve_log_eq_l383_383819


namespace number_of_selection_schemes_l383_383838

/-- Define the set of candidates --/
inductive Candidate
| A | B | C | D | E
deriving DecidableEq, Fintype

/-- Define the set of tasks --/
inductive Task
| Translation | TourGuiding | Protocol | Driving
deriving DecidableEq, Fintype

/-- A and B can only do Translation, TourGuiding, Protocol --/
def ValidTasksForA : Set Task := {Task.Translation, Task.TourGuiding, Task.Protocol}
def ValidTasksForB : Set Task := {Task.Translation, Task.TourGuiding, Task.Protocol}

/-- Candidates C, D, E can do any task --/
def ValidTasksForCDE : Set Task := {Task.Translation, Task.TourGuiding, Task.Protocol, Task.Driving}

/-- Prove the number of different selection schemes --/
theorem number_of_selection_schemes : 
    ∃ schemes : Finset (Candidate × Task),
      schemes.card = 72 := 
begin
  sorry
end

end number_of_selection_schemes_l383_383838


namespace lcm_gcf_ratio_280_450_l383_383407

open Nat

theorem lcm_gcf_ratio_280_450 :
  let a := 280
  let b := 450
  lcm a b / gcd a b = 1260 :=
by
  let a := 280
  let b := 450
  sorry

end lcm_gcf_ratio_280_450_l383_383407


namespace find_vector_value_l383_383684

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b (m : ℝ) : ℝ × ℝ := (-2, m)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem find_vector_value (m : ℝ) (h : dot_product a (b m) = 0) : 
  norm (a + (2, 2) * (b m)) = 5 := by
  sorry

end find_vector_value_l383_383684


namespace pastry_problem_minimum_n_l383_383070

theorem pastry_problem_minimum_n (fillings : Finset ℕ) (n : ℕ) : 
    fillings.card = 10 →
    (∃ pairs : Finset (ℕ × ℕ), pairs.card = 45 ∧ ∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ≠ p.2 ∧ p.1 ∈ fillings ∧ p.2 ∈ fillings) →
    (∀ (remaining_pies : Finset (ℕ × ℕ)), remaining_pies.card = 45 - n → 
     ∃ f1 f2, (f1, f2) ∈ remaining_pies → (f1 ∈ fillings ∧ f2 ∈ fillings)) →
    n = 36 :=
by
  intros h_fillings h_pairs h_remaining_pies
  sorry

end pastry_problem_minimum_n_l383_383070


namespace compute_f_neg3_plus_f_0_l383_383041

-- Define that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = -f(x)

noncomputable def f : ℝ → ℝ := sorry

-- Define the conditions
axiom odd_f : is_odd_function f
axiom f_at_3 : f 3 = -2

-- State the main theorem
theorem compute_f_neg3_plus_f_0 : f (-3) + f 0 = 2 :=
by
  sorry

end compute_f_neg3_plus_f_0_l383_383041


namespace water_level_in_cubic_tank_l383_383455

theorem water_level_in_cubic_tank
  (volume_of_water : ℝ)
  (capacity_ratio : ℝ)
  (is_cubic : ∀ s: ℝ, s^3 = volume_of_tank)
  (volume_of_tank : ℝ)
  (water_level : ℝ) :
  volume_of_water = 50 → capacity_ratio = 0.4 →
  volume_of_tank = volume_of_water / capacity_ratio →
  water_level = volume_of_water / (∛volume_of_tank)^2 →
  water_level = 2 :=
by
  sorry

end water_level_in_cubic_tank_l383_383455


namespace exists_x_geq_zero_l383_383321

theorem exists_x_geq_zero (h : ∀ x : ℝ, x^2 + x - 1 < 0) : ∃ x : ℝ, x^2 + x - 1 ≥ 0 :=
sorry

end exists_x_geq_zero_l383_383321


namespace problem_statement_l383_383225

def f (x : ℝ) : ℝ := Real.exp x
def g (x : ℝ) : ℝ := (1/2 : ℝ) * x^2 + x + 1

theorem problem_statement : ∀ x : ℝ, x ≥ 0 → f x ≥ g x := 
by
  sorry

end problem_statement_l383_383225


namespace starting_player_wins_l383_383906

def game_board : Prop :=
  ∃ (num_squares den_squares : ℕ), 
  num_squares = 1010 ∧ 
  den_squares = 1011

def number_range : finset ℕ := finset.range 2022 \ {0}

def winning_condition (f : ℕ → ℕ → ℚ) : Prop :=
  abs (f 1010 1011 - 1) < 10^(-6)

def starting_player_strategy (f : ℕ → ℕ → ℚ) : Prop :=
  ∀ (num_squares den_squares : ℕ) (x ∈ number_range), 
  ∃ (y ∈ number_range), 
  winning_condition (λ n d, if n + d = 1010 + 1011 
  then (f n d) 
  else (f (n + x) (d + y)))

theorem starting_player_wins :
  ∃ f : ℕ → ℕ → ℚ, 
  game_board ∧ 
  winning_condition f ∧ 
  starting_player_strategy f :=
sorry

end starting_player_wins_l383_383906


namespace parts_of_milk_in_drink_A_l383_383446

-- Definitions
variables (m : ℕ) (partsA : ℕ := m + 3) (partsB : ℕ := 7) (litersA : ℕ := 42) (addedJuice : ℕ := 14)
variables (totalJuiceA : ℕ := 3)
variables (finalJuice : ℕ := totalJuiceA + addedJuice)

-- The condition that the final mixture should match the 4:3 ratio of drink B
variables (ratio : ℕ → ℕ → Prop) (drinkB_ratio : ratio 4 3)

-- The drink A's ratio with added juice should also match drink B's ratio after adding 14 liters of fruit juice
def ratio (a b : ℕ) : Prop := a * 3 = b * 4

-- Problem Statement: Prove the number of parts of milk in drink A.
theorem parts_of_milk_in_drink_A : m + 3 = 42 → finalJuice = 17 → ratio finalJuice m → m = 13 := by
  intros h1 h2 h3
  sorry

end parts_of_milk_in_drink_A_l383_383446


namespace p_necessary_not_sufficient_for_q_l383_383987

variables (a b : ℝ) (p q : Prop)

-- Conditions
def p : Prop := a ≠ 0
def q : Prop := a * b ≠ 0

-- Statement
theorem p_necessary_not_sufficient_for_q (ha : p) (hb : q) : p → q → (q → p ∧ ¬(p → q)) :=
by
  sorry

end p_necessary_not_sufficient_for_q_l383_383987


namespace find_value_of_a_l383_383710

theorem find_value_of_a (a : ℚ) (h : a + a / 4 - 1 / 2 = 2) : a = 2 :=
by
  sorry

end find_value_of_a_l383_383710


namespace graph_symmetric_about_y_equals_x_l383_383671

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem graph_symmetric_about_y_equals_x :
  ∀ x : ℝ, f(f(x)) = x := 
begin 
  sorry 
end

end graph_symmetric_about_y_equals_x_l383_383671


namespace no_y_satisfies_both_inequalities_l383_383616

variable (y : ℝ)

theorem no_y_satisfies_both_inequalities :
  ¬ (3 * y^2 - 4 * y - 5 < (y + 1)^2 ∧ (y + 1)^2 < 4 * y^2 - y - 1) :=
by
  sorry

end no_y_satisfies_both_inequalities_l383_383616


namespace trick_proof_l383_383061

-- Defining the number of fillings and total pastries based on combinations
def num_fillings := 10

def total_pastries : ℕ := (num_fillings * (num_fillings - 1)) / 2

-- Definition stating that the smallest number of pastries n such that Vasya can always determine at least one filling of any remaining pastry
def min_n := 36

-- The theorem stating the proof problem
theorem trick_proof (n m: ℕ) (h1: n = 10) (h2: m = (n * (n - 1)) / 2) : min_n = 36 :=
by
  sorry

end trick_proof_l383_383061


namespace smallest_number_top_block_l383_383942

theorem smallest_number_top_block : 
  ∃ n : ℕ, n = 48 ∧ 
  (∀ (blocks : List ℕ),
    blocks.length = 15 →
    (∀ i j k l,
      (i ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) →
      (j ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) →
      (k ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) →
      (l ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) →
      l = 2 * (blocks.nthLe i h_i + blocks.nthLe j h_j + blocks.nthLe k h_k) →
      blocks.nthLe l h_l = 48)) :=
by 
  sorry

end smallest_number_top_block_l383_383942


namespace Δ_unbounded_l383_383291

-- We define ℕ as the set of positive integers.
def pos_nat := { n : ℕ // 0 < n }

-- Define function f from positive integers to positive integers
variable {f : pos_nat → pos_nat}

-- Define Δ(m, n) as given in the problem
def Δ (m n : pos_nat) : ℤ :=
  let f_iter (x : pos_nat) (times : ℕ) : pos_nat :=
    (nat.rec_on times x (λ _ r, f r))
  in (f_iter m (f n).val).val - (f_iter n (f m).val).val

-- Assume Δ(m, n) ≠ 0 for any distinct m, n in pos_nat
axiom Δ_nonzero : ∀ {m n : pos_nat}, m ≠ n → Δ m n ≠ 0

-- Prove that Δ is unbounded.
theorem Δ_unbounded : ∀ (C : ℤ), ∃ (m n : pos_nat), |Δ m n| > C :=
by
  sorry

end Δ_unbounded_l383_383291


namespace solve_for_y_l383_383886

theorem solve_for_y (y : ℕ) (h : 9^y = 3^14) : y = 7 := 
by
  sorry

end solve_for_y_l383_383886


namespace curve_C1_general_equation_curve_C2_cartesian_equation_PA_PB_abs_diff_l383_383275

noncomputable def curve_C1 := { p : ℝ × ℝ | ∃ (θ : ℝ), p.1 = 2 + sqrt 3 * Real.cos θ ∧ p.2 = sqrt 3 * Real.sin θ }
noncomputable def curve_C2 := { p : ℝ × ℝ | p.2 = (1 / sqrt 3) * p.1 }

theorem curve_C1_general_equation : ∀ (x y : ℝ), ((x = 2 + sqrt 3 * Real.cos θ) ∧ (y = sqrt 3 * Real.sin θ)) → (x - 2)^2 + y^2 = 3 := 
sorry

theorem curve_C2_cartesian_equation : ∀ (x y : ℝ), (y = (1 / sqrt 3) * x) ↔ (θ = π / 6) :=
sorry

theorem PA_PB_abs_diff : 
  let P : ℝ × ℝ := (3, sqrt 3),
      A : ℝ × ℝ := ((3 + sqrt 6) / 2, (sqrt 3 + sqrt 2) / 2),
      B : ℝ × ℝ := ((3 - sqrt 6) / 2, (sqrt 3 - sqrt 2) / 2) in
  abs ((Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)) - (Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2))) = 2 * sqrt 2 :=
sorry

end curve_C1_general_equation_curve_C2_cartesian_equation_PA_PB_abs_diff_l383_383275


namespace convex_polygon_triangles_impossible_l383_383269

theorem convex_polygon_triangles_impossible :
  ∀ (a b c : ℕ), 2016 + 2 * b + c - 2014 = 0 → a + b + c = 2014 → a = 1007 → false :=
sorry

end convex_polygon_triangles_impossible_l383_383269


namespace ratio_constant_value_of_9_l383_383293

noncomputable def moving_point_trajectory_condition1 : Prop :=
    ∀ P : ℝ × ℝ, 
    let distK := λ P, real.sqrt ((P.1 - 1) ^ 2 + P.2 ^ 2) 
    let distF := λ P, real.sqrt ((P.1 + 1) ^ 2 + P.2 ^ 2)
    (distK P + distF P = 4)

noncomputable def moving_point_trajectory_condition2 : Prop :=
    ∀ P : ℝ × ℝ, 
    let distT := λ P, real.sqrt ((P.1 + 1) ^ 2 + P.2 ^ 2)
    let distL := abs (P.1 + 4)
    (distT P / distL == 1/2)

noncomputable def moving_point_trajectory_condition3 : Prop :=
    ∀ E P G : ℝ × ℝ,
    let C := (0, 0) -- Center of the circle O
    (E.1^2 + E.2^2 = 4) ∧ 
    (G = (E.1, 0)) ∧
    (P.1 - G.1 = (sqrt 3 / 2) * (E.1 - C.1)) ∧ 
    (P.2 - G.2 = (sqrt 3 / 2) * (E.2 - C.2))

theorem ratio_constant_value_of_9
  (P T M N K A B Q : ℝ × ℝ)
  (hT : T = (-1, 0))
  (hK : K = (1, 0))
  (hGamma : ∀ xy, xy.1^2 / 4 + xy.2^2 / 3 = 1)
  (hM : ∀ t, M = (t + 1, √(3 - 3 * t^2)))
  (hN : ∀ t, N = (-t + 1, -√(3 - 3 * t^2)))
  (hA : A = (-2, 0))
  (hB : B = ( 2, 0))
  (P_from_AM : M.2 = ((M.1 + 2) / P.1))
  (Q_from_BN : N.2 = ((N.1 + 2) / Q.1)) :
  P.2 / Q.2 = 1/9 :=
sorry


end ratio_constant_value_of_9_l383_383293


namespace parallel_vectors_l383_383984

noncomputable def e1 : ℝ := sorry
noncomputable def e2 : ℝ := sorry
noncomputable def λ : ℝ := sorry

def a : ℝ × ℝ := (e1, -4 * e2)
def b (k : ℝ) : ℝ × ℝ := (2 * e1, k * e2)

-- Vectors e1 and e2 are not collinear
axiom not_collinear (e1 e2 : ℝ) : e1 ≠ 0 ∧ e2 ≠ 0

-- Prove that if a is parallel to b then k = -8
theorem parallel_vectors (k : ℝ) (h : a = λ • (b k)) : k = -8 :=
sorry

end parallel_vectors_l383_383984


namespace solve_special_sequence_l383_383742

noncomputable def special_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1010 ∧ a 2 = 1015 ∧ ∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = 2 * n + 1

theorem solve_special_sequence :
  ∃ a : ℕ → ℕ, special_sequence a ∧ a 1000 = 1676 :=
by
  sorry

end solve_special_sequence_l383_383742


namespace hypotenuse_right_triangle_l383_383545

theorem hypotenuse_right_triangle (a b : ℕ) (h1 : a = 15) (h2 : b = 36) :
  ∃ c, c ^ 2 = a ^ 2 + b ^ 2 ∧ c = 39 :=
by
  sorry

end hypotenuse_right_triangle_l383_383545


namespace coeff_x2_l383_383171

variable {R : Type*} [Semiring R]

def poly1 (c : R) : R[X] :=
  c * X^3 + 5 * X^2 - 2 * X

def poly2 (b : R) : R[X] :=
  b * X^2 - 7 * X - 5

theorem coeff_x2 (b c : R) : coefficient (poly1 c * poly2 b) 2 = -11 :=
by
  sorry

end coeff_x2_l383_383171


namespace circumcircle_intersection_and_fixed_point_l383_383890

-- Define the triangles
variables {A B C A₁ B₁ C₁ : Type}

structure Triangle (α : Type) := 
(vertices : α → α → α → Prop)

variables (ABC : Triangle ℝ) (AB₁C₁ A₁BC₁ A₁B₁C : Triangle ℝ)

-- Conditions from part (a)
def circumcircle_intersection_part_a (A B C A₁ B₁ C₁ : ℝ) : Prop :=
  AB₁C₁ ∧ A₁BC₁ ∧ A₁B₁C → 
  ∃ P, P ∈ circumcircle_of AB₁C₁ ∧ 
       P ∈ circumcircle_of A₁BC₁ ∧
       P ∈ circumcircle_of A₁B₁C

-- Define similarity and fixed intersection from part (b)
def circumcircle_fixed_intersection (A B C A₁ B₁ C₁ : ℝ) (similar_triangle : Triangle ℝ) : Prop :=
  ∀ (A₁ B₁ C₁ : ℝ), 
    (similar A₁B₁C₁ similar_triangle → 
     ∃ P, P ∈ circumcircle_of AB₁C₁ ∧
          P ∈ circumcircle_of A₁BC₁ ∧
          P ∈ circumcircle_of A₁B₁C) → 
  ∃ Q, ∀ (A₁ B₁ C₁ : ℝ), 
        triangle A₁ B₁ C₁ → 
        (similar A₁B₁C₁ similar_triangle) → 
        P = Q

-- Combined statement for both parts
theorem circumcircle_intersection_and_fixed_point (A B C A₁ B₁ C₁ : ℝ) (similar_triangle : Triangle ℝ) :
  circumcircle_intersection_part_a A B C A₁ B₁ C₁ ∧ 
  circumcircle_fixed_intersection A B C A₁ B₁ C₁ similar_triangle :=
by 
  sorry

end circumcircle_intersection_and_fixed_point_l383_383890


namespace circle_sector_cones_sum_radii_l383_383450

theorem circle_sector_cones_sum_radii :
  let r := 5
  let a₁ := 1
  let a₂ := 2
  let a₃ := 3
  let total_area := π * r * r
  let θ₁ := (a₁ / (a₁ + a₂ + a₃)) * 2 * π
  let θ₂ := (a₂ / (a₁ + a₂ + a₃)) * 2 * π
  let θ₃ := (a₃ / (a₁ + a₂ + a₃)) * 2 * π
  let r₁ := (a₁ / (a₁ + a₂ + a₃)) * r
  let r₂ := (a₂ / (a₁ + a₂ + a₃)) * r
  let r₃ := (a₃ / (a₁ + a₂ + a₃)) * r
  r₁ + r₂ + r₃ = 5 :=
by {
  sorry
}

end circle_sector_cones_sum_radii_l383_383450


namespace ratio_areas_l383_383644

-- Define the points A, B, C, D
def A : (ℝ × ℝ) := (0, 0)
def B : (ℝ × ℝ) := (0, 2)
def C : (ℝ × ℝ) := (3, 2)
def D : (ℝ × ℝ) := (3, 0)

-- Define point E as the midpoint of BD
def E : (ℝ × ℝ) := ((B.1 + D.1) / 2, (B.2 + D.2) / 2)

-- DF = 1/4 * DA, thus F is located on DA
def F : (ℝ × ℝ) := (3 - 3 / 4, 0)  -- since DA = 3, DF = 3/4, so F = (3 - 3/4, 0)

-- Function to calculate the area of a triangle given its vertices
def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

-- Areas of the specific triangles and quadrilateral
def area_DFE := triangle_area D F E
def area_ABE := triangle_area A B E
def area_AEF := triangle_area A E F
def area_ABEF := area_ABE + area_AEF

-- Prove the ratio of area_DFE to area_ABEF is 1 / 7
theorem ratio_areas :
  (area_DFE / area_ABEF) = (1 / 7) := by
  sorry

end ratio_areas_l383_383644


namespace pastry_problem_minimum_n_l383_383066

theorem pastry_problem_minimum_n (fillings : Finset ℕ) (n : ℕ) : 
    fillings.card = 10 →
    (∃ pairs : Finset (ℕ × ℕ), pairs.card = 45 ∧ ∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ≠ p.2 ∧ p.1 ∈ fillings ∧ p.2 ∈ fillings) →
    (∀ (remaining_pies : Finset (ℕ × ℕ)), remaining_pies.card = 45 - n → 
     ∃ f1 f2, (f1, f2) ∈ remaining_pies → (f1 ∈ fillings ∧ f2 ∈ fillings)) →
    n = 36 :=
by
  intros h_fillings h_pairs h_remaining_pies
  sorry

end pastry_problem_minimum_n_l383_383066


namespace hypotenuse_right_triangle_l383_383550

theorem hypotenuse_right_triangle (a b : ℕ) (h1 : a = 15) (h2 : b = 36) :
  ∃ c, c ^ 2 = a ^ 2 + b ^ 2 ∧ c = 39 :=
by
  sorry

end hypotenuse_right_triangle_l383_383550


namespace Mn_nonempty_iff_odd_Mn_sum_equality_l383_383771

open Finset

noncomputable def Mn (n : ℕ) : Set (Equiv.Perm (Fin n)) :=
{σ | ∃ τ : Equiv.Perm (Fin n), ∀ i : Fin n, ∃ k : ℤ, (σ i).val + (τ i).val = k} ∧ 
  {n % 2 = 1}

theorem Mn_nonempty_iff_odd (n : ℕ) :
  (Mn n ≠ ∅ ↔ (n % 2 = 1)) :=
sorry

theorem Mn_sum_equality (n : ℕ) (σ₁ σ₂ : Equiv.Perm (Fin n)) (hσ₁ : σ₁ ∈ Mn n) (hσ₂ : σ₂ ∈ Mn n) :
  ∑ k in range n, k * σ₁ (Fin k) = ∑ k in range n, k * σ₂ (Fin k) :=
sorry

end Mn_nonempty_iff_odd_Mn_sum_equality_l383_383771


namespace algebra_problem_l383_383981

noncomputable def expression (a b : ℝ) : ℝ :=
  (3 * a + b / 3)⁻¹ * ((3 * a)⁻¹ + (b / 3)⁻¹)

theorem algebra_problem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  expression a b = (a * b)⁻¹ :=
by
  sorry

end algebra_problem_l383_383981


namespace river_width_l383_383915

noncomputable def convert_kmph_to_mpm (kmph : ℝ) : ℝ := kmph * 1000 / 60

theorem river_width (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℕ)
  (h_depth : depth = 3)
  (h_flow_rate_kmph : flow_rate_kmph = 1)
  (h_volume_per_minute : volume_per_minute = 2750) :
  (volume_per_minute : ℝ) / (depth * convert_kmph_to_mpm flow_rate_kmph) ≈ 55 :=
by 
  sorry

end river_width_l383_383915


namespace matching_red_pair_probability_l383_383861

def total_socks := 8
def red_socks := 4
def blue_socks := 2
def green_socks := 2

noncomputable def total_pairs := Nat.choose total_socks 2
noncomputable def red_pairs := Nat.choose red_socks 2
noncomputable def blue_pairs := Nat.choose blue_socks 2
noncomputable def green_pairs := Nat.choose green_socks 2
noncomputable def total_matching_pairs := red_pairs + blue_pairs + green_pairs
noncomputable def probability_red := (red_pairs : ℚ) / total_matching_pairs

theorem matching_red_pair_probability : probability_red = 3 / 4 :=
  by sorry

end matching_red_pair_probability_l383_383861


namespace range_of_a_l383_383214

variable {R : Type} [LinearOrderedField R]

def is_even (f : R → R) : Prop := ∀ x, f x = f (-x)
def is_monotone_increasing_on_non_neg (f : R → R) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem range_of_a 
  (f : R → R) 
  (even_f : is_even f)
  (mono_f : is_monotone_increasing_on_non_neg f)
  (ineq : ∀ a, f (a + 1) ≤ f 4) : 
  ∀ a, -5 ≤ a ∧ a ≤ 3 :=
by
  sorry

end range_of_a_l383_383214


namespace problem_geometric_sequence_sum_bn_l383_383202

theorem problem_geometric_sequence 
  (a : ℕ → ℝ)
  (h1 : a 1 = 511) 
  (h2 : ∀ n ≥ 2, 4 * (a n) = a (n - 1) - 3) :
  ∃ r : ℝ, ∃ b : ℕ → ℝ, (∀ n, b n = a n + 1) ∧ (∃ c : ℝ, b 1 = c ∧ (∀ n ≥ 2, b n = r * b (n - 1))) :=
sorry

theorem sum_bn 
  (a : ℕ → ℝ)
  (h1 : a 1 = 511) 
  (h2 : ∀ n ≥ 2, 4 * (a n) = a (n - 1) - 3)
  (b : ℕ → ℝ)
  (hb : ∀ n, b n = | real.log (a n + 1) / real.log 2 |) :
  ∀ n : ℕ, 
    let T : ℕ → ℝ := λ n, 10 * n - n ^ 2 in
    let T5: ℝ := 10 * 5 - 5 ^ 2 in
    T n :=
      if n ≤ 5 then T n
      else (2 * T 5 - T n) :=
sorry

end problem_geometric_sequence_sum_bn_l383_383202


namespace worker_A_completion_time_l383_383743

-- Define the work rates
def work_rate_B (WB : ℝ) : ℝ := WB
def work_rate_A (WB : ℝ) : ℝ := 3 * WB
def work_rate_C (WB : ℝ) : ℝ := 2 * WB

-- Define the work cycle completion time
def work_cycle_days : ℝ := 3

-- Define the number of cycles in 36 days
def number_of_cycles (total_days : ℝ) : ℝ := total_days / work_cycle_days

-- Define the total work done in 36 days
def total_work_done (WB : ℝ) (cycles : ℝ) : ℝ := cycles * (work_rate_A WB + work_rate_B WB + work_rate_C WB)

-- Define the time required for worker A to complete the work alone
def days_for_A_alone (total_work : ℝ) (rate_A : ℝ) : ℝ := total_work / rate_A

theorem worker_A_completion_time (WB : ℝ) (total_days : ℝ) (cycles : ℝ) :
    days_for_A_alone (total_work_done WB cycles) (work_rate_A WB) = 24 :=
by
  have h1 : work_cycle_days = 3 := by rfl
  have h2 : number_of_cycles total_days = 12 := by sorry 
  have h3 : total_work_done WB cycles = 72 * WB := by sorry
  have h4 : work_rate_A WB = 3 * WB := by rfl
  have h5 : days_for_A_alone (72 * WB) (3 * WB) = 24 := by 
    rw [days_for_A_alone, h3, h4]
    have : (72 * WB) / (3 * WB) = 24 := by { field_simp [WB], norm_num }
    exact this
  exact h5

end worker_A_completion_time_l383_383743


namespace Albert_has_more_rocks_than_Jose_l383_383767

noncomputable def Joshua_rocks : ℕ := 80
noncomputable def Jose_rocks : ℕ := Joshua_rocks - 14
noncomputable def Albert_rocks : ℕ := Joshua_rocks + 6

theorem Albert_has_more_rocks_than_Jose :
  Albert_rocks - Jose_rocks = 20 := by
  sorry

end Albert_has_more_rocks_than_Jose_l383_383767


namespace sum_of_interior_angles_l383_383108

theorem sum_of_interior_angles {n : ℕ} (h1 : ∀ i, i < n → (interior_angle i : ℝ) = 144) : 
  (sum_of_polygon_interior_angles n = 1440) :=
sorry

end sum_of_interior_angles_l383_383108


namespace common_tangent_range_l383_383254

theorem common_tangent_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, (x₁ ≠ x₂ ∧ 2*a*x₁ = Real.exp x₂ 
   ∧ Real.exp x₂ = (Real.exp x₂ - a*x₁^2)/(x₂ - x₁)))
  ↔ a ∈ set.Ici (Real.exp 2 / 4) :=
by
  sorry

end common_tangent_range_l383_383254


namespace max_sides_polygon_from_plane_intersection_with_cube_l383_383896

theorem max_sides_polygon_from_plane_intersection_with_cube (P : Plane) (C : Cube) :
  ∃ p : Polygon, (p ⊆ P ∩ C) ∧ (num_sides p = 6) :=
sorry

end max_sides_polygon_from_plane_intersection_with_cube_l383_383896


namespace work_completion_time_for_A_l383_383899

-- Define the conditions
def B_completion_time : ℕ := 30
def joint_work_days : ℕ := 4
def work_left_fraction : ℚ := 2 / 3

-- Define the required proof statement
theorem work_completion_time_for_A (x : ℚ) : 
  (4 * (1 / x + 1 / B_completion_time) = 1 / 3) → x = 20 := 
by
  sorry

end work_completion_time_for_A_l383_383899


namespace determine_x_l383_383604

noncomputable def mean (s : List ℕ) : ℚ :=
s.sum / s.length

def median (s : List ℕ) : ℕ :=
let sorted := s.qsort (· ≤ ·)
sorted.get (sorted.length / 2 - 1)

def mode (s : List ℕ) : ℕ :=
s.groupBy id |> List.map (fun g => (g.head!, g.length)) |> List.maximumBy (·.2 <= ·.2) |>.1

theorem determine_x (x : ℕ) :
(x ∉ {2, 3, 4, 5, 8}) →
median [2, 3, 4, x, 5, 5, 8] = 5 →
mean [2, 3, 4, x, 5, 5, 8] = 5 →
mode [2, 3, 4, x, 5, 5, 8] = 5 →
x = 8 := by
  sorry

end determine_x_l383_383604


namespace find_radius_l383_383451

def radius_approx (r : ℝ) : Prop :=
  let walk_width : ℝ := 3
  let cost_per_m2 : ℝ := 2
  let total_cost : ℝ := 659.7344572538564
  (π * ((r + walk_width)^2 - r^2) * cost_per_m2 = total_cost)

theorem find_radius : ∃ r : ℝ, radius_approx r ∧ abs (r - 16.004) < 0.001 := by
  sorry

end find_radius_l383_383451


namespace area_of_sector_l383_383916

theorem area_of_sector (r : ℝ) (n : ℝ) (h_r : r = 3) (h_n : n = 120) : 
  (n / 360) * π * r^2 = 3 * π :=
by
  rw [h_r, h_n] -- Plugin in the given values first
  norm_num     -- Normalize numerical expressions
  sorry        -- Placeholder for further simplification if needed. 

end area_of_sector_l383_383916


namespace good_point_exists_l383_383199

def label := Int
-- Define point labels as 1 or -1
def is_label (l : label) : Prop := l = 1 ∨ l = -1

def is_good_point (labels : List label) (i : Nat) : Prop :=
  (∀ j, j ≠ i → j < labels.length → 
    (List.sum (labels.rotate (j - i)) > 0 ∨
     List.sum (labels.rotate (i - j)) > 0))

theorem good_point_exists (labels : List label) :
  labels.length = 1985 → (labels.count (-1) < 662) →
  ∃ i, is_good_point labels i :=
by
  sorry

end good_point_exists_l383_383199


namespace grace_earnings_in_september_l383_383695

theorem grace_earnings_in_september
  (hours_mowing : ℕ) (hours_pulling_weeds : ℕ) (hours_putting_mulch : ℕ)
  (rate_mowing : ℕ) (rate_pulling_weeds : ℕ) (rate_putting_mulch : ℕ)
  (total_hours_mowing : hours_mowing = 63) (total_hours_pulling_weeds : hours_pulling_weeds = 9) (total_hours_putting_mulch : hours_putting_mulch = 10)
  (rate_for_mowing : rate_mowing = 6) (rate_for_pulling_weeds : rate_pulling_weeds = 11) (rate_for_putting_mulch : rate_putting_mulch = 9) :
  hours_mowing * rate_mowing + hours_pulling_weeds * rate_pulling_weeds + hours_putting_mulch * rate_putting_mulch = 567 :=
by
  intros
  sorry

end grace_earnings_in_september_l383_383695


namespace sunny_fun_never_stops_l383_383801

theorem sunny_fun_never_stops 
  (k : ℕ) 
  (a : ℤ → ℝ) 
  (P : ℤ → ℝ → ℂ) 
  (hP_def : ∀ n : ℤ, P n t = ∏ (i : ℤ) in (finset.range (n+1 - k)).map (λ i => -k + i), (1 - a i * t))
  (h_a_def : ∀ n : ℤ, a n = (i * P n i - i * P n (-i)) / (P n i + P n (-i)))
  (m n : ℕ) (hmn : m > n)
  (h_repeat : a m = a n) :
  (∀ j > n, ∃ i ≤ n, a j = a i) ∧
  ∀ k : ℕ, P k i + P k (-i) ≠ 0 := sorry

end sunny_fun_never_stops_l383_383801


namespace day_100_days_from_friday_l383_383393

-- Define the days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Define a function to get the day of the week after a given number of days
def dayOfWeekAfter (start : Day) (n : ℕ) : Day :=
  match start with
  | Sunday    => match n % 7 with
                  | 0 => Sunday
                  | 1 => Monday
                  | 2 => Tuesday
                  | 3 => Wednesday
                  | 4 => Thursday
                  | 5 => Friday
                  | 6 => Saturday
                  | _ => start
  | Monday    => match n % 7 with
                  | 0 => Monday
                  | 1 => Tuesday
                  | 2 => Wednesday
                  | 3 => Thursday
                  | 4 => Friday
                  | 5 => Saturday
                  | 6 => Sunday
                  | _ => start
  | Tuesday   => match n % 7 with
                  | 0 => Tuesday
                  | 1 => Wednesday
                  | 2 => Thursday
                  | 3 => Friday
                  | 4 => Saturday
                  | 5 => Sunday
                  | 6 => Monday
                  | _ => start
  | Wednesday => match n % 7 with
                  | 0 => Wednesday
                  | 1 => Thursday
                  | 2 => Friday
                  | 3 => Saturday
                  | 4 => Sunday
                  | 5 => Monday
                  | 6 => Tuesday
                  | _ => start
  | Thursday  => match n % 7 with
                  | 0 => Thursday
                  | 1 => Friday
                  | 2 => Saturday
                  | 3 => Sunday
                  | 4 => Monday
                  | 5 => Tuesday
                  | 6 => Wednesday
                  | _ => start
  | Friday    => match n % 7 with
                  | 0 => Friday
                  | 1 => Saturday
                  | 2 => Sunday
                  | 3 => Monday
                  | 4 => Tuesday
                  | 5 => Wednesday
                  | 6 => Thursday
                  | _ => start
  | Saturday  => match n % 7 with
                  | 0 => Saturday
                  | 1 => Sunday
                  | 2 => Monday
                  | 3 => Tuesday
                  | 4 => Wednesday
                  | 5 => Thursday
                  | 6 => Friday
                  | _ => start

-- The proof problem as a Lean theorem
theorem day_100_days_from_friday : dayOfWeekAfter Friday 100 = Sunday := by
  -- Proof will go here
  sorry

end day_100_days_from_friday_l383_383393


namespace number_of_true_propositions_is_two_l383_383834

theorem number_of_true_propositions_is_two :
  (∀ (angles : Type) 
     (verticalAnglesEqual : angles → angles → Prop) 
     (correspondingAnglesEqual : angles → angles → Prop) 
     (parallel : angles → angles → Prop)
     (parallelTransitive : ∀ a b c, parallel a b → parallel b c → parallel a c)
     (sidesParallelToSideanglesEqual : angles → angles → Prop),
     
     (∃ angle1 angle2 angle3 angle4 : angles,
       verticalAnglesEqual angle1 angle2 ∧
       ¬ correspondingAnglesEqual angle2 angle3 ∧
       parallelTransitive angle3 angle4 angle1 angle4 ∧
       ¬ sidesParallelToSideanglesEqual angle3 angle4) 
        → 2) :=
sorry

end number_of_true_propositions_is_two_l383_383834


namespace center_of_mass_distance_l383_383373

-- Definitions for the problem
def disk_sequence_radius (i : ℕ) : ℝ := 2 * (1/2)^(i - 1)
def disk_sequence_mass (i : ℕ) : ℝ := (disk_sequence_radius i)^2
def total_mass : ℝ := ∑' i, disk_sequence_mass i

-- Center of mass calculation
def center_of_mass_y : ℝ :=
  (∑' i, disk_sequence_mass i * disk_sequence_radius i) / total_mass

-- The main theorem stating the distance from the center of the largest disk
theorem center_of_mass_distance : center_of_mass_y = 6 / 7 :=
by
  sorry

end center_of_mass_distance_l383_383373


namespace polyhedron_calculation_l383_383104

def faces := 32
def triangular := 10
def pentagonal := 8
def hexagonal := 14
def edges := 79
def vertices := 49
def T := 1
def P := 2

theorem polyhedron_calculation : 
  100 * P + 10 * T + vertices = 249 := 
sorry

end polyhedron_calculation_l383_383104


namespace hyperbola_asymptotes_angle_l383_383185

noncomputable def hyperbola_ratio (a b : Real) (h : a > b) (theta : Real) [Fact (theta = π / 6)] : Real := 
a / b

theorem hyperbola_asymptotes_angle (a b : Real) (h : a > b) (theta : Real) [Fact (theta = π / 6)] :
  hyperbola_ratio a b h theta = 2 - Real.sqrt 3 := 
sorry

end hyperbola_asymptotes_angle_l383_383185


namespace coinciding_rest_days_l383_383576

-- Definitions of the cycles
def Alice_cycle := 6
def Bob_cycle := 6

-- Definitions of rest days within the cycle
def Alice_rest_days := [5, 6]
def Bob_rest_days := [6]

-- Function to check if a day is a rest day for Alice
def is_Alice_rest_day (d : ℕ) : Bool :=
  (d % Alice_cycle + 1) ∈ Alice_rest_days

-- Function to check if a day is a rest day for Bob
def is_Bob_rest_day (d : ℕ) : Bool :=
  (d % Bob_cycle + 1) ∈ Bob_rest_days

-- The theorem to prove
theorem coinciding_rest_days (n : ℕ) : n = 1000 →
  (∃ (count : ℕ), (∀ (d : ℕ), 1 ≤ d ∧ d ≤ n → is_Alice_rest_day (d - 1) = tt ∧ is_Bob_rest_day (d - 1) = tt).count d = count → count = 167) :=
by {
  sorry
}

end coinciding_rest_days_l383_383576


namespace train_length_is_120_l383_383462

noncomputable def train_length (jogger_speed : ℚ) (train_speed : ℚ) (head_start : ℚ) (time_to_pass : ℚ) : ℚ :=
  let relative_speed := train_speed - jogger_speed
  in (relative_speed * time_to_pass) - head_start

theorem train_length_is_120 :
  ∀ (jogger_speed train_speed head_start time_to_pass : ℚ),
    jogger_speed = 9 ∧ 
    train_speed = 45 ∧ 
    head_start = 190 ∧ 
    time_to_pass = 31 →
    train_length jogger_speed train_speed head_start time_to_pass = 120 := 
by
  intros jogger_speed train_speed head_start time_to_pass
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  calc
    train_length jogger_speed train_speed head_start time_to_pass 
        = (train_speed - jogger_speed) * time_to_pass - head_start : by sorry
    ... = (45 - 9) * 31 - 190 : by rw [h1, h3, h5]
    ... = 120 : by norm_num

end train_length_is_120_l383_383462


namespace umbrella_numbers_count_l383_383921

theorem umbrella_numbers_count :
  ∃ (A : ℕ → ℕ → ℕ), (A 2 2 + A 2 3 + A 2 4 + A 2 5 = 40) :=
by
  let A := Nat.perm
  use A
  sorry

end umbrella_numbers_count_l383_383921


namespace math_proof_problem_l383_383991
open Classical

noncomputable section

variables {A B C A' B' C' : Point} -- Define variables for points

-- Defining conditions
def is_triangle (A B C : Point) : Prop :=
  ∃ circle : Circle, A ∈ circle ∧ B ∈ circle ∧ C ∈ circle

def symmetric_points_on_circle (A B C : Point) : Prop :=
  ∀ (H : Point), (is_orthocenter_of H A B C) →
  ∀ side : Line, (is_symmetric_relative_to_side H side) ∈ circle

def symmetric_points_on_line (A B C : Point) (P : Point) : Prop :=
  is_on_circle A B C P →
  ∀ line : Line, is_symmetric_points_line (P, line) (A, B, C)

def circles_intersect_at_one_point (A B C A' B' C' : Point) : Prop :=
  ∃ P : Point, circle A B' C' ∋ P ∧ circle B C' A' ∋ P ∧ circle C A' B' ∋ P

-- Full statement in Lean
theorem math_proof_problem (A B C A' B' C' : Point) :
  is_triangle A B C →
  symmetric_points_on_circle A B C →
  symmetric_points_on_line A B C any_point →
  circles_intersect_at_one_point A B C A' B' C' := sorry

end math_proof_problem_l383_383991


namespace solve_equation_l383_383337

noncomputable def equation_conditions (x : ℝ) :=
  real.sqrt (7 * x - 3) + real.sqrt (2 * x - 2) = 5

theorem solve_equation : equation_conditions 20.14 :=
by 
  sorry

end solve_equation_l383_383337


namespace pastry_trick_l383_383080

theorem pastry_trick (fillings : Fin 10) (n : ℕ) :
  ∃ n, (n = 36 ∧ ∀ remaining_pastries, 
    (remaining_pastries.length = 45 - n) → 
    (∃ remaining_filling ∈ fillings, true)) := 
sorry

end pastry_trick_l383_383080


namespace pastry_problem_minimum_n_l383_383067

theorem pastry_problem_minimum_n (fillings : Finset ℕ) (n : ℕ) : 
    fillings.card = 10 →
    (∃ pairs : Finset (ℕ × ℕ), pairs.card = 45 ∧ ∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ≠ p.2 ∧ p.1 ∈ fillings ∧ p.2 ∈ fillings) →
    (∀ (remaining_pies : Finset (ℕ × ℕ)), remaining_pies.card = 45 - n → 
     ∃ f1 f2, (f1, f2) ∈ remaining_pies → (f1 ∈ fillings ∧ f2 ∈ fillings)) →
    n = 36 :=
by
  intros h_fillings h_pairs h_remaining_pies
  sorry

end pastry_problem_minimum_n_l383_383067


namespace red_light_max_probability_l383_383132

theorem red_light_max_probability {m : ℕ} (h1 : m > 0) (h2 : m < 35) :
  m = 3 ∨ m = 15 ∨ m = 30 ∨ m = 40 → m = 30 :=
by
  sorry

end red_light_max_probability_l383_383132


namespace day_100_days_from_friday_l383_383394

-- Define the days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Define a function to get the day of the week after a given number of days
def dayOfWeekAfter (start : Day) (n : ℕ) : Day :=
  match start with
  | Sunday    => match n % 7 with
                  | 0 => Sunday
                  | 1 => Monday
                  | 2 => Tuesday
                  | 3 => Wednesday
                  | 4 => Thursday
                  | 5 => Friday
                  | 6 => Saturday
                  | _ => start
  | Monday    => match n % 7 with
                  | 0 => Monday
                  | 1 => Tuesday
                  | 2 => Wednesday
                  | 3 => Thursday
                  | 4 => Friday
                  | 5 => Saturday
                  | 6 => Sunday
                  | _ => start
  | Tuesday   => match n % 7 with
                  | 0 => Tuesday
                  | 1 => Wednesday
                  | 2 => Thursday
                  | 3 => Friday
                  | 4 => Saturday
                  | 5 => Sunday
                  | 6 => Monday
                  | _ => start
  | Wednesday => match n % 7 with
                  | 0 => Wednesday
                  | 1 => Thursday
                  | 2 => Friday
                  | 3 => Saturday
                  | 4 => Sunday
                  | 5 => Monday
                  | 6 => Tuesday
                  | _ => start
  | Thursday  => match n % 7 with
                  | 0 => Thursday
                  | 1 => Friday
                  | 2 => Saturday
                  | 3 => Sunday
                  | 4 => Monday
                  | 5 => Tuesday
                  | 6 => Wednesday
                  | _ => start
  | Friday    => match n % 7 with
                  | 0 => Friday
                  | 1 => Saturday
                  | 2 => Sunday
                  | 3 => Monday
                  | 4 => Tuesday
                  | 5 => Wednesday
                  | 6 => Thursday
                  | _ => start
  | Saturday  => match n % 7 with
                  | 0 => Saturday
                  | 1 => Sunday
                  | 2 => Monday
                  | 3 => Tuesday
                  | 4 => Wednesday
                  | 5 => Thursday
                  | 6 => Friday
                  | _ => start

-- The proof problem as a Lean theorem
theorem day_100_days_from_friday : dayOfWeekAfter Friday 100 = Sunday := by
  -- Proof will go here
  sorry

end day_100_days_from_friday_l383_383394


namespace set_of_points_with_projections_l383_383876

def orthogonal_projection_onto_lines (α β : Plane) (P : Point) : Prop :=
  (exists l1 l2 : Line, orthogonal_projection P α = l1 ∧ orthogonal_projection P β = l2)

theorem set_of_points_with_projections (α β : Plane) (h_intersecting_planes : α ≠ β) :
  {P : Point | orthogonal_projection_onto_lines α β P} = {P : Point | P ∈ α ∧ P ∈ β} :=
by 
  sorry

end set_of_points_with_projections_l383_383876


namespace Mrs_Brown_points_l383_383133

-- Conditions given
variables (points_William points_Adams points_Daniel points_mean: ℝ) (num_classes: ℕ)

-- Define the conditions
def Mrs_William_points := points_William = 50
def Mr_Adams_points := points_Adams = 57
def Mrs_Daniel_points := points_Daniel = 57
def mean_condition := points_mean = 53.3
def num_classes_condition := num_classes = 4

-- Define the problem to prove
theorem Mrs_Brown_points :
  Mrs_William_points points_William ∧ Mr_Adams_points points_Adams ∧ Mrs_Daniel_points points_Daniel ∧ mean_condition points_mean ∧ num_classes_condition num_classes →
  ∃ (points_Brown: ℝ), points_Brown = 49 :=
by
  sorry

end Mrs_Brown_points_l383_383133


namespace num_distinct_pairs_l383_383602

theorem num_distinct_pairs : ∃ (s : Finset (ℝ × ℝ)), 
  (∀ (p : ℝ × ℝ), p ∈ s ↔ (let (x, y) := p in x = x^2 + y^2 ∧ y = 3 * x * y)) ∧ 
  s.card = 4 :=
by
  sorry

end num_distinct_pairs_l383_383602


namespace rate_of_interest_is_4_l383_383367

theorem rate_of_interest_is_4 (R : ℝ) : 
  ∀ P : ℝ, ∀ T : ℝ, P = 3000 → T = 5 → (P * R * T / 100 = P - 2400) → R = 4 :=
by
  sorry

end rate_of_interest_is_4_l383_383367


namespace range_of_u_l383_383197

variable (a b u : ℝ)

theorem range_of_u (h1 : a > 0) (h2 : b > 0) (h3 : (1 / a) + (9 / b) = 1) : u ≤ 16 :=
by
  sorry

end range_of_u_l383_383197


namespace least_m_lcm_l383_383973

theorem least_m_lcm (m : ℕ) (h : m > 0) : Nat.lcm 15 m = Nat.lcm 42 m → m = 70 := by
  sorry

end least_m_lcm_l383_383973


namespace exists_non_perfect_square_pair_l383_383297

noncomputable def is_perfect_square (n : ℤ) : Prop :=
∃ (m : ℤ), m * m = n

theorem exists_non_perfect_square_pair 
  (d : ℤ) (hd : 0 < d) (hd_ne2 : d ≠ 2) (hd_ne5 : d ≠ 5) (hd_ne13 : d ≠ 13) : 
    ∃ a b : ℤ, a ≠ b ∧ a ∈ {2, 5, 13, d} ∧ b ∈ {2, 5, 13, d} ∧ ¬ is_perfect_square (a * b - 1) :=
sorry

end exists_non_perfect_square_pair_l383_383297


namespace Parkway_Elementary_girls_not_playing_soccer_l383_383755

/-
  In the fifth grade at Parkway Elementary School, there are 500 students. 
  350 students are boys and 250 students are playing soccer.
  86% of the students that play soccer are boys.
  Prove that the number of girl students that are not playing soccer is 115.
-/
theorem Parkway_Elementary_girls_not_playing_soccer
  (total_students : ℕ)
  (boys : ℕ)
  (playing_soccer : ℕ)
  (percentage_boys_playing_soccer : ℝ)
  (H1 : total_students = 500)
  (H2 : boys = 350)
  (H3 : playing_soccer = 250)
  (H4 : percentage_boys_playing_soccer = 0.86) :
  ∃ (girls_not_playing_soccer : ℕ), girls_not_playing_soccer = 115 :=
by
  sorry

end Parkway_Elementary_girls_not_playing_soccer_l383_383755


namespace triangle_area_correct_l383_383263

noncomputable def area_of_triangle_given_conditions (a b c C : ℝ) (h1 : c = 2) (h2 : b = 2 * a) (h3 : C = real.pi / 3) : ℝ :=
  (1 / 2) * a * b * real.sin C

theorem triangle_area_correct (a : ℝ) 
  (ha : a = (2 : ℝ) * real.sqrt 3 / 3) 
  (b : ℝ) 
  (hb : b = (4 : ℝ) * real.sqrt 3 / 3) 
  (c : ℝ) 
  (hc : c = 2) 
  (C : ℝ) 
  (hC : C = real.pi / 3) 
  : area_of_triangle_given_conditions a b c C hc hb hC = (2 * real.sqrt 3 / 3) := 
sorry

end triangle_area_correct_l383_383263


namespace remainder_of_67_pow_67_plus_67_mod_68_l383_383428

theorem remainder_of_67_pow_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  -- Add the conditions and final proof step
  sorry

end remainder_of_67_pow_67_plus_67_mod_68_l383_383428


namespace tangent_vertical_a_value_three_intersection_points_l383_383222

noncomputable def f (x a : ℝ) := -x^3 + a*x^2 + 1

noncomputable def f' (x a : ℝ) := -3*x^2 + 2*a*x

theorem tangent_vertical_a_value :
  (∀ (a : ℝ), f' (2/3) a = 0 → a = 1) :=
begin
  intros a ha,
  sorry
end

example (a : ℝ) : (∀ {x : ℝ}, x ≥ -2 ∧ x ≤ 3 → f' x a = 0 → x = -2 ∨ x = 3) → a ≠ 0 ∧ -3 < a ∧ a < 9/2 :=
begin
  intro h,
  sorry
end

noncomputable def g (x m : ℝ) := x^4 - 5*x^3 + (2-m)*x^2 + 1

theorem three_intersection_points :
  (∃ (m : ℝ), (2-m = 0 ∨ ∃ (x : ℝ), f x 1 = g x m)) → (m ≠ 1 ∧ m > -3) :=
begin
  intro h_m,
  sorry
end

end tangent_vertical_a_value_three_intersection_points_l383_383222


namespace longest_pole_in_room_l383_383974

theorem longest_pole_in_room :
  ∀ (l w h : ℕ), l = 12 → w = 8 → h = 9 → sqrt (l^2 + w^2 + h^2) = 17 :=
by
  intros l w h hl hw hh
  rw [hl, hw, hh]
  norm_num
  sorry

end longest_pole_in_room_l383_383974


namespace hundred_days_from_friday_is_sunday_l383_383398

def days_from_friday (n : ℕ) : Nat :=
  (n + 5) % 7  -- 0 corresponds to Sunday, starting from Friday (5 + 0 % 7 = 5 which is Friday)

theorem hundred_days_from_friday_is_sunday :
  days_from_friday 100 = 0 := by
  sorry

end hundred_days_from_friday_is_sunday_l383_383398


namespace triangle_sequence_count_l383_383701

theorem triangle_sequence_count : 
  let letters := ['T', 'R', 'I', 'A', 'N', 'L'] in
  let sequences := ∀ g e t r i a n l p1 p2 p3, 
    p1 = 'G' ∧ p3 = 'E' ∧ 
    (g = 'T' ∨ g = 'R' ∨ g = 'I' ∨ g = 'A' ∨ g = 'N' ∨ g = 'L') ∧
    (e = 'T' ∨ e = 'R' ∨ e = 'I' ∨ e = 'A' ∨ e = 'N' ∨ e = 'L') ∧
    (t = 'T' ∨ t = 'R' ∨ t = 'I' ∨ t = 'A' ∨ t = 'N' ∨ t = 'L') ∧
    (r = 'T' ∨ r = 'R' ∨ r = 'I' ∨ r = 'A' ∨ r = 'N' ∨ r = 'L') ∧
    (i = 'T' ∨ i = 'R' ∨ i = 'I' ∨ i = 'A' ∨ i = 'N' ∨ i = 'L') ∧
    (a = 'T' ∨ a = 'R' ∨ a = 'I' ∨ a = 'A' ∨ a = 'N' ∨ a = 'L') ∧
    (n = 'T' ∨ n = 'R' ∨ n = 'I' ∨ n = 'A' ∨ n = 'N' ∨ n = 'L') ∧
    (l = 'T' ∨ l = 'R' ∨ l = 'I' ∨ l = 'A' ∨ l = 'N' ∨ l = 'L') ∧
    g ≠ e ∧ g ≠ t ∧ g ≠ r ∧ g ≠ i ∧ g ≠ a ∧ g ≠ n ∧ g ≠ l ∧
    e ≠ t ∧ e ≠ r ∧ e ≠ i ∧ e ≠ a ∧ e ≠ n ∧ e ≠ l ∧
    t ≠ r ∧ t ≠ i ∧ t ≠ a ∧ t ≠ n ∧ t ≠ l ∧
    r ≠ i ∧ r ≠ a ∧ r ≠ n ∧ r ≠ l ∧
    i ≠ a ∧ i ≠ n ∧ i ≠ l ∧
    a ≠ n ∧ a ≠ l ∧
    n ≠ l in
  sequences = 120 :=
  begin
    sorry
  end

end triangle_sequence_count_l383_383701


namespace max_f_value_l383_383737

noncomputable theory
open Real

def f (x : ℝ) : ℝ := abs (x - 1) + 2 * abs (x - 3)

theorem max_f_value (x θ : ℝ) (h1 : log 2 x = 1 + cos θ) (h2 : -π / 2 ≤ θ ∧ θ ≤ 0) :
  ∃ y, f y = 5 :=
by
  -- The proof starts here but is omitted as per instructions 
  sorry

end max_f_value_l383_383737


namespace karlsson_malysh_jam_time_l383_383768

theorem karlsson_malysh_jam_time
(karlsson_jam_3_honey_1_25 : 3 * j + h = 25)
(karlsson_jam_1_honey_3_35 : j + 3 * h = 35)
(malysh_jam_3_honey_1_55 : 3 * j' + h' = 55)
(malysh_jam_1_honey_3_85 : j' + 3 * h' = 85) :
  let karlsson_jam_rate := 1 / 5,
      malysh_jam_rate := 1 / 10,
      combined_rate := karlsson_jam_rate + malysh_jam_rate in
  let total_time := 6 / combined_rate in
  total_time = 20 :=
by
  sorry

end karlsson_malysh_jam_time_l383_383768


namespace fg_of_3_eq_79_l383_383250

def g (x : ℤ) : ℤ := x ^ 3
def f (x : ℤ) : ℤ := 3 * x - 2

theorem fg_of_3_eq_79 : f (g 3) = 79 := by
  sorry

end fg_of_3_eq_79_l383_383250


namespace area_shaded_region_is_75_l383_383111

-- Define the side length of the larger square
def side_length_large_square : ℝ := 10

-- Define the side length of the smaller square
def side_length_small_square : ℝ := 5

-- Define the area of the larger square
def area_large_square : ℝ := side_length_large_square ^ 2

-- Define the area of the smaller square
def area_small_square : ℝ := side_length_small_square ^ 2

-- Define the area of the shaded region
def area_shaded_region : ℝ := area_large_square - area_small_square

-- The theorem that states the area of the shaded region is 75 square units
theorem area_shaded_region_is_75 : area_shaded_region = 75 := by
  -- The proof will be filled in here when required
  sorry

end area_shaded_region_is_75_l383_383111


namespace spadesuit_evaluation_l383_383736

-- Define the operation
def spadesuit (x y : ℚ) : ℚ := x - (1 / y)

-- Prove the main statement
theorem spadesuit_evaluation : spadesuit 3 (spadesuit 3 (3 / 2)) = 18 / 7 :=
by
  sorry

end spadesuit_evaluation_l383_383736


namespace minimum_value_g_l383_383620

noncomputable def g (x : ℝ) : ℝ :=
  x + (2 * x) / (x^2 + 1) + (x * (x + 3)) / (x^2 + 3) + (3 * (x + 1)) / (x * (x^2 + 3))

theorem minimum_value_g : ∀ x : ℝ, x > 0 → g x ≥ 7 :=
by
  intros x hx
  sorry

end minimum_value_g_l383_383620


namespace vasya_pastry_trick_l383_383072

theorem vasya_pastry_trick :
  ∀ (pastries : Finset (Finset Nat))
    (filling_set : Finset Nat),
    (filling_set.card = 10) →
    (pastries.card = 45) →
    (∀ p ∈ pastries, p.card = 2 ∧ p ⊆ filling_set) →
    ∃ n, n = 36 ∧
    ∀ remain_p ∈ (pastries \ pastries.sort (λ x y, x < y)).take (45 - n), 
      ∃ f ∈ filling_set, f ∈ remain_p :=
begin
  sorry

end vasya_pastry_trick_l383_383072


namespace sum_of_interior_angles_l383_383107

theorem sum_of_interior_angles (n : ℕ) (interior_angle : ℝ) :
  (interior_angle = 144) → (180 - 144) * n = 360 → n = 10 → (n - 2) * 180 = 1440 :=
by
  intros h1 h2 h3
  sorry

end sum_of_interior_angles_l383_383107


namespace problem_solution_l383_383874

theorem problem_solution : 
  (∀ (x y z : ℤ), x = 2 ∧ y = -3 ∧ z = 1 → 2 * x^2 + 3 * y^2 - z^2 + 4 * x * y = 10) :=
by
  intros x y z h,
  cases h with hx hy hz,
  rw [hx, hy, hz],
  sorry

end problem_solution_l383_383874


namespace right_triangle_hypotenuse_l383_383520

theorem right_triangle_hypotenuse (a b : ℕ) (ha : a = 15) (hb : b = 36) : 
  ∃ h : ℕ, h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  . exact rfl
  . rw [ha, hb]
    norm_num
    sorry

end right_triangle_hypotenuse_l383_383520


namespace total_visible_surface_area_correct_l383_383607

-- Define the volumes of the cubes
def volumes : List ℕ := [1, 27, 64, 125, 216, 343, 512, 729]

-- Define the function to calculate the side length of a cube given its volume
def side_length (v : ℕ) : ℕ := Nat.cbrt v

-- Define the total visible surface area function
def total_visible_surface_area (volumes : List ℕ) : ℕ :=
  let side_lengths := volumes.map side_length
  let surface_areas := side_lengths.map (λ s => 6 * s^2)
  let adjusted_surface_areas := (surface_areas.headI :: 
    surface_areas.tailI.zipWith (λ s v => s - (side_length v)^2) volumes.tail!).reduce (· + ·)
  adjusted_surface_areas

-- The theorem to prove the total visible surface area
theorem total_visible_surface_area_correct : 
  total_visible_surface_area volumes = 1406 := 
by 
  sorry -- Proof is omitted.

end total_visible_surface_area_correct_l383_383607


namespace minimum_strip_width_l383_383017

-- Definitions based on given conditions
def infinite_strip : Type := ℝ
def triangle (A B C : infinite_strip) : Prop := A ≠ B ∧ B ≠ C ∧ A ≠ C

-- The area of a given triangle is 1
def area (A B C : infinite_strip) : Prop :=
  ∃ a : ℝ, (a^2 * Real.sqrt 3 / 4 = 1 ∧
            A ≠ B ∧ B ≠ C ∧ A ≠ C)

-- Definition of the height of a triangle function
def height (A B C : infinite_strip) :=
  (Real.sqrt 3 / 2) * (2 / Real.sqrt 3)

-- Proof problem
theorem minimum_strip_width (strip_width : ℝ) :
  strip_width = Real.sqrt (Real.sqrt 3) ↔
  ∀ (A B C : infinite_strip), triangle A B C →
  area A B C → 
  height A B C = Real.sqrt (Real.sqrt 3) :=
begin
  sorry
end

end minimum_strip_width_l383_383017


namespace repaired_shoes_last_time_l383_383901

theorem repaired_shoes_last_time :
  let cost_of_repair := 13.50
  let cost_of_new := 32.00
  let duration_of_new := 2.0
  let surcharge := 0.1852
  let avg_cost_new := cost_of_new / duration_of_new
  let avg_cost_repair (T : ℝ) := cost_of_repair / T
  (avg_cost_new = (1 + surcharge) * avg_cost_repair 1) ↔ T = 1 := 
by
  sorry

end repaired_shoes_last_time_l383_383901


namespace graph_of_g_abs_l383_383679

def g (x : ℝ) : ℝ :=
if x ≤ 1 then 2 * x + 1
else if x ≤ 2 then -sqrt (1 - (x - 1)^2) + 1
else if x ≤ 3 then x - 2
else 0  -- Define g(x) as 0 for other cases to make it total

theorem graph_of_g_abs :
  ∀ (x : ℝ),
  g (|x|) = 
    if x ≥ 0 then g x 
    else if x ≤ -2 then -x - 2
    else if x ≤ -1 then -sqrt (1 - (-x - 1)^2) + 1
    else -2 * x + 1 :=
by
  sorry

end graph_of_g_abs_l383_383679


namespace range_of_a_l383_383258

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 1| + |x + 1| > a) ↔ a < 2 := 
sorry

end range_of_a_l383_383258


namespace correct_propositions_l383_383669

theorem correct_propositions (a b c d m : ℝ) :
  (ab > 0 → a > b → (1 / a < 1 / b)) ∧
  (a > |b| → a ^ 2 > b ^ 2) ∧
  ¬ (a > b ∧ c < d → a - d > b - c) ∧
  ¬ (a < b ∧ m > 0 → a / b < (a + m) / (b + m)) :=
by sorry

end correct_propositions_l383_383669


namespace hypotenuse_length_l383_383489

-- Definitions for the problem
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def leg1 : ℕ := 15
def leg2 : ℕ := 36
def hypotenuse : ℕ := 39

-- Lean 4 statement
theorem hypotenuse_length (a b c : ℕ) (h : is_right_triangle a b c) (ha : a = leg1) (hb : b = leg2) :
  c = hypotenuse :=
begin
  sorry
end

end hypotenuse_length_l383_383489


namespace gcd_digits_bounded_by_lcm_l383_383732

theorem gcd_digits_bounded_by_lcm (a b : ℕ) (h_a : 10^6 ≤ a ∧ a < 10^7) (h_b : 10^6 ≤ b ∧ b < 10^7) (h_lcm : 10^10 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^11) : Nat.gcd a b < 10^4 :=
by
  sorry

end gcd_digits_bounded_by_lcm_l383_383732


namespace num_valid_configurations_l383_383272

-- Definitions used in the problem
def grid := (Fin 8) × (Fin 8)
def knights_tell_truth := true
def knaves_lie := true
def statement (i j : Fin 8) (r c : grid → ℕ) := (c ⟨0,j⟩ > r ⟨i,0⟩)

-- The theorem statement to prove
theorem num_valid_configurations : ∃ n : ℕ, n = 255 :=
sorry

end num_valid_configurations_l383_383272


namespace tangent_length_is_five_sqrt_two_l383_383596

noncomputable def point := (ℝ × ℝ)
noncomputable def distance (p1 p2 : point) : ℝ := (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2
noncomputable def is_on_circle (p : point) (center : point) (radius : ℝ) : Prop :=
  distance p center = radius ^ 2

noncomputable def circumncircle_radius (A B C : point) : ℝ := 
  let h := ... -- some expression to determine h
  let k := ... -- some expression to determine k
  let r := distance A (h,k) -- radius from one of the points to the center
  r

noncomputable def length_of_tangent (P A B C : point) : ℝ :=
  let r := circumncircle_radius A B C
  let dist_PA := sqrt $ distance P A
  let dist_PB := sqrt $ distance P B
  sqrt (dist_PA * dist_PB)

noncomputable def length_of_segment_tangent : ℝ :=
  length_of_tangent (1,1) (4,5) (7,9) (6,14)

theorem tangent_length_is_five_sqrt_two :
  length_of_segment_tangent = 5 * sqrt 2 := by
  sorry

end tangent_length_is_five_sqrt_two_l383_383596


namespace intersection_coordinates_proof_max_area_triangle_OAB_proof_l383_383653

def curve1 (θ : ℝ) : ℝ × ℝ := (-1 + cos θ, sin θ)
def curve2 (θ : ℝ) : ℝ := 2 * sin θ 
def point_on_curve1 (p : ℝ × ℝ) : Prop := (p.1 + 1)^2 + p.2^2 = 1
def point_on_curve2 (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 = 2 * p.2

-- Definitions for the intersection points
def intersection_points : set (ℝ × ℝ) := {p | point_on_curve1 p ∧ point_on_curve2 p}

-- Definition for the maximum area of triangle OAB
def max_area_triangle_OAB : ℝ := (√2 + 1) / 2

theorem intersection_coordinates_proof :
  intersection_points = {(0, 0), (-1, 1)} :=
sorry

theorem max_area_triangle_OAB_proof :
  ∀ (A B : ℝ × ℝ),
    point_on_curve1 A → point_on_curve2 B → 
    |A.1 - B.1|^2 + |A.2 - B.2|^2 = (2 + √2)^2 →
    ∃ S, S = max_area_triangle_OAB ∧ S = (1 / 2) * (2 + √2) * (√2 / 2) :=
sorry

end intersection_coordinates_proof_max_area_triangle_OAB_proof_l383_383653


namespace gcd_digit_bound_l383_383728

theorem gcd_digit_bound (a b : ℕ) (h1 : a < 10^7) (h2 : b < 10^7) (h3 : 10^10 ≤ Nat.lcm a b) :
  Nat.gcd a b < 10^4 :=
by
  sorry

end gcd_digit_bound_l383_383728


namespace wang_hua_withdrawal_correct_l383_383274

noncomputable def wang_hua_withdrawal : ℤ :=
  let d : ℤ := 14
  let c : ℤ := 32
  -- The amount Wang Hua was supposed to withdraw in yuan
  (d * 100 + c)

theorem wang_hua_withdrawal_correct (d c : ℤ) :
  let initial_amount := (100 * d + c)
  let incorrect_amount := (100 * c + d)
  let amount_spent := 350
  let remaining_amount := incorrect_amount - amount_spent
  let expected_remaining := 2 * initial_amount
  remaining_amount = expected_remaining ∧ 
  d = 14 ∧ 
  c = 32 :=
by
  sorry

end wang_hua_withdrawal_correct_l383_383274


namespace root_implies_value_of_m_l383_383244

theorem root_implies_value_of_m (x: ℝ) (m: ℝ) 
  (h: 2 * x^2 - 3 * m * x + 1 = 0) : x = -1 → m = -1 :=
begin
  intros hx,
  rw hx at h,
  simp at h,
  -- Proving 3m + 3 = 0
  linarith,
end

end root_implies_value_of_m_l383_383244


namespace sixty_percent_of_40_greater_than_four_fifths_of_25_l383_383031

theorem sixty_percent_of_40_greater_than_four_fifths_of_25 :
  let x := (60 / 100 : ℝ) * 40
  let y := (4 / 5 : ℝ) * 25
  x - y = 4 :=
by
  sorry

end sixty_percent_of_40_greater_than_four_fifths_of_25_l383_383031


namespace james_drank_13_ounces_l383_383764

theorem james_drank_13_ounces (gallons_milk : ℕ) (ounces_in_gallon : ℕ) (remaining_ounces : ℕ) 
    (initial_gallons : gallons_milk = 3) (ounces_per_gallon : ounces_in_gallon = 128) 
    (milk_left : remaining_ounces = 371) : 
    let total_ounces := gallons_milk * ounces_in_gallon in
    total_ounces - remaining_ounces = 13 := 
by
  simp [initial_gallons, ounces_per_gallon, milk_left]
  sorry

end james_drank_13_ounces_l383_383764


namespace sum_of_solutions_of_quadratic_l383_383956

theorem sum_of_solutions_of_quadratic :
  let f := λ x : ℝ, x ^ 2 - 8 * x - 26
  let solutions := {x : ℝ | f x = 0}
  ∃ a b : ℝ, a ∈ solutions ∧ b ∈ solutions ∧ a ≠ b ∧ a + b = 8 → ∑ solutions = 8 := sorry

end sum_of_solutions_of_quadratic_l383_383956


namespace exists_larger_integer_l383_383386

theorem exists_larger_integer (a b : Nat) (h1 : b > a) (h2 : b - a = 5) (h3 : a * b = 88) :
  b = 11 :=
sorry

end exists_larger_integer_l383_383386


namespace countGoodTernarySequences_l383_383155

def isTernarySequence (seq : List Nat) : Prop :=
  ∀ x ∈ seq, x = 0 ∨ x = 1 ∨ x = 2

def sequenceDifference (seq : List Nat) : List Nat :=
  seq.zipWith (λ x y => abs (x - y)) (seq.tail)

def featureValue (seq : List Nat) : Nat :=
  match seq with
  | [] => 0
  | [x] => x
  | xs => featureValue (sequenceDifference xs)

def goodSequenceCount : Nat :=
  3^(2023) + 3^(1959) / 2 - 2^(2022)

theorem countGoodTernarySequences : 
  ∃ (sequences : Set (List Nat)), 
    (∀ seq ∈ sequences, isTernarySequence seq ∧ seq.length = 2023 ∧ featureValue seq = 0) ∧
    sequences.card = goodSequenceCount :=
sorry

end countGoodTernarySequences_l383_383155


namespace sum_of_distinct_roots_l383_383872

theorem sum_of_distinct_roots: 
  let P : Polynomial ℝ := Polynomial.Coeff [ -6, 13, -7, -1, 1 ]
  let roots := {x : ℝ | Polynomial.eval x P = 0}
  ∑ x in roots.toFinset, x = 0 :=
by
  sorry

end sum_of_distinct_roots_l383_383872


namespace min_value_expression_l383_383179

theorem min_value_expression : ∀ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 9 :=
by
  intro x y
  sorry

end min_value_expression_l383_383179


namespace log_expression_value_l383_383022

theorem log_expression_value :
  (log 3 243 / log 27 3) - (log 3 729 / log 9 3) = 3 :=
by sorry

end log_expression_value_l383_383022


namespace general_term_sequence_value_of_q_l383_383203

-- Given conditions for the arithmetic sequence
variable {a : ℕ → ℝ} {d q : ℝ}

-- Given conditions for specific problem statements
variable {a1 S3 : ℝ} {n : ℕ}

-- Definitions based on problem conditions
def S (n : ℕ) : ℝ := ∑ i in finset.range n, a i * q ^ (i - 1)

-- Question 1
theorem general_term_sequence (h_q : q = 1) (h_a1 : a 1 = 1) (h_S3 : S 3 = 15) :
  a n = 4 * n - 3 :=
sorry

-- Question 2
theorem value_of_q (h_a1d : a 1 = d) (h_geo_seq : S 2 * S 2 = S 1 * S 3) :
  q = -2 :=
sorry

end general_term_sequence_value_of_q_l383_383203


namespace pastry_trick_l383_383081

theorem pastry_trick (fillings : Fin 10) (n : ℕ) :
  ∃ n, (n = 36 ∧ ∀ remaining_pastries, 
    (remaining_pastries.length = 45 - n) → 
    (∃ remaining_filling ∈ fillings, true)) := 
sorry

end pastry_trick_l383_383081


namespace arrangement_of_six_students_l383_383334

theorem arrangement_of_six_students (students : Fin 6) (rows columns : ℕ) (A B : students)
  (h_rows : rows = 2) (h_columns : columns = 3)
  (h_A_not_same_row_or_column : ∀(i j : Fin 6), i ≠ j → (A ≠ B ∧ (A / 3 ≠ B / 3) ∧ (A % 3 ≠ B % 3))) :
  ∃ n : ℕ, n = 288 :=
by
  sorry

end arrangement_of_six_students_l383_383334


namespace inequality_for_cuberoot_solutions_l383_383156

noncomputable def cuberoot (x : ℚ) : ℝ := real.cbrt (x : ℝ)

theorem inequality_for_cuberoot_solutions 
    (R : ℚ) (hR : 0 ≤ R)
    (a b c d e f : ℤ)
    (h1 : (a = 1 ∧ b = 1 ∧ c = 2 ∧ d = 1 ∧ e = 1 ∧ f = 1) 
       ∨ (a = -1 ∧ b = -1 ∧ c = -2 ∧ d = -1 ∧ e = -1 ∧ f = -1)) :
    abs ((a * R^2 + b * R + c) / (d * R^2 + e * R + f) - cuberoot 2) < abs (R - cuberoot 2) := 
by
  sorry

end inequality_for_cuberoot_solutions_l383_383156


namespace largest_real_x_is_120_over_11_l383_383176

noncomputable def largest_real_x (x : ℝ) : Prop :=
  floor x / x = 11 / 12

theorem largest_real_x_is_120_over_11 :
  ∃ x, largest_real_x x ∧ x ≤ 120 / 11 :=
sorry

end largest_real_x_is_120_over_11_l383_383176


namespace coefficient_of_x3_in_expansion_l383_383351

theorem coefficient_of_x3_in_expansion :
  let f := (1 - x)^6 * (2 - x) 
  coefficient_of_x_power f 3 = -35 :=
sorry

end coefficient_of_x3_in_expansion_l383_383351


namespace solve_quadratic_1_solve_quadratic_2_l383_383820

theorem solve_quadratic_1 (x : ℝ) : 3 * x^2 - 8 * x + 4 = 0 ↔ x = 2/3 ∨ x = 2 := by
  sorry

theorem solve_quadratic_2 (x : ℝ) : (2 * x - 1)^2 = (x - 3)^2 ↔ x = 4/3 ∨ x = -2 := by
  sorry

end solve_quadratic_1_solve_quadratic_2_l383_383820


namespace roll_2_four_times_last_not_2_l383_383716

def probability_of_rolling_2_four_times_last_not_2 : ℚ :=
  (1/6)^4 * (5/6)

theorem roll_2_four_times_last_not_2 :
  probability_of_rolling_2_four_times_last_not_2 = 5 / 7776 := 
by
  sorry

end roll_2_four_times_last_not_2_l383_383716


namespace algebraic_sum_of_coefficients_vn_l383_383302

theorem algebraic_sum_of_coefficients_vn :
  ∃ (a b c : ℤ), (∀ (n : ℤ), n ≥ 1 → v (n + 1) - v n = 6 * n - 1)
  ∧ v 1 = 3
  ∧ a + b + c = 3 :=
sorry

end algebraic_sum_of_coefficients_vn_l383_383302


namespace median_and_mode_l383_383741

def scores : List ℕ := [100, 99, 99, 97, 90, 88]

theorem median_and_mode (s : List ℕ) (hs : s = scores) : 
  let sorted_scores := s.sort (λ a b => a > b) in
  let median := (sorted_scores.nth 2 + sorted_scores.nth 3) / 2 in
  let mode := (sorted_scores.nth 1) in
  median = 98 ∧ mode = 99 :=
by
  sorry

end median_and_mode_l383_383741


namespace vasya_pastry_trick_l383_383075

theorem vasya_pastry_trick :
  ∀ (pastries : Finset (Finset Nat))
    (filling_set : Finset Nat),
    (filling_set.card = 10) →
    (pastries.card = 45) →
    (∀ p ∈ pastries, p.card = 2 ∧ p ⊆ filling_set) →
    ∃ n, n = 36 ∧
    ∀ remain_p ∈ (pastries \ pastries.sort (λ x y, x < y)).take (45 - n), 
      ∃ f ∈ filling_set, f ∈ remain_p :=
begin
  sorry

end vasya_pastry_trick_l383_383075


namespace chessboard_tiling_impossible_l383_383003

theorem chessboard_tiling_impossible :
  let board_size := 8
  let total_squares := board_size * board_size
  let dominos_needed := 31
  let domino_squares := 2
  let remaining_squares := total_squares - 2
  let black_squares := 32
  let white_squares := 32
in
  (remaining_squares = 62) ∧ (black_squares - 1 = 31) ∧ (white_squares - 1 = 31) ∧
  (remaining_squares / domino_squares = dominos_needed) → false :=
by
  sorry

end chessboard_tiling_impossible_l383_383003


namespace Bianca_pictures_distribution_l383_383584

theorem Bianca_pictures_distribution 
(pictures_total : ℕ) 
(pictures_in_one_album : ℕ) 
(albums_remaining : ℕ) 
(h1 : pictures_total = 33)
(h2 : pictures_in_one_album = 27)
(h3 : albums_remaining = 3)
: (pictures_total - pictures_in_one_album) / albums_remaining = 2 := 
by 
  sorry

end Bianca_pictures_distribution_l383_383584


namespace gcd_digit_bound_l383_383734

theorem gcd_digit_bound (a b : ℕ) (h₁ : 10^6 ≤ a) (h₂ : a < 10^7) (h₃ : 10^6 ≤ b) (h₄ : b < 10^7) 
  (h₅ : 10^{10} ≤ lcm a b) (h₆ : lcm a b < 10^{11}) : 
  gcd a b < 10^4 :=
sorry

end gcd_digit_bound_l383_383734


namespace rectangular_prism_volume_l383_383319

theorem rectangular_prism_volume (h : ℝ) : 
  ∃ (V : ℝ), V = 120 * h :=
by
  sorry

end rectangular_prism_volume_l383_383319


namespace count_distinct_real_numbers_satisfying_g_g_c_eq_7_l383_383783

noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem count_distinct_real_numbers_satisfying_g_g_c_eq_7 : 
  ∃ S : set ℝ, (∀ c ∈ S, g (g c) = 7) ∧ S.finite ∧ S.card = 4 :=
by
  sorry

end count_distinct_real_numbers_satisfying_g_g_c_eq_7_l383_383783


namespace max_sum_fibonacci_l383_383979

-- Definitions and conditions
def fibonacci : ℕ → ℕ 
| 0     := 0
| 1     := 1
| (n+2) := fibonacci n + fibonacci (n+1)

def max_sum (n : ℕ) (a : ℕ → ℕ) : ℕ :=
finset.sum (finset.range (n + 1)) a

-- The problem statement
theorem max_sum_fibonacci (n : ℕ) (a : ℕ → ℕ) (h1 : a 0 = 1) (h2 : ∀ i, i ≤ n - 2 → a i ≥ a (i+1) + a (i+2)) (h3 : ∀ i ≤ n, 0 ≤ a i) (hn : 2 ≤ n) : 
∃ m : ℕ, max_sum n a = (fibonacci (n+1)) / (fibonacci (n-1)) := 
sorry

end max_sum_fibonacci_l383_383979


namespace simplified_value_of_dagger_l383_383860

def dagger (a b : ℚ) : ℚ := (a.num * b.num : ℤ) * (a.den : ℚ) / b.den

theorem simplified_value_of_dagger :
  dagger (Rat.mk 5 9) (Rat.mk 12 4) = 135 := by
  sorry

end simplified_value_of_dagger_l383_383860


namespace concentrate_amount_l383_383121

def parts_concentrate : ℤ := 1
def parts_water : ℤ := 5
def part_ratio : ℤ := parts_concentrate + parts_water -- Total parts
def servings : ℤ := 375
def volume_per_serving : ℤ := 150
def total_volume : ℤ := servings * volume_per_serving -- Total volume of orange juice
def volume_per_part : ℤ := total_volume / part_ratio -- Volume per part of mixture

theorem concentrate_amount :
  volume_per_part = 9375 :=
by
  sorry

end concentrate_amount_l383_383121


namespace least_number_to_add_l383_383866

theorem least_number_to_add (x : ℕ) (h : 1055 % 23 = 20) : x = 3 :=
by
  -- Proof goes here.
  sorry

end least_number_to_add_l383_383866


namespace right_triangle_hypotenuse_length_l383_383556

theorem right_triangle_hypotenuse_length (a b : ℝ) (h_triangle : a = 15 ∧ b = 36) :
  ∃ (h : ℝ), h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  · exact rfl
  · rw [h_triangle.1, h_triangle.2]
    norm_num

end right_triangle_hypotenuse_length_l383_383556


namespace bd_le_q2_l383_383686

theorem bd_le_q2 (a b c d p q : ℝ) (h1 : a * b + c * d = 2 * p * q) (h2 : a * c ≥ p^2 ∧ p^2 > 0) : b * d ≤ q^2 :=
sorry

end bd_le_q2_l383_383686


namespace vasya_pastry_trick_l383_383074

theorem vasya_pastry_trick :
  ∀ (pastries : Finset (Finset Nat))
    (filling_set : Finset Nat),
    (filling_set.card = 10) →
    (pastries.card = 45) →
    (∀ p ∈ pastries, p.card = 2 ∧ p ⊆ filling_set) →
    ∃ n, n = 36 ∧
    ∀ remain_p ∈ (pastries \ pastries.sort (λ x y, x < y)).take (45 - n), 
      ∃ f ∈ filling_set, f ∈ remain_p :=
begin
  sorry

end vasya_pastry_trick_l383_383074


namespace perpendicular_segments_l383_383574

variables (A B B' M M' N : Point)
variable (circle : Circle)
variable (chord : Chord circle)
variable (arc : Arc circle)
variables (midpoint_M : is_midpoint M arc)
variables (midpoint_N : is_midpoint N arc)
variables (rotated_B : RotatedPoint A B B')
variables (rotated_M : RotatedPoint A M M')
variable (midpoint_BB' : is_midpoint (midpoint B B') (segment B B'))
variables (angle_perpendicular : Perpendicular (segment (midpoint B B') N) (segment (midpoint B B') M'))

theorem perpendicular_segments : angle_perpendicular :=
sorry

end perpendicular_segments_l383_383574


namespace four_digit_number_condition_l383_383377

theorem four_digit_number_condition (x n : ℕ) (h1 : n = 2000 + x) (h2 : 10 * x + 2 = 2 * n + 66) : n = 2508 :=
sorry

end four_digit_number_condition_l383_383377


namespace total_cost_of_tickets_l383_383059

theorem total_cost_of_tickets (num_family_members num_adult_tickets num_children_tickets : ℕ)
    (cost_adult_ticket cost_children_ticket total_cost : ℝ) 
    (h1 : num_family_members = 7) 
    (h2 : cost_adult_ticket = 21) 
    (h3 : cost_children_ticket = 14) 
    (h4 : num_adult_tickets = 4) 
    (h5 : num_children_tickets = num_family_members - num_adult_tickets) 
    (h6 : total_cost = num_adult_tickets * cost_adult_ticket + num_children_tickets * cost_children_ticket) :
    total_cost = 126 :=
by
  sorry

end total_cost_of_tickets_l383_383059


namespace right_triangle_hypotenuse_length_l383_383505

theorem right_triangle_hypotenuse_length (a b : ℕ) (h1 : a = 15) (h2 : b = 36) : 
  ∃ c : ℕ, c * c = a * a + b * b ∧ c = 39 := 
by
  have hyp_square := 225 + 1296 
  have h_calculation : 15 * 15 + 36 * 36 = 1521 := by
    calc
      15 * 15 = 225 : rfl
      36 * 36 = 1296 : rfl
      225 + 1296 = 1521 : rfl
  use 39
  split
  exact h_calculation
  rfl

end right_triangle_hypotenuse_length_l383_383505


namespace minimum_sum_PA_PF_l383_383682

noncomputable theory

open Real

-- Definition of parabola and points
def parabola (P : ℝ × ℝ) : Prop :=
  (1 / 4) * P.snd ^ 2 = P.fst

-- Definition of Euclidean distance
def dist (P Q : ℝ × ℝ) : ℝ :=
  ((P.fst - Q.fst) ^ 2 + (P.snd - Q.snd) ^ 2).sqrt

-- Define constants in the problem
def F : ℝ × ℝ := (1, 0)         -- Assuming focus for the parabolic form given
def A : ℝ × ℝ := (2, 2)

-- Main statement of the theorem
theorem minimum_sum_PA_PF :
  ∃ P : ℝ × ℝ, parabola P → dist P A + dist P F = 3 := sorry  -- the 'sorry' here is used to skip proof.

end minimum_sum_PA_PF_l383_383682


namespace sqrt_31_between_5_and_6_l383_383162

theorem sqrt_31_between_5_and_6
  (h1 : Real.sqrt 25 = 5)
  (h2 : Real.sqrt 36 = 6)
  (h3 : 25 < 31)
  (h4 : 31 < 36) :
  5 < Real.sqrt 31 ∧ Real.sqrt 31 < 6 :=
sorry

end sqrt_31_between_5_and_6_l383_383162


namespace triangle_x_value_l383_383914

-- Declare that the function will not be computed
noncomputable def similar_triangles (a b c d : ℝ) : Prop :=
  a / b = c / d

theorem triangle_x_value :
  (similar_triangles 12 x 9 6) → (x = 8) :=
by
  intro h
  have h_ratio := (by simp [similar_triangles] at h; exact h)
  rw [h_ratio] at ⊢
  norm_num
  have h1 : 1.5 * x = 12 := by rw [←h_ratio]
  simpa using (by linarith)

end triangle_x_value_l383_383914


namespace sum_of_products_nonzero_l383_383744

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2*k + 1

noncomputable def row_product (mat : ℕ → ℕ → ℤ) (i : ℕ) (n : ℕ) : ℤ :=
  ∏ j in finset.range(n), mat i j

noncomputable def col_product (mat : ℕ → ℕ → ℤ) (j : ℕ) (n : ℕ) : ℤ :=
  ∏ i in finset.range(n), mat i j

theorem sum_of_products_nonzero (n : ℕ) (mat : ℕ → ℕ → ℤ) 
  (h_odd : is_odd n) 
  (h_elements : ∀ i j, mat i j = 1 ∨ mat i j = -1) :
  ∑ i in finset.range(n), row_product mat i n + 
  ∑ j in finset.range(n), col_product mat j n ≠ 0 :=
sorry

end sum_of_products_nonzero_l383_383744


namespace cost_of_watch_l383_383152

variable (saved amount_needed total_cost : ℕ)

-- Conditions
def connie_saved : Prop := saved = 39
def connie_needs : Prop := amount_needed = 16

-- Theorem to prove
theorem cost_of_watch : connie_saved saved → connie_needs amount_needed → total_cost = 55 :=
by
  sorry

end cost_of_watch_l383_383152


namespace find_r_minus4_l383_383110

noncomputable def p : ℕ → ℚ := sorry -- Define p(x) symbolically

theorem find_r_minus4 :
  let r := λ x : ℚ, (1 / 10) * x^2 - (9 / 10) * x + (19 / 10),
  r (-4) = 71 / 10 :=
by
  have p1 := 2
  have p3 := -1
  have pm2 := 5
  have r1 : r (1:ℚ) = 2 := sorry -- These are the conditions we derived from problem
  have r2 : r (3:ℚ) = -1 := sorry
  have r3 : r (-2:ℚ) = 5 := sorry
  show r (-4:ℚ) = 71 / 10
  sorry

end find_r_minus4_l383_383110


namespace stratified_sampling_sum_l383_383919

theorem stratified_sampling_sum (
  grains : ℕ := 40,
  vegetable_oils : ℕ := 10,
  animal_foods : ℕ := 30,
  fruits_vegetables : ℕ := 20,
  sample_size : ℕ := 20
) : 
  let total_foods := grains + vegetable_oils + animal_foods + fruits_vegetables in
  let sampling_ratio := sample_size / total_foods in
  (vegetable_oils * sampling_ratio + fruits_vegetables * sampling_ratio) = 6 :=
by
  sorry

end stratified_sampling_sum_l383_383919


namespace sum_of_roots_l383_383786

open Real

theorem sum_of_roots (x1 x2 k c : ℝ) (h1 : 4 * x1^2 - k * x1 = c) (h2 : 4 * x2^2 - k * x2 = c) (h3 : x1 ≠ x2) :
  x1 + x2 = k / 4 :=
by
  sorry

end sum_of_roots_l383_383786


namespace hex_conversion_sum_l383_383154

-- Convert hexadecimal E78 to decimal
def hex_to_decimal (h : String) : Nat :=
  match h with
  | "E78" => 3704
  | _ => 0

-- Convert decimal to radix 7
def decimal_to_radix7 (d : Nat) : String :=
  match d with
  | 3704 => "13541"
  | _ => ""

-- Convert radix 7 to decimal
def radix7_to_decimal (r : String) : Nat :=
  match r with
  | "13541" => 3704
  | _ => 0

-- Convert decimal to hexadecimal
def decimal_to_hex (d : Nat) : String :=
  match d with
  | 3704 => "E78"
  | 7408 => "1CF0"
  | _ => ""

theorem hex_conversion_sum :
  let initial_hex : String := "E78"
  let final_decimal := 3704 
  let final_hex := decimal_to_hex (final_decimal)
  let final_sum := hex_to_decimal initial_hex + final_decimal
  (decimal_to_hex final_sum) = "1CF0" :=
by
  sorry

end hex_conversion_sum_l383_383154


namespace num_divisors_of_30_l383_383238

theorem num_divisors_of_30 : 
  (∀ n : ℕ, n > 0 → (30 = 2^1 * 3^1 * 5^1) → (∀ k : ℕ, 0 < k ∧ k ∣ 30 → ∃ m : ℕ, k = 2^m ∧ k ∣ 30)) → 
  ∃ num_divisors : ℕ, num_divisors = 8 := 
by 
  sorry

end num_divisors_of_30_l383_383238


namespace term_in_sequence_is_24_l383_383691

theorem term_in_sequence_is_24 {n : ℕ} :
  (n : ℝ) / (n + 1) = 0.96 → n = 24 :=
by
  -- Placeholder for the proof.
  sorry

end term_in_sequence_is_24_l383_383691


namespace find_phi_l383_383256

theorem find_phi (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π)
  (h3 : (∀ x, 3 * sin (2 * x + φ) = 3 * sin (2 * (2 * π / 3 - x) + φ))) :
  φ = π / 3 :=
by sorry

end find_phi_l383_383256


namespace least_positive_integer_l383_383618

theorem least_positive_integer (x : ℕ) (h : x + 5600 ≡ 325 [MOD 15]) : x = 5 :=
sorry

end least_positive_integer_l383_383618


namespace right_triangle_hypotenuse_length_l383_383526

theorem right_triangle_hypotenuse_length :
  ∀ (a b h : ℕ), a = 15 → b = 36 → h^2 = a^2 + b^2 → h = 39 :=
by
  intros a b h ha hb hyp
  -- In the proof, we would use ha, hb, and hyp to show h = 39
  sorry

end right_triangle_hypotenuse_length_l383_383526


namespace vasya_pastry_trick_l383_383076

theorem vasya_pastry_trick :
  ∀ (pastries : Finset (Finset Nat))
    (filling_set : Finset Nat),
    (filling_set.card = 10) →
    (pastries.card = 45) →
    (∀ p ∈ pastries, p.card = 2 ∧ p ⊆ filling_set) →
    ∃ n, n = 36 ∧
    ∀ remain_p ∈ (pastries \ pastries.sort (λ x y, x < y)).take (45 - n), 
      ∃ f ∈ filling_set, f ∈ remain_p :=
begin
  sorry

end vasya_pastry_trick_l383_383076


namespace hypotenuse_length_l383_383499

theorem hypotenuse_length (a b : ℤ) (h₀ : a = 15) (h₁ : b = 36) : 
  ∃ c : ℤ, c^2 = a^2 + b^2 ∧ c = 39 := 
by {
  sorry
}

end hypotenuse_length_l383_383499


namespace faith_change_l383_383166

def flour_cost : ℕ := 5
def cake_stand_cost : ℕ := 28
def two_twenty_bills : ℕ := 2 * 20
def loose_coins : ℕ := 3
def total_cost : ℕ := flour_cost + cake_stand_cost
def total_given : ℕ := two_twenty_bills + loose_coins
def change : ℕ := total_given - total_cost

theorem faith_change : change = 10 := by
  sorry

end faith_change_l383_383166


namespace find_m_value_l383_383209

theorem find_m_value (m x y : ℝ) (hx : x = 2) (hy : y = -1) (h_eq : m * x - y = 3) : m = 1 :=
by
  sorry

end find_m_value_l383_383209


namespace cargo_loaded_in_Bahamas_l383_383918

theorem cargo_loaded_in_Bahamas (initial final : ℕ) (h1 : initial = 5973) (h2 : final = 14696) :
  final - initial = 8723 := by
  rw [h1, h2]
  norm_num

end cargo_loaded_in_Bahamas_l383_383918


namespace gcd_digit_bound_l383_383729

theorem gcd_digit_bound (a b : ℕ) (h1 : a < 10^7) (h2 : b < 10^7) (h3 : 10^10 ≤ Nat.lcm a b) :
  Nat.gcd a b < 10^4 :=
by
  sorry

end gcd_digit_bound_l383_383729


namespace sum_of_sines_larger_in_smaller_difference_equilateral_triangle_max_sum_of_sines_l383_383806

-- Conditions
variables {α β γ β' γ' : ℝ}

-- Noncomputable because we are dealing with transcendental functions like sin and cos
noncomputable def sum_sines_triangle (α β γ : ℝ) : ℝ :=
  sin α + sin β + sin γ

noncomputable def difference_between_angles (β γ : ℝ) : ℝ :=
  abs (β - γ)

-- Main theorem statement
theorem sum_of_sines_larger_in_smaller_difference
  (h_shared_angle : α = α)
  (h_condition : sum_sines_triangle 0 β γ > sum_sines_triangle 0 β' γ') :
  sum_sines_triangle α β γ > sum_sines_triangle α β' γ' :=
sorry

-- Secondary theorem statement
theorem equilateral_triangle_max_sum_of_sines (α : ℝ) :
  is_equilateral α →  sum_sines_triangle α α α = 3 * sin α :=
sorry

end sum_of_sines_larger_in_smaller_difference_equilateral_triangle_max_sum_of_sines_l383_383806


namespace minimum_pies_for_trick_l383_383090

-- Definitions from conditions
def num_fillings : ℕ := 10
def num_pastries := (num_fillings * (num_fillings - 1)) / 2
def min_pies_for_trick (n : ℕ) : Prop :=
  ∀ remaining_pies : ℕ, remaining_pies = num_pastries - n → remaining_pies ≤ 9

theorem minimum_pies_for_trick : ∃ n : ℕ, min_pies_for_trick n ∧ n = 36 :=
by
  -- We need to show that there exists n such that,
  -- min_pies_for_trick holds and n = 36
  existsi (36 : ℕ)
  -- remainder of the proof (step solution) skipped
  sorry

end minimum_pies_for_trick_l383_383090


namespace hypotenuse_right_triangle_l383_383546

theorem hypotenuse_right_triangle (a b : ℕ) (h1 : a = 15) (h2 : b = 36) :
  ∃ c, c ^ 2 = a ^ 2 + b ^ 2 ∧ c = 39 :=
by
  sorry

end hypotenuse_right_triangle_l383_383546


namespace bamboo_tube_rice_capacity_l383_383435

theorem bamboo_tube_rice_capacity :
  ∃ (a d : ℝ), 3 * a + 3 * d * (1 + 2) = 4.5 ∧ 
               4 * (a + 5 * d) + 4 * d * (6 + 7 + 8) = 3.8 ∧ 
               (a + 3 * d) + (a + 4 * d) = 2.5 :=
by
  sorry

end bamboo_tube_rice_capacity_l383_383435


namespace fraction_product_is_729_l383_383413

-- Define the sequence of fractions
noncomputable def fraction_sequence : List ℚ :=
  [1/3, 9, 1/27, 81, 1/243, 729, 1/729, 6561, 1/2187, 59049, 1/6561, 59049/81]

-- Define the condition that each supposed pair product is 3
def pair_product_condition (seq : List ℚ) (i : ℕ) : Prop :=
  seq[2 * i] * seq[2 * i + 1] = 3

-- State the main theorem
theorem fraction_product_is_729 :
  fraction_sequence.prod = 729 := by
  sorry

end fraction_product_is_729_l383_383413


namespace neg_of_exists_eq_one_l383_383685

theorem neg_of_exists_eq_one : 
  (¬ (∃ x_0 : ℝ, 2 ^ x_0 = 1)) = (∀ x : ℝ, 2 ^ x ≠ 1) :=
by
  sorry

end neg_of_exists_eq_one_l383_383685


namespace cow_water_consumption_l383_383799

theorem cow_water_consumption :
  (∀ (C : ℕ), (40 * C + 400 * (C / 4)) * 7 = 78400 → C = 80) := 
by 
  intro C,
  assume h : (40 * C + 400 * (C / 4)) * 7 = 78400,
  sorry

end cow_water_consumption_l383_383799


namespace length_of_goods_train_l383_383459

-- Define the problem conditions
def train_speed_kmph : ℕ := 72
def platform_length_m : ℕ := 260
def time_to_cross_platform_s : ℕ := 26

-- Define the expected length of the train
def length_of_train_m : ℕ := 260

-- Lean 4 theorem statement
theorem length_of_goods_train :
  let speed_m_per_s := train_speed_kmph * 5 / 18 in
  let distance_covered_m := speed_m_per_s * time_to_cross_platform_s in
  length_of_train_m = distance_covered_m - platform_length_m := by
  -- Proof to be provided
  sorry

end length_of_goods_train_l383_383459


namespace time_to_cross_signal_post_l383_383048

-- Definition of the conditions
def length_of_train : ℝ := 600  -- in meters
def time_to_cross_bridge : ℝ := 8  -- in minutes
def length_of_bridge : ℝ := 7200  -- in meters

-- Equivalent statement
theorem time_to_cross_signal_post (constant_speed : ℝ) (t : ℝ) 
  (h1 : constant_speed * t = length_of_train) 
  (h2 : constant_speed * time_to_cross_bridge = length_of_train + length_of_bridge) : 
  t * 60 = 36.9 := 
sorry

end time_to_cross_signal_post_l383_383048


namespace number_of_outfits_l383_383327

theorem number_of_outfits :
  let trousers := 5 in
  let shirts := 8 in
  let jackets := 2 in
  trousers * shirts * jackets = 80 :=
by
  trivial

end number_of_outfits_l383_383327


namespace f_is_periodic_and_positive_subtraction_l383_383212

def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

def f (x : ℝ) : ℝ := if 0 ≤ x ∧ x ≤ 1 then x - Math.sin x else sorry

theorem f_is_periodic_and_positive_subtraction
  (h_even_f : is_even_function f)
  (h_even_f_shifted : is_even_function (λ x, f (x+1))) :
  f (-3/2) - f (Float.pi / 2) > 0 :=
by {
  sorry
}

end f_is_periodic_and_positive_subtraction_l383_383212


namespace cupcake_price_l383_383339

theorem cupcake_price
  (x : ℝ)
  (h1 : 5 * x + 6 * 1 + 4 * 2 + 15 * 0.6 = 33) : x = 2 :=
by
  sorry

end cupcake_price_l383_383339


namespace day_of_week_in_100_days_l383_383403

theorem day_of_week_in_100_days (start_day : ℕ) (h : start_day = 5) : 
  (start_day + 100) % 7 = 0 := 
by
  cases h with 
  | rfl => -- start_day is Friday, which is represented as 5
  sorry

end day_of_week_in_100_days_l383_383403


namespace smallest_n_divisible_by_100_l383_383791

theorem smallest_n_divisible_by_100 (n : ℕ) :
  (∀ m : ℕ+, ∃ S ⊆ (finset.range n.succ), (S.prod id % 100 = m % 100)) ↔ n = 17 :=
begin
  sorry
end

end smallest_n_divisible_by_100_l383_383791


namespace find_a_b_sum_l383_383363

-- Definitions for the conditions
def equation1 (a : ℝ) : Prop := 3 = (1 / 3) * 6 + a
def equation2 (b : ℝ) : Prop := 6 = (1 / 3) * 3 + b

theorem find_a_b_sum : 
  ∃ (a b : ℝ), equation1 a ∧ equation2 b ∧ (a + b = 6) :=
sorry

end find_a_b_sum_l383_383363


namespace problem_solution_l383_383923

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_strictly_decreasing (f : ℝ → ℝ) : Prop := ∀ ⦃x y⦄, 0 < x → x < y → f y < f x

def ln_inv_abs (x : ℝ) : ℝ := Real.log (1 / |x|)
def x_cubed (x : ℝ) : ℝ := x ^ 3
def ln_add_sqrt (x : ℝ) : ℝ := Real.log (x + Real.sqrt (x ^ 2 + 1))
noncomputable def sin_squared (x : ℝ) : ℝ := Real.sin x ^ 2

theorem problem_solution :
  (is_even ln_inv_abs ∧ is_strictly_decreasing ln_inv_abs) ∧
  ¬ (is_even x_cubed) ∧
  ¬ (is_even ln_add_sqrt) ∧ 
  (is_even sin_squared ∧ ¬ is_strictly_decreasing sin_squared) :=
by 
  sorry

end problem_solution_l383_383923


namespace find_range_of_m_l383_383998

variables {f : ℝ → ℝ}

-- Proposition p
def prop_p (m : ℝ) : Prop :=
  monotone_decreasing f ∧ f (m + 1) < f (3 - 2 * m)

-- Proposition q
def prop_q (m : ℝ) : Prop :=
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ (π / 2) ∧ m = -(sin x)^2 - 2*sin x + 1

-- The combined proposition
theorem find_range_of_m (m : ℝ) :
  prop_p m ∧ prop_q m → m ∈ Ioc (2 / 3 : ℝ) 1 :=
sorry

end find_range_of_m_l383_383998


namespace minimize_angle_AXB_perpendicular_l383_383800

-- Definitions used in the Lean 4 proof statement
variables {circle : Type} [metric_space circle] [normed_group circle] [normed_space ℝ circle]

-- Lean translation of the math proof problem
theorem minimize_angle_AXB_perpendicular
  (O X : circle) 
  (Y : circle) 
  (hO_center : is_center O circle)
  (hX_on_circumference : on_circumference X circle)
  (hY_on_diameter : on_diameter Y X O)
  : ∃ (A B : circle), on_chord A B Y ∧ is_perpendicular (line O X) (line A B) ∧ ∀ (C D : circle) (hC : on_chord C D Y), ∠ A X B ≤ ∠ C X D := 
sorry

end minimize_angle_AXB_perpendicular_l383_383800


namespace sum_of_sines_larger_in_smaller_difference_equilateral_triangle_max_sum_of_sines_l383_383805

-- Conditions
variables {α β γ β' γ' : ℝ}

-- Noncomputable because we are dealing with transcendental functions like sin and cos
noncomputable def sum_sines_triangle (α β γ : ℝ) : ℝ :=
  sin α + sin β + sin γ

noncomputable def difference_between_angles (β γ : ℝ) : ℝ :=
  abs (β - γ)

-- Main theorem statement
theorem sum_of_sines_larger_in_smaller_difference
  (h_shared_angle : α = α)
  (h_condition : sum_sines_triangle 0 β γ > sum_sines_triangle 0 β' γ') :
  sum_sines_triangle α β γ > sum_sines_triangle α β' γ' :=
sorry

-- Secondary theorem statement
theorem equilateral_triangle_max_sum_of_sines (α : ℝ) :
  is_equilateral α →  sum_sines_triangle α α α = 3 * sin α :=
sorry

end sum_of_sines_larger_in_smaller_difference_equilateral_triangle_max_sum_of_sines_l383_383805


namespace evaluate_statements_l383_383628

theorem evaluate_statements (a m n M N : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  ¬ (a^m + a^n = a^(m+n)) ∧
  ¬ ((a^m)^n = a^(m^n)) ∧
  (M = N → log a M = log a N) ∧
  (log a (M^2) = log a (N^2) → M = N) :=
by
  sorry

end evaluate_statements_l383_383628


namespace right_triangle_hypotenuse_length_l383_383528

theorem right_triangle_hypotenuse_length :
  ∀ (a b h : ℕ), a = 15 → b = 36 → h^2 = a^2 + b^2 → h = 39 :=
by
  intros a b h ha hb hyp
  -- In the proof, we would use ha, hb, and hyp to show h = 39
  sorry

end right_triangle_hypotenuse_length_l383_383528


namespace n_divisible_by_3_l383_383905

theorem n_divisible_by_3 (n : ℕ) (triangulated : convex_ngon_triangulation n) :
  (∀ v ∈ triangulated.vertices, odd (triangulated.vertex_degree v)) → 3 ∣ n :=
by
  sorry

end n_divisible_by_3_l383_383905


namespace factorial_comparison_l383_383718

theorem factorial_comparison (n k : ℕ) (h_pos_n : 0 < n) (h_pos_k : 0 < k)
  (h_ineq : ∀ n k : ℕ, 0 < n ∧ 0 < k → (n+1) * (n+2) * ... * (n+k) < (n+k)^k) :
  (100!)^(99!) * (99!)^(100!) > (100!)! :=
by 
  -- Sorry replaces the proof steps
  sorry

end factorial_comparison_l383_383718


namespace moles_of_water_formed_l383_383621

open Nat

theorem moles_of_water_formed
  (moles_methane : ℕ) (moles_oxygen : ℕ)
  (h_eq : moles_methane = 3) (h_eq2 : moles_oxygen = 6)
  (balanced_eq : ∀ (moles_methane : ℕ) (moles_oxygen : ℕ), 1 * moles_methane <= 2 * moles_oxygen) :
  2 * moles_methane = 6 :=
by
  have h : 2 * 3 = 6 := by norm_num
  exact h

end moles_of_water_formed_l383_383621


namespace hypotenuse_length_l383_383475

theorem hypotenuse_length (a b c : ℕ) (h₀ : a = 15) (h₁ : b = 36) (h₂ : a^2 + b^2 = c^2) : c = 39 :=
by
  -- Proof is omitted
  sorry

end hypotenuse_length_l383_383475


namespace marks_proof_l383_383127

noncomputable def max_marks : ℕ :=
  let passing_marks := 130 + 14
  in 100 * passing_marks / 36

theorem marks_proof : max_marks = 400 :=
by
  simp [max_marks]
  norm_num
  sorry

end marks_proof_l383_383127


namespace option_C_odd_monotonic_increasing_l383_383580

-- Definition of the function f
def f (x : ℝ) : ℝ := x + 1/x

-- Definition of monotonic increasing on an interval
def monotonic_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x ≤ f y

-- Definition of an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = - (f x)

theorem option_C_odd_monotonic_increasing :
  odd_function f ∧ monotonic_increasing_on f (Set.Ioi 1) :=
sorry

end option_C_odd_monotonic_increasing_l383_383580


namespace sum_of_1_to_17_is_odd_l383_383762

-- Define the set of natural numbers from 1 to 17
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

-- Proof that the sum of these numbers is odd
theorem sum_of_1_to_17_is_odd : (List.sum nums) % 2 = 1 := 
by
  sorry  -- Proof goes here

end sum_of_1_to_17_is_odd_l383_383762


namespace right_triangle_hypotenuse_l383_383538

theorem right_triangle_hypotenuse (a b : ℝ) (h : a^2 + b^2 = 39^2) : a = 15 ∧ b = 36 := by
  sorry

end right_triangle_hypotenuse_l383_383538


namespace right_triangle_hypotenuse_l383_383537

theorem right_triangle_hypotenuse (a b : ℝ) (h : a^2 + b^2 = 39^2) : a = 15 ∧ b = 36 := by
  sorry

end right_triangle_hypotenuse_l383_383537


namespace complex_poly_eq_zero_l383_383246

theorem complex_poly_eq_zero (z : ℂ) (h : z = 1 - complex.i) : z ^ 2 - 2 * z + 2 = 0 :=
by
  sorry

end complex_poly_eq_zero_l383_383246


namespace union_complement_eq_l383_383305

def U : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 2, 3}

theorem union_complement_eq : M ∪ (U \ N) = {0, 1, 2} := by
  sorry

end union_complement_eq_l383_383305


namespace part1_part2_l383_383636

-- Definitions
def A (x : ℝ) : Prop := 3 * x^2 - 8 * x + 4 > 0
def B (x a : ℝ) : Prop := (-2) / (x^2 - a * x - 2 * a^2) < 0

-- The two proof goals
theorem part1 (a : ℝ) :
  (∀ x : ℝ, (x < 2 / 3 ∨ x > 2) ∨ (a > 0 ∧ (x > 2 * a ∨ x < -a) ∨ a < 0 ∧ (x > -a ∨ x < 2 * a))) → 
  (a ∈ (-∞, -2] ∪ [1, +∞)) := sorry

theorem part2 (a : ℝ) :
  ((∀ x : ℝ, A x → B x a) ∧ ∃ x : ℝ, ¬ (B x a → A x)) →
  (a ∈ (-∞, -2] ∪ [1, +∞)) := sorry

end part1_part2_l383_383636


namespace baker_cakes_l383_383136

theorem baker_cakes (cakes_made cakes_sold : ℕ) (h1 : cakes_made = 149) (h2 : cakes_sold = 10) : 
  cakes_made - cakes_sold = 139 := 
by {
  rw [h1, h2],
  rfl,
}

end baker_cakes_l383_383136


namespace tyler_meals_l383_383858

noncomputable def meals : ℕ :=
  let meats := 4 in
  let vegetables := Nat.choose 5 3 in
  let desserts := Nat.choose 5 2 in
  meats * vegetables * desserts

theorem tyler_meals : meals = 400 := by
  sorry

end tyler_meals_l383_383858


namespace Carol_weight_equals_nine_l383_383311

-- conditions in Lean definitions
def Mildred_weight : ℤ := 59
def weight_difference : ℤ := 50

-- problem statement to prove in Lean 4
theorem Carol_weight_equals_nine (Carol_weight : ℤ) :
  Mildred_weight = Carol_weight + weight_difference → Carol_weight = 9 :=
by
  sorry

end Carol_weight_equals_nine_l383_383311


namespace prob_two_equals_one_third_of_set_l383_383102

noncomputable def prob_d (d : ℕ) : ℝ := log10 ((d^2 + 1) / d^2)

theorem prob_two_equals_one_third_of_set :
  prob_d 2 = 1 / 3 * (prob_d 2 + prob_d 5 + prob_d 6) :=
by
  sorry

end prob_two_equals_one_third_of_set_l383_383102


namespace length_AD_l383_383281

-- Define the points and their distances according to the conditions
structure Quadrilateral :=
  (A B C D: ℝ × ℝ)

def AB (q : Quadrilateral) := (q.A.1 - q.B.1)^2 + (q.A.2 - q.B.2)^2
def BC (q : Quadrilateral) := (q.B.1 - q.C.1)^2 + (q.B.2 - q.C.2)^2
def CD (q : Quadrilateral) := (q.C.1 - q.D.1)^2 + (q.C.2 - q.D.2)^2
def angle_ABC (q : Quadrilateral) := 120 * Real.pi / 180 -- Convert degrees to radians
def angle_BCD (q : Quadrilateral) := 90 * Real.pi / 180

-- State the theorem
theorem length_AD (q : Quadrilateral)
  (h_AB : AB q = 1^2) (h_BC : BC q = 2^2) (h_CD : CD q = (sqrt 3)^2)
  (h_angle_ABC : angle_ABC q = 2 * Real.pi / 3) (h_angle_BCD : angle_BCD q = Real.pi / 2) :
  sqrt ((q.A.1 - q.D.1)^2 + (q.A.2 - q.D.2)^2) = sqrt 7 :=
sorry

end length_AD_l383_383281


namespace fraction_product_is_243_l383_383410

/- Define the sequence of fractions -/
def fraction_seq : list ℚ :=
  [1/3, 9, 1/27, 81, 1/243, 243, 1/729, 729, 1/2187, 6561]

/- Define the product of the sequence of fractions -/
def product_fractions : ℚ :=
  (fraction_seq.foldl (*) 1)

/- The theorem we want to prove -/
theorem fraction_product_is_243 : product_fractions = 243 := 
  sorry

end fraction_product_is_243_l383_383410


namespace range_of_a_l383_383639

variable (a : ℝ)

def A : set ℝ := { x | x^2 - 2*x - 3 ≤ 0 }

theorem range_of_a (h : a ∈ A) : -1 ≤ a ∧ a ≤ 3 :=
sorry

end range_of_a_l383_383639


namespace suitcase_lock_settings_l383_383566

-- Define the conditions as Lean definitions
def dials := 4
def digits := finset.range 8  -- Digits from 0 to 7
def all_digits_different (s : finset ℕ) : Prop := s.card = dials

-- The theorem statement
theorem suitcase_lock_settings : 
  (finset.univ.filter all_digits_different).card = 1680 :=
sorry

end suitcase_lock_settings_l383_383566


namespace find_f_at_2_l383_383990

def f : ℝ → ℝ := 
λ x, if x ≤ 1 then 1 / (x - 2) else f (x - 4)

theorem find_f_at_2 : f 2 = -1 / 4 := 
by
  sorry

end find_f_at_2_l383_383990


namespace Andrea_stops_after_HH_l383_383927

noncomputable def probability_stops_after_HH : ℝ :=
by
  let P_H := (1/2 : ℝ)
  let P_HH := P_H * P_H
  exact P_HH

theorem Andrea_stops_after_HH (P_H := (1/2 : ℝ)) : probability_stops_after_HH = 1/4 := 
by
  unfold probability_stops_after_HH
  rw [mul_self, mul_one_half, mul_one_half]
  exact rfl

/- Definitions used i.e., P_H and P_HH are directly from the conditions of flipping coins and stopping criteria -/
/- The math proof problem now ensures that the probability calculation for Andrea stopping after flipping HH is shown to be 1/4 -/

end Andrea_stops_after_HH_l383_383927


namespace find_C_coordinates_l383_383608

-- Define the coordinates of the vertices
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨ 2, 0 ⟩
def B : Point := ⟨ 0, 4 ⟩

-- Define the Euler line equation
def euler_line_eq (P : Point) : Prop :=
  P.x - P.y + 2 = 0

-- Define the centroid of triangle ABC
def centroid (A B C : Point) : Point :=
  ⟨(A.x + B.x + C.x) / 3, (A.y + B.y + C.y) / 3⟩

-- Theorem stating the required proof
theorem find_C_coordinates (C : Point) (h : euler_line_eq (centroid A B C)) : C = ⟨-4, 0⟩ :=
sorry

end find_C_coordinates_l383_383608


namespace exists_equal_length_segments_l383_383370

theorem exists_equal_length_segments (n : ℕ) (A : Fin (2 * n) → Point) (h: regular_polygon A (2 * n)) (hn : n % 4 = 2 ∨ n % 4 = 3) :
  ∃ (i j k l: Fin (2 * n)), i ≠ j ∧ k ≠ l ∧ segment_length (A i) (A j) = segment_length (A k) (A l) := 
sorry

end exists_equal_length_segments_l383_383370


namespace june_10_2013_is_tuesday_l383_383815

-- Declare that June 10, 2010 is a Friday
def initial_year : ℕ := 2010
def initial_weekday : ℕ := 5 -- Representing Friday as 5

-- Define function to determine if a year is a leap year
def is_leap_year (year : ℕ) : Prop := 
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

-- Calculate the weekday of June 10 in the next year based on leap or non-leap year
def next_weekday (year : ℕ) (current_weekday : ℕ) : ℕ :=
  if is_leap_year year then (current_weekday + 2) % 7
  else (current_weekday + 1) % 7

-- Mathematical problem statement to proof
theorem june_10_2013_is_tuesday : 
  ∃ year : ℕ, year > initial_year ∧ 
  (∀ y ∈ [2010, 2011, 2012, 2013], ∀ wd, wd = initial_weekday → 
    ∃ wd', wd' = if is_leap_year y then (wd + 2) % 7 else (wd + 1) % 7 ∧ 
    wd' = if y = 2013 then 2 else wd') :=
sorry

end june_10_2013_is_tuesday_l383_383815


namespace fg_of_3_eq_79_l383_383249

def g (x : ℤ) : ℤ := x ^ 3
def f (x : ℤ) : ℤ := 3 * x - 2

theorem fg_of_3_eq_79 : f (g 3) = 79 := by
  sorry

end fg_of_3_eq_79_l383_383249


namespace exists_convex_polygon_l383_383646

theorem exists_convex_polygon {S : Finset (ℝ × ℝ)} (h_n : 3 ≤ S.card)
  (h_noncollinear : ¬∃ l : AffineSubspace ℝ S, l.is_line ∧ ∀ p ∈ S, p ∈ l) :
  ∃ P : Finset (ℝ × ℝ), P ⊆ S ∧ is_convex_hull P S :=
sorry

end exists_convex_polygon_l383_383646


namespace smallest_n_for_trick_l383_383084

theorem smallest_n_for_trick (fillings : Finset Fin 10)
  (pastries : Finset (Fin 45)) 
  (has_pairs : ∀ p ∈ pastries, ∃ f1 f2 ∈ fillings, f1 ≠ f2 ∧ p = pair f1 f2) : 
  ∃ n (tray : Finset (Fin 45)), 
    (tray.card = n ∧ n = 36 ∧ 
    ∀ remaining_p ∈ pastries \ tray, ∃ f ∈ fillings, f ∈ remaining_p) :=
by
  sorry

end smallest_n_for_trick_l383_383084


namespace MikaGaveStickers_l383_383310

theorem MikaGaveStickers :
    ∀ (initial bought birthday used left gave total accounted: ℕ),
    initial = 20 →
    bought = 26 →
    birthday = 20 →
    used = 58 →
    left = 2 →
    total = initial + bought + birthday →
    accounted = used + left →
    gave = total - accounted →
    gave = 6 :=
by
  intros initial bought birthday used left gave total accounted
  assume h1 : initial = 20
  assume h2 : bought = 26
  assume h3 : birthday = 20
  assume h4 : used = 58
  assume h5 : left = 2
  assume h6 : total = initial + bought + birthday
  assume h7 : accounted = used + left
  assume h8 : gave = total - accounted
  sorry

end MikaGaveStickers_l383_383310


namespace pastry_trick_l383_383078

theorem pastry_trick (fillings : Fin 10) (n : ℕ) :
  ∃ n, (n = 36 ∧ ∀ remaining_pastries, 
    (remaining_pastries.length = 45 - n) → 
    (∃ remaining_filling ∈ fillings, true)) := 
sorry

end pastry_trick_l383_383078


namespace inequality_proof_l383_383782

-- Definitions of the angles
def alpha : ℝ := real.arcsin 0.5 -- replace with the correct expression for arcsin
def beta : ℝ := real.arctan 0.5 -- replace with the correct expression for arctan
def gamma : ℝ := real.arccos (-1) -- replace with the correct expression for arccos
def d : ℝ := real.arccot 0.5 -- replace with the correct expression for arccot

-- Definition of the function f
def f (x : ℝ) : ℝ := x^2 - real.pi * x

-- Assumptions based on the problem
lemma order_of_angles : 0 < alpha ∧ alpha < beta ∧ beta < gamma ∧ gamma < d ∧ d < real.pi / 2 :=
sorry

-- The final statement to prove option (B) is correct
theorem inequality_proof : f(alpha) > f(d) ∧ f(d) > f(beta) ∧ f(beta) > f(gamma) :=
sorry

end inequality_proof_l383_383782


namespace right_triangle_hypotenuse_l383_383539

theorem right_triangle_hypotenuse (a b : ℝ) (h : a^2 + b^2 = 39^2) : a = 15 ∧ b = 36 := by
  sorry

end right_triangle_hypotenuse_l383_383539


namespace right_triangle_side_sums_l383_383969

theorem right_triangle_side_sums (a b c : ℕ) (h1 : a + b = c + 6) (h2 : a^2 + b^2 = c^2) :
  (a = 7 ∧ b = 24 ∧ c = 25) ∨ (a = 8 ∧ b = 15 ∧ c = 17) ∨ (a = 9 ∧ b = 12 ∧ c = 15) :=
sorry

end right_triangle_side_sums_l383_383969


namespace first_wing_hall_rooms_l383_383307

theorem first_wing_hall_rooms
    (total_rooms : ℕ) (first_wing_floors : ℕ) (first_wing_halls_per_floor : ℕ)
    (second_wing_floors : ℕ) (second_wing_halls_per_floor : ℕ) (second_wing_rooms_per_hall : ℕ)
    (hotel_total_rooms : ℕ) (first_wing_total_rooms : ℕ) :
    hotel_total_rooms = total_rooms →
    first_wing_floors = 9 →
    first_wing_halls_per_floor = 6 →
    second_wing_floors = 7 →
    second_wing_halls_per_floor = 9 →
    second_wing_rooms_per_hall = 40 →
    hotel_total_rooms = first_wing_total_rooms + (second_wing_floors * second_wing_halls_per_floor * second_wing_rooms_per_hall) →
    first_wing_total_rooms = first_wing_floors * first_wing_halls_per_floor * 32 :=
by
  sorry

end first_wing_hall_rooms_l383_383307


namespace cube_root_digit_sum_equality_l383_383889

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem cube_root_digit_sum_equality :
  ∀ n, (n = 5832 ∨ n = 17576 ∨ n = 19683) →
    (∛n = sum_of_digits n) :=
by {
  -- Proof skipped
  sorry
}

end cube_root_digit_sum_equality_l383_383889


namespace smallest_positive_period_of_f_l383_383625

def f (x : ℝ) : ℝ := cos (2 * x - (π / 6))

theorem smallest_positive_period_of_f : ∃ T > 0, T = π ∧ ∀ x, f (x + T) = f x := 
by
  sorry

end smallest_positive_period_of_f_l383_383625


namespace maximize_sine_sum_of_equilateral_triangle_l383_383807

theorem maximize_sine_sum_of_equilateral_triangle
  (α β γ β' γ' : ℝ)
  (hβγ : β + γ = π - α)
  (hβ'γ' : β' + γ' = π - α)
  (hβγ_sum : sin β + sin γ > sin β' + sin γ')
  (hcommon_angle : α ≠ 0 ∧ α ≠ π) :
  sin α + sin β + sin γ = sin (π / 3) + sin (π / 3) + sin (π / 3) → 
  β = γ ∧ α = β ∧ α = γ :=
sorry

end maximize_sine_sum_of_equilateral_triangle_l383_383807


namespace fox_weight_l383_383371

variable {F D : ℝ}

-- Conditions
def condition1 : Prop := 3 * F + 5 * D = 65
def condition2 : Prop := D = F + 5

-- Goal
theorem fox_weight (h1 : condition1) (h2 : condition2) : 5 * F = 25 := by
  sorry

end fox_weight_l383_383371


namespace pm_eq_ps_l383_383562

variables {α : Type*} [EuclideanGeometry α]

/-- Define a semicircle, points, and their properties -/
def semicircle (A B : α) (M : α) := M = midpoint A B ∧ semicircle_over_segment A B

/-- Point P lies on the semicircle -/
def on_semicircle (P : α) (A B : α) (sc : semicircle A B (midpoint A B)) := 
  P ∈ sc ∧ P ≠ A ∧ P ≠ B

/-- Point Q is the midpoint of arc AP -/
def midpoint_of_arc (Q : α) (A P : α) (sc : semicircle A P (midpoint A P)) := 
  Q = midpoint_of_arc A P

/-- Define intersection point S -/
def intersection_point (S : α) (P Q M : α) (A B : α) := 
  let l1 := line_through P B,
      l2 := parallel_line_through M (line_through P Q) in
    S = intersection l1 l2

theorem pm_eq_ps 
  {A B M P Q S : α} 
  (sc : semicircle A B M) 
  (hmid : M = midpoint A B)
  (hP : on_semicircle P A B sc) 
  (hQ : midpoint_of_arc Q A P sc) 
  (hS : intersection_point S P Q M A B) 
  : dist P M = dist P S :=
sorry

end pm_eq_ps_l383_383562


namespace greatest_possible_average_speed_l383_383930

-- Defining the initial conditions
def initial_odometer : ℕ := 12321
def driving_hours : ℕ := 4
def speed_limit : ℕ := 65

-- Defining the palindrome condition
def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

-- Hypothesis for feasible distance
def max_distance : ℕ := driving_hours * speed_limit

-- Next possible palindrome after odometer reading
def next_palindromes := [12421, 12521, 12621, 12721]

-- Filtering feasible palindromes
def feasible_palindromes : List ℕ :=
  next_palindromes.filter (λ n, n - initial_odometer ≤ max_distance)

-- Proving the correct average speed
theorem greatest_possible_average_speed : (feasible_palindromes.maximum! - initial_odometer) / driving_hours = 50 :=
by
  sorry

end greatest_possible_average_speed_l383_383930


namespace right_triangle_hypotenuse_length_l383_383552

theorem right_triangle_hypotenuse_length (a b : ℝ) (h_triangle : a = 15 ∧ b = 36) :
  ∃ (h : ℝ), h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  · exact rfl
  · rw [h_triangle.1, h_triangle.2]
    norm_num

end right_triangle_hypotenuse_length_l383_383552


namespace right_triangle_hypotenuse_l383_383513

theorem right_triangle_hypotenuse (a b : ℕ) (ha : a = 15) (hb : b = 36) : 
  ∃ h : ℕ, h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  . exact rfl
  . rw [ha, hb]
    norm_num
    sorry

end right_triangle_hypotenuse_l383_383513


namespace quadruples_count_l383_383341

-- Define the conditions as functions within Lean
def condition1 (a b c : ℝ) : Prop :=
  ∀ x : ℝ, 2 * sin(3 * x - π / 3) = a * sin(b * x + c)

-- Define the function representing the intersection condition
def is_intersection (d : ℝ) : Prop :=
  (d ∈ Icc 0 (3 * π)) ∧ (sin (2 * d) = cos d)

-- Define the main theorem statement
theorem quadruples_count : 
  (∃! (a b c d : ℝ), condition1 a b c ∧ is_intersection d) = 28 :=
sorry

end quadruples_count_l383_383341


namespace right_triangle_hypotenuse_length_l383_383523

theorem right_triangle_hypotenuse_length :
  ∀ (a b h : ℕ), a = 15 → b = 36 → h^2 = a^2 + b^2 → h = 39 :=
by
  intros a b h ha hb hyp
  -- In the proof, we would use ha, hb, and hyp to show h = 39
  sorry

end right_triangle_hypotenuse_length_l383_383523


namespace ratio_of_perimeters_l383_383821

theorem ratio_of_perimeters (s : ℝ) (hs : s > 0) :
  let small_triangle_perimeter := s + (s / 2) + (s / 2)
  let large_rectangle_perimeter := 2 * (s + (s / 2))
  small_triangle_perimeter / large_rectangle_perimeter = 2 / 3 :=
by
  sorry

end ratio_of_perimeters_l383_383821


namespace hypotenuse_length_l383_383498

theorem hypotenuse_length (a b : ℤ) (h₀ : a = 15) (h₁ : b = 36) : 
  ∃ c : ℤ, c^2 = a^2 + b^2 ∧ c = 39 := 
by {
  sorry
}

end hypotenuse_length_l383_383498


namespace certain_multiple_l383_383615

theorem certain_multiple (n m : ℤ) (h : n = 5) (eq : 7 * n - 15 = m * n + 10) : m = 2 :=
by
  sorry

end certain_multiple_l383_383615


namespace pastry_problem_minimum_n_l383_383071

theorem pastry_problem_minimum_n (fillings : Finset ℕ) (n : ℕ) : 
    fillings.card = 10 →
    (∃ pairs : Finset (ℕ × ℕ), pairs.card = 45 ∧ ∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ≠ p.2 ∧ p.1 ∈ fillings ∧ p.2 ∈ fillings) →
    (∀ (remaining_pies : Finset (ℕ × ℕ)), remaining_pies.card = 45 - n → 
     ∃ f1 f2, (f1, f2) ∈ remaining_pies → (f1 ∈ fillings ∧ f2 ∈ fillings)) →
    n = 36 :=
by
  intros h_fillings h_pairs h_remaining_pies
  sorry

end pastry_problem_minimum_n_l383_383071


namespace total_weight_marble_purchased_l383_383802

theorem total_weight_marble_purchased (w1 w2 w3 : ℝ) (h1 : w1 = 0.33) (h2 : w2 = 0.33) (h3 : w3 = 0.08) :
  w1 + w2 + w3 = 0.74 := by
  sorry

end total_weight_marble_purchased_l383_383802


namespace geom_seq_properties_sum_of_seq_lt_3_l383_383993

-- define the sequence {a_n} as a geometric sequence
noncomputable def a_n (n: ℕ) : ℕ := 3^(n-1)

-- sequence properties
theorem geom_seq_properties :
  ∀ (q : ℝ), (q ≠ 0 ∧ a_n 1 = 1 ∧ ∀ n, a_n (n+1) = q * a_n n ∧ monotone a_n) ∧ 
    ((a_n 3, (5/3) * a_n 4, a_n 5) form_arithmetic_seq) →
    a_n = λ n, 3^(n-1) := sorry

-- define the sum of the first n terms of the sequence { (2n-1)/a_n }
noncomputable def S_n (n : ℕ) : ℝ := ∑ i in finset.range n, (2i + 1) / (a_n (i + 1))

-- prove that S_n < 3
theorem sum_of_seq_lt_3 (n : ℕ) : S_n n < 3 := sorry

end geom_seq_properties_sum_of_seq_lt_3_l383_383993


namespace find_a_in_expansion_l383_383182

theorem find_a_in_expansion (a : ℝ) : 
  (let x : ℝ := 2 in (4 + 6 * a = 16)) → a = 2 :=
by
  intro h
  sorry

end find_a_in_expansion_l383_383182


namespace distance_point_to_line_l383_383354

theorem distance_point_to_line : 
  let x0 := 2
  let y0 := -1
  let A := 1
  let B := -1
  let C := 3
  let d := (abs (A * x0 + B * y0 + C)) / (real.sqrt (A ^ 2 + B ^ 2))
  d = 3 * real.sqrt 2 :=
by
  let x0 := 2
  let y0 := -1
  let A := 1
  let B := -1
  let C := 3
  let d := (abs (A * x0 + B * y0 + C)) / (real.sqrt (A ^ 2 + B ^ 2))
  show d = 3 * real.sqrt 2
  sorry

end distance_point_to_line_l383_383354


namespace problem_solution_l383_383245

theorem problem_solution (x : ℝ) (h : x * Real.log 4 / Real.log 3 = 1) : 
  2^x + 4^(-x) = 1 / 3 + Real.sqrt 3 :=
by 
  sorry

end problem_solution_l383_383245


namespace point_of_tangency_l383_383835

theorem point_of_tangency (x y : ℝ) (h : (y = x^3 + x - 2)) (slope : 4 = 3 * x^2 + 1) : (x, y) = (-1, -4) := 
sorry

end point_of_tangency_l383_383835


namespace count_valid_sequences_l383_383947

-- Define the length of the sequence
def sequence_length : ℕ := 10

-- Definitions to represent the conditions
def all_consecutive_zero_or_one (s : List ℕ) : Prop :=
  (∃ k l, k + l = sequence_length ∧
          (∀ i < k, s.nth i = some 0) ∧
          (∀ i, sequence_length - k ≤ i ∧ i < sequence_length → s.nth i = some 1)) ∨
  (∃ k l, k + l = sequence_length ∧
          (∀ i < k, s.nth i = some 1) ∧
          (∀ i, sequence_length - k ≤ i ∧ i < sequence_length → s.nth i = some 0))

def has_separate_zero (s : List ℕ) : Prop :=
  ∃ i j, i < j ∧ s.nth i = some 0 ∧ s.nth j = some 1

def has_separate_one (s : List ℕ) : Prop :=
  ∃ i j, i < j ∧ s.nth i = some 1 ∧ s.nth j = some 0

def valid_sequence (s : List ℕ) : Prop :=
  s.length = sequence_length ∧
  all_consecutive_zero_or_one s ∧
  has_separate_zero s ∧
  has_separate_one s

-- The statement to be proved
theorem count_valid_sequences : 
  ∃ n, n = 108 ∧ ∀ s : List ℕ, valid_sequence s →
  count_sequences_with_property s = n :=
sorry

end count_valid_sequences_l383_383947


namespace correct_statements_l383_383813

def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem correct_statements :
  (∀ x, f (x - Real.pi / 3) = -f (- (x - Real.pi / 3)) → false) ∧
  (∀ T, T > 0 → ∀ x, f (x + T) = f x → T = Real.pi → false) ∧
  (∃ x : ℝ, x = -Real.pi / 6 ∧ f (x - Real.pi / 3) = f x) ∧
  (∀ x, ℕ, x = n ∗ Real.pi / 2 + Real.pi / 12 → false) :=
by
  sorry

end correct_statements_l383_383813


namespace air_conditioner_consumption_l383_383581

theorem air_conditioner_consumption :
  ∀ (total_consumption_8_hours : ℝ)
    (hours_8 : ℝ)
    (hours_per_day : ℝ)
    (days : ℝ),
    total_consumption_8_hours / hours_8 * hours_per_day * days = 27 :=
by
  intros total_consumption_8_hours hours_8 hours_per_day days
  sorry

end air_conditioner_consumption_l383_383581


namespace max_distance_origin_to_line_l383_383220

open Real

-- Definitions from the conditions
def line_eq (a b c : ℝ) (x y : ℝ) : ℝ := a * x + b * y + c

def is_arithmetic_sequence (a b c : ℝ) : Prop := a + c = 2 * b

-- The theorem we aim to prove
theorem max_distance_origin_to_line (a b c : ℝ) (h_arithmetic : is_arithmetic_sequence a b c) :
  ∃ d : ℝ, d = sqrt 5 ∧ ∀ (x y : ℝ), line_eq a b c x y = 0 → sqrt (x^2 + y^2) ≤ d := by
  sorry

end max_distance_origin_to_line_l383_383220


namespace correct_output_statement_l383_383124

-- defining the various statement types as inductive types
inductive StatementType
| input : StatementType
| print : StatementType
| conditional_if : StatementType
| conditional_while : StatementType

open StatementType

-- Question: Which one is an output statement?

noncomputable def is_output_statement : StatementType → Prop
| print := true
| _ := false

-- Proving the correct answer
theorem correct_output_statement : is_output_statement print = true :=
by
  sorry

end correct_output_statement_l383_383124


namespace pascal_triangle_45th_number_l383_383010

theorem pascal_triangle_45th_number (n k : ℕ) (h1 : n = 47) (h2 : k = 44) : 
  Nat.choose (n - 1) k = 1035 :=
by
  sorry

end pascal_triangle_45th_number_l383_383010


namespace find_d_in_line_intercepts_l383_383758

theorem find_d_in_line_intercepts 
  (d : ℝ)
  (h : let x_intercept := -d / 3 
           y_intercept := -d / 5 
        in x_intercept + y_intercept = 16) : 
  d = -30 :=
by
  sorry

end find_d_in_line_intercepts_l383_383758


namespace restaurant_june_productions_l383_383470

theorem restaurant_june_productions
  (weekdays : ℕ := 20)
  (weekends : ℕ := 10)
  (weekday_hotdogs : ℕ := 60)
  (weekend_hotdogs : ℕ := 50)
  (weekday_cheese_offset : ℕ := 40)
  (weekend_cheese_offset : ℕ := 30)
  (weekday_beef_hotdogs : ℕ := 30)
  (weekday_chicken_hotdogs : ℕ := 30)
  (weekend_beef_hotdogs : ℕ := 20)
  (weekend_chicken_hotdogs : ℕ := 30)
  (weekend_bbq_chicken_pizzas : ℕ := 25)
  (weekend_veggie_pizzas : ℕ := 15) :
  ∃ 
  (june_cheese_pizzas : ℕ = weekdays * (weekday_hotdogs + weekday_cheese_offset) + weekends * (weekend_hotdogs + weekend_cheese_offset))
  (june_pepperoni_pizzas : ℕ = weekdays * 2 * (weekday_hotdogs + weekday_cheese_offset) + weekends * 2 * (weekend_hotdogs + weekend_cheese_offset))
  (june_beef_hotdogs : ℕ = weekdays * weekday_beef_hotdogs + weekends * weekend_beef_hotdogs)
  (june_chicken_hotdogs : ℕ = weekdays * weekday_chicken_hotdogs + weekends * weekend_chicken_hotdogs)
  (june_bbq_chicken_pizzas : ℕ = weekends * weekend_bbq_chicken_pizzas)
  (june_veggie_pizzas : ℕ = weekends * weekend_veggie_pizzas), 
  true :=
by
  sorry

end restaurant_june_productions_l383_383470


namespace relationship_among_abc_l383_383637

noncomputable def a : ℝ := Real.log 4 / Real.log 3
noncomputable def b : ℝ := Real.log 3 / Real.log 0.4
noncomputable def c : ℝ := 0.4 ^ 3

theorem relationship_among_abc : a > c ∧ c > b := by
  sorry

end relationship_among_abc_l383_383637


namespace age_ratio_in_9_years_l383_383184

-- Initial age definitions for Mike and Sam
def ages (m s : ℕ) : Prop :=
  (m - 5 = 2 * (s - 5)) ∧ (m - 12 = 3 * (s - 12))

-- Proof that in 9 years the ratio of their ages will be 3:2
theorem age_ratio_in_9_years (m s x : ℕ) (h_ages : ages m s) :
  (m + x) * 2 = 3 * (s + x) ↔ x = 9 :=
by {
  sorry
}

end age_ratio_in_9_years_l383_383184


namespace hypotenuse_length_l383_383483

-- Definitions for the problem
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def leg1 : ℕ := 15
def leg2 : ℕ := 36
def hypotenuse : ℕ := 39

-- Lean 4 statement
theorem hypotenuse_length (a b c : ℕ) (h : is_right_triangle a b c) (ha : a = leg1) (hb : b = leg2) :
  c = hypotenuse :=
begin
  sorry
end

end hypotenuse_length_l383_383483


namespace sin_pi_over_9_sin_2pi_over_9_sin_4pi_over_9_l383_383594

theorem sin_pi_over_9_sin_2pi_over_9_sin_4pi_over_9 :
  let ζ := Complex.exp (Complex.i * (2 * Real.pi / 9)) in
  ζ ^ 9 = 1 →
  Real.sin (Real.pi / 9) * Real.sin (2 * Real.pi / 9) * Real.sin (4 * Real.pi / 9) = -1 / 8 :=
by
  intro ζ
  intro hζ
  sorry

end sin_pi_over_9_sin_2pi_over_9_sin_4pi_over_9_l383_383594


namespace evaluate_f_g_3_l383_383247

def g (x : ℝ) := x^3
def f (x : ℝ) := 3 * x - 2

theorem evaluate_f_g_3 : f (g 3) = 79 := by
  sorry

end evaluate_f_g_3_l383_383247


namespace sequence_properties_l383_383992

theorem sequence_properties (a : ℝ) (a_seq : ℕ → ℝ)
  (h1 : a > 2)
  (h2 : a_seq 0 = a)
  (h3 : ∀ n : ℕ, a_seq (n + 1) = (a_seq n)^2 / (2 * (a_seq n - 1))) :
  (∀ n : ℕ, a_seq n > a_seq (n + 1) ∧ a_seq (n + 1) > 2) ∧ (∀ n : ℕ, (∑ i in finset.range (n + 1), a_seq i) < 2 * (n + a - 2)) :=
by
  sorry

end sequence_properties_l383_383992


namespace simplify_and_multiply_roots_l383_383405

theorem simplify_and_multiply_roots :
  (∛27) * (∜81) * (√9) = 27 :=
by 
  sorry

end simplify_and_multiply_roots_l383_383405


namespace right_triangle_hypotenuse_l383_383536

theorem right_triangle_hypotenuse (a b : ℝ) (h : a^2 + b^2 = 39^2) : a = 15 ∧ b = 36 := by
  sorry

end right_triangle_hypotenuse_l383_383536


namespace records_given_l383_383814

theorem records_given (X : ℕ) (started_with : ℕ) (bought : ℕ) (days_per_record : ℕ) (total_days : ℕ)
  (h1 : started_with = 8) (h2 : bought = 30) (h3 : days_per_record = 2) (h4 : total_days = 100) :
  X = 12 := by
  sorry

end records_given_l383_383814


namespace quadrilateral_inequality_l383_383770

theorem quadrilateral_inequality
  {A B C D : Type*}
  (a b c d : A)
  (BD : ℝ)
  (AB AD CB CD : ℝ)
  (h1 : ∠A = 90)
  (h2 : ∠C = 90) :
  (1 / BD) * (AB + BC + CD + DA) + BD ^ 2 * ((1 / (AB * AD)) + (1 / (CB * CD)))
  ≥ 2 * (2 + real.sqrt 2) :=
sorry

end quadrilateral_inequality_l383_383770


namespace angle_qpr_right_triangle_l383_383740

noncomputable def magnitude_of_angle (P Q R S : Type) [metric_space P] [metric_space Q] [metric_space R] [metric_space S] 
  {a b c : ℝ} (h₁ : a = 90) (h₂ : b = c) (h₃ : a = b) : ℝ :=
  45

theorem angle_qpr_right_triangle (P Q R S : Type) [metric_space P] [metric_space Q] [metric_space R] [metric_space S] 
  (a b c : ℝ) (h₁ : a = 90) (h₂ : b = c) (h₃ : a = b) : 
  magnitude_of_angle P Q R S h₁ h₂ h₃ = 45 :=
by
  sorry

end angle_qpr_right_triangle_l383_383740


namespace perpendicular_lines_values_of_a_l383_383207

theorem perpendicular_lines_values_of_a (a : ℝ) :
  (∃ (a : ℝ), (∀ x y : ℝ, a * x - y + 2 * a = 0 ∧ (2 * a - 1) * x + a * y = 0) 
    ↔ (a = 0 ∨ a = 1))
  := sorry

end perpendicular_lines_values_of_a_l383_383207


namespace range_of_a_l383_383257

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4 * a) ↔ a ∈ Set.union (Set.Iio 1) (Set.Ioi 3) :=
by
  sorry

end range_of_a_l383_383257


namespace fruit_seller_stock_l383_383907

-- Define the given conditions
def remaining_oranges : ℝ := 675
def remaining_percentage : ℝ := 0.25

-- Define the problem function
def original_stock (O : ℝ) : Prop :=
  remaining_percentage * O = remaining_oranges

-- Prove the original stock of oranges was 2700 kg
theorem fruit_seller_stock : original_stock 2700 :=
by
  sorry

end fruit_seller_stock_l383_383907


namespace number_of_relatively_prime_to_18_l383_383702

theorem number_of_relatively_prime_to_18 : 
  ∃ N : ℕ, N = 30 ∧ ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 → Nat.gcd n 18 = 1 ↔ false :=
by
  sorry

end number_of_relatively_prime_to_18_l383_383702


namespace problem_statement_l383_383816

noncomputable def a_seq (n : ℕ) : ℝ := 
  if n = 0 then real.sqrt(2) / 2 else real.sqrt(2) / 2 * real.sqrt(1 - real.sqrt(1 - (a_seq (n - 1))^2))

noncomputable def b_seq (n : ℕ) : ℝ := 
  if n = 0 then 1 else (real.sqrt(1 + (b_seq (n - 1)) ^ 2) - 1) / (b_seq (n - 1))

theorem problem_statement : ∀ n : ℕ, 2^ (n + 2) * a_seq n < real.pi ∧ real.pi < 2^ (n + 2) * b_seq n :=
sorry

end problem_statement_l383_383816


namespace part1_l383_383676

def f (x : ℝ) : ℝ := abs (x + 2) + abs (x - 3)

theorem part1 (x : ℝ) : f(x) ≥ 2 * x ↔ x ∈ Set.Iic (5 / 2) := sorry

end part1_l383_383676


namespace fraction_of_green_marbles_half_l383_383049

-- Definitions based on given conditions
def initial_fraction (x : ℕ) : ℚ := 1 / 3

-- Number of blue, red, and green marbles initially
def blue_marbles (x : ℕ) : ℚ := initial_fraction x * x
def red_marbles (x : ℕ) : ℚ := initial_fraction x * x
def green_marbles (x : ℕ) : ℚ := initial_fraction x * x

-- Number of green marbles after doubling
def doubled_green_marbles (x : ℕ) : ℚ := 2 * green_marbles x

-- New total number of marbles
def new_total_marbles (x : ℕ) : ℚ := blue_marbles x + red_marbles x + doubled_green_marbles x

-- New fraction of green marbles after doubling
def new_fraction_of_green_marbles (x : ℕ) : ℚ := doubled_green_marbles x / new_total_marbles x

theorem fraction_of_green_marbles_half (x : ℕ) (hx : x > 0) :
  new_fraction_of_green_marbles x = 1 / 2 :=
by
  sorry

end fraction_of_green_marbles_half_l383_383049


namespace fraction_product_l383_383418

theorem fraction_product :
  (∏ n in Finset.range 6, (1 / 3^(n+1) * 3^(2*(n+1))) = 729) :=
sorry

end fraction_product_l383_383418


namespace quadratic_relationship_l383_383690

theorem quadratic_relationship :
  ∀ (x z : ℕ), (x = 1 ∧ z = 5) ∨ (x = 2 ∧ z = 12) ∨ (x = 3 ∧ z = 23) ∨ (x = 4 ∧ z = 38) ∨ (x = 5 ∧ z = 57) →
  z = 2 * x^2 + x + 2 :=
by
  sorry

end quadratic_relationship_l383_383690


namespace solve_for_x_l383_383719

theorem solve_for_x (x: ℚ) (h: (3/5 - 1/4) = 4/x) : x = 80/7 :=
by
  sorry

end solve_for_x_l383_383719


namespace divisor_is_20_l383_383421

theorem divisor_is_20 (D : ℕ) 
  (h1 : 242 % D = 11) 
  (h2 : 698 % D = 18) 
  (h3 : 940 % D = 9) :
  D = 20 :=
sorry

end divisor_is_20_l383_383421


namespace proof_bd_leq_q2_l383_383688

variables {a b c d p q : ℝ}

theorem proof_bd_leq_q2 
  (h1 : ab + cd = 2pq)
  (h2 : ac ≥ p^2)
  (h3 : p^2 > 0) :
  bd ≤ q^2 :=
sorry

end proof_bd_leq_q2_l383_383688


namespace hypotenuse_length_l383_383493

theorem hypotenuse_length (a b : ℤ) (h₀ : a = 15) (h₁ : b = 36) : 
  ∃ c : ℤ, c^2 = a^2 + b^2 ∧ c = 39 := 
by {
  sorry
}

end hypotenuse_length_l383_383493


namespace calc_x6_plus_inv_x6_l383_383640

theorem calc_x6_plus_inv_x6 (x : ℝ) (hx : x + (1 / x) = 7) : x^6 + (1 / x^6) = 103682 := by
  sorry

end calc_x6_plus_inv_x6_l383_383640


namespace sequence_sum_l383_383563

-- Definitions
def sequence : ℕ → ℕ
| 0 := 2
| 1 := 4
| 2 := 12
| (n + 3) := 3 * (n+1)^2 - 7 * (n+1) + 6  -- Extend based on identified pattern

def sum_first_n_terms (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, sequence i

-- Statement of the problem
theorem sequence_sum (n : ℕ) : sum_first_n_terms n = n * (n^2 - 2 * n + 6) :=
by
  sorry

end sequence_sum_l383_383563


namespace log_inverse_proof_l383_383252

theorem log_inverse_proof (x : ℝ) (h : log 81 (x - 6) = 1 / 4) : 1 / log x 3 = 2 := 
  sorry

end log_inverse_proof_l383_383252


namespace min_sum_ab_l383_383779

theorem min_sum_ab {a b : ℤ} (h : a * b = 36) : a + b ≥ -37 := sorry

end min_sum_ab_l383_383779


namespace weight_feel_when_lowered_l383_383380

-- Conditions from the problem
def num_plates : ℕ := 10
def weight_per_plate : ℝ := 30
def technology_increase : ℝ := 0.20
def incline_increase : ℝ := 0.15

-- Calculate the contributions
def total_weight_without_factors : ℝ := num_plates * weight_per_plate
def weight_with_technology : ℝ := total_weight_without_factors * (1 + technology_increase)
def weight_with_incline : ℝ := weight_with_technology * (1 + incline_increase)

-- Theorem statement we want to prove
theorem weight_feel_when_lowered : weight_with_incline = 414 := by
  sorry

end weight_feel_when_lowered_l383_383380


namespace sum_dist_equals_a_locus_diff_dist_equals_b_locus_l383_383619

open EuclideanGeometry

-- Given two lines l1 and l2 and the values a and b.
variables {l1 l2 : Line} (a b : ℝ)

-- Definitions for distances from a point to lines l1 and l2
def dist_to_lines (P : Point) : ℝ × ℝ :=
  (distance P (projection l1 P), distance P (projection l2 P))

-- Definition for sum of distances condition
def sum_of_distances (P : Point) (a : ℝ) : Prop :=
  (dist_to_lines P).fst + (dist_to_lines P).snd = a

-- Definition for difference of distances condition
def diff_of_distances (P : Point) (b : ℝ) : Prop :=
  (dist_to_lines P).fst - (dist_to_lines P).snd = b

-- Theorems for the locus of points
theorem sum_dist_equals_a_locus :
  ∃ (Rectangle : set Point), ∀ P, sum_of_distances P a ↔ P ∈ Rectangle := sorry

theorem diff_dist_equals_b_locus :
  ∃ (ExtendedLines : set Point), ∀ P, diff_of_distances P b ↔ P ∈ ExtendedLines := sorry

end sum_dist_equals_a_locus_diff_dist_equals_b_locus_l383_383619


namespace focus_of_parabola_l383_383683

theorem focus_of_parabola (p : ℝ) (h₀ : p > 0) (h₁ : p = 2) : 
  ∃ (x y : ℝ), (x, y) = (1, 0) :=
by
  use (1, 0)
  sorry

end focus_of_parabola_l383_383683


namespace equilateral_triangle_label_properties_l383_383582

theorem equilateral_triangle_label_properties
  (n : ℕ)
  (labels : Fin (n*(n+1)/2) → ℝ)
  (a b c : ℝ)
  (h_labels : ∀ {A B C D : ℝ}, 
    (labels ⟨A, sorry⟩ = a) ∧
    (labels ⟨B, sorry⟩ = b) ∧ 
    (labels ⟨C, sorry⟩ = c) → 
    labels ⟨D, sorry⟩ = (labels ⟨A, sorry⟩ + labels ⟨B, sorry⟩ + labels ⟨C, sorry⟩ - labels ⟨B, sorry⟩ - labels ⟨C, sorry⟩)) :
  (shortest_distance : ℝ) 
  (sum_of_labels : ℝ) :
  shortest_distance = n 
  ∧ sum_of_labels = (n + 1) * (n + 2) * (a + b + c) / 6 := sorry

end equilateral_triangle_label_properties_l383_383582


namespace right_triangle_hypotenuse_length_l383_383555

theorem right_triangle_hypotenuse_length (a b : ℝ) (h_triangle : a = 15 ∧ b = 36) :
  ∃ (h : ℝ), h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  · exact rfl
  · rw [h_triangle.1, h_triangle.2]
    norm_num

end right_triangle_hypotenuse_length_l383_383555


namespace packages_yesterday_l383_383057

variable x : ℕ -- The number of packages received yesterday
variable total : ℕ -- The total number of packages received over two days

-- Conditions
def packages_today := 2 * x
def total_packages := x + packages_today

-- Given that the total number of packages is 240
axiom total_eq : total_packages = 240

-- We need to prove that x = 80
theorem packages_yesterday : x = 80 :=
by
  -- Here we will eventually prove that x = 80 based on the provided conditions
  sorry

end packages_yesterday_l383_383057


namespace gcd_digit_bound_l383_383735

theorem gcd_digit_bound (a b : ℕ) (h₁ : 10^6 ≤ a) (h₂ : a < 10^7) (h₃ : 10^6 ≤ b) (h₄ : b < 10^7) 
  (h₅ : 10^{10} ≤ lcm a b) (h₆ : lcm a b < 10^{11}) : 
  gcd a b < 10^4 :=
sorry

end gcd_digit_bound_l383_383735


namespace vasya_password_combinations_l383_383001

theorem vasya_password_combinations :
  let count := (λ (A B C : ℕ), 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    A ≠ 2 ∧ B ≠ 2 ∧ C ≠ 2 ∧ 
    (A ∈ [0, 1, 3, 4, 5, 6, 7, 8, 9]) ∧ 
    (B ∈ [0, 1, 3, 4, 5, 6, 7, 8, 9]) ∧ 
    (C ∈ [0, 1, 3, 4, 5, 6, 7, 8, 9])) 
  in ∃! n, n = 9 * 8 * 7 ∧
  (∃ A B C, count A B C) := 
by
  sorry

end vasya_password_combinations_l383_383001


namespace train_travel_time_l383_383461

def travel_time (departure arrival : Nat) : Nat :=
  arrival - departure

theorem train_travel_time : travel_time 425 479 = 54 := by
  sorry

end train_travel_time_l383_383461


namespace problem_A_eq_7_problem_A_eq_2012_l383_383949

open Nat

-- Problem statement for A = 7
theorem problem_A_eq_7 (n k : ℕ) :
  (n! + 7 * n = n^k) ↔ ((n, k) = (2, 4) ∨ (n, k) = (3, 3)) :=
sorry

-- Problem statement for A = 2012
theorem problem_A_eq_2012 (n k : ℕ) :
  ¬ (n! + 2012 * n = n^k) :=
sorry

end problem_A_eq_7_problem_A_eq_2012_l383_383949


namespace comprehensive_survey_selection_l383_383924

-- Definitions based on the conditions
def comprehensive_survey_method (option: String) : Prop :=
  option = "Understanding the '50-meter dash' performance of a class of students" ∨
  option = "Understanding the lifespan of a batch of light bulbs" ∨
  option = "Understanding whether a batch of bagged food contains preservatives"

def suitable_for_comprehensive (option: String) : Prop :=
  (option = "Understanding the '50-meter dash' performance of a class of students")

-- Statement of the problem as a theorem
theorem comprehensive_survey_selection : ∀ option,
  comprehensive_survey_method option → suitable_for_comprehensive option :=
begin
  assume option H,
  -- The proof would go here, showing that only option A is suitable.
  sorry -- Placeholder for the actual proof.
end

end comprehensive_survey_selection_l383_383924


namespace length_of_each_train_l383_383035

theorem length_of_each_train (L : ℝ) (s1 : ℝ) (s2 : ℝ) (t : ℝ)
    (h1 : s1 = 46) (h2 : s2 = 36) (h3 : t = 144) (h4 : 2 * L = ((s1 - s2) * (5 / 18)) * t) :
    L = 200 := 
sorry

end length_of_each_train_l383_383035


namespace optionA_optionB_optionC_optionD_l383_383933

theorem optionA: (sqrt 2 * sin (real.angle.degrees 15) + sqrt 2 * cos (real.angle.degrees 15)) = sqrt 3 := 
  sorry

theorem optionB: (cos (real.angle.degrees 15))^2 - (sin (real.angle.degrees 15) * cos (real.angle.degrees 75)) ≠ sqrt 3 := 
  sorry

theorem optionC: (tan (real.angle.degrees 30) / (1 - tan (real.angle.degrees 30)^2)) ≠ sqrt 3 := 
  sorry

theorem optionD: ((1 + tan (real.angle.degrees 15)) / (1 - tan (real.angle.degrees 15))) = sqrt 3 := 
  sorry

end optionA_optionB_optionC_optionD_l383_383933


namespace most_stable_scores_l383_383053

-- Definitions for the variances of students A, B, and C
def s_A_2 : ℝ := 6
def s_B_2 : ℝ := 24
def s_C_2 : ℝ := 50

-- The proof that student A has the most stable math scores
theorem most_stable_scores : 
  s_A_2 < s_B_2 ∧ s_B_2 < s_C_2 → 
  ("Student A has the most stable scores" = "Student A has the most stable scores") :=
by
  intros h
  sorry

end most_stable_scores_l383_383053


namespace square_locus_proof_l383_383994

noncomputable def square_locus (A B C D M : Point) : Prop :=
  ∃ O₁ O₂ r₁ r₂, circle O₁ r₁ ∧ circle O₂ r₂ ∧
  A ∈ (circle_points O₁ r₁) ∧ C ∈ (circle_points O₁ r₁) ∧
  B ∈ (circle_points O₂ r₂) ∧ D ∈ (circle_points O₂ r₂) ∧
  ((M ∈ (circle_points O₁ r₁)) ∨ (M ∈ (circle_points O₂ r₂)))

theorem square_locus_proof (A B C D M : Point) (h_square : is_square A B C D) :
  (∠ A M B = ∠ C M D) ↔ square_locus A B C D M :=
by
  sorry

end square_locus_proof_l383_383994


namespace evaluation_f_at_2_l383_383198

theorem evaluation_f_at_2 (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (3 ^ x) = 2 * x * real.log 3 / real.log 2) : 
  f 2 = 2 := 
sorry

end evaluation_f_at_2_l383_383198


namespace b_catches_A_distance_l383_383885

noncomputable def speed_A := 10 -- kmph
noncomputable def speed_B := 20 -- kmph
noncomputable def time_diff := 7 -- hours
noncomputable def distance_A := speed_A * time_diff -- km
noncomputable def relative_speed := speed_B - speed_A -- kmph
noncomputable def catch_up_time := distance_A / relative_speed -- hours
noncomputable def distance_B := speed_B * catch_up_time -- km

theorem b_catches_A_distance :
  distance_B = 140 := by
  sorry

end b_catches_A_distance_l383_383885


namespace necessary_but_not_sufficient_l383_383988

theorem necessary_but_not_sufficient (x : ℝ) (p : Prop) (q : Prop) 
  (h₁ : p ↔ (1 / x < 1))
  (h₂ : q ↔ (x > 1)) :
  (∀ x, q x → p x) ∧ ¬(∀ x, p x → q x) :=
by {
  sorry
}

end necessary_but_not_sufficient_l383_383988


namespace not_dividable_by_wobbly_l383_383573

-- Define a wobbly number
def is_wobbly_number (n : ℕ) : Prop :=
  n > 0 ∧ (∀ k : ℕ, k < (Nat.log 10 n) → 
    (n / (10^k) % 10 ≠ 0 → n / (10^(k+1)) % 10 = 0) ∧
    (n / (10^k) % 10 = 0 → n / (10^(k+1)) % 10 ≠ 0))

-- Define sets of multiples of 10 and 25
def multiples_of (m : ℕ) (k : ℕ): Prop :=
  ∃ q : ℕ, k = q * m

def is_multiple_of_10 (k : ℕ) : Prop := multiples_of 10 k
def is_multiple_of_25 (k : ℕ) : Prop := multiples_of 25 k

theorem not_dividable_by_wobbly (n : ℕ) : 
  ¬ ∃ w : ℕ, is_wobbly_number w ∧ n ∣ w ↔ is_multiple_of_10 n ∨ is_multiple_of_25 n :=
by
  sorry

end not_dividable_by_wobbly_l383_383573


namespace graph_passes_through_point_l383_383837

-- Define the linear function 
def linear_function (a x : ℝ) : ℝ := a * x - 3 * a + 1

-- The point of interest
def point_of_interest := (3, 1 : ℝ × ℝ)

-- Prove that the graph passes through the point (3, 1)
theorem graph_passes_through_point : ∀ (a : ℝ), linear_function a 3 = 1 := 
by
  -- To be filled with the proof
  sorry

end graph_passes_through_point_l383_383837


namespace minimum_pies_for_trick_l383_383091

-- Definitions from conditions
def num_fillings : ℕ := 10
def num_pastries := (num_fillings * (num_fillings - 1)) / 2
def min_pies_for_trick (n : ℕ) : Prop :=
  ∀ remaining_pies : ℕ, remaining_pies = num_pastries - n → remaining_pies ≤ 9

theorem minimum_pies_for_trick : ∃ n : ℕ, min_pies_for_trick n ∧ n = 36 :=
by
  -- We need to show that there exists n such that,
  -- min_pies_for_trick holds and n = 36
  existsi (36 : ℕ)
  -- remainder of the proof (step solution) skipped
  sorry

end minimum_pies_for_trick_l383_383091


namespace monotonicity_of_f_l383_383960

noncomputable def f (a x : ℝ) : ℝ := (a - 1) * Real.log x + a * x^2 + 1

theorem monotonicity_of_f (a : ℝ) :
  (a ≥ 1 → ∀ x ∈ Ioi 0, 0 < (a - 1)/x + 2 * a * x) ∧
  (a ≤ 0 → ∀ x ∈ Ioi 0, (a - 1)/x + 2 * a * x < 0) ∧
  (0 < a ∧ a < 1 →
    (∀ x ∈ Ioo 0 (Real.sqrt ((1 - a) / (2 * a))), (a - 1)/x + 2 * a * x < 0) ∧
    (∀ x ∈ Ioi (Real.sqrt ((1 - a) / (2 * a))), 0 < (a - 1)/x + 2 * a * x)) :=
by
  sorry

end monotonicity_of_f_l383_383960


namespace general_formula_an_sum_bn_seq_l383_383042

-- Given conditions
variable (a : ℕ → ℤ)
variable (S : ℕ → ℤ)
variable (b : ℕ → ℤ)
variable (T : ℕ → ℤ)

-- Conditions for the problems
axiom a1_eq : a 1 = 1
axiom S3_eq : S 3 = 9
axiom Sn_def : ∀ n, S n = n^2

-- (1) Find the general formula for the sequence {a_n}
theorem general_formula_an : ∀ n, a n = 2 * n - 1 := 
sorry

-- (2) Find sum of the first n terms of the sequence {b_n}
theorem sum_bn_seq : ∀ n, T n = 2 * (n * n - ⌊n/2⌋) := 
sorry

end general_formula_an_sum_bn_seq_l383_383042


namespace germs_left_percentage_l383_383054

-- Defining the conditions
def first_spray_kill_percentage : ℝ := 0.50
def second_spray_kill_percentage : ℝ := 0.25
def overlap_percentage : ℝ := 0.05
def total_kill_percentage : ℝ := first_spray_kill_percentage + second_spray_kill_percentage - overlap_percentage

-- The statement to be proved
theorem germs_left_percentage :
  1 - total_kill_percentage = 0.30 :=
by
  -- The proof would go here.
  sorry

end germs_left_percentage_l383_383054


namespace find_d_plus_f_l383_383376

noncomputable def complex_numbers (a b c d e f : ℝ) : Prop :=
  b = 1 ∧ 
  e = -a - 2 * c ∧
  (a + complex.I * b) + (c + complex.I * d) + (e + complex.I * f) = 3 + 2 * complex.I

theorem find_d_plus_f (a b c d e f : ℝ) (h : complex_numbers a b c d e f) : d + f = 1 :=
by sorry

end find_d_plus_f_l383_383376


namespace pascal_triangle_45th_number_l383_383014

theorem pascal_triangle_45th_number :
  let row := List.range (46 + 1) in
  row.nth 44 = some 1035 :=
by
  let row := List.range (46 + 1)
  have binom_46_2 : nat.binom 46 2 = 1035 := by
    -- Calculations for binomials can be validated here
    calc
      nat.binom 46 2 = 46 * 45 / (2 * 1) : by norm_num
      _ = 1035 : by norm_num
  show row.nth 44 = some (nat.binom 46 2) from by
    rw binom_46_2
    simp only [List.nth_range, option.some_eq_coe, nat.lt_succ_iff, nat.le_refl]
  sorry -- Additional reasoning if necessary

end pascal_triangle_45th_number_l383_383014


namespace camila_weeks_needed_l383_383147

/--
Camila has only gone hiking 7 times.
Amanda has gone on 8 times as many hikes as Camila.
Steven has gone on 15 more hikes than Amanda.
Camila plans to go on 4 hikes a week.

Prove that it will take Camila 16 weeks to achieve her goal of hiking as many times as Steven.
-/
noncomputable def hikes_needed_to_match_steven : ℕ :=
  let camila_hikes := 7
  let amanda_hikes := 8 * camila_hikes
  let steven_hikes := amanda_hikes + 15
  let additional_hikes_needed := steven_hikes - camila_hikes
  additional_hikes_needed / 4

theorem camila_weeks_needed : hikes_needed_to_match_steven = 16 := 
  sorry

end camila_weeks_needed_l383_383147


namespace fraction_square_l383_383597

theorem fraction_square : (123456 * 123456) / (24691 * 24691) = 25 := by
  have h1 : 123456 / 24691 = 5 := by norm_num
  have h2 : (123456 * 123456) / (24691 * 24691) = (5 * 5)
  calc 
    (123456 * 123456) / (24691 * 24691) = ((123456 / 24691) ^ 2) : sorry
    ... = (5 ^ 2) : by rw [h1]
  norm_num
  sorry

end fraction_square_l383_383597


namespace criminal_is_B_l383_383160

-- Define the suspects
inductive Suspect : Type
| A
| B
| C
| D

open Suspect

-- Define statements made by each suspect
def statement (s : Suspect) (criminal : Suspect) : Prop :=
match s with
| A => criminal = B ∨ criminal = C ∨ criminal = D
| B => criminal ≠ B ∧ criminal = C
| C => criminal = A ∨ criminal = B
| D => criminal ≠ B ∧ criminal = C
end

-- Define the truth-telling condition
def tells_truth (s : Suspect) (criminal : Suspect) : Prop := 
if s = C then statement s criminal
else ¬ (statement s criminal)

-- Problem's condition: exactly two suspects tell the truth, and one of them is the criminal
def conditions (criminal : Suspect) : Prop :=
∃ (l : list Suspect), l.length = 2 ∧ ∀ x ∈ l, tells_truth x criminal ∧ ∀ y ∉ l, tells_truth y criminal = false

-- The theorem to prove
theorem criminal_is_B : ∃ (criminal : Suspect), conditions criminal ∧ criminal = B :=
sorry

end criminal_is_B_l383_383160


namespace hypotenuse_length_l383_383472

theorem hypotenuse_length (a b c : ℕ) (h₀ : a = 15) (h₁ : b = 36) (h₂ : a^2 + b^2 = c^2) : c = 39 :=
by
  -- Proof is omitted
  sorry

end hypotenuse_length_l383_383472


namespace odd_equivalence_classes_iff_n_eq_2_l383_383037

-- Defining the binary n-tuples and cyclic permutations
def B_n (n : ℕ) : Finset (Vector Bool n) :=
  Finset.univ

def cyclic_permutation {n : ℕ} (v : Vector Bool n) : Finset (Vector Bool n) :=
  Finset.image (λ k => v.rotate k) (Finset.range n)

-- The theorem statement proving the given problem
theorem odd_equivalence_classes_iff_n_eq_2 :
  ∀ (n : ℕ), n ≥ 2 →
  ((B_n n).card / n % 2 = 1) ↔ n = 2 :=
by
  sorry

end odd_equivalence_classes_iff_n_eq_2_l383_383037


namespace apples_to_grapes_equivalent_l383_383822

-- Definitions based on the problem conditions
def apples := ℝ
def grapes := ℝ

-- Given conditions
def given_condition : Prop := (3 / 4) * 12 = 9

-- Question to prove
def question : Prop := (1 / 2) * 6 = 3

-- The theorem statement combining given conditions to prove the question
theorem apples_to_grapes_equivalent : given_condition → question := 
by
    intros
    sorry

end apples_to_grapes_equivalent_l383_383822


namespace calculate_expression_l383_383589

theorem calculate_expression :
  round (4.75 * 3.023 + 5.393 / 1.432 - 2.987^2) 2 = 9.20 :=
by
  sorry

end calculate_expression_l383_383589


namespace hypotenuse_length_l383_383486

-- Definitions for the problem
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def leg1 : ℕ := 15
def leg2 : ℕ := 36
def hypotenuse : ℕ := 39

-- Lean 4 statement
theorem hypotenuse_length (a b c : ℕ) (h : is_right_triangle a b c) (ha : a = leg1) (hb : b = leg2) :
  c = hypotenuse :=
begin
  sorry
end

end hypotenuse_length_l383_383486


namespace interval_between_doses_l383_383284

noncomputable def dose_mg : ℕ := 2 * 375

noncomputable def total_mg_per_day : ℕ := 3000

noncomputable def hours_in_day : ℕ := 24

noncomputable def doses_per_day := total_mg_per_day / dose_mg

noncomputable def hours_between_doses := hours_in_day / doses_per_day

theorem interval_between_doses : hours_between_doses = 6 :=
by
  sorry

end interval_between_doses_l383_383284


namespace train_length_correct_l383_383884

noncomputable def speed_kmh_to_mps (kmh : ℝ) : ℝ :=
  (kmh * 1000) / 3600

noncomputable def length_of_train (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  speed_kmh_to_mps(speed_kmh) * time_sec

theorem train_length_correct (speed_kmh : ℝ) (time_sec : ℝ) (expected_length : ℝ) :
  speed_kmh = 60 → time_sec = 9 → expected_length = 150.03 → length_of_train speed_kmh time_sec = expected_length :=
by
  intros h_speed h_time h_length
  rw [h_speed, h_time, h_length]
  sorry

end train_length_correct_l383_383884


namespace find_f_prime_one_l383_383985

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * (deriv f 1) * x

theorem find_f_prime_one :
  (deriv f 1) = 1 :=
sorry

end find_f_prime_one_l383_383985


namespace function_y_neg3x_plus_1_quadrants_l383_383633

theorem function_y_neg3x_plus_1_quadrants :
  ∀ (x : ℝ), (∃ y : ℝ, y = -3 * x + 1) ∧ (
    (x < 0 ∧ y > 0) ∨ -- Second quadrant
    (x > 0 ∧ y > 0) ∨ -- First quadrant
    (x > 0 ∧ y < 0)   -- Fourth quadrant
  )
:= sorry

end function_y_neg3x_plus_1_quadrants_l383_383633


namespace correct_division_result_l383_383941

theorem correct_division_result {x : ℕ} (h : 3 * x = 90) : x / 3 = 10 :=
by
  -- placeholder for the actual proof
  sorry

end correct_division_result_l383_383941


namespace largest_x_eq_120_div_11_l383_383173

theorem largest_x_eq_120_div_11 (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 11 / 12) : x ≤ 120 / 11 :=
sorry

end largest_x_eq_120_div_11_l383_383173


namespace probability_of_drawing_l383_383273

noncomputable def total_balls (red yellow : Nat) : Nat := red + yellow

def probability_red (red yellow : Nat) : ℚ := red / (total_balls red yellow)
def probability_yellow (red yellow : Nat) : ℚ := yellow / (total_balls red yellow)

def make_prob_equal (initial_red initial_yellow added_red : Nat) : Prop :=
  let total := total_balls (initial_red + added_red) (initial_yellow + (7 - added_red))
  probability_red (initial_red + added_red) (initial_yellow + (7 - added_red)) = probability_yellow (initial_red + added_red) (initial_yellow + (7 - added_red))

theorem probability_of_drawing (initial_red initial_yellow : Nat) (added_red added_yellow : Nat) :
  initial_red = 9 ∧ initial_yellow = 6 ∧ added_red = 2 ∧ added_yellow = 5 →
  probability_red initial_red initial_yellow = 3/5 ∧ 
  probability_yellow initial_red initial_yellow = 2/5 ∧
  make_prob_equal initial_red initial_yellow added_red :=
by
  sorry

end probability_of_drawing_l383_383273


namespace statement1_statement2_l383_383809

def is_pow_of_two (a : ℕ) : Prop := ∃ n : ℕ, a = 2^(n + 1)
def in_A (a : ℕ) : Prop := is_pow_of_two a
def not_in_A (a : ℕ) : Prop := ¬ in_A a ∧ a ≠ 1

theorem statement1 : 
  ∀ (a : ℕ), in_A a → ∀ (b : ℕ), b < 2 * a - 1 → ¬ (2 * a ∣ b * (b + 1)) := 
by {
  sorry
}

theorem statement2 :
  ∀ (a : ℕ), not_in_A a → ∃ (b : ℕ), b < 2 * a - 1 ∧ (2 * a ∣ b * (b + 1)) :=
by {
  sorry
}

end statement1_statement2_l383_383809


namespace final_digit_invariance_l383_383314

theorem final_digit_invariance (A B C : ℕ) :
  ∃ k : ℕ, 
  (∀ n : ℕ, 
   (n < k → (∀ a b c : ℕ, (a + b + c = 3 * n + 2) → (a mod 2 = 0 → b mod 2 = 0 → c mod 2 = 0 → 
   ∃ m : ℕ, m = 1 → A + B + C = 3 * k + 2 → m = 2))) :=
sorry

end final_digit_invariance_l383_383314


namespace minimum_a_l383_383347

-- Definitions using the conditions from Step a)
noncomputable def p (a : ℕ) : ℚ :=
  (choose (45 - a) 2 + choose (a - 1) 2) / 1225

-- The main statement of the proof problem
theorem minimum_a (a m n : ℕ) (h_relprime : nat.coprime m n):
  (p a ≥ 1 / 2) ∧ (a = 14 → p 14 = (62 : ℚ) / 81) → (m + n) = 143 :=
sorry

end minimum_a_l383_383347


namespace cube_cut_perfect_hexagons_l383_383708

-- Define the structure of the cube and the concept of midpoints
structure Cube :=
  (vertices : Finset (ℝ × ℝ × ℝ))
  (edges : Finset (Finset (ℝ × ℝ × ℝ)))
  (midpoints : Finset (ℝ × ℝ × ℝ))

-- Define the cube instance
def myCube : Cube :=
  { vertices := {(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)},
    edges := {{(-1, -1, -1), (-1, -1, 1)}, {(-1, -1, 1), (-1, 1, 1)}, {(-1, 1, 1), (-1, 1, -1)}, {(-1, 1, -1), (-1, -1, -1)},
            {(1, -1, -1), (1, -1, 1)}, {(1, -1, 1), (1, 1, 1)}, {(1, 1, 1), (1, 1, -1)}, {(1, 1, -1), (1, -1, -1)},
            {(-1, -1, -1), (1, -1, -1)}, {(-1, -1, 1), (1, -1, 1)}, {(-1, 1, -1), (1, 1, -1)}, {(-1, 1, 1), (1, 1, 1)}},
    midpoints := {(0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1), (1, 0, 0), (-1, 0, 0)}
  }

-- Prove that a single straight cut through these midpoints results in two new surfaces that are regular hexagons
theorem cube_cut_perfect_hexagons (c : Cube) :
  (∃ p : Plane, ∀ M ∈ c.midpoints, M ∈ p) →
  (∀ s, (s ∈ cut_surfaces p c) → regular_hexagon s) :=
by
  sorry

end cube_cut_perfect_hexagons_l383_383708


namespace paint_needed_to_buy_l383_383614

def total_paint := 333
def existing_paint := 157

theorem paint_needed_to_buy : total_paint - existing_paint = 176 := by
  sorry

end paint_needed_to_buy_l383_383614


namespace solve_for_q_l383_383335

theorem solve_for_q (m n q : ℕ) (h1 : 7/8 = m/96) (h2 : 7/8 = (n + m)/112) (h3 : 7/8 = (q - m)/144) :
  q = 210 :=
sorry

end solve_for_q_l383_383335


namespace sum_first_10_terms_geometric_sequence_l383_383356

theorem sum_first_10_terms_geometric_sequence :
  ∀ {a : ℕ → ℕ}, a 1 = 2 → (∀ n, a (n + 1) = a n * 1) → (Σ n in finset.range 10, a n) = 20 := 
by
  intros a h1 hq
  sorry

end sum_first_10_terms_geometric_sequence_l383_383356


namespace day_of_week_in_100_days_l383_383404

theorem day_of_week_in_100_days (start_day : ℕ) (h : start_day = 5) : 
  (start_day + 100) % 7 = 0 := 
by
  cases h with 
  | rfl => -- start_day is Friday, which is represented as 5
  sorry

end day_of_week_in_100_days_l383_383404


namespace probability_of_sum_11_l383_383355

theorem probability_of_sum_11 (d1 d2 : ℕ) (h1 : 1 ≤ d1 ∧ d1 ≤ 6) (h2 : 1 ≤ d2 ∧ d2 ≤ 6) :
  (∃ (outcomes : finset (ℕ × ℕ)), outcomes = {(5, 6), (6, 5)} ∧
    ∃ (total : ℕ), total = 6 * 6 ∧
    ∃ (prob : ℚ), prob = ((outcomes.card : ℚ) / total) ∧ prob = 1 / 18) := sorry

end probability_of_sum_11_l383_383355


namespace probability_of_green_ball_l383_383945

/-- Container I contains 10 red balls and 5 green balls.
    Container II contains 3 red balls and 6 green balls.
    Container III has 4 red balls and 8 green balls.
    A container is randomly chosen, and a ball is randomly selected from that container.
    Prove that the probability that the ball selected is green is 5/9. -/
theorem probability_of_green_ball :
  let P := λ (red green total : ℕ), green.to_rat / total.to_rat,
      P_container := λ (red green total : ℕ), (total.to_rat / 45).to_rat in
  P_container 10 5 15 * P 10 5 15 +
  P_container 3 6 9 * P 3 6 9 +
  P_container 4 8 12 * P 4 8 12 = 5 / 9 :=
by sorry

end probability_of_green_ball_l383_383945


namespace sum_distances_to_foci_l383_383219

-- Definition of the ellipse and the conditions
def ellipse (x y m : ℝ) : Prop := (x^2 / 4) + (y^2 / m) = 1

theorem sum_distances_to_foci (m : ℝ) (h1 : ellipse 0 4 m) : 
  ∀ (x y : ℝ), ellipse x y m → (distance (x, y) (2, 0) + distance (x, y) (-2, 0) = 8) :=
sorry

end sum_distances_to_foci_l383_383219


namespace statement_c_statement_d_l383_383206

variable {Point Line Plane : Type}
variable (l m : Line) (α β : Plane)
variable contains_line : Plane → Line → Prop
variable parallel_lines : Line → Line → Prop
variable perpendicular_lines : Line → Line → Prop
variable parallel_planes : Plane → Plane → Prop
variable perpendicular_planes : Plane → Plane → Prop

variables (hl : contains_line α l) (hm : contains_line β m)

-- Definitions for conditions
def contains_line (p : Plane) (l : Line) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry

-- Statements C and D
theorem statement_c (h : perpendicular_lines l β) : perpendicular_planes α β := sorry
theorem statement_d (h : parallel_planes α β) : parallel_lines l β := sorry

end statement_c_statement_d_l383_383206


namespace area_of_triangle_ABC_l383_383364

noncomputable def point_refl_yaxis (p : (ℝ × ℝ)) : (ℝ × ℝ) :=
  (-p.1, p.2)

noncomputable def point_refl_xyline (p : (ℝ × ℝ)) : (ℝ × ℝ) :=
  (p.2, p.1)

def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def area_of_triangle (p1 p2 p3 : (ℝ × ℝ)) : ℝ :=
  (1 / 2) * Real.abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

theorem area_of_triangle_ABC :
  let A := (2, 5)
  let B := point_refl_yaxis A
  let C := point_refl_xyline B
  area_of_triangle A B C = 4 * Real.sqrt 2 :=
by
  let A := (2, 5)
  have B := point_refl_yaxis A
  have C := point_refl_xyline B
  have area_ABC := area_of_triangle A B C
  show area_ABC = 4 * Real.sqrt 2, from sorry

end area_of_triangle_ABC_l383_383364


namespace determine_p_q_l383_383777

def vector_a (p : ℚ) : (ℚ × ℚ × ℚ) := (4, p, -2)
def vector_b (q : ℚ) : (ℚ × ℚ × ℚ) := (3, 2, q)

def dot_product (v1 v2 : ℚ × ℚ × ℚ) : ℚ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v : ℚ × ℚ × ℚ) : ℚ :=
  (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

theorem determine_p_q :
  ∃ (p q : ℚ), dot_product (vector_a p) (vector_b q) = 0 ∧ magnitude (vector_a p) = magnitude (vector_b q) ∧ p = -29/12 ∧ q = 50/12 :=
by
  sorry

end determine_p_q_l383_383777


namespace find_incorrect_statement_l383_383879

-- Definitions based on conditions
def contraposition (P Q : Prop) := ¬Q → ¬P

def sufficient_but_not_necessary (P Q : Prop) := P → Q ∧ ¬(Q → P)

def is_false_prop (P Q : Prop) := P = False ∧ Q = False

def neg_of_existential (P : ℝ → Prop) := (∀ x, ¬ P x) → ¬ (∃ x, P x)

-- Main proposition proof statement
theorem find_incorrect_statement (P Q : Prop) (R S : ℝ → Prop) : 
  (contraposition (λ x, x^2 - 4 * x + 3 = 0) (λ x, x = 3)) ∧
  (sufficient_but_not_necessary (λ x, x > 1) (λ x, abs x > 0)) ∧
  (¬ is_false_prop P Q) ∧
  (neg_of_existential R (λ x, x^2 + x + 1 < 0)) 
  → C = "C" := 
by
  intro cont sf_np not_ff neg_exist
  have incorr : "C" = "C" := sorry
  exact incorr

end find_incorrect_statement_l383_383879


namespace number_of_ordered_pairs_l383_383236

theorem number_of_ordered_pairs (a b : ℤ) : 
  (a^2 + b^2 < 16) ∧ (a^2 + b^2 < 8 * a) ∧ (a^2 + b^2 < 8 * b) → 
  {p : ℤ × ℤ | (p.1, p.2).fst ^ 2 + (p.1, p.2).snd ^ 2 < 16 ∧ 
                (p.1, p.2).fst ^ 2 + (p.1, p.2).snd ^ 2 < 8 * (p.1, p.2).fst ∧ 
                (p.1, p.2).fst ^ 2 + (p.1, p.2).snd ^ 2 < 8 * (p.1, p.2).snd }.card = 6 :=
sorry

end number_of_ordered_pairs_l383_383236


namespace log_x_base_13_l383_383717

theorem log_x_base_13 (x : ℝ) (h : log 7 (x - 3) = 2) : log 13 x = (log 52) / (log 13) :=
by {
  sorry
}

end log_x_base_13_l383_383717


namespace marked_price_correct_l383_383880

-- Set up the variables and conditions
variables (marked_price selling_price cost_price profit : ℝ)
variable (discount : ℝ := 0.1)

-- Define the conditions
def selling_price_discounted (marked_price : ℝ) : ℝ := marked_price - discount * marked_price
def profit_condition (selling_price cost_price profit : ℝ) : Prop :=
  selling_price = cost_price + profit

-- Given conditions
def given_conditions :=
  profit_condition (selling_price_discounted marked_price) cost_price profit ∧
  profit = 30 ∧
  cost_price = 60

-- The theorem to be proven
theorem marked_price_correct : given_conditions → marked_price = 100 :=
by
  intros
  sorry

end marked_price_correct_l383_383880


namespace number_of_pairs_is_1734_l383_383157

theorem number_of_pairs_is_1734 :
  (∃ n m : ℕ, (1 ≤ m ∧ m ≤ 4024) ∧ (3^n < 2^m ∧ 2^m < 2^(m+1) ∧ 2^(m+1) < 3^(n+1))) ∧
  (3^1734 > 2^4023 ∧ 3^1734 < 2^4024) →
  {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 4024 ∧ 3^p.2 < 2^p.1 ∧ 2^p.1 < 2^(p.1 + 1) ∧ 2^(p.1 + 1) < 3^(p.2 + 1)}.card = 1734 :=
sorry

end number_of_pairs_is_1734_l383_383157


namespace right_triangle_hypotenuse_length_l383_383531

theorem right_triangle_hypotenuse_length :
  ∀ (a b h : ℕ), a = 15 → b = 36 → h^2 = a^2 + b^2 → h = 39 :=
by
  intros a b h ha hb hyp
  -- In the proof, we would use ha, hb, and hyp to show h = 39
  sorry

end right_triangle_hypotenuse_length_l383_383531


namespace min_value_a_l383_383630

theorem min_value_a (a b c : ℤ) (α β : ℝ)
  (h_a_pos : a > 0) 
  (h_eq : ∀ x : ℝ, a * x^2 + b * x + c = 0 → (x = α ∨ x = β))
  (h_alpha_beta_order : 0 < α ∧ α < β ∧ β < 1) :
  a ≥ 5 :=
sorry

end min_value_a_l383_383630


namespace trajectory_of_M_l383_383226

-- Definitions for the given conditions
def point (x y : ℝ) := (x, y)
def ellipse (x y : ℝ) := x^2 / 2 + y^2 = 1
def line_through (P A B : ℝ × ℝ) := A.1 * (B.2 - P.2) = B.1 * (A.2 - P.2)

-- Problem statement
theorem trajectory_of_M (P : ℝ × ℝ) (A B M : ℝ × ℝ)
    (hP : P = (2, 2))
    (hE1 : ellipse A.1 A.2)
    (hE2 : ellipse B.1 B.2)
    (hM : M = (A.1 + B.1) / 2, (A.2 + B.2) / 2)
    (hLine : line_through P A B) : 
    (M.1 - 1)^2 + 2 * (M.2 - 1)^2 = 3 := 
sorry

end trajectory_of_M_l383_383226


namespace number_of_Joes_in_crowd_l383_383270

noncomputable def num_people_named_Joe (barrys kevins julies joes : ℕ) : ℕ :=
  joes

axiom all_people_are_nice (barrys kevins julies joes nice_barrys nice_kevins nice_julies nice_joes : ℕ) :
  nice_barrys = barrys →
  nice_kevins = kevins / 2 →
  nice_julies = 3 * julies / 4 →
  nice_joes = joes / 10 →
  nice_barrys + nice_kevins + nice_julies + nice_joes = 99 →
  barrys = 24 →
  kevins = 20 →
  julies = 80 →
  num_people_named_Joe barrys kevins julies joes = 50

theorem number_of_Joes_in_crowd : ∀ (barrys kevins julies nice_barrys nice_kevins nice_julies : ℕ),
  (∃ joes, all_people_are_nice barrys kevins julies joes nice_barrys nice_kevins nice_julies (nice_joes := joes / 10)) →
  num_people_named_Joe barrys kevins julies 50 = 50 :=
by
  intros barrys kevins julies nice_barrys nice_kevins nice_julies h
  obtain ⟨joes, h_all⟩ := h
  exact rfl

end number_of_Joes_in_crowd_l383_383270


namespace hundred_days_from_friday_is_sunday_l383_383400

def days_from_friday (n : ℕ) : Nat :=
  (n + 5) % 7  -- 0 corresponds to Sunday, starting from Friday (5 + 0 % 7 = 5 which is Friday)

theorem hundred_days_from_friday_is_sunday :
  days_from_friday 100 = 0 := by
  sorry

end hundred_days_from_friday_is_sunday_l383_383400


namespace speed_of_stream_l383_383368

theorem speed_of_stream (x : ℝ) (boat_speed : ℝ) (distance_one_way : ℝ) (total_time : ℝ) 
  (h1 : boat_speed = 16) 
  (h2 : distance_one_way = 7560) 
  (h3 : total_time = 960) 
  (h4 : (distance_one_way / (boat_speed + x)) + (distance_one_way / (boat_speed - x)) = total_time) 
  : x = 2 := 
  sorry

end speed_of_stream_l383_383368


namespace min_expression_l383_383656

theorem min_expression (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1/a + 1/b = 1) : 
  (∃ x : ℝ, x = min ((1 / (a - 1)) + (4 / (b - 1))) 4) :=
sorry

end min_expression_l383_383656


namespace log_domain_is_pos_real_l383_383026

noncomputable def domain_log : Set ℝ := {x | x > 0}
noncomputable def domain_reciprocal : Set ℝ := {x | x ≠ 0}
noncomputable def domain_sqrt : Set ℝ := {x | x ≥ 0}
noncomputable def domain_exp : Set ℝ := {x | true}

theorem log_domain_is_pos_real :
  (domain_log = {x : ℝ | 0 < x}) ∧ 
  (domain_reciprocal = {x : ℝ | x ≠ 0}) ∧ 
  (domain_sqrt = {x : ℝ | 0 ≤ x}) ∧ 
  (domain_exp = {x : ℝ | true}) →
  domain_log = {x : ℝ | 0 < x} :=
by
  intro h
  sorry

end log_domain_is_pos_real_l383_383026


namespace part1_inequality_l383_383438

theorem part1_inequality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (|3 * a + b| + |a - b|) ≥ (|a| * (|x + 1| + |x - 1|)) 
  (x : ℝ) : (x ∈ set.Icc (-2 : ℝ) (2 : ℝ)) :=
by sorry

end part1_inequality_l383_383438


namespace supremum_s_l383_383189

theorem supremum_s (n : ℕ) (h : 2 ≤ n) (x : ℕ → ℝ)
  (h1 : ∀ i : ℕ, 1 ≤ i → i ≤ n → 1 < x i)
  (h2 : ∀ i : ℕ, 1 ≤ i → i ≤ n → (x i) ^ 2 / (x i - 1) ≥ ∑ j in (Finset.range n).filter (λ j, 1 ≤ j ∧ j ≤ n), x j):
  supS = (n * n) / (n - 1) :=
sorry

end supremum_s_l383_383189


namespace fraction_between_stops_l383_383159

/-- Prove that the fraction of the remaining distance traveled between Maria's first and second stops is 1/4. -/
theorem fraction_between_stops (total_distance first_stop_distance remaining_distance final_leg_distance : ℝ)
  (h_total : total_distance = 400)
  (h_first_stop : first_stop_distance = total_distance / 2)
  (h_remaining : remaining_distance = total_distance - first_stop_distance)
  (h_final_leg : final_leg_distance = 150)
  (h_second_leg : remaining_distance - final_leg_distance = 50) :
  50 / remaining_distance = 1 / 4 :=
by
  { sorry }

end fraction_between_stops_l383_383159


namespace set_contains_one_implies_values_l383_383229

theorem set_contains_one_implies_values (x : ℝ) (A : Set ℝ) (hA : A = {x, x^2}) (h1 : 1 ∈ A) : x = 1 ∨ x = -1 := by
  sorry

end set_contains_one_implies_values_l383_383229


namespace suitable_dishes_fraction_l383_383312

theorem suitable_dishes_fraction (total_dishes : ℕ) (vegan_fraction : ℚ) (gluten_fraction : ℚ)
  (nuts_fraction : ℚ) (h_total : total_dishes = 30) (h_vegan : vegan_fraction = 1/3)
  (h_gluten : gluten_fraction = 2/5) (h_nuts : nuts_fraction = 1/4) :
  (∃ suitable_fraction : ℚ, suitable_fraction = 1/10) :=
by
  have vegan_dishes := vegan_fraction * total_dishes,
  have vegan_gluten_dishes := gluten_fraction * vegan_dishes,
  have vegan_nuts_dishes := nuts_fraction * vegan_dishes,
  have unsuitable_dishes := vegan_gluten_dishes + vegan_nuts_dishes,
  have suitable_dishes := vegan_dishes - unsuitable_dishes,
  use suitable_dishes / total_dishes,
  sorry

end suitable_dishes_fraction_l383_383312


namespace vasya_password_combinations_l383_383000

/-- A function to count the number of valid 4-digit passwords as per the given constraints. -/
def count_valid_passwords : Nat := 
  let digits := {0, 1, 3, 4, 5, 6, 7, 8, 9}
  let valid_A := digits.toFinset.card  -- 9
  let valid_B := valid_A - 1            -- 8 (excluding A)
  let valid_C := valid_B - 1            -- 7 (excluding A and B)
  valid_A * valid_B * valid_C

theorem vasya_password_combinations : count_valid_passwords = 504 := by
  sorry

end vasya_password_combinations_l383_383000


namespace rectangle_area_stage_5_l383_383720

-- Definitions based on the conditions in the problem
def side_length (n : ℕ) : ℕ :=
  n + 2

def length_at_stage (stage : ℕ) : ℕ :=
  ∑ i in Finset.range stage, side_length i

def width_at_stage (stage : ℕ) : ℕ :=
  side_length (stage - 1)

def area_of_rectangle (stage : ℕ) : ℕ :=
  length_at_stage stage * width_at_stage stage

-- Statement of the problem rewritten in Lean 4
theorem rectangle_area_stage_5 : area_of_rectangle 5 = 175 :=
by
  -- proof to be provided
  sorry

end rectangle_area_stage_5_l383_383720


namespace distinct_left_views_l383_383114

/-- Consider 10 small cubes each having dimension 1 cm × 1 cm × 1 cm.
    Each pair of adjacent cubes shares at least one edge (1 cm) or one face (1 cm × 1 cm).
    The cubes must not be suspended in the air and each cube's edges should be either
    perpendicular or parallel to the horizontal lines. Prove that the number of distinct
    left views of any arrangement of these 10 cubes is 16. -/
theorem distinct_left_views (cube_count : ℕ) (dimensions : ℝ) 
  (shared_edge : (ℝ × ℝ) → Prop) (no_suspension : Prop) (alignment : Prop) :
  cube_count = 10 →
  dimensions = 1 →
  (∀ x y, shared_edge (x, y) ↔ x = y ∨ x - y = 1) →
  no_suspension →
  alignment →
  distinct_left_views_count = 16 :=
by
  sorry

end distinct_left_views_l383_383114


namespace hypotenuse_length_l383_383495

theorem hypotenuse_length (a b : ℤ) (h₀ : a = 15) (h₁ : b = 36) : 
  ∃ c : ℤ, c^2 = a^2 + b^2 ∧ c = 39 := 
by {
  sorry
}

end hypotenuse_length_l383_383495


namespace original_treadmill_price_l383_383316

-- Given conditions in Lean definitions
def discount_rate : ℝ := 0.30
def plate_cost : ℝ := 50
def num_plates : ℕ := 2
def total_paid : ℝ := 1045

noncomputable def treadmill_price :=
  let plate_total := num_plates * plate_cost
  let treadmill_discount := (1 - discount_rate)
  (total_paid - plate_total) / treadmill_discount

theorem original_treadmill_price :
  treadmill_price = 1350 := by
  sorry

end original_treadmill_price_l383_383316


namespace polynomial_division_l383_383623

-- Define the polynomials
def f : Polynomial ℤ := Polynomial.C 1 * Polynomial.X ^ 6 + Polynomial.C 3
def g : Polynomial ℤ := Polynomial.X - Polynomial.C 2
def q : Polynomial ℤ := Polynomial.X ^ 5 + Polynomial.C 2 * Polynomial.X ^ 4 + Polynomial.C 4 * Polynomial.X ^ 3 + Polynomial.C 8 * Polynomial.X ^ 2 + Polynomial.C 16 * Polynomial.X + Polynomial.C 32
def r : Polynomial ℤ := Polynomial.C 67

-- Prove the relationship between them
theorem polynomial_division :
  f = g * q + r :=
by
  sorry

end polynomial_division_l383_383623


namespace gcd_of_A_and_B_l383_383726

theorem gcd_of_A_and_B (A B : ℕ) (h_lcm : Nat.lcm A B = 120) (h_ratio : A * 4 = B * 3) : Nat.gcd A B = 10 :=
sorry

end gcd_of_A_and_B_l383_383726


namespace concyclic_points_l383_383295

/-- Let L be the midpoint of the smaller arc AC of the circumcircle
    of an acute-angled triangle ABC. Let M be the midpoint of AB
    and N be the midpoint of BC. Drop a perpendicular from B to
    the tangent at L, and let the foot of this perpendicular be P.
    Prove that points P, L, M, and N lie on the same circle. -/
theorem concyclic_points
  {A B C P L M N : Type*}
  (hL : L is the midpoint of the smaller arc ⟨A, C⟩ of the circumcircle of ⟨A, B, C⟩)
  (hM : M is the midpoint of ⟨A, B⟩)
  (hN : N is the midpoint of ⟨B, C⟩)
  (hP : make_perpendicular_from_to B P (tangent_at ⟨circumcircle_of ⟨A, B, C⟩, L⟩)):
  concyclic {P, L, M, N} :=
sorry

end concyclic_points_l383_383295


namespace distance_between_points_l383_383019

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points :
  distance 3 (-2) (-7) 4 = 2 * Real.sqrt 34 := by
  sorry

end distance_between_points_l383_383019


namespace find_a1_find_Sn_l383_383775

noncomputable theory

def S : ℕ → ℕ
def a : ℕ → ℕ

axiom S_2_eq_3 : S 2 = 3
axiom a_recurrence : ∀ n ∈ ℕ, a (n + 1) = S n + 1

theorem find_a1 : a 1 = 1 := 
sorry

theorem find_Sn : ∀ n ∈ ℕ, S n = 2^n - 1 :=
sorry

end find_a1_find_Sn_l383_383775


namespace necessary_condition_for_acute_angle_l383_383693

-- Defining vectors a and b
def vec_a (x : ℝ) : ℝ × ℝ := (x - 3, 2)
def vec_b : ℝ × ℝ := (1, 1)

-- Condition for the dot product to be positive
def dot_product_positive (x : ℝ) : Prop :=
  let (ax1, ax2) := vec_a x
  let (bx1, bx2) := vec_b
  ax1 * bx1 + ax2 * bx2 > 0

-- Statement for necessary condition
theorem necessary_condition_for_acute_angle (x : ℝ) :
  (dot_product_positive x) → (1 < x) :=
sorry

end necessary_condition_for_acute_angle_l383_383693


namespace polar_to_cartesian_l383_383365

theorem polar_to_cartesian (ρ : ℝ) (θ : ℝ) (hx : ρ = 3) (hy : θ = π / 6) :
  (ρ * Real.cos θ, ρ * Real.sin θ) = (3 * Real.cos (π / 6), 3 * Real.sin (π / 6)) := by
  sorry

end polar_to_cartesian_l383_383365


namespace faith_change_l383_383167

def flour_cost : ℕ := 5
def cake_stand_cost : ℕ := 28
def two_twenty_bills : ℕ := 2 * 20
def loose_coins : ℕ := 3
def total_cost : ℕ := flour_cost + cake_stand_cost
def total_given : ℕ := two_twenty_bills + loose_coins
def change : ℕ := total_given - total_cost

theorem faith_change : change = 10 := by
  sorry

end faith_change_l383_383167


namespace intersection_count_l383_383953

theorem intersection_count :
  ∀ {x y : ℝ}, (2 * x - 2 * y + 4 = 0 ∨ 6 * x + 2 * y - 8 = 0) ∧ (y = -x^2 + 2 ∨ 4 * x - 10 * y + 14 = 0) → 
  (x ≠ 0 ∨ y ≠ 2) ∧ (x ≠ -1 ∨ y ≠ 1) ∧ (x ≠ 1 ∨ y ≠ -1) ∧ (x ≠ 2 ∨ y ≠ 2) → 
  ∃! (p : ℝ × ℝ), (p = (0, 2) ∨ p = (-1, 1) ∨ p = (1, -1) ∨ p = (2, 2)) := sorry

end intersection_count_l383_383953


namespace hypotenuse_length_l383_383481

theorem hypotenuse_length (a b c : ℕ) (h₀ : a = 15) (h₁ : b = 36) (h₂ : a^2 + b^2 = c^2) : c = 39 :=
by
  -- Proof is omitted
  sorry

end hypotenuse_length_l383_383481


namespace smallest_n_for_Tn_gt_2006_over_2016_l383_383848

-- Definitions from the given problem
def Sn (n : ℕ) : ℚ := n^2 / (n + 1)
def an (n : ℕ) : ℚ := if n = 1 then 1 / 2 else Sn n - Sn (n - 1)
def bn (n : ℕ) : ℚ := an n / (n^2 + n - 1)

-- Definition of Tn sum
def Tn (n : ℕ) : ℚ := (Finset.range n).sum (λ k => bn (k + 1))

-- The main statement
theorem smallest_n_for_Tn_gt_2006_over_2016 : ∃ n : ℕ, Tn n > 2006 / 2016 := by
  sorry

end smallest_n_for_Tn_gt_2006_over_2016_l383_383848


namespace math_problem_l383_383672

noncomputable def f (a x : ℝ) := log a (x + 2) + log a (3 - x)

theorem math_problem 
  (a : ℝ) (h : 0 < a ∧ a < 1) 
  (h_min : ∀ x, -2 < x ∧ x < 3 → f a x ≥ -4)
  (min_val : ∃ x, f a x = -4) :
  (∀ x, f a x ≥ -4 → -2 < x ∧ x < 3) ∧ a = real.sqrt 10 / 5 :=
by sorry

end math_problem_l383_383672


namespace slope_of_line_through_midpoints_l383_383868

theorem slope_of_line_through_midpoints :
  let mid1 := (0 + 3) / 2, (0 + 4) / 2,
            mid2 := (6 + 7) / 2, (0 + 4) / 2
  mid1.1 = 3 / 2 → mid1.2 = 2 → mid2.1 = 13 / 2 → mid2.2 = 2 → 
  (mid2.2 - mid1.2) / (mid2.1 - mid1.1) = 0 :=
by
  sorry

end slope_of_line_through_midpoints_l383_383868


namespace limit_tangent_logarithm_l383_383935

theorem limit_tangent_logarithm (a : ℝ) (hla : a ≠ 0) :
  (filter.tendsto (λ x, (tan x - tan a) / (log x - log a)) (nhds a) (nhds (a / cos a ^ 2))) :=
sorry

end limit_tangent_logarithm_l383_383935


namespace algebraic_expression_value_l383_383999

theorem algebraic_expression_value (x y : ℝ) (h : |x - 2| + (y + 3)^2 = 0) : (x + y)^2023 = -1 := by
  sorry

end algebraic_expression_value_l383_383999


namespace tangent_length_is_five_sqrt_two_l383_383595

noncomputable def point := (ℝ × ℝ)
noncomputable def distance (p1 p2 : point) : ℝ := (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2
noncomputable def is_on_circle (p : point) (center : point) (radius : ℝ) : Prop :=
  distance p center = radius ^ 2

noncomputable def circumncircle_radius (A B C : point) : ℝ := 
  let h := ... -- some expression to determine h
  let k := ... -- some expression to determine k
  let r := distance A (h,k) -- radius from one of the points to the center
  r

noncomputable def length_of_tangent (P A B C : point) : ℝ :=
  let r := circumncircle_radius A B C
  let dist_PA := sqrt $ distance P A
  let dist_PB := sqrt $ distance P B
  sqrt (dist_PA * dist_PB)

noncomputable def length_of_segment_tangent : ℝ :=
  length_of_tangent (1,1) (4,5) (7,9) (6,14)

theorem tangent_length_is_five_sqrt_two :
  length_of_segment_tangent = 5 * sqrt 2 := by
  sorry

end tangent_length_is_five_sqrt_two_l383_383595


namespace hypotenuse_length_l383_383488

-- Definitions for the problem
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def leg1 : ℕ := 15
def leg2 : ℕ := 36
def hypotenuse : ℕ := 39

-- Lean 4 statement
theorem hypotenuse_length (a b c : ℕ) (h : is_right_triangle a b c) (ha : a = leg1) (hb : b = leg2) :
  c = hypotenuse :=
begin
  sorry
end

end hypotenuse_length_l383_383488


namespace train_crosses_pole_in_11_11_seconds_l383_383569

noncomputable def train_crossing_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

theorem train_crosses_pole_in_11_11_seconds :
  train_crossing_time 700 62.99999999999999 ≈ 11.11 :=
by
  -- Proof would go here
  sorry

end train_crosses_pole_in_11_11_seconds_l383_383569


namespace infinitely_many_orange_primes_l383_383772

-- Define the concept of an orange prime
def is_orange_prime (q p : ℕ) [nat.prime q] [nat.prime p] : Prop :=
  ∀ a : ℤ, ∃ r : ℤ, r^q ≡ a [ZMOD p]

-- The main theorem to be proved: there are infinitely many orange primes
theorem infinitely_many_orange_primes (q : ℕ) [fact (nat.prime q)] (hq : q % 2 = 1) : 
  ∃ᶠ p in filter.at_top, is_orange_prime q p := 
sorry

end infinitely_many_orange_primes_l383_383772


namespace cost_per_sq_m_plastering_l383_383567

theorem cost_per_sq_m_plastering
  (length width depth total_cost : ℝ)
  (h_length : length = 25)
  (h_width : width = 12)
  (h_depth : depth = 6)
  (h_total_cost : total_cost = 558) :
  let area_long_walls := 2 * (length * depth)
      area_wide_walls := 2 * (width * depth)
      area_bottom := length * width
      total_area := area_long_walls + area_wide_walls + area_bottom
      cost_per_sq_m := total_cost / total_area
  in cost_per_sq_m = 0.75 :=
by
  rw [h_length, h_width, h_depth, h_total_cost]
  let area_long_walls := 2 * (25 * 6)
  let area_wide_walls := 2 * (12 * 6)
  let area_bottom := 25 * 12
  let total_area := area_long_walls + area_wide_walls + area_bottom
  let cost_per_sq_m := 558 / total_area
  have h_area_long_walls : area_long_walls = 300 := by norm_num
  have h_area_wide_walls : area_wide_walls = 144 := by norm_num
  have h_area_bottom : area_bottom = 300 := by norm_num
  have h_total_area : total_area = 744 := by
    rw [h_area_long_walls, h_area_wide_walls, h_area_bottom]
    norm_num
  rw [h_total_area]
  have h_cost_per_sq_m : 558 / 744 = 0.75 := by norm_num
  exact h_cost_per_sq_m

end cost_per_sq_m_plastering_l383_383567


namespace friends_ranking_l383_383125

-- Define friends and their relative ages
inductive Friend : Type
| Amy | Bill | Celine | David
deriving DecidableEq

open Friend

-- Only one of these statements can be true
def Statement : Type
| I | II | III | IV
deriving DecidableEq

open Statement

def exactly_one_true (P Q R S : Prop) : Prop :=
  (P ∧ ¬Q ∧ ¬R ∧ ¬S) ∨ (¬P ∧ Q ∧ ¬R ∧ ¬S) ∨ (¬P ∧ ¬Q ∧ R ∧ ¬S) ∨ (¬P ∧ ¬Q ∧ ¬R ∧ S)

def is_oldest : Friend → Prop := λ f,
  match f with
  | Amy    => λ hb ha hc hd => false
  | Bill   => λ hb ha hc hd => ¬hb ∧ ha ∧ ¬hc ∧ ¬hd
  | Celine => λ hb ha hc hd => ¬hb ∧ ¬ha ∧ ¬hc ∧ ¬hd
  | David  => λ hb ha hc hd => ¬hb ∧ ¬ha ∧ ¬hc ∧ hd

def is_youngest : Friend → Prop := λ f,
  match f with
  | Amy    => λ hb ha hc hd => false
  | Bill   => λ hb ha hc hd => hb ∧ ¬ha ∧ ha ∧ ¬hd
  | Celine => λ hb ha hc hd => ¬hb ∧ ¬ha ∧ ¬hc ∧ ¬hd
  | David  => λ hb ha hc hd => ¬hb ∧ ¬ha ∧ hc ∧ hd

/-- Problem statement: Rank the friends from oldest to youngest given the conditions. -/
theorem friends_ranking :
  ∃ (o1 o2 o3 o4 : Friend),
  exactly_one_true
    (¬ is_youngest Bill) -- I: Bill is not the youngest
    (is_oldest Amy)      -- II: Amy is the oldest
    (¬ is_oldest Celine) -- III: Celine is not the oldest
    (is_youngest David)  -- IV: David is the youngest
  ∧ o1 = Celine
  ∧ o2 = Bill
  ∧ o3 = Amy
  ∧ o4 = David :=
sorry

end friends_ranking_l383_383125


namespace parabola_point_coordinates_l383_383466

theorem parabola_point_coordinates :
  ∃ (P : ℝ × ℝ), 
    P.1 > 0 ∧ P.2 > 0 ∧ -- P is in the first quadrant
    (P.1^2 = 4 * P.2) ∧ -- The relationship x^2 = 4y
    (math.sqrt(P.1^2 + (P.2 - 1)^2) = 101) ∧ -- Distance from P to Focus is 101
    P = (20, 100) := -- Coordinates of P
sorry

end parabola_point_coordinates_l383_383466


namespace oblique_circle_equation_l383_383948

-- Define the oblique coordinate system with given conditions
noncomputable def oblique_coord_system (x y : ℝ) : Prop :=
  let e1 := (1 : ℝ) -- unit vector along x-axis
  let e2 := (1 : ℝ) -- unit vector along y-axis
  let dot_product := cos (120 : ℝ) -- dot product from the angle
  dot_product = -1/2
  
-- Define the coordinates of point C
def point_C (x y : ℝ) : Prop := 
  x = 2 ∧ y = 3

-- Define the circle equation in the oblique coordinate system
def circle_eq (x y : ℝ) : Prop := 
  x^2 + y^2 - x*y - x - 4*y + 3 = 0
  
-- The main theorem asserting the correct circle equation given the conditions
theorem oblique_circle_equation : 
  ∀ x y : ℝ, oblique_coord_system x y ∧ point_C x y → circle_eq x y :=
begin
  sorry
end

end oblique_circle_equation_l383_383948


namespace missing_digit_in_mean_of_number_set_l383_383944

noncomputable def number_set : Finset ℝ :=
Finset.range 9 |>.map (λ k, ((8:ℝ) * (10 ^ (k + 1) - 1) / 9))

noncomputable def arithmetic_mean (s : Finset ℝ) : ℝ :=
s.sum / s.card

theorem missing_digit_in_mean_of_number_set : 
  ∀ M : ℝ, M = arithmetic_mean number_set → (∃ d : ℕ, d ∈ (Finset.range 10) ∧ ¬(d : ℝ) ∈ Finset.coe (Finset.image (λ x, (x : ℝ).fract) (LinOrder.range 10)) → d = 0) :=
by
  sorry

end missing_digit_in_mean_of_number_set_l383_383944


namespace Milo_siblings_l383_383844

structure Child :=
  (name : String)
  (eye_color : String)
  (hair_color : String)
  (age : Nat)

def Caitlin := Child.mk "Caitlin" "Green" "Red" 10
def Lucas := Child.mk "Lucas" "Gray" "Brown" 12
def Milo := Child.mk "Milo" "Gray" "Red" 13
def Emma := Child.mk "Emma" "Green" "Brown" 11
def Sophia := Child.mk "Sophia" "Gray" "Red" 10
def Noah := Child.mk "Noah" "Green" "Red" 14
def Olivia := Child.mk "Olivia" "Gray" "Brown" 12

def siblings (c1 c2 c3 : Child) : Prop :=
  (c1.eye_color = c2.eye_color ∨ c1.hair_color = c2.hair_color ∨ (abs (c1.age - c2.age) ≤ 2)) ∧
  (c1.eye_color = c3.eye_color ∨ c1.hair_color = c3.hair_color ∨ (abs (c1.age - c3.age) ≤ 2)) ∧
  (c2.eye_color = c3.eye_color ∨ c2.hair_color = c3.hair_color ∨ (abs (c2.age - c3.age) ≤ 2))

theorem Milo_siblings : siblings Milo Lucas Olivia := by
  -- proof steps would go here, but are omitted
  sorry

end Milo_siblings_l383_383844


namespace points_concyclic_l383_383434

-- Definitions based on the conditions
noncomputable def O (AB : set ℝ) : set ℝ := sorry
def C (AB : set ℝ) : set ℝ := sorry
def D (C : set ℝ) : set ℝ := sorry
def E (C : set ℝ) : set ℝ := sorry
noncomputable def F (O1 : set ℝ) : set ℝ := sorry
def G (C F : set ℝ) : set ℝ := sorry

-- Prove that points O, A, E, G are concyclic
theorem points_concyclic 
  (AB : set ℝ) (O : set ℝ) (C : set ℝ) (D : set ℝ) (E : set ℝ) 
  (O1 : set ℝ) (F : set ℝ) (G : set ℝ)
  (h1 : is_diameter O AB)
  (h2 : C ∈ line_extension AB)
  (h3 : D ∈ line_through_two_points C O)
  (h4 : E ∈ line_through_two_points C O)
  (h5 : is_diameter O1 F)
  (h6 : G ∈ circle_inter_distribution O1 (line_extension CF)) :
  is_concyclic {O, A, E, G} :=
sorry

end points_concyclic_l383_383434


namespace camila_weeks_needed_l383_383146

/--
Camila has only gone hiking 7 times.
Amanda has gone on 8 times as many hikes as Camila.
Steven has gone on 15 more hikes than Amanda.
Camila plans to go on 4 hikes a week.

Prove that it will take Camila 16 weeks to achieve her goal of hiking as many times as Steven.
-/
noncomputable def hikes_needed_to_match_steven : ℕ :=
  let camila_hikes := 7
  let amanda_hikes := 8 * camila_hikes
  let steven_hikes := amanda_hikes + 15
  let additional_hikes_needed := steven_hikes - camila_hikes
  additional_hikes_needed / 4

theorem camila_weeks_needed : hikes_needed_to_match_steven = 16 := 
  sorry

end camila_weeks_needed_l383_383146


namespace max_value_sin_cos_l383_383839

theorem max_value_sin_cos (x : ℝ) :
  ∃ M, M = sin x * cos x + sin x + cos x ∧ ∀ y, y = sin x * cos x + sin x + cos x → y ≤ (1/2 + sqrt 2) :=
sorry

end max_value_sin_cos_l383_383839


namespace hundred_days_from_friday_is_sunday_l383_383391

/-- Given that today is Friday, determine that 100 days from now is Sunday. -/
theorem hundred_days_from_friday_is_sunday (today : ℕ) (days_in_week : ℕ := 7) 
(friday : ℕ := 0) (sunday : ℕ := 2) : (((today + 100) % days_in_week) = sunday) :=
sorry

end hundred_days_from_friday_is_sunday_l383_383391


namespace expression_value_at_neg1_l383_383420

theorem expression_value_at_neg1
  (p q : ℤ)
  (h1 : p + q = 2016) :
  p * (-1)^3 + q * (-1) - 10 = -2026 := by
  sorry

end expression_value_at_neg1_l383_383420


namespace expression_value_cardinality_l383_383056

theorem expression_value_cardinality :
  let expr := ["*", "*", "*", "*", "*", "*", "*"]
  let ops := ["+", "-", "/", "*"]
  -- Define a function that evaluates the expression based on the given order of operations
  let eval_expr (expr : List String) (ops : List String) : Int := sorry 
  -- Define a function that generates all possible combinations of operators applied to the given expression
  let generate_combinations (expr : List String) (ops : List String) : List (List String) := sorry
  -- The number of distinct results we can obtain by replacing each \( * \) in the expression with one of the operators \( +, -, /, \times \)
  List.card (List.uniqs (List.map (λ c => eval_expr c ops) (generate_combinations expr ops))) = 15 := sorry

end expression_value_cardinality_l383_383056


namespace instantaneous_velocity_at_3_l383_383253

-- Define the position function s(t)
def s (t : ℝ) : ℝ := 3 * t^2

-- Define the velocity function v(t) as the derivative of s(t)
def v (t : ℝ) : ℝ := derivative s t

-- Define a theorem to state the problem: Prove that the instantaneous velocity at t = 3 is 18
theorem instantaneous_velocity_at_3 : v 3 = 18 := sorry

end instantaneous_velocity_at_3_l383_383253


namespace find_square_number_divisible_by_9_between_40_and_90_l383_383959

theorem find_square_number_divisible_by_9_between_40_and_90 :
  ∃ x : ℕ, (∃ n : ℕ, x = n^2) ∧ (9 ∣ x) ∧ 40 < x ∧ x < 90 ∧ x = 81 :=
by
  sorry

end find_square_number_divisible_by_9_between_40_and_90_l383_383959


namespace total_earnings_in_september_l383_383697

theorem total_earnings_in_september (
  mowing_rate: ℕ := 6
  mowing_hours: ℕ := 63
  weeds_rate: ℕ := 11
  weeds_hours: ℕ := 9
  mulch_rate: ℕ := 9
  mulch_hours: ℕ := 10
): 
  mowing_rate * mowing_hours + weeds_rate * weeds_hours + mulch_rate * mulch_hours = 567 := 
by
  sorry

end total_earnings_in_september_l383_383697


namespace weave_time_for_8_mat_weaves_l383_383442

-- Define conditions
def rate_per_mat_weave : ℝ := 1 / 4  -- One mat-weave can weave 1 mat in 4 days

-- Define the given problem as an equality that needs to be proven
theorem weave_time_for_8_mat_weaves :
  let rate_for_8_weaves := 8 * rate_per_mat_weave in
  let mats_to_weave := 16 in
  let tot_days := mats_to_weave / rate_for_8_weaves in
  tot_days = 8 := sorry

end weave_time_for_8_mat_weaves_l383_383442


namespace Part1_Part2_l383_383674

section Part1

def f_a2 (x : ℝ) : ℝ := |x + 2| + |x - 3|

theorem Part1 (x : ℝ) : f_a2 x ≥ 2 * x ↔ x ∈ set.Iic (5 / 2) :=
by
  sorry

end Part1

section Part2

variable (a : ℝ)
def f_a (x : ℝ) : ℝ := |x + a| + |x - 3|

theorem Part2 : (∃ x : ℝ, f_a a x ≤ (1 / 2) * a + 5) ↔ a ∈ set.Icc (-16 / 3) 4 :=
by
  sorry

end Part2

end Part1_Part2_l383_383674


namespace right_triangle_hypotenuse_l383_383515

theorem right_triangle_hypotenuse (a b : ℕ) (ha : a = 15) (hb : b = 36) : 
  ∃ h : ℕ, h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  . exact rfl
  . rw [ha, hb]
    norm_num
    sorry

end right_triangle_hypotenuse_l383_383515


namespace part1_part2_l383_383677

-- Part 1
theorem part1 (a m : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = |x - a|):
  (∀ x, -1 ≤ x ∧ x ≤ 5 ↔ f x ≤ m) →
  a = 2 ∧ m = 3 := 
by
  sorry

-- Part 2
theorem part2 (t x : ℝ) 
  (h1 : a = 2) 
  (h2 : t ≥ 0) :
  (∀ x, |x - 2| + t ≥ |x + 2t - 2|) →
  (t = 0 → ∀ x, x ∈ ℝ ) ∧
  (t > 0 → ∀ x, x ≤ 2 - t / 2) := 
by
  sorry

end part1_part2_l383_383677


namespace longest_interval_green_l383_383112

-- Definitions for the conditions
def light_cycle_duration : ℕ := 180 -- total cycle duration in seconds
def green_duration : ℕ := 90 -- green light duration in seconds
def red_delay : ℕ := 10 -- red light delay between consecutive lights in seconds
def num_lights : ℕ := 8 -- number of lights

-- Theorem statement to be proved
theorem longest_interval_green (h1 : ∀ i : ℕ, i < num_lights → 
  ∃ t : ℕ, t < light_cycle_duration ∧ (∀ k : ℕ, i + k < num_lights → t + k * red_delay < light_cycle_duration ∧ t + k * red_delay + green_duration <= light_cycle_duration)):
  ∃ interval : ℕ, interval = 20 :=
sorry

end longest_interval_green_l383_383112


namespace total_cars_produced_l383_383940

def CarCompanyA_NorthAmerica := 3884
def CarCompanyA_Europe := 2871
def CarCompanyA_Asia := 1529

def CarCompanyB_NorthAmerica := 4357
def CarCompanyB_Europe := 3690
def CarCompanyB_Asia := 1835

def CarCompanyC_NorthAmerica := 2937
def CarCompanyC_Europe := 4210
def CarCompanyC_Asia := 977

def TotalNorthAmerica :=
  CarCompanyA_NorthAmerica + CarCompanyB_NorthAmerica + CarCompanyC_NorthAmerica

def TotalEurope :=
  CarCompanyA_Europe + CarCompanyB_Europe + CarCompanyC_Europe

def TotalAsia :=
  CarCompanyA_Asia + CarCompanyB_Asia + CarCompanyC_Asia

def TotalProduction := TotalNorthAmerica + TotalEurope + TotalAsia

theorem total_cars_produced : TotalProduction = 26290 := 
by sorry

end total_cars_produced_l383_383940


namespace max_min_theorem_root_values_sum_real_roots_l383_383360

namespace TrigFunctionProof

theorem max_min_theorem
  (A ω : ℝ) (φ : ℝ)
  (hA_pos : A > 0) (hω_pos : ω > 0) (hφ : |φ| < Real.pi / 2)
  (h_max : A * Real.sin(ω * (Real.pi / 4) + φ) = 2)
  (h_min : A * Real.sin(ω * (7 * Real.pi / 12) + φ) = -2) :
  A = 2 ∧ ω = 3 ∧ φ = -Real.pi / 4 :=
sorry

theorem root_values (x : ℝ) :
  (x ∈ Set.Icc 0 (2 * Real.pi) ∧ 2 * Real.sin(3 * x - Real.pi / 4) = Real.sqrt 3) ↔
  x ∈ {7 * Real.pi / 36, 11 * Real.pi / 36, 31 * Real.pi / 36, 35 * Real.pi / 36, 55 * Real.pi / 36, 59 * Real.pi / 36} :=
sorry

theorem sum_real_roots (a : ℝ) (ha : 1 < a ∧ a < 2) :
  ∑ x in {7 * Real.pi / 36, 11 * Real.pi / 36, 31 * Real.pi / 36, 35 * Real.pi / 36, 55 * Real.pi / 36, 59 * Real.pi / 36}, x = 11 * Real.pi / 2 :=
sorry

end TrigFunctionProof

end max_min_theorem_root_values_sum_real_roots_l383_383360


namespace high_school_competition_arrangements_l383_383439

theorem high_school_competition_arrangements :
  let students := [1, 2, 3, 4, 5]
  let subjects := ["Mathematics", "Physics", "Chemistry"]
  ∃ (arrangements : Nat), arrangements = 180 :=
by
  sorry

end high_school_competition_arrangements_l383_383439


namespace water_usage_l383_383747

theorem water_usage (payment : ℝ) (usage : ℝ) : 
  payment = 7.2 → (usage ≤ 6 → payment = usage * 0.8) → (usage > 6 → payment = 4.8 + (usage - 6) * 1.2) → usage = 8 :=
by
  sorry

end water_usage_l383_383747


namespace clock_hands_overlap_twice_in_24_hours_l383_383706

noncomputable def angular_velocity (hand : String) : ℕ :=
  if hand = "hour" then 1
  else if hand = "minute" then 12
  else if hand = "second" then 720
  else 0

def hand_position (hand : String) (t : ℝ) : ℝ :=
  (angular_velocity hand) * t % 360

def hands_overlap (t : ℝ) : Prop :=
  hand_position "hour" t = hand_position "minute" t ∧
  hand_position "minute" t = hand_position "second" t

theorem clock_hands_overlap_twice_in_24_hours :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧
  hands_overlap t1 ∧ hands_overlap t2 ∧
  ∀ t : ℝ, t ∈ [0, 24] → hands_overlap t → (t = t1 ∨ t = t2) :=
begin
  sorry
end

end clock_hands_overlap_twice_in_24_hours_l383_383706


namespace right_triangle_hypotenuse_l383_383535

theorem right_triangle_hypotenuse (a b : ℝ) (h : a^2 + b^2 = 39^2) : a = 15 ∧ b = 36 := by
  sorry

end right_triangle_hypotenuse_l383_383535


namespace son_age_l383_383465

theorem son_age (S M : ℕ) (h1 : M = S + 30) (h2 : M + 2 = 2 * (S + 2)) : S = 28 := 
by
  -- The proof can be filled in here.
  sorry

end son_age_l383_383465


namespace g_x_values_l383_383168

theorem g_x_values :
  ∃ x : ℝ, ∃ y : ℝ, 
  (y = arctan (2 * x) + arctan ((2 - 3 * x) / (1 + 3 * x))) ∧ 
  (y = π / 4 ∨ y = 5 * π / 4) := 
begin
  sorry
end

end g_x_values_l383_383168


namespace perimeter_of_triangle_l383_383841

theorem perimeter_of_triangle
  (P : ℝ)
  (r : ℝ := 1.5)
  (A : ℝ := 29.25)
  (h : A = r * (P / 2)) :
  P = 39 :=
by
  sorry

end perimeter_of_triangle_l383_383841


namespace expected_consecutive_different_digits_l383_383320

-- Define the problem statement
theorem expected_consecutive_different_digits :
  let n := 4095;
  let num_bits := 12;
  -- Define helper function to represent the probability (1/2 in this case)
  let prob_diff : ℕ → ℚ := λ i, if i < n then 1/2 else 0;
  -- Define R_i as a random variable which is 1 if the i-th and (i+1)-st digits differ, and 0 otherwise
  let R_i : ℕ → ℚ := λ i, prob_diff i;
  -- Compute the expected value for the sum of R_i from 1 to 11
  let expected_value := ∑ i in Finset.range (num_bits - 1), R_i i;
  expected_value = 20481 / 4096 :=
by
  sorry

end expected_consecutive_different_digits_l383_383320


namespace probability_of_convex_number_l383_383115

def is_convex (x y z : ℕ) : Prop :=
  y > x ∧ y > z

def convex_prob : Prop :=
  (∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x ∈ {1, 2, 3, 4} ∧ y ∈ {1, 2, 3, 4} ∧ z ∈ {1, 2, 3, 4} ∧ is_convex x y z) / 
  (∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x ∈ {1, 2, 3, 4} ∧ y ∈ {1, 2, 3, 4} ∧ z ∈ {1, 2, 3, 4}) = 1 / 3

theorem probability_of_convex_number : convex_prob :=
by
  sorry

end probability_of_convex_number_l383_383115


namespace probability_same_number_l383_383585

theorem probability_same_number :
  let n_billy := 500 / 30
  let n_bobbi := 500 / 45
  let n_common := 500 / Nat.lcm 30 45
  (n_common : ℚ) / (n_billy * n_bobbi) = 5 / 176 :=
by
  sorry

end probability_same_number_l383_383585


namespace distinct_right_angles_l383_383036

theorem distinct_right_angles (n : ℕ) (rectangles : Fin n → Set (ℝ × ℝ)) (h : ∀ i, is_rectangle (rectangles i)) :
  number_of_distinct_right_angles (set_of_right_angles rectangles) ≥ ⌊4 * sqrt n⌋ :=
sorry

end distinct_right_angles_l383_383036


namespace right_triangle_hypotenuse_l383_383533

theorem right_triangle_hypotenuse (a b : ℝ) (h : a^2 + b^2 = 39^2) : a = 15 ∧ b = 36 := by
  sorry

end right_triangle_hypotenuse_l383_383533


namespace find_m_l383_383659

open Set

theorem find_m (m : ℝ) (A B : Set ℝ) (hA : A = {1, m - 2}) (hB : B = {2, 3}) (hAB : A ∩ B = {2}) :
  m = 4 :=
by
  sorry

end find_m_l383_383659


namespace find_principal_amount_l383_383970

def compound_interest {P A : ℝ} (CI r t : ℝ) (n : ℕ) : Prop := 
  A = P * (1 + r / n) ^ (n * t) ∧ A = P + CI 

noncomputable def principal_amount (CI r t : ℝ) (n : ℕ) : ℝ := 
  CI / ((1 + r / n) ^ (n * t) - 1)

theorem find_principal_amount : 
  ∃ P : ℝ, compound_interest 181.78817648189806 0.04 1.5 2 → 
    P ≈ 2970.00 :=
by 
  sorry

end find_principal_amount_l383_383970


namespace hypotenuse_length_l383_383473

theorem hypotenuse_length (a b c : ℕ) (h₀ : a = 15) (h₁ : b = 36) (h₂ : a^2 + b^2 = c^2) : c = 39 :=
by
  -- Proof is omitted
  sorry

end hypotenuse_length_l383_383473


namespace pascal_triangle_45th_number_l383_383012

theorem pascal_triangle_45th_number :
  let row := List.range (46 + 1) in
  row.nth 44 = some 1035 :=
by
  let row := List.range (46 + 1)
  have binom_46_2 : nat.binom 46 2 = 1035 := by
    -- Calculations for binomials can be validated here
    calc
      nat.binom 46 2 = 46 * 45 / (2 * 1) : by norm_num
      _ = 1035 : by norm_num
  show row.nth 44 = some (nat.binom 46 2) from by
    rw binom_46_2
    simp only [List.nth_range, option.some_eq_coe, nat.lt_succ_iff, nat.le_refl]
  sorry -- Additional reasoning if necessary

end pascal_triangle_45th_number_l383_383012


namespace product_of_consecutive_even_numbers_l383_383850

theorem product_of_consecutive_even_numbers
  (a b c : ℤ)
  (h : a + b + c = 18 ∧ 2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c ∧ a < b ∧ b < c ∧ b - a = 2 ∧ c - b = 2) :
  a * b * c = 192 :=
sorry

end product_of_consecutive_even_numbers_l383_383850


namespace exponentiation_rule_l383_383137

variable {a b : ℝ}

theorem exponentiation_rule (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 := by
  sorry

end exponentiation_rule_l383_383137


namespace sum_abs_val_less_5_l383_383847

theorem sum_abs_val_less_5 : (∑ x in (Finset.filter (λ x : ℤ, |x| < 5) (Finset.range 9)), x) = 0 := 
sorry

end sum_abs_val_less_5_l383_383847


namespace number_divisible_by_seven_l383_383862

theorem number_divisible_by_seven (x : ℕ) (h : x ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  (666666666666666666666666666666666666666666666666?555555555555555555555555555555555555555555555555) % 7 = 0 ↔ x = 2 ∨ x = 9 := sorry

end number_divisible_by_seven_l383_383862


namespace john_pretzels_l383_383852

-- Definitions and assumptions based on the conditions
variables (pretzels_in_bowl : ℕ) (pretzels_john : ℕ) (pretzels_alan : ℕ) (pretzels_marcus : ℕ)

-- Conditions
def initial_conditions : Prop :=
  pretzels_in_bowl = 95 ∧
  pretzels_alan = pretzels_john - 9 ∧
  pretzels_marcus = pretzels_john + 12 ∧
  pretzels_marcus = 40

-- Theorem to prove: John ate 28 pretzels
theorem john_pretzels : initial_conditions → pretzels_john = 28 :=
  by
    intro h
    cases h,
    cases h_right,
    cases h_right_right,
    sorry

end john_pretzels_l383_383852


namespace total_unique_pickers_l383_383425

def people_picked_fruits : Set String :=
  { "grandpa", "dad", "granduncle", "aunt", "Xiao Yu", "uncle", "grandma", "mom" }

def passion_fruit_pickers : Set String :=
  { "grandpa", "dad", "granduncle", "aunt", "Xiao Yu", "uncle" }

def strawberry_pickers : Set String :=
  { "grandma", "mom", "grandpa", "Xiao Yu" }

theorem total_unique_pickers : (passion_fruit_pickers ∪ strawberry_pickers).card = 8 :=
by
  -- TODO: provide proof here
  sorry

end total_unique_pickers_l383_383425


namespace hypotenuse_length_l383_383496

theorem hypotenuse_length (a b : ℤ) (h₀ : a = 15) (h₁ : b = 36) : 
  ∃ c : ℤ, c^2 = a^2 + b^2 ∧ c = 39 := 
by {
  sorry
}

end hypotenuse_length_l383_383496


namespace gain_percent_l383_383887

-- Let C be the cost price of one chocolate
-- Let S be the selling price of one chocolate
-- Given: 35 * C = 21 * S
-- Prove: The gain percent is 66.67%

theorem gain_percent (C S : ℝ) (h : 35 * C = 21 * S) : (S - C) / C * 100 = 200 / 3 :=
by sorry

end gain_percent_l383_383887


namespace tom_reading_speed_l383_383963

noncomputable def normal_reading_speed := 12

theorem tom_reading_speed:
  ∃ P : ℕ, (∀ (factor total_pages : ℕ), factor = 3 → total_pages = 72 → 2 * factor * P = total_pages) → P = 12 :=
begin
  let P := normal_reading_speed,
  use P,
  intros factor total_pages h_factor h_total_pages,
  rw [h_factor, h_total_pages],
  linarith,
end

end tom_reading_speed_l383_383963


namespace right_triangle_hypotenuse_length_l383_383511

theorem right_triangle_hypotenuse_length (a b : ℕ) (h1 : a = 15) (h2 : b = 36) : 
  ∃ c : ℕ, c * c = a * a + b * b ∧ c = 39 := 
by
  have hyp_square := 225 + 1296 
  have h_calculation : 15 * 15 + 36 * 36 = 1521 := by
    calc
      15 * 15 = 225 : rfl
      36 * 36 = 1296 : rfl
      225 + 1296 = 1521 : rfl
  use 39
  split
  exact h_calculation
  rfl

end right_triangle_hypotenuse_length_l383_383511


namespace percentage_reduction_distance_l383_383765

theorem percentage_reduction_distance :
  let length := 3
  let width := 4
  let j := length + width
  let s := Math.sqrt (length^2 + width^2)
  let reduction := (j - s) / j * 100
  reduction ≈ 28.57 :=
by
  let length := 3
  let width := 4
  let j := length + width
  let s := Math.sqrt (length^2 + width^2)
  let reduction := (j - s) / j * 100
  sorry

end percentage_reduction_distance_l383_383765


namespace rank_from_right_l383_383113

theorem rank_from_right (rank_from_left total_students : ℕ) (h1 : rank_from_left = 5) (h2 : total_students = 10) :
  total_students - rank_from_left + 1 = 6 :=
by 
  -- Placeholder for the actual proof.
  sorry

end rank_from_right_l383_383113


namespace matrix_pow_2020_l383_383150

-- Define the matrix type and basic multiplication rule
def M : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 0], ![3, 1]]

theorem matrix_pow_2020 :
  M ^ 2020 = ![![1, 0], ![6060, 1]] := by
  sorry

end matrix_pow_2020_l383_383150


namespace g_monotonic_intervals_and_min_value_l383_383638

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := f x + (deriv f) x

theorem g_monotonic_intervals_and_min_value :
  (∀ x : ℝ, 0 < x → g'(x) = (x-1) / x^2) ∧
  (∀ x : ℝ, 0 < x < 1 → g'(x) < 0) ∧
  (∀ x : ℝ, 1 < x → g'(x) > 0) ∧
  (g 1 = 1) :=
by
  sorry

end g_monotonic_intervals_and_min_value_l383_383638


namespace maximize_sine_sum_of_equilateral_triangle_l383_383808

theorem maximize_sine_sum_of_equilateral_triangle
  (α β γ β' γ' : ℝ)
  (hβγ : β + γ = π - α)
  (hβ'γ' : β' + γ' = π - α)
  (hβγ_sum : sin β + sin γ > sin β' + sin γ')
  (hcommon_angle : α ≠ 0 ∧ α ≠ π) :
  sin α + sin β + sin γ = sin (π / 3) + sin (π / 3) + sin (π / 3) → 
  β = γ ∧ α = β ∧ α = γ :=
sorry

end maximize_sine_sum_of_equilateral_triangle_l383_383808


namespace determine_x_l383_383958

theorem determine_x
  (w : ℤ) (z : ℤ) (y : ℤ) (x : ℤ)
  (h₁ : w = 90)
  (h₂ : z = w + 25)
  (h₃ : y = z + 12)
  (h₄ : x = y + 7) : x = 134 :=
by
  sorry

end determine_x_l383_383958


namespace GCF_of_48_180_98_l383_383865

theorem GCF_of_48_180_98 : Nat.gcd (Nat.gcd 48 180) 98 = 2 :=
by
  sorry

end GCF_of_48_180_98_l383_383865


namespace hyperbola_eccentricity_proof_l383_383660

noncomputable def ellipse := {a : ℝ // a = 2}
noncomputable def foci_dist := {c : ℝ // c = sqrt 3}
noncomputable def hyperbola_eccentricity : ℝ :=
let f1f2_distance := 2 * sqrt 3,
    af1 := 2 - sqrt 2,
    af2 := 2 + sqrt 2,
    real_axis_length := 2 * sqrt 2,
    focus_distance := {c // c = sqrt 3} in
sqrt 3 / sqrt 2

theorem hyperbola_eccentricity_proof : 
  ∃ (e : ℝ), ellipse 2 ∧ foci_dist (sqrt 3) 
  ∧ (f1_distance := 2; 
      (∃ (af1 af2 : ℝ), af1 + af2 = 4 ∧ 
       af1^2 + af2^2 = 12 ∧ 
       abs (af2 - af1) = 2*sqrt 2 ∧ 
       f1f2_distance = 2*sqrt 3) ∧ 
      e = sqrt 3 / sqrt 2 ∧ 
      e = sqrt 6 / 2) :=
begin
  sorry
end

end hyperbola_eccentricity_proof_l383_383660


namespace hypotenuse_length_l383_383485

-- Definitions for the problem
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def leg1 : ℕ := 15
def leg2 : ℕ := 36
def hypotenuse : ℕ := 39

-- Lean 4 statement
theorem hypotenuse_length (a b c : ℕ) (h : is_right_triangle a b c) (ha : a = leg1) (hb : b = leg2) :
  c = hypotenuse :=
begin
  sorry
end

end hypotenuse_length_l383_383485


namespace hypotenuse_right_triangle_l383_383544

theorem hypotenuse_right_triangle (a b : ℕ) (h1 : a = 15) (h2 : b = 36) :
  ∃ c, c ^ 2 = a ^ 2 + b ^ 2 ∧ c = 39 :=
by
  sorry

end hypotenuse_right_triangle_l383_383544


namespace normal_prob_l383_383344

open Real
open ProbabilityTheory

noncomputable def normal_dist := Normal(1, σ^2)

theorem normal_prob :
  (P({x | 0 < x ∧ x < 1} : Set Real) = 0.3) :=
sorry

end normal_prob_l383_383344


namespace propA_necessary_not_sufficient_propB_l383_383795

variable (a : ℝ)

def proposition_A : Prop := ∀ x : ℝ, a * x ^ 2 + 2 * a * x + 1 > 0

def proposition_B : Prop := 0 < a ∧ a < 1

-- Proof problem statement
theorem propA_necessary_not_sufficient_propB : (proposition_B → proposition_A) ∧ ¬ (proposition_A → proposition_B) :=
by
  sorry

end propA_necessary_not_sufficient_propB_l383_383795


namespace part1_part2_l383_383739

-- Part I
theorem part1 {a b c : ℝ} {A B C : ℝ}
  (h1 : a * Real.sin B = b * Real.cos A)
  (h2 : a = c * Real.sin B)
  (h3 : b = c * Real.sin A)
  : A = 45 :=
sorry

-- Part II
def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin x * Real.cos x - (Real.cos x) ^ 2

theorem part2 {B : ℝ}
  (h1 : 0 < B) (h2 : B < 135)
  : -((1 + Real.sqrt 3) / 2) < f B ∧ f B <= 1 / 2 :=
sorry

end part1_part2_l383_383739


namespace marigolds_total_sale_l383_383317

theorem marigolds_total_sale (day1_sale day2_add day3_factor : ℕ)
  (h1 : day1_sale = 14)
  (h2 : day2_add = 25)
  (h3 : day3_factor = 2) :
  let day2_sale := day1_sale + day2_add in
  let day3_sale := day2_add * day3_factor in
  day1_sale + day2_sale + day3_sale = 89 :=
by
  sorry

end marigolds_total_sale_l383_383317


namespace amount_C_l383_383328

-- Define the variables and conditions.
variables (A B C : ℝ)
axiom h1 : A = (2 / 3) * B
axiom h2 : B = (1 / 4) * C
axiom h3 : A + B + C = 544

-- State the theorem.
theorem amount_C (A B C : ℝ) (h1 : A = (2 / 3) * B) (h2 : B = (1 / 4) * C) (h3 : A + B + C = 544) : C = 384 := 
sorry

end amount_C_l383_383328


namespace circle_equation_with_diameter_endpoints_l383_383832

theorem circle_equation_with_diameter_endpoints (A B : ℝ × ℝ) (x y : ℝ) :
  A = (1, 4) → B = (3, -2) → (x-2)^2 + (y-1)^2 = 10 :=
by
  sorry

end circle_equation_with_diameter_endpoints_l383_383832


namespace project_selection_probability_l383_383982

/-- Each employee can randomly select one project from four optional assessment projects. -/
def employees : ℕ := 4

def projects : ℕ := 4

def total_events (e : ℕ) (p : ℕ) : ℕ := p^e

def choose_exactly_one_project_not_selected_probability (e : ℕ) (p : ℕ) : ℚ :=
  (Nat.choose p 2 * Nat.factorial 3) / (p^e : ℚ)

theorem project_selection_probability :
  choose_exactly_one_project_not_selected_probability employees projects = 9 / 16 :=
by
  sorry

end project_selection_probability_l383_383982


namespace smallest_integer_inequality_l383_383408

theorem smallest_integer_inequality:
  ∃ x : ℤ, (2 * x < 3 * x - 10) ∧ ∀ y : ℤ, (2 * y < 3 * y - 10) → y ≥ 11 := by
  sorry

end smallest_integer_inequality_l383_383408


namespace point_P_in_quadrant_II_l383_383750

-- Definitions of the coordinates and the quadrants
def Point (x : Int) (y : Int) : Type := (x, y)

def Quadrant : Type
| I | II | III | IV deriving DecidableEq

def quadrant_of_point : Point → Quadrant
| (x, y) => 
  if x > 0 ∧ y > 0 then Quadrant.I
  else if x < 0 ∧ y > 0 then Quadrant.II
  else if x < 0 ∧ y < 0 then Quadrant.III
  else Quadrant.IV

-- Assertion that the point P(-3, 2) lies in the second quadrant
theorem point_P_in_quadrant_II : 
  quadrant_of_point (Point.mk (-3) 2) = Quadrant.II := 
sorry

end point_P_in_quadrant_II_l383_383750


namespace inequality_solution_set_l383_383955

noncomputable def solution_set := { x : ℝ | (x < -1 ∨ 1 < x) ∧ x ≠ 4 }

theorem inequality_solution_set : 
  { x : ℝ | (x^2 - 1) / (4 - x)^2 ≥ 0 } = solution_set :=
  by 
    sorry

end inequality_solution_set_l383_383955


namespace right_triangle_hypotenuse_length_l383_383559

theorem right_triangle_hypotenuse_length (a b : ℝ) (h_triangle : a = 15 ∧ b = 36) :
  ∃ (h : ℝ), h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  · exact rfl
  · rw [h_triangle.1, h_triangle.2]
    norm_num

end right_triangle_hypotenuse_length_l383_383559


namespace tony_solving_puzzles_time_l383_383384

theorem tony_solving_puzzles_time : ∀ (warm_up_time long_puzzle_ratio num_long_puzzles : ℕ),
  warm_up_time = 10 →
  long_puzzle_ratio = 3 →
  num_long_puzzles = 2 →
  (warm_up_time + long_puzzle_ratio * warm_up_time * num_long_puzzles) = 70 :=
by
  intros
  sorry

end tony_solving_puzzles_time_l383_383384


namespace prove_similar_triangles_l383_383761

open Classical

noncomputable def triangles_similar (A B C A' B' C' : Type*) [Metric A] [Metric B] [Metric C] [Metric A'] [Metric B'] [Metric C'] : Prop :=
  ∃ (α : A) (α' : A') (AB : B) (A'B' : B') (AC : C) (A'C' : C'), 
    ∠α = ∠α' ∧ (A'B' / AB = A'C' / AC) ∧ 
    (∠α = ∠α') ∧ ((A'B' / AB) = (A'C' / AC)) → 
    ∀ (P Q R : Triangle A) (P' Q' R' : Triangle A'), Triangle.similar P Q R P' Q' R'

axiom similarity_theorem: ∀ (A B C A' B' C' : Type*) [Metric A] [Metric B] [Metric C] [Metric A'] [Metric B'] [Metric C'], 
 ∀ (α : A) (α' : A') (AB : B) (A'B' : B') (AC : C) (A'C' : C'), 
    ∠α = ∠α' ∧ (A'B' / AB = A'C' / AC) →
    triangles_similar A B C A' B' C'
    
theorem prove_similar_triangles (A B C A' B' C' : Type*) [Metric A] [Metric B] [Metric C] [Metric A'] [Metric B'] [Metric C'] :
    ∀ (α : A) (α' : A') (AB : B) (A'B' : B') (AC : C) (A'C' : C'), 
    ∠α = ∠α' ∧ (A'B' / AB = A'C' / AC) → 
    Triangle.similar (Triangle.mk A B C) (Triangle.mk A' B' C') :=
by
  intro α α' AB A'B' AC A'C'
  intro h
  sorry

end prove_similar_triangles_l383_383761


namespace max_number_of_airlines_l383_383266

theorem max_number_of_airlines (n : ℕ) (k : ℕ) 
  (h1 : n = 50) 
  (h2 : ∀ (i j : ℕ) (hij : i ≠ j), ∃ (airline : ℕ), airline < k ∧ connected_by_airline i j airline)
  (h3 : ∀ (i j : ℕ), reachable i j) : k = 25 :=
  sorry

end max_number_of_airlines_l383_383266


namespace lemonade_glasses_l383_383148

def lemonade_glasses_proof : Prop :=
  let g := 5 in
  let p := 6 in
  g * p = 30

theorem lemonade_glasses : lemonade_glasses_proof :=
by
  sorry

end lemonade_glasses_l383_383148


namespace simplify_expression_l383_383331

theorem simplify_expression (x : ℝ) : (2 * x)^5 - (3 * x^2 * x^3) = 29 * x^5 := 
  sorry

end simplify_expression_l383_383331


namespace hypotenuse_length_l383_383474

theorem hypotenuse_length (a b c : ℕ) (h₀ : a = 15) (h₁ : b = 36) (h₂ : a^2 + b^2 = c^2) : c = 39 :=
by
  -- Proof is omitted
  sorry

end hypotenuse_length_l383_383474


namespace carter_ate_12_green_mms_l383_383593

theorem carter_ate_12_green_mms :
  (∃ G, 
  let green_before := 20,
      red_before := 20,
      yellow_added := 14,
      green_after := green_before - G,
      red_after := red_before / 2,
      total_after := green_after + red_after + yellow_added in
  0.25 = green_after / total_after) → 
  G = 12 :=
sorry

end carter_ate_12_green_mms_l383_383593


namespace part1_part2_l383_383051

-- Definitions for Part 1
def originalBillingStandard (t : ℝ) := 2 * t
def newBillingStandard (a : ℝ) (t : ℝ) : ℝ :=
  if t ≤ 10 then 1.8 * t else 1.8 * 10 + a * (t - 10)

-- Definitions for the problem conditions
def differenceInBilling (a : ℝ) := newBillingStandard a 20 - originalBillingStandard 20 = 8

-- Definitions for Part 2
def costForGivenUsage (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 10 then 1.8 * x else 1.8 * 10 + a * (x - 10)

-- Lean proof statements

-- Part 1
theorem part1 : ∃ a > 1.8, differenceInBilling a :=
by
  use 3
  split
  sorry -- Proof that 3 > 1.8
  sorry -- Proof that newBillingStandard 3 20 - originalBillingStandard 20 = 8

-- Part 2
theorem part2 : ∀ x : ℝ, x ≥ 0 → 
  (if x ≤ 10 then costForGivenUsage 3 x = 1.8 * x 
  else costForGivenUsage 3 x = 3 * x - 12) :=
by
  intro x hx
  split_ifs
  {
    sorry -- Proof for the case x ≤ 10
  }
  {
    sorry -- Proof for the case x > 10
  }

end part1_part2_l383_383051


namespace find_fourth_point_l383_383754

-- Define the equations and given points
def curve (x y : ℝ) : Prop := x * y = 2

def given_points := [(4, 1/2), (-2, -1), (2/3, 3)]

-- Define the unknown fourth point
def fourth_point (p : ℝ × ℝ) : Prop :=
  let x := p.fst
  let y := p.snd in
  curve x y ∧
  (4 * -2 * (2/3) * x = 4) ∧
  y = 2 / x

theorem find_fourth_point : ∃ p : ℝ × ℝ, fourth_point p :=
by
  use (-1/2, -4)
  unfold fourth_point curve
  simp
  split
  case left => exact (by norm_num : (-1/2) * (-4) = 2)
  norm_num

end find_fourth_point_l383_383754


namespace general_term_min_S9_and_S10_sum_b_seq_l383_383649

-- Definitions for the arithmetic sequence {a_n}
def a_seq (n : ℕ) : ℤ := 2 * ↑n - 20

-- Conditions provided in the problem
def cond1 : Prop := a_seq 4 = -12
def cond2 : Prop := a_seq 8 = -4

-- The sum of the first n terms S_n of the arithmetic sequence {a_n}
def S_n (n : ℕ) : ℤ := n * (a_seq 1 + a_seq n) / 2

-- Definitions for the new sequence {b_n}
def b_seq (n : ℕ) : ℤ := 2^n - 20

-- The sum of the first n terms of the new sequence {b_n}
def T_n (n : ℕ) : ℤ := (2^(n + 1) - 2) - 20 * n

-- Lean 4 theorem statements
theorem general_term (h1 : cond1) (h2 : cond2) : ∀ n : ℕ, a_seq n = 2 * ↑n - 20 :=
sorry

theorem min_S9_and_S10 (h1 : cond1) (h2 : cond2) : S_n 9 = -90 ∧ S_n 10 = -90 :=
sorry

theorem sum_b_seq (n : ℕ) : ∀ k : ℕ, (k < n) → T_n k = (2^(k+1) - 20 * k - 2) :=
sorry

end general_term_min_S9_and_S10_sum_b_seq_l383_383649


namespace exponential_monotonic_l383_383123

-- Define the function set we are considering
def functions : List (ℝ → ℝ) := [λ x, x^3, λ x, 3^x, λ x, Real.log x, λ x, Real.tan x]

-- Define the property f(x + y) = f(x) * f(y)
def is_exponential (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x + y) = f(x) * f(y)

-- Define the property of being monotonically increasing
def is_monotonic_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f(x) < f(y)

-- The statement we want to prove
theorem exponential_monotonic :
  ∃ f : ℝ → ℝ, f = (λ x, 3^x) ∧ is_exponential f ∧ is_monotonic_increasing f :=
by
  -- Placeholder for the proof
  sorry

end exponential_monotonic_l383_383123


namespace tony_total_puzzle_time_l383_383382

def warm_up_puzzle_time : ℕ := 10
def number_of_puzzles : ℕ := 2
def multiplier : ℕ := 3
def time_per_puzzle : ℕ := warm_up_puzzle_time * multiplier
def total_time : ℕ := warm_up_puzzle_time + number_of_puzzles * time_per_puzzle

theorem tony_total_puzzle_time : total_time = 70 := 
by
  sorry

end tony_total_puzzle_time_l383_383382


namespace impossible_task_l383_383745

-- Definitions based on problem conditions
def ellipse (F1 F2 : ℝ × ℝ) (a : ℝ) := 
  { P : ℝ × ℝ | dist P F1 + dist P F2 = 2 * a }

def on_edge (A : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (a : ℝ) : Prop :=
  A ∈ ellipse F1 F2 a

def on_segment (B F1 F2 : ℝ × ℝ) : Prop :=
  ∃ t ∈ Icc (0 : ℝ) 1, B = (1 - t) • F1 + t • F2

def no_cross (A : ℝ × ℝ) (F1 F2 B : ℝ × ℝ) : Prop := 
  -- A function to check if A does not cross segment s before bouncing
  sorry

-- Main statement
theorem impossible_task (A B F1 F2 : ℝ × ℝ) (a : ℝ) :
  on_edge A F1 F2 a →
  on_segment B F1 F2 →
  no_cross A F1 F2 B →
  ¬ ∃ P : ℝ × ℝ, reflection_point A P F1 F2 ∧ hits_target A P B :=
sorry

-- Additional function definition_placeholder for reflection point and hitting target property
def reflection_point (A P F1 F2 : ℝ × ℝ) : Prop := 
  -- check reflection point P based on A and foci F1, F2
  sorry

def hits_target (A P B : ℝ × ℝ) : Prop := 
  -- check if A hits B after reflecting at P
  sorry

end impossible_task_l383_383745


namespace sulfuric_acid_moles_used_l383_383976

-- Definitions and conditions
def iron_moles : ℕ := 2
def iron_ii_sulfate_moles_produced : ℕ := 2
def sulfuric_acid_to_iron_ratio : ℕ := 1

-- Proof statement
theorem sulfuric_acid_moles_used {H2SO4_moles : ℕ} 
  (h_fe_reacts : H2SO4_moles = iron_moles * sulfuric_acid_to_iron_ratio) 
  (h_fe produces: iron_ii_sulfate_moles_produced = iron_moles) : H2SO4_moles = 2 :=
by
  sorry

end sulfuric_acid_moles_used_l383_383976


namespace solve_for_y_l383_383336

theorem solve_for_y (y : ℕ) (h1 : 40 = 2^3 * 5) (h2 : 8 = 2^3) :
  40^3 = 8^y ↔ y = 3 :=
by sorry

end solve_for_y_l383_383336


namespace find_f2015_l383_383223

-- Define the function f and the conditions.
variable (f : ℝ → ℝ)
variable h₁ : ∀ x : ℝ, f(x + 3) = -1 / f(x)
variable h₂ : f(2) = 1 / 2

-- The theorem to prove that f(2015) = -2 given the conditions.
theorem find_f2015 : f 2015 = -2 := by
  sorry

end find_f2015_l383_383223


namespace triangle_area_120_l383_383571

theorem triangle_area_120 (a b c : ℕ) (h1 : a = 10) (h2 : b = 24) (h3 : c = 26) 
    (h4 : a^2 + b^2 = c^2) : 1/2 * a * b = 120 :=
by {
  rw [h1, h2],  -- Substitute a = 10, b = 24
  norm_num,     -- Evaluate the arithmetic
  sorry
}

end triangle_area_120_l383_383571


namespace min_val_n_sum_prime_l383_383016

theorem min_val_n_sum_prime (n : ℕ) (h : n ≤ 100) 
    (H : ∀ (s : Finset ℕ), s.card = n → ∀ (a b ∈ s), a ≠ b → a + b ∈ Finset.range 100 → Nat.Prime (a + b)) 
    (H₁ : n ≥ 51) : 
  ∃ (s : Finset ℕ), s.card = n ∧ ∀ (a b ∈ s), a ≠ b → Nat.Prime (a + b) :=
by sorry


end min_val_n_sum_prime_l383_383016


namespace exchange_ways_count_l383_383851

-- Conditions
variable (m : ℕ)
variable (p : ℕ → ℕ) -- p is a function that returns the value of the ith type of currency
variable (r : ℕ)

-- The definition of the generating function
def generating_function (t : ℕ) : ℕ :=
  ∏ i in finset.range m, (1 - t ^ (p i))⁻¹

-- The statement of the proof problem
theorem exchange_ways_count :
  ∃ a_r : ℕ, a_r = coefficient r (generating_function t) :=
sorry

end exchange_ways_count_l383_383851


namespace auction_theorem_l383_383699

def auctionProblem : Prop :=
  let starting_value := 300
  let harry_bid_round1 := starting_value + 200
  let alice_bid_round1 := harry_bid_round1 * 2
  let bob_bid_round1 := harry_bid_round1 * 3
  let highest_bid_round1 := bob_bid_round1
  let carol_bid_round2 := highest_bid_round1 * 1.5
  let sum_previous_increases := (harry_bid_round1 - starting_value) + 
                                 (alice_bid_round1 - harry_bid_round1) + 
                                 (bob_bid_round1 - harry_bid_round1)
  let dave_bid_round2 := carol_bid_round2 + sum_previous_increases
  let highest_other_bid_round3 := dave_bid_round2
  let harry_final_bid_round3 := 6000
  let difference := harry_final_bid_round3 - highest_other_bid_round3
  difference = 2050

theorem auction_theorem : auctionProblem :=
by
  sorry

end auction_theorem_l383_383699


namespace trick_proof_l383_383064

-- Defining the number of fillings and total pastries based on combinations
def num_fillings := 10

def total_pastries : ℕ := (num_fillings * (num_fillings - 1)) / 2

-- Definition stating that the smallest number of pastries n such that Vasya can always determine at least one filling of any remaining pastry
def min_n := 36

-- The theorem stating the proof problem
theorem trick_proof (n m: ℕ) (h1: n = 10) (h2: m = (n * (n - 1)) / 2) : min_n = 36 :=
by
  sorry

end trick_proof_l383_383064


namespace bd_le_q2_l383_383687

theorem bd_le_q2 (a b c d p q : ℝ) (h1 : a * b + c * d = 2 * p * q) (h2 : a * c ≥ p^2 ∧ p^2 > 0) : b * d ≤ q^2 :=
sorry

end bd_le_q2_l383_383687


namespace right_triangle_hypotenuse_length_l383_383506

theorem right_triangle_hypotenuse_length (a b : ℕ) (h1 : a = 15) (h2 : b = 36) : 
  ∃ c : ℕ, c * c = a * a + b * b ∧ c = 39 := 
by
  have hyp_square := 225 + 1296 
  have h_calculation : 15 * 15 + 36 * 36 = 1521 := by
    calc
      15 * 15 = 225 : rfl
      36 * 36 = 1296 : rfl
      225 + 1296 = 1521 : rfl
  use 39
  split
  exact h_calculation
  rfl

end right_triangle_hypotenuse_length_l383_383506


namespace pascal_triangle_45th_number_l383_383011

theorem pascal_triangle_45th_number (n k : ℕ) (h1 : n = 47) (h2 : k = 44) : 
  Nat.choose (n - 1) k = 1035 :=
by
  sorry

end pascal_triangle_45th_number_l383_383011


namespace gcd_digits_bounded_by_lcm_l383_383730

theorem gcd_digits_bounded_by_lcm (a b : ℕ) (h_a : 10^6 ≤ a ∧ a < 10^7) (h_b : 10^6 ≤ b ∧ b < 10^7) (h_lcm : 10^10 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^11) : Nat.gcd a b < 10^4 :=
by
  sorry

end gcd_digits_bounded_by_lcm_l383_383730


namespace odd_positive_93rd_l383_383005

theorem odd_positive_93rd : 
  (2 * 93 - 1) = 185 := 
by sorry

end odd_positive_93rd_l383_383005


namespace chip_permutations_l383_383045

-- Field identifiers and chip colors
inductive Field : Type
| One | Two | Three | Four | Five | Six 
| Seven | Eight | Nine | Ten | Eleven | Twelve

inductive Color : Type
| Red | Yellow | Green | Blue

-- Initial positions of the chips on fields
def initial_positions : Field -> Option Color
| Field.One   => some Color.Red
| Field.Two   => some Color.Yellow
| Field.Three => some Color.Green
| Field.Four  => some Color.Blue
| _           => none

-- Definition of valid moves: four positions away in either direction
def valid_move (f1 f2 : Field) : Prop :=
  let idx := match f1 with
             | Field.One   => 1 | Field.Two   => 2 | Field.Three => 3
             | Field.Four  => 4 | Field.Five  => 5 | Field.Six   => 6
             | Field.Seven => 7 | Field.Eight => 8 | Field.Nine  => 9
             | Field.Ten   => 10 | Field.Eleven => 11 | Field.Twelve => 12
  let offsets := [4, 8]  -- 4 positions clockwise or counterclockwise
  f2 = if idx + 4 > 12 then Field.mk (idx + 4 - 12) else Field.mk (idx + 4) ∨
  f2 = if idx - 4 <= 0 then Field.mk (idx - 4 + 12) else Field.mk (idx - 4)

-- Statement to verify the permutations of the chips
theorem chip_permutations :
  ∀ perms, (perms = [(Field.One, Field.Two, Field.Three, Field.Four),
                      (Field.Two, Field.Three, Field.Four, Field.One),
                      (Field.Three, Field.Four, Field.One, Field.Two),
                      (Field.Four, Field.One, Field.Two, Field.Three)])
             → true :=
by
  intros perms h
  sorry

end chip_permutations_l383_383045


namespace part1_part2a_part2b_l383_383658

def A (x : ℝ) : Prop := (x - 3) / (x + 1) > 0
def B (x : ℝ) : Prop := x ≤ 4
def U := set.univ

theorem part1 : A x ↔ (x > 3 ∨ x < -1) :=
by sorry

theorem part2a : A x ∧ B x ↔ (3 < x ∧ x ≤ 4) :=
by sorry

theorem part2b : (¬ A x ∨ B x) ↔ x ≤ 4 :=
by sorry

end part1_part2a_part2b_l383_383658


namespace C_rate_proof_l383_383882

def work_done (W : ℕ) := W

def A_rate (W : ℕ) := W / 3
def B_C_rate (W : ℕ) := W / 2
def A_B_rate (W : ℕ) := W / 2

theorem C_rate_proof (W : ℕ) :
  let B_rate := A_B_rate W - A_rate W in
  let C_rate := B_C_rate W - B_rate in
  C_rate = W / 3 :=
by
  sorry

end C_rate_proof_l383_383882


namespace max_value_y_over_x_l383_383722

theorem max_value_y_over_x
  (x y : ℝ) 
  (h : (x - 2) ^ 2 + y ^ 2 = 3) : 
  ∃ k : ℝ, k = sqrt 3 ∧ k = y / x :=
sorry

end max_value_y_over_x_l383_383722


namespace train_crossing_first_platform_l383_383570

theorem train_crossing_first_platform
  (L : ℕ) (P1 : ℕ) (P2 : ℕ) (t2 : ℕ) (v : ℕ) (t1 : ℕ)
  (hL : L = 150)
  (hP1 : P1 = 150)
  (hP2 : P2 = 250)
  (ht2 : t2 = 20)
  (hv : v = 20)
  (ht1 : t1 = 15) :
  t1 = (L + P1) / v :=
by
  rw [hL, hP1, hv, ht1]
  norm_num
  sorry

end train_crossing_first_platform_l383_383570


namespace hypotenuse_right_triangle_l383_383547

theorem hypotenuse_right_triangle (a b : ℕ) (h1 : a = 15) (h2 : b = 36) :
  ∃ c, c ^ 2 = a ^ 2 + b ^ 2 ∧ c = 39 :=
by
  sorry

end hypotenuse_right_triangle_l383_383547


namespace third_vertex_y_coordinate_l383_383925

-- Define the isosceles triangle conditions
def is_isosceles (A B C : Point) (a b c : ℝ) : Prop :=
  (a = b ∧ c ≠ b ∧ c ≠ a) ∨ (a = c ∧ b ≠ a ∧ b ≠ c) ∨ (b = c ∧ a ≠ b ∧ a != c)

-- Define points for triangle vertices
structure Point where
  x : ℝ
  y : ℝ

-- Vertices of the triangle
def A : Point := ⟨0, 5⟩
def B : Point := ⟨3, 5⟩
def C : Point := ⟨13, 5⟩

-- Given the two equal sides
def side_lengths : Point → Point → ℝ :=
  λ P Q, (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Define the property of equilateral triangle
def is_equilateral (A B C : Point) : Prop :=
  side_lengths A B = side_lengths B C ∧ side_lengths B C = side_lengths C A

-- Calculate the distance between two points
def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

-- Calculate the y-coordinate of the third vertex (A) for the given conditions
theorem third_vertex_y_coordinate :
  is_isosceles A B C ∧ (distance B C = 10) →
  (A.y + 5 * real.sqrt 3) = 5 + 5 * real.sqrt 3 :=
by
  sorry

end third_vertex_y_coordinate_l383_383925


namespace correct_statement_among_given_conditions_l383_383027

theorem correct_statement_among_given_conditions
  (A : ∀ (T1 T2 : Triangle), congruent T1 T2 → symmetrical_about_line T1 T2)
  (B : ∀ (T : IsoscelesTriangle), symmetrical_about_median T)
  (C : ∀ (T1 T2 : Triangle), symmetrical_about_line T1 T2 → congruent T1 T2)
  (D : ∀ (L : LineSegment), symmetrical_about_midpoint L) :
  C (T1 T2 : Triangle) :=
sorry

end correct_statement_among_given_conditions_l383_383027


namespace solution_set_of_abs_inequality_l383_383845

theorem solution_set_of_abs_inequality :
  { x : ℝ | |x^2 - 2| < 2 } = { x : ℝ | -2 < x ∧ x < 0 ∨ 0 < x ∧ x < 2 } :=
sorry

end solution_set_of_abs_inequality_l383_383845


namespace smallest_n_for_trick_l383_383087

theorem smallest_n_for_trick (fillings : Finset Fin 10)
  (pastries : Finset (Fin 45)) 
  (has_pairs : ∀ p ∈ pastries, ∃ f1 f2 ∈ fillings, f1 ≠ f2 ∧ p = pair f1 f2) : 
  ∃ n (tray : Finset (Fin 45)), 
    (tray.card = n ∧ n = 36 ∧ 
    ∀ remaining_p ∈ pastries \ tray, ∃ f ∈ fillings, f ∈ remaining_p) :=
by
  sorry

end smallest_n_for_trick_l383_383087


namespace minimum_pies_for_trick_l383_383094

-- Definitions from conditions
def num_fillings : ℕ := 10
def num_pastries := (num_fillings * (num_fillings - 1)) / 2
def min_pies_for_trick (n : ℕ) : Prop :=
  ∀ remaining_pies : ℕ, remaining_pies = num_pastries - n → remaining_pies ≤ 9

theorem minimum_pies_for_trick : ∃ n : ℕ, min_pies_for_trick n ∧ n = 36 :=
by
  -- We need to show that there exists n such that,
  -- min_pies_for_trick holds and n = 36
  existsi (36 : ℕ)
  -- remainder of the proof (step solution) skipped
  sorry

end minimum_pies_for_trick_l383_383094


namespace find_constant_b_l383_383943

theorem find_constant_b (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_cycles : ∃ k : ℕ, k = 5)
  (h_interval : ∃ I : ℝ, I = 6 * Real.pi)
  (h_range : ∃ l u : ℝ, l = -Real.pi ∧ u = 5 * Real.pi)
  (h_period : ∃ T : ℝ, T = 2 * Real.pi / b)
  (h_coverage : h_interval = (l, u) → (I / k) = T) :
  b = 5 / 3 :=
sorry

end find_constant_b_l383_383943


namespace probability_of_two_hits_out_of_three_is_0_25_l383_383215

-- Definition of the probability and the conditions
def probability_of_hit := 0.4
def probability_of_miss := 0.6 -- implicitly derived from definition but can help clarify
def groups := [
    [9,0,7], [9,6,6], [1,9,1], [9,2,5], [2,7,1], [9,3,2], [8,1,2], [4,5,8], [5,6,9], [6,8,3],
    [4,3,1], [2,5,7], [3,9,3], [0,2,7], [5,5,6], [4,8,8], [7,3,0], [1,1,3], [5,3,7], [9,8,9]
]

-- Function to determine a hit or miss based on the given rules
def is_hit (n : Nat) : Bool :=
  n > 0 ∧ n ≤ 4

-- Function to determine the number of hits in a group
def count_hits (group : List Nat) : Nat :=
  group.countp is_hit

-- Function to count the groups with exactly two hits
def count_groups_with_exactly_two_hits (groups : List (List Nat)) : Nat :=
  groups.countp (λ g => count_hits g = 2)

-- Define the expected probability
def expected_probability : Float :=
  (count_groups_with_exactly_two_hits groups).to_float / groups.length.to_float

-- Prove that the estimated probability matches the correct answer
theorem probability_of_two_hits_out_of_three_is_0_25 :
  expected_probability = 0.25 :=
by
  -- skip proof
  sorry

end probability_of_two_hits_out_of_three_is_0_25_l383_383215


namespace find_f_inv_of_3_l383_383797

noncomputable def f (x : ℝ) := x^2 + 2

noncomputable def f_inv (y : ℝ) := -Real.sqrt (y - 2)

theorem find_f_inv_of_3 : f_inv (f (-1)) = -1 :=
by
  have h1 : ∀ x, f x = x^2 + 2 := by intro x; rfl
  have h2 : ∀ y, f_inv y = -Real.sqrt (y - 2) := by intro y; rfl
  rw [h1 (-1), h2 3]
  sorry

end find_f_inv_of_3_l383_383797


namespace fraction_product_is_729_l383_383415

-- Define the sequence of fractions
noncomputable def fraction_sequence : List ℚ :=
  [1/3, 9, 1/27, 81, 1/243, 729, 1/729, 6561, 1/2187, 59049, 1/6561, 59049/81]

-- Define the condition that each supposed pair product is 3
def pair_product_condition (seq : List ℚ) (i : ℕ) : Prop :=
  seq[2 * i] * seq[2 * i + 1] = 3

-- State the main theorem
theorem fraction_product_is_729 :
  fraction_sequence.prod = 729 := by
  sorry

end fraction_product_is_729_l383_383415


namespace elmer_car_savings_l383_383965

theorem elmer_car_savings (x c : ℝ) :
  let old_efficiency := x,
      new_efficiency := (8 / 5) * x,
      old_cost_per_km := c / old_efficiency,
      new_cost_per_km := (1.25 * c) / new_efficiency,
      cost_difference := old_cost_per_km - new_cost_per_km,
      percent_savings := (cost_difference / old_cost_per_km) * 100
  in percent_savings = 21.875 := by
  sorry

end elmer_car_savings_l383_383965


namespace find_three_digit_number_l383_383170

-- Definitions of digit constraints and the number representation
def is_three_digit_number (N : ℕ) (a b c : ℕ) : Prop :=
  N = 100 * a + 10 * b + c ∧ 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9

-- Definition of the problem condition
def sum_of_digits_condition (N : ℕ) (a b c : ℕ) : Prop :=
  a + b + c = N / 11

-- Lean theorem statement
theorem find_three_digit_number (N a b c : ℕ) :
  is_three_digit_number N a b c ∧ sum_of_digits_condition N a b c → N = 198 :=
by
  sorry

end find_three_digit_number_l383_383170


namespace right_triangle_hypotenuse_length_l383_383527

theorem right_triangle_hypotenuse_length :
  ∀ (a b h : ℕ), a = 15 → b = 36 → h^2 = a^2 + b^2 → h = 39 :=
by
  intros a b h ha hb hyp
  -- In the proof, we would use ha, hb, and hyp to show h = 39
  sorry

end right_triangle_hypotenuse_length_l383_383527


namespace interest_rate_proof_l383_383721

noncomputable def compound_interest_rate (P A : ℝ) (n : ℕ) (r : ℝ) : Prop :=
  A = P * (1 + r)^n

noncomputable def interest_rate (initial  final: ℝ) (years : ℕ) : ℝ := 
  (4: ℝ)^(1/(years: ℝ)) - 1

theorem interest_rate_proof :
  compound_interest_rate 8000 32000 36 (interest_rate 8000 32000 36) ∧
  abs (interest_rate 8000 32000 36 * 100 - 3.63) < 0.01 :=
by
  -- Conditions from the problem for compound interest
  -- Using the formula for interest rate and the condition checks
  sorry

end interest_rate_proof_l383_383721


namespace original_number_divisibility_l383_383023

theorem original_number_divisibility (N : ℤ) : (∃ k : ℤ, N = 9 * k + 3) ↔ (∃ m : ℤ, (N + 3) = 9 * m) := sorry

end original_number_divisibility_l383_383023


namespace max_distance_min_distance_l383_383854

-- Definitions based on conditions
def bus_stops : Fin 10 → ℝ := fun n => n
def distance_between_adjacent_stops (a : ℝ) : ℝ := a

-- Problem statement
theorem max_distance (a : ℝ) : 
  let S : ℝ := |bus_stops 0 - bus_stops 9 * a| + 
               |bus_stops 9 * a - bus_stops 1 * a| + 
               |bus_stops 1 * a - bus_stops 8 * a| + 
               |bus_stops 8 * a - bus_stops 2 * a| + 
               |bus_stops 2 * a - bus_stops 7 * a| + 
               |bus_stops 7 * a - bus_stops 3 * a| + 
               |bus_stops 3 * a - bus_stops 6 * a| + 
               |bus_stops 6 * a - bus_stops 4 * a| + 
               |bus_stops 4 * a - bus_stops 5 * a| + 
               |bus_stops 5 * a - bus_stops 0|
  in
  S = 50 * a := sorry

theorem min_distance (a : ℝ) : 
  let S : ℝ := |bus_stops 0 - bus_stops 1 * a| + 
               |bus_stops 1 * a - bus_stops 2 * a| + 
               |bus_stops 2 * a - bus_stops 3 * a| + 
               |bus_stops 3 * a - bus_stops 4 * a| + 
               |bus_stops 4 * a - bus_stops 5 * a| + 
               |bus_stops 5 * a - bus_stops 6 * a| + 
               |bus_stops 6 * a - bus_stops 7 * a| + 
               |bus_stops 7 * a - bus_stops 8 * a| + 
               |bus_stops 8 * a - bus_stops 9 * a| + 
               |bus_stops 9 * a - bus_stops 0|
  in
  S = 18 * a := sorry

end max_distance_min_distance_l383_383854


namespace smallest_n_for_trick_l383_383089

theorem smallest_n_for_trick (fillings : Finset Fin 10)
  (pastries : Finset (Fin 45)) 
  (has_pairs : ∀ p ∈ pastries, ∃ f1 f2 ∈ fillings, f1 ≠ f2 ∧ p = pair f1 f2) : 
  ∃ n (tray : Finset (Fin 45)), 
    (tray.card = n ∧ n = 36 ∧ 
    ∀ remaining_p ∈ pastries \ tray, ∃ f ∈ fillings, f ∈ remaining_p) :=
by
  sorry

end smallest_n_for_trick_l383_383089


namespace pumps_work_hours_l383_383441

theorem pumps_work_hours (d : ℕ) (h_d_pos : d > 0) : 6 * (8 / d) * d = 48 :=
by
  -- The proof is omitted
  sorry

end pumps_work_hours_l383_383441


namespace pastry_trick_l383_383079

theorem pastry_trick (fillings : Fin 10) (n : ℕ) :
  ∃ n, (n = 36 ∧ ∀ remaining_pastries, 
    (remaining_pastries.length = 45 - n) → 
    (∃ remaining_filling ∈ fillings, true)) := 
sorry

end pastry_trick_l383_383079


namespace least_number_to_subtract_l383_383875

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (r : ℕ) (h : n = 427398) (k : d = 13) (r_val : r = 2) : 
  ∃ x : ℕ, (n - x) % d = 0 ∧ r = x :=
by sorry

end least_number_to_subtract_l383_383875


namespace number_of_red_balls_l383_383746

theorem number_of_red_balls (total_balls : ℕ) (prob_red : ℚ) (h : total_balls = 20 ∧ prob_red = 0.25) : ∃ x : ℕ, x = 5 :=
by
  sorry

end number_of_red_balls_l383_383746


namespace positive_integer_fraction_condition_l383_383188

theorem positive_integer_fraction_condition :
  (∀ n : ℕ, (0 < n ∧ n < 25) → (50 - 2 * n) ∣ n) →
  (2 = finset.card { n : ℕ | 0 < n ∧ n < 25 ∧ (50 - 2 * n) ∣ n }) :=
by
  sorry

end positive_integer_fraction_condition_l383_383188


namespace sqrt_inequality_l383_383379

theorem sqrt_inequality :
  (sqrt 2 + sqrt 7) ^ 2 < (sqrt 3 + sqrt 6) ^ 2 → sqrt 2 - sqrt 3 < sqrt 6 - sqrt 7 :=
sorry

end sqrt_inequality_l383_383379


namespace grassy_width_excluding_path_l383_383468

theorem grassy_width_excluding_path
  (l : ℝ) (w : ℝ) (p : ℝ)
  (h1: l = 110) (h2: w = 65) (h3: p = 2.5) :
  w - 2 * p = 60 :=
by
  sorry

end grassy_width_excluding_path_l383_383468


namespace right_triangle_hypotenuse_length_l383_383561

theorem right_triangle_hypotenuse_length (a b : ℝ) (h_triangle : a = 15 ∧ b = 36) :
  ∃ (h : ℝ), h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  · exact rfl
  · rw [h_triangle.1, h_triangle.2]
    norm_num

end right_triangle_hypotenuse_length_l383_383561


namespace right_triangle_hypotenuse_l383_383519

theorem right_triangle_hypotenuse (a b : ℕ) (ha : a = 15) (hb : b = 36) : 
  ∃ h : ℕ, h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  . exact rfl
  . rw [ha, hb]
    norm_num
    sorry

end right_triangle_hypotenuse_l383_383519


namespace sum_of_fractions_to_decimal_l383_383142

theorem sum_of_fractions_to_decimal :
  (\frac{3}{50} + \frac{5}{500} + \frac{7}{5000}) = 0.0714 := sorry

end sum_of_fractions_to_decimal_l383_383142


namespace pq_square_sum_l383_383784

theorem pq_square_sum (p q : ℝ) (h1 : p * q = 9) (h2 : p + q = 6) : p^2 + q^2 = 18 := 
by
  sorry

end pq_square_sum_l383_383784


namespace day_100_days_from_friday_l383_383395

-- Define the days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Define a function to get the day of the week after a given number of days
def dayOfWeekAfter (start : Day) (n : ℕ) : Day :=
  match start with
  | Sunday    => match n % 7 with
                  | 0 => Sunday
                  | 1 => Monday
                  | 2 => Tuesday
                  | 3 => Wednesday
                  | 4 => Thursday
                  | 5 => Friday
                  | 6 => Saturday
                  | _ => start
  | Monday    => match n % 7 with
                  | 0 => Monday
                  | 1 => Tuesday
                  | 2 => Wednesday
                  | 3 => Thursday
                  | 4 => Friday
                  | 5 => Saturday
                  | 6 => Sunday
                  | _ => start
  | Tuesday   => match n % 7 with
                  | 0 => Tuesday
                  | 1 => Wednesday
                  | 2 => Thursday
                  | 3 => Friday
                  | 4 => Saturday
                  | 5 => Sunday
                  | 6 => Monday
                  | _ => start
  | Wednesday => match n % 7 with
                  | 0 => Wednesday
                  | 1 => Thursday
                  | 2 => Friday
                  | 3 => Saturday
                  | 4 => Sunday
                  | 5 => Monday
                  | 6 => Tuesday
                  | _ => start
  | Thursday  => match n % 7 with
                  | 0 => Thursday
                  | 1 => Friday
                  | 2 => Saturday
                  | 3 => Sunday
                  | 4 => Monday
                  | 5 => Tuesday
                  | 6 => Wednesday
                  | _ => start
  | Friday    => match n % 7 with
                  | 0 => Friday
                  | 1 => Saturday
                  | 2 => Sunday
                  | 3 => Monday
                  | 4 => Tuesday
                  | 5 => Wednesday
                  | 6 => Thursday
                  | _ => start
  | Saturday  => match n % 7 with
                  | 0 => Saturday
                  | 1 => Sunday
                  | 2 => Monday
                  | 3 => Tuesday
                  | 4 => Wednesday
                  | 5 => Thursday
                  | 6 => Friday
                  | _ => start

-- The proof problem as a Lean theorem
theorem day_100_days_from_friday : dayOfWeekAfter Friday 100 = Sunday := by
  -- Proof will go here
  sorry

end day_100_days_from_friday_l383_383395


namespace twentieth_prime_number_l383_383664

theorem twentieth_prime_number (h : ∃ n, prime n ∧ (List.drop 10 (List.filter prime (List.range 100))) = 31) : 
  List.nth (List.filter prime (List.range 100)) 19 = some 71 :=
sorry

end twentieth_prime_number_l383_383664


namespace hypotenuse_right_triangle_l383_383551

theorem hypotenuse_right_triangle (a b : ℕ) (h1 : a = 15) (h2 : b = 36) :
  ∃ c, c ^ 2 = a ^ 2 + b ^ 2 ∧ c = 39 :=
by
  sorry

end hypotenuse_right_triangle_l383_383551


namespace count_congruent_to_4_mod_7_l383_383239

theorem count_congruent_to_4_mod_7 (n : ℕ) (hn : n < 1500) (hc : n % 7 = 4) : 
  214 :=
by
  sorry

end count_congruent_to_4_mod_7_l383_383239


namespace center_of_mass_distance_l383_383372

-- Definitions for the problem
def disk_sequence_radius (i : ℕ) : ℝ := 2 * (1/2)^(i - 1)
def disk_sequence_mass (i : ℕ) : ℝ := (disk_sequence_radius i)^2
def total_mass : ℝ := ∑' i, disk_sequence_mass i

-- Center of mass calculation
def center_of_mass_y : ℝ :=
  (∑' i, disk_sequence_mass i * disk_sequence_radius i) / total_mass

-- The main theorem stating the distance from the center of the largest disk
theorem center_of_mass_distance : center_of_mass_y = 6 / 7 :=
by
  sorry

end center_of_mass_distance_l383_383372


namespace palindrome_contains_7_percentage_l383_383912

theorem palindrome_contains_7_percentage (a b : ℕ) :
  (1 ≤ a ∧ a ≤ 9) → (0 ≤ b ∧ b ≤ 9) →
  let total_palindromes := 100 in
  let palindromes_with_7 := (if a = 7 then 10 else 0) + (if b = 7 then 9 else 0) in
  (total_palindromes ≠ 0) →
  (palindromes_with_7 / total_palindromes.to_rat) * 100 = 19 := by
  sorry

end palindrome_contains_7_percentage_l383_383912


namespace largest_circle_cut_30_over_7_l383_383191

noncomputable def largest_circle_diameter (D1 D2 D3 : ℝ) (h1 : D1 = 30) (h2 : D2 = 20) (h3 : D3 = 10) : ℝ :=
  let A_initial := real.pi * (D1 / 2)^2
  let A_smaller1 := real.pi * (D2 / 2)^2
  let A_smaller2 := real.pi * (D3 / 2)^2
  let A_remaining := A_initial - A_smaller1 - A_smaller2
  let R := real.sqrt (A_remaining / real.pi)
  2 * R

theorem largest_circle_cut_30_over_7 :
  largest_circle_diameter 30 20 10 = 30/7 :=
sorry

end largest_circle_cut_30_over_7_l383_383191


namespace cos_sum_identity_l383_383221

theorem cos_sum_identity (α β : ℝ) 
  (h : (cos α * cos (β / 2)) / cos (α - β / 2) + (cos β * cos (α / 2)) / cos (β - α / 2) - 1 = 0) 
  : cos α + cos β = 1 :=
sorry

end cos_sum_identity_l383_383221


namespace correct_number_of_elements_l383_383350

theorem correct_number_of_elements 
  (n : ℕ) 
  (incorrect_avg : ℕ := 46) 
  (correct_incorrect_value : ℕ := 40)
  (correct_avg : ℕ := 50)
  (h1 : (incorrect_avg * n + correct_incorrect_value) / n = correct_avg) : 
  n = 10 := 
begin
  linarith,
end

end correct_number_of_elements_l383_383350


namespace faith_change_l383_383165

def flour_cost : ℕ := 5
def cake_stand_cost : ℕ := 28
def given_amount : ℕ := 20 * 2 + 3
def total_cost := flour_cost + cake_stand_cost
def change := given_amount - total_cost

theorem faith_change : change = 10 :=
by
  -- the proof goes here
  sorry

end faith_change_l383_383165


namespace sign_pyramid_valid_combinations_l383_383153

-- Define the structure of the sign pyramid and helper functions
def same_sign (x y : Int) : Bool := (x = y)
def sign_of (x y : Int) : Int :=
  if same_sign x y then 1 else -1

-- A function to propagate signs from the bottom row to the top
def pyramid_sign (a b c d e : Int) : Int :=
  let ab := sign_of a b
  let bc := sign_of b c
  let cd := sign_of c d
  let de := sign_of d e
  let second_row_left := sign_of ab bc
  let second_row_right := sign_of bc cd
  let top_sign := sign_of second_row_left second_row_right
  top_sign

-- Define the main theorem problem:
theorem sign_pyramid_valid_combinations : ∃(n : Nat), n = 17 ∧ (n = (List.filter 
  (λ (l : List Int), 
    l.length = 5 ∧ 
    (pyramid_sign l.head (l.get 1) (l.get 2) (l.get 3) l.get 4) = 1) 
  (List.replicateM 5 [1, -1])).length) :=
by
  sorry  -- Proof to be filled in

end sign_pyramid_valid_combinations_l383_383153


namespace part1_part2_part2_corr_part3_l383_383575

theorem part1 (a b c : ℝ) (h1 : a + b + c = 0.18) (h2 : 2 * b = a + c) :
  b = 0.06 :=
sorry

theorem part2 (x : Fin 6 → ℝ) (y : Fin 6 → ℝ) (hx : ∑ i, x i * y i = 0.146)
  (hx2 : (∑ i, x i ^ 2 - 6 * (∑ i, x i / 6) ^ 2) = 0.044^2)
  (hy2 : (∑ i, y i ^ 2 - 6 * (∑ i, y i / 6) ^ 2) = 0.303^2) :
  ∑ i, x i = 0.348 :=
sorry

theorem part2_corr (x y : Fin 6 → ℝ) (hx : ∑ i, x i * y i = 0.146) 
  (x_avg : ℝ) (y_avg : ℝ) 
  (hx2 : (∑ i, x i ^ 2 - 6 * x_avg ^ 2) ≈ 0.044^2)
  (hy2 : (∑ i, y i ^ 2 - 6 * y_avg ^ 2) ≈ 0.303^2) 
  (x_avg := 0.058) (y_avg := 0.382) :
  let r := (hx - 6 * x_avg * y_avg) / (0.044 * 0.303) in r > 0.75 :=
sorry

theorem part3 (a c : ℝ) (ha : 6.75 * a - 0.0095 = 0.35)
  (hc : 6.75 * c - 0.0095 = 0.43) :
  a ≈ 0.05 ∧ c ≈ 0.07 :=
sorry

end part1_part2_part2_corr_part3_l383_383575


namespace area_triangle_ANM_of_isosceles_trapezoid_l383_383757

theorem area_triangle_ANM_of_isosceles_trapezoid
  (ABCD : Trapezoid ABCD)
  (isosceles : is_isosceles_trapezoid ABCD)
  (base_angles_eq_30 : base_angles_eq ABCD (30 : ℝ))
  (diag_bisects : bisects_diagonal AC ∠BAD)
  (bisect_bisects_base_at_point_M : bisects_diagonal_at_point (angle_bisector_of ∠BCD) AD M)
  (BM_intersects_AC_at_N : intersects_at BM AC N)
  (trapezoid_area : area ABCD = 2 + sqrt 3) :
  area (triangle ANM) = 3 * (sqrt 3 - 1) / 2 := 
sorry

end area_triangle_ANM_of_isosceles_trapezoid_l383_383757


namespace pascal_triangle_45th_number_l383_383007

theorem pascal_triangle_45th_number : nat.choose 46 44 = 1035 := 
by sorry

end pascal_triangle_45th_number_l383_383007


namespace sin_condition_necessary_but_not_sufficient_l383_383040

noncomputable def condition_sin (α β : ℝ) : Prop :=
  sin (α + β) = 0

noncomputable def condition_sum (α β : ℝ) : Prop :=
  α + β = 0

theorem sin_condition_necessary_but_not_sufficient (α β : ℝ) :
  (condition_sin α β → ∃ k : ℤ, α + β = k * π) ∧ (condition_sum α β → condition_sin α β) → 
  (¬(condition_sin α β → condition_sum α β)) ∧ (condition_sum α β → condition_sin α β) :=
by
  sorry

end sin_condition_necessary_but_not_sufficient_l383_383040


namespace average_gas_mileage_round_trip_l383_383055

def distance_to_home : ℝ := 150
def mileage_to_home : ℝ := 35
def distance_return : ℝ := 130
def mileage_return : ℝ := 25
def total_distance : ℝ := distance_to_home + distance_return
def gas_usage_to_home : ℝ := distance_to_home / mileage_to_home
def gas_usage_return : ℝ := distance_return / mileage_return
def total_gas_used : ℝ := gas_usage_to_home + gas_usage_return
def average_mileage : ℝ := total_distance / total_gas_used

theorem average_gas_mileage_round_trip : average_mileage ≈ 30 := sorry

end average_gas_mileage_round_trip_l383_383055


namespace total_uneaten_pizza_pieces_l383_383583

-- Define the initial number of pizza pieces.
def total_pieces : ℕ := 6

-- Define the portions eaten by each individual.
def ann_consumed_pct : ℝ := 0.4
def bill_consumed_pct : ℝ := 0.4
def cate_consumed_pct : ℝ := 0.7
def dale_consumed_pct : ℝ := 0.7
def eddie_consumed_pct : ℝ := 0.6

-- Calculate the number of pieces eaten by each individual.
def ann_eaten : ℝ := total_pieces * ann_consumed_pct
def bill_eaten : ℝ := total_pieces * bill_consumed_pct
def cate_eaten : ℝ := total_pieces * cate_consumed_pct
def dale_eaten : ℝ := total_pieces * dale_consumed_pct
def eddie_eaten : ℝ := total_pieces * eddie_consumed_pct

-- Calculate the number of pieces left uneaten by each individual.
def ann_uneaten : ℝ := total_pieces - ann_eaten
def bill_uneaten : ℝ := total_pieces - bill_eaten
def cate_uneaten : ℝ := total_pieces - cate_eaten
def dale_uneaten : ℝ := total_pieces - dale_eaten
def eddie_uneaten : ℝ := total_pieces - eddie_eaten

-- Calculate the total number of uneaten pieces.
def total_uneaten : ℝ := ann_uneaten + bill_uneaten + cate_uneaten + dale_uneaten + eddie_uneaten

-- Round down to the nearest whole number.
noncomputable def total_uneaten_int : ℕ := total_uneaten.to_int

-- The theorem statement to be proven.
theorem total_uneaten_pizza_pieces : total_uneaten_int = 13 := by
  sorry

end total_uneaten_pizza_pieces_l383_383583


namespace combined_weight_l383_383306

-- We define the variables and the conditions
variables (x y : ℝ)

-- First condition 
def condition1 : Prop := y = (16 - 4) + (30 - 6) + (x - 3)

-- Second condition
def condition2 : Prop := y = 12 + 24 + (x - 3)

-- The statement to prove
theorem combined_weight (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) : y = x + 33 :=
by
  -- Skipping the proof part
  sorry

end combined_weight_l383_383306


namespace fraction_of_juniors_studying_japanese_l383_383131

-- Definitions based on the problem conditions
def seniorClassSize (J : ℕ) : ℕ := 3 * J
def fractionSeniorsStudyingJapanese : ℚ := 1 / 3
def fractionTotalStudyingJapanese : ℚ := 0.4375

-- The problem statement in Lean 4
theorem fraction_of_juniors_studying_japanese (J : ℕ) (x : ℚ) :
  let S := seniorClassSize J in
  ((fractionSeniorsStudyingJapanese * S) + (x * J)) / (S + J) = fractionTotalStudyingJapanese →
  x = 3 / 4 :=
by
  -- Here we state that the proof goes here, but say sorry for now to put a placeholder
  sorry

end fraction_of_juniors_studying_japanese_l383_383131


namespace a_5_eq_31_l383_383759

def seq (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ (∀ n, a (n + 1) = 2 * a n + 1)

theorem a_5_eq_31 (a : ℕ → ℕ) (h : seq a) : a 5 = 31 :=
by
  sorry
 
end a_5_eq_31_l383_383759


namespace sum_of_interior_angles_l383_383106

theorem sum_of_interior_angles (n : ℕ) (interior_angle : ℝ) :
  (interior_angle = 144) → (180 - 144) * n = 360 → n = 10 → (n - 2) * 180 = 1440 :=
by
  intros h1 h2 h3
  sorry

end sum_of_interior_angles_l383_383106


namespace volume_ratio_of_divided_tetrahedron_l383_383378

theorem volume_ratio_of_divided_tetrahedron 
  (P A B C : Point)
  (G : Point)
  (hG_base: is_centroid_of_tetrahedron_base P A B C G)
  (planes_parallel_to_base: (Plane G).parallel_to (Plane P A B C)) :
  ∃v₁ v₂ v₃ v₄ v₅ v₆ v₇, 
  (1:1):(1:1):(1:1):(6:6):(6:6):(6:6):(6:6) := 
  sorry

end volume_ratio_of_divided_tetrahedron_l383_383378


namespace solution_set_of_inequality_l383_383643

theorem solution_set_of_inequality {f : ℝ → ℝ}
  (h1 : ∀ x, f x ∈ ℝ)
  (h2 : ∀ x1 x2 : ℝ, x1 < x2 → x1 <= 1 → x2 <= 1 → f x1 < f x2)
  (h3 : ∀ x, f (x + 1) = f (-x + 1))
  (h4 : f 3 = 0) :
  {x | f x > 0} = set.Ioo (-1) 3 :=
by
  sorry

end solution_set_of_inequality_l383_383643


namespace u_n_greater_than_1_l383_383290

noncomputable def u_sequence : ℕ → ℝ → ℝ
| 0, u := 1 + u
| (n + 1), u := 1 / (u_sequence n u) + u

theorem u_n_greater_than_1 (u : ℝ) (n : ℕ) (h : 0 < u ∧ u < 1) : 1 < u_sequence n u :=
by {
  sorry
}

end u_n_greater_than_1_l383_383290


namespace pastry_problem_minimum_n_l383_383068

theorem pastry_problem_minimum_n (fillings : Finset ℕ) (n : ℕ) : 
    fillings.card = 10 →
    (∃ pairs : Finset (ℕ × ℕ), pairs.card = 45 ∧ ∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ≠ p.2 ∧ p.1 ∈ fillings ∧ p.2 ∈ fillings) →
    (∀ (remaining_pies : Finset (ℕ × ℕ)), remaining_pies.card = 45 - n → 
     ∃ f1 f2, (f1, f2) ∈ remaining_pies → (f1 ∈ fillings ∧ f2 ∈ fillings)) →
    n = 36 :=
by
  intros h_fillings h_pairs h_remaining_pies
  sorry

end pastry_problem_minimum_n_l383_383068


namespace problem_value_l383_383419

theorem problem_value:
  3^(1+3+4) - (3^1 + 3^3 + 3^4) = 6450 :=
by sorry

end problem_value_l383_383419


namespace simplest_quadratic_surd_l383_383423

-- Definition of the expressions
def expr_A := Real.sqrt 9
def expr_B := Real.sqrt 20
def expr_C := Real.sqrt 7
def expr_D := Real.sqrt (1 / 3)

-- The statement of the proof problem
theorem simplest_quadratic_surd :
  expr_C = Real.sqrt 7 ∧
  (∀ x, x ∈ {expr_A, expr_B, expr_C, expr_D} → 
    (x = Real.sqrt 7 ∨ (x = Real.sqrt 9 → ¬ Real.is_surd x) ∨
    (x = Real.sqrt 20 → ¬ ∃ m, Real.sqrt 20 = m * (Real.sqrt 5)) ∨
    (x = Real.sqrt (1 / 3) → ¬ Real.is_simplest_surd (Real.sqrt (1 / 3))))) :=
by
  sorry

end simplest_quadratic_surd_l383_383423


namespace reflex_angle_at_T_l383_383983

-- Defining the points and angles
variables (P Q R S T : Type) [point : is_point P] [point : is_point Q]
          [point : is_point R] [point : is_point S] [point : is_point T]

-- Collinearity and angle conditions
axiom collinear_PQRS : collinear P Q R S
axiom PQT_angle : angle P Q T = 100
axiom RTS_angle : angle R T S = 70

-- Theorem statement
theorem reflex_angle_at_T : reflex_angle Q R T = 350 :=
by sorry

end reflex_angle_at_T_l383_383983


namespace minimum_points_to_immobilize_body_in_plane_l383_383237

-- Define what it means for points to be fixed and non-collinear
def are_non_collinear (p1 p2 p3 : Point) : Prop := 
  ¬collinear p1 p2 p3

-- Define the main problem as a theorem
theorem minimum_points_to_immobilize_body_in_plane : 
  ∃ (p1 p2 p3 : Point) (non_collinear : are_non_collinear p1 p2 p3), 
  ∀ (body : Body) (fixed_points : set Point),
  {p1, p2, p3} ⊆ fixed_points -> body_immobile body fixed_points :=
sorry

end minimum_points_to_immobilize_body_in_plane_l383_383237


namespace floor_add_ceil_eq_five_l383_383849

theorem floor_add_ceil_eq_five (x : ℝ) :
  (⌊x⌋ : ℝ) + (⌈x⌉ : ℝ) = 5 ↔ 2 < x ∧ x < 3 :=
by sorry

end floor_add_ceil_eq_five_l383_383849


namespace interval_solution_l383_383668

theorem interval_solution (a : ℝ) (x : ℝ) :
  (6 - 2 * a = 0 → a = 3) →
  (4 / a < 0 → a < 0) →
  ((a < 0 → x = (2 - sqrt (4 - 6 * a + 2 * a ^ 2) / a) ^ 2) ∧
  (a = 0 → x = 2.25) ∧
  (1 < a ∧ a < 2 → x = (2 - sqrt (4 - 6 * a + 2 * a ^ 2) / a) ^ 2) ∧
  (3 < a → x = (2 + sqrt (4 - 6 * a + 2 * a ^ 2) / a) ^ 2)) :=
sorry

end interval_solution_l383_383668


namespace proof_bd_leq_q2_l383_383689

variables {a b c d p q : ℝ}

theorem proof_bd_leq_q2 
  (h1 : ab + cd = 2pq)
  (h2 : ac ≥ p^2)
  (h3 : p^2 > 0) :
  bd ≤ q^2 :=
sorry

end proof_bd_leq_q2_l383_383689


namespace cost_price_one_meter_l383_383116

-- Definitions and Conditions
def total_price : ℝ := 8925
def total_meters_of_cloth : ℕ := 85
def profit_per_meter : ℝ := 35
def discount_rate : ℝ := 0.05
def tax_rate : ℝ := 0.03

-- Calculation of the result
def cost_price_per_meter (total_price : ℝ) (total_meters_of_cloth : ℕ) (profit_per_meter : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let sp := total_price / total_meters_of_cloth
  let discount := sp * discount_rate
  let sp_after_discount := sp - discount
  let tax := sp_after_discount * tax_rate
  let sp_after_tax := sp_after_discount + tax
  sp_after_tax - profit_per_meter

-- Proof of the result
theorem cost_price_one_meter :
  cost_price_per_meter total_price total_meters_of_cloth profit_per_meter discount_rate tax_rate = 67.7425 :=
sorry

end cost_price_one_meter_l383_383116


namespace calculation_correct_l383_383931

-- Defining the initial values
def a : ℕ := 20 ^ 10
def b : ℕ := 20 ^ 9
def c : ℕ := 10 ^ 6
def d : ℕ := 2 ^ 12

-- The expression we need to prove
theorem calculation_correct : ((a / b) ^ 3 * c) / d = 1953125 :=
by
  sorry

end calculation_correct_l383_383931


namespace right_triangle_hypotenuse_length_l383_383560

theorem right_triangle_hypotenuse_length (a b : ℝ) (h_triangle : a = 15 ∧ b = 36) :
  ∃ (h : ℝ), h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  · exact rfl
  · rw [h_triangle.1, h_triangle.2]
    norm_num

end right_triangle_hypotenuse_length_l383_383560


namespace machine_c_more_bottles_l383_383853

theorem machine_c_more_bottles (A B C : ℕ) 
  (hA : A = 12)
  (hB : B = A - 2)
  (h_total : 10 * A + 10 * B + 10 * C = 370) :
  C - B = 5 :=
by
  sorry

end machine_c_more_bottles_l383_383853


namespace sum_of_reversible_primes_is_zero_l383_383873

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_reversible_prime (n : ℕ) : Prop :=
  is_prime n ∧ is_prime (n / 10 + (n % 10) * 10)

def sum_digits_prime (n : ℕ) : Prop :=
  is_prime (n / 10 + n % 10)

theorem sum_of_reversible_primes_is_zero :
  ∑ p in Finset.filter (λ n, is_reversible_prime n ∧ sum_digits_prime n) (Finset.filter is_prime (Finset.range 100)), n = 0 :=
by
  sorry

end sum_of_reversible_primes_is_zero_l383_383873


namespace right_triangle_hypotenuse_l383_383534

theorem right_triangle_hypotenuse (a b : ℝ) (h : a^2 + b^2 = 39^2) : a = 15 ∧ b = 36 := by
  sorry

end right_triangle_hypotenuse_l383_383534


namespace right_triangle_hypotenuse_length_l383_383554

theorem right_triangle_hypotenuse_length (a b : ℝ) (h_triangle : a = 15 ∧ b = 36) :
  ∃ (h : ℝ), h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  · exact rfl
  · rw [h_triangle.1, h_triangle.2]
    norm_num

end right_triangle_hypotenuse_length_l383_383554


namespace sum_of_interior_angles_l383_383109

theorem sum_of_interior_angles {n : ℕ} (h1 : ∀ i, i < n → (interior_angle i : ℝ) = 144) : 
  (sum_of_polygon_interior_angles n = 1440) :=
sorry

end sum_of_interior_angles_l383_383109


namespace Petya_cannot_exchange_l383_383454

theorem Petya_cannot_exchange (N : ℕ) :
  (2 * N + 3 * N = 2001) → false :=
by {
  intro h,
  have h_mod : 5 ∣ 2001 := by rw [←h]; exact dvd_mul_right 5 N,
  sorry
}

end Petya_cannot_exchange_l383_383454


namespace books_on_shelf_l383_383039

theorem books_on_shelf (original_books : ℕ) (books_added : ℕ) (total_books : ℕ) (h1 : original_books = 38) 
(h2 : books_added = 10) : total_books = 48 :=
by 
  sorry

end books_on_shelf_l383_383039


namespace hypotenuse_length_l383_383477

theorem hypotenuse_length (a b c : ℕ) (h₀ : a = 15) (h₁ : b = 36) (h₂ : a^2 + b^2 = c^2) : c = 39 :=
by
  -- Proof is omitted
  sorry

end hypotenuse_length_l383_383477


namespace solve_for_a_l383_383661

noncomputable def f (x : ℝ) (g : ℝ → ℝ) (a : ℝ) : ℝ := a^x * g x
noncomputable def f' (x : ℝ) (g : ℝ → ℝ) (g' : ℝ → ℝ) (a : ℝ) : ℝ := 
  a^x * (log a * g x + g' x)

theorem solve_for_a (a : ℝ) (g g' : ℝ → ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x, g x ≠ 0) (h4 : ∀ x, f'(x, g, g', a) * g x - f x * g' x > 0) 
  (h5 : ∀ (t : ℝ), t = f 1 g a / g 1 + f (-1) g a / g (-1) → t = 10 / 3) : 
  a = 1 / 3 :=
sorry

end solve_for_a_l383_383661


namespace infinite_squares_in_arithmetic_sequence_l383_383324

open Nat Int

theorem infinite_squares_in_arithmetic_sequence
  (a d : ℤ) (h_d_nonneg : d ≥ 0) (x : ℤ) 
  (hx_square : ∃ n : ℕ, a + n * d = x * x) :
  ∃ (infinitely_many_n : ℕ → Prop), 
    (∀ k : ℕ, ∃ n : ℕ, infinitely_many_n n ∧ a + n * d = (x + k * d) * (x + k * d)) :=
sorry

end infinite_squares_in_arithmetic_sequence_l383_383324


namespace product_eq_5832_l383_383190

theorem product_eq_5832 (P Q R S : ℕ) 
(h1 : P + Q + R + S = 48)
(h2 : P + 3 = Q - 3)
(h3 : Q - 3 = R * 3)
(h4 : R * 3 = S / 3) :
P * Q * R * S = 5832 := sorry

end product_eq_5832_l383_383190


namespace ratio_of_doctors_to_lawyers_l383_383826

variable (d l : ℕ) -- number of doctors and lawyers
variable (h1 : (40 * d + 55 * l) / (d + l) = 45) -- overall average age condition

theorem ratio_of_doctors_to_lawyers : d = 2 * l :=
by
  sorry

end ratio_of_doctors_to_lawyers_l383_383826


namespace multiple_of_birds_is_two_l383_383900

theorem multiple_of_birds_is_two (birds : ℕ) (x : ℕ) (h1 : birds = 20) (h2 : x * birds + 10 = 50) :
  x = 2 :=
by {
  -- Here, we would write the proof steps if needed, but they are omitted as per the provided requirements.
  sorry,
}

end multiple_of_birds_is_two_l383_383900


namespace largest_angle_of_triangle_l383_383349

def isTriangle (α β γ : ℝ) : Prop := (α + β + γ = π) ∧ (0 < α) ∧ (0 < β) ∧ (0 < γ)

def cotHalfAngle (θ : ℝ) : ℝ := (real.tan (θ / 2))⁻¹

theorem largest_angle_of_triangle (α β γ : ℝ) (h : isTriangle α β γ) 
  (h1 : ∃ n : ℤ,  cotHalfAngle α = n ∧ cotHalfAngle β = n - 1 ∧ cotHalfAngle γ = n - 2) : 
  γ = π / 2 :=
sorry

end largest_angle_of_triangle_l383_383349


namespace vasya_pastry_trick_l383_383077

theorem vasya_pastry_trick :
  ∀ (pastries : Finset (Finset Nat))
    (filling_set : Finset Nat),
    (filling_set.card = 10) →
    (pastries.card = 45) →
    (∀ p ∈ pastries, p.card = 2 ∧ p ⊆ filling_set) →
    ∃ n, n = 36 ∧
    ∀ remain_p ∈ (pastries \ pastries.sort (λ x y, x < y)).take (45 - n), 
      ∃ f ∈ filling_set, f ∈ remain_p :=
begin
  sorry

end vasya_pastry_trick_l383_383077


namespace bicycle_final_price_l383_383050

theorem bicycle_final_price : 
  let original_price := 200 
  let weekend_discount := 0.40 * original_price 
  let price_after_weekend_discount := original_price - weekend_discount 
  let wednesday_discount := 0.20 * price_after_weekend_discount 
  let final_price := price_after_weekend_discount - wednesday_discount 
  final_price = 96 := 
by 
  sorry

end bicycle_final_price_l383_383050


namespace pascal_triangle_45th_number_l383_383009

theorem pascal_triangle_45th_number (n k : ℕ) (h1 : n = 47) (h2 : k = 44) : 
  Nat.choose (n - 1) k = 1035 :=
by
  sorry

end pascal_triangle_45th_number_l383_383009


namespace seating_arrangements_l383_383161

-- Define the number of people and cars
def numPeople := 8
def numCars := 3

-- Define the conditions for each car
def minPeoplePerCar := 1
def maxPeoplePerCar := 4

-- Lean proof statement
theorem seating_arrangements : 
  (∃ (f : Fin numPeople → Fin numCars), 
   (∀ c : Fin numCars, minPeoplePerCar ≤ (Finset.filter (λ p, f p = c) (Finset.univ : Finset (Fin numPeople)).card) ∧
                        (Finset.filter (λ p, f p = c) (Finset.univ : Finset (Fin numPeople)).card ≤ maxPeoplePerCar))) →
  ((number of possible different groupings) * (number of ways to assign these groups to cars) = 4620) :=
sorry

end seating_arrangements_l383_383161


namespace percentage_price_l383_383920

theorem percentage_price (C T : ℝ) 
  (hT : T = 52.5) 
  (hCT : T + C = 60) : 
  (2 * C + T) / (C + 2 * T) * 100 = 60 :=
by
  have hC : C = 60 - T := by linarith [hCT]
  rw [hT] at hC
  rw [hC, hT]
  rw [show 60 - 52.5 = 7.5, by norm_num]
  rw [show 2 * 7.5 + 52.5 = 67.5, by norm_num]
  rw [show 7.5 + 2 * 52.5 = 112.5, by norm_num]
  norm_num
  sorry

end percentage_price_l383_383920


namespace right_triangle_hypotenuse_length_l383_383508

theorem right_triangle_hypotenuse_length (a b : ℕ) (h1 : a = 15) (h2 : b = 36) : 
  ∃ c : ℕ, c * c = a * a + b * b ∧ c = 39 := 
by
  have hyp_square := 225 + 1296 
  have h_calculation : 15 * 15 + 36 * 36 = 1521 := by
    calc
      15 * 15 = 225 : rfl
      36 * 36 = 1296 : rfl
      225 + 1296 = 1521 : rfl
  use 39
  split
  exact h_calculation
  rfl

end right_triangle_hypotenuse_length_l383_383508


namespace smallest_n_for_trick_l383_383088

theorem smallest_n_for_trick (fillings : Finset Fin 10)
  (pastries : Finset (Fin 45)) 
  (has_pairs : ∀ p ∈ pastries, ∃ f1 f2 ∈ fillings, f1 ≠ f2 ∧ p = pair f1 f2) : 
  ∃ n (tray : Finset (Fin 45)), 
    (tray.card = n ∧ n = 36 ∧ 
    ∀ remaining_p ∈ pastries \ tray, ∃ f ∈ fillings, f ∈ remaining_p) :=
by
  sorry

end smallest_n_for_trick_l383_383088


namespace work_done_proof_l383_383128

-- Given conditions
variable (k a : ℝ)

-- Define displacement as a function of time
def s (t : ℝ) : ℝ := 1/2 * t^2

-- Define velocity as the derivative of displacement
def v (t : ℝ) : ℝ := t  -- Since ds/dt = t

-- Define the resistance force function
def F (s : ℝ) : ℝ := 2 * k * s  -- Since F = kv² and v² = 2s

-- Define the work done by integrating the force over the displacement
def work_done (a : ℝ) : ℝ := ∫ (0)^(a) (F s) ds

-- The theorem that needs to be proven
theorem work_done_proof : work_done k a = k * a^2 :=
by sorry

end work_done_proof_l383_383128


namespace power_rule_example_l383_383139

variable {R : Type*} [Ring R] (a b : R)

theorem power_rule_example : (a * b^3) ^ 2 = a^2 * b^6 :=
sorry

end power_rule_example_l383_383139


namespace number_of_pencils_broken_l383_383929

theorem number_of_pencils_broken
  (initial_pencils : ℕ)
  (misplaced_pencils : ℕ)
  (found_pencils : ℕ)
  (bought_pencils : ℕ)
  (final_pencils : ℕ)
  (h_initial : initial_pencils = 20)
  (h_misplaced : misplaced_pencils = 7)
  (h_found : found_pencils = 4)
  (h_bought : bought_pencils = 2)
  (h_final : final_pencils = 16) :
  (initial_pencils - misplaced_pencils + found_pencils + bought_pencils - final_pencils) = 3 := 
by
  sorry

end number_of_pencils_broken_l383_383929


namespace hundred_days_from_friday_is_sunday_l383_383390

/-- Given that today is Friday, determine that 100 days from now is Sunday. -/
theorem hundred_days_from_friday_is_sunday (today : ℕ) (days_in_week : ℕ := 7) 
(friday : ℕ := 0) (sunday : ℕ := 2) : (((today + 100) % days_in_week) = sunday) :=
sorry

end hundred_days_from_friday_is_sunday_l383_383390


namespace hypotenuse_length_l383_383484

-- Definitions for the problem
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def leg1 : ℕ := 15
def leg2 : ℕ := 36
def hypotenuse : ℕ := 39

-- Lean 4 statement
theorem hypotenuse_length (a b c : ℕ) (h : is_right_triangle a b c) (ha : a = leg1) (hb : b = leg2) :
  c = hypotenuse :=
begin
  sorry
end

end hypotenuse_length_l383_383484


namespace bonus_distribution_plans_l383_383097

theorem bonus_distribution_plans (x y : ℕ) (A B : ℕ) 
  (h1 : x + y = 15)
  (h2 : x = 2 * y)
  (h3 : 10 * A + 5 * B = 20000)
  (hA : A ≥ B)
  (hB : B ≥ 800)
  (hAB_mult_100 : ∃ (k m : ℕ), A = k * 100 ∧ B = m * 100) :
  (x = 10 ∧ y = 5) ∧
  ((A = 1600 ∧ B = 800) ∨
   (A = 1500 ∧ B = 1000) ∨
   (A = 1400 ∧ B = 1200)) :=
by
  -- The proof should be provided here
  sorry

end bonus_distribution_plans_l383_383097


namespace pascal_triangle_45th_number_l383_383006

theorem pascal_triangle_45th_number : nat.choose 46 44 = 1035 := 
by sorry

end pascal_triangle_45th_number_l383_383006


namespace prove_mouse_cost_l383_383613

variable (M K : ℕ)

theorem prove_mouse_cost (h1 : K = 3 * M) (h2 : M + K = 64) : M = 16 :=
by
  sorry

end prove_mouse_cost_l383_383613


namespace tangent_line_at_2_eq_l383_383601

noncomputable def f (x : ℝ) : ℝ := x^3 - 4 * x^2 + 5 * x - 4

theorem tangent_line_at_2_eq :
  let x := (2 : ℝ)
  let slope := (deriv f) x
  let y := f x
  ∃ (m y₀ : ℝ), m = slope ∧ y₀ = y ∧ 
    (∀ (x y : ℝ), y = m * (x - 2) + y₀ → x - y - 4 = 0)
:= sorry

end tangent_line_at_2_eq_l383_383601


namespace calculate_value_l383_383599

def op (a b : ℝ) : ℝ := (a - b) / (1 - a * b)

theorem calculate_value :
  ∀ (y : ℝ), y ≠ 1 / 2 →
    2 * (op 3 (op 4 (op ... (op 1000 1001) ... ))) = (2 - y) / (1 - 2 * y) := by
  intros y hy
  sorry

end calculate_value_l383_383599


namespace ellipse_condition_necessary_but_not_sufficient_l383_383749

variables {F_1 F_2 P : ℝ × ℝ} {a : ℝ}

-- Definitions based on provided conditions:

def dist (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def is_ellipse_with_foci_major_axis (F_1 F_2 : ℝ × ℝ) (P : ℝ × ℝ) (a : ℝ) : Prop :=
  dist P F_1 + dist P F_2 = 2 * a

-- Rewrite the statement:
theorem ellipse_condition_necessary_but_not_sufficient 
  (h_fixed_points : F_1 ≠ F_2)
  (h_moving_point : true)
  (h_positive_a : a > 0) :
  (∀ P, is_ellipse_with_foci_major_axis F_1 F_2 P a) →
  (dist F_1 F_2 < 2 * a) → 
  false :=
sorry

-- Note: The statement is designed to assert that the ellipse condition
-- is necessary (∀ P, is_ellipse_with_foci_major_axis F_1 F_2 P a) implies the distance condition,
-- but not sufficient because the false statement will follow without the distance constraint.

end ellipse_condition_necessary_but_not_sufficient_l383_383749


namespace tangency_of_parabolas_l383_383622

theorem tangency_of_parabolas :
  ∃ x y : ℝ, y = x^2 + 12*x + 40
  ∧ x = y^2 + 44*y + 400
  ∧ x = -11 / 2
  ∧ y = -43 / 2 := by
sorry

end tangency_of_parabolas_l383_383622


namespace fourth_power_l383_383864

noncomputable def sqrt_inner : ℝ := real.sqrt(2)
noncomputable def sqrt_middle : ℝ := real.sqrt(1 + sqrt_inner)
noncomputable def sqrt_outer : ℝ := real.sqrt(1 + sqrt_middle)
def x : ℝ := sqrt_outer
def x_squared : ℝ := x * x
def x_quartic : ℝ := x_squared * x_squared

theorem fourth_power : x_quartic = 4 + 2 * real.sqrt 3 :=
by
  sorry -- Proof would go here

end fourth_power_l383_383864


namespace area_of_triangle_between_lines_l383_383015

theorem area_of_triangle_between_lines (x y : ℝ) :
  let line1 := λ x, 3 * x - 6,
      line2 := λ x, -2 * x + 8,
      intersection_x := (8 + 6) / (3 + 2),
      intersection_y := line1 intersection_x,
      vertex1 := (0 : ℝ, -6),
      vertex2 := (0 : ℝ, 8),
      vertex3 := (intersection_x, intersection_y),
      base := 8 - (-6),
      height := intersection_x,
      area := 1 / 2 * base * height
  in area = 19.6 :=
by
  let line1 := λ x, 3 * x - 6
  let line2 := λ x, -2 * x + 8
  let intersection_x := (8 + 6) / (3 + 2)
  let intersection_y := line1 intersection_x
  let vertex1 := (0 : ℝ, -6)
  let vertex2 := (0 : ℝ, 8)
  let vertex3 := (intersection_x, intersection_y)
  let base := 8 - (-6)
  let height := intersection_x
  let area := 1 / 2 * base * height
  exact (by norm_num : area = 19.6)

end area_of_triangle_between_lines_l383_383015


namespace find_a_l383_383205

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

noncomputable def line_eq (x y a : ℝ) : Prop := x + a * y + 1 = 0

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, circle_eq x y → line_eq x y a → (x - 1)^2 + (y - 2)^2 = 4) →
  ∃ a, (a = -1) :=
sorry

end find_a_l383_383205


namespace chord_length_l383_383099

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ :=
  (-2 + (1/2) * t, real.sqrt 3 + (real.sqrt 3 / 2) * t)

def curve (x y : ℝ) : Prop :=
  y^2 = -x - 1

def is_on_curve (p : ℝ × ℝ) : Prop :=
  curve p.1 p.2

theorem chord_length :
  ∀ A B : ℝ × ℝ,
  is_on_curve A →
  is_on_curve B →
  A ≠ B →
  ( ∃ t_A t_B, parametric_line t_A = A ∧ parametric_line t_B = B ) →
  (abs (A.1 - B.1)) * real.sqrt (1 + 3) = (10 : ℝ) / 3 :=
sorry

end chord_length_l383_383099


namespace median_and_area_of_triangle_l383_383748

-- Declare the conditions
variables {X Y Z N : Type}
variables [euclidean_geometry.{v}]
variables (XY YZ XZ YN : ℝ)
variable (right_triangle : ∆ XYZ)
variable (midpoint_N : midpoint N (XZ : n))

theorem median_and_area_of_triangle
  (h_right : ∡XYZ = π / 2)
  (h_XY : XY = 5)
  (h_YZ : YZ = 12)
  (h_midpoint_N: N.midpoint XZ)
  (h_XZ : XZ = sqrt (XY^2 + YZ^2))
  :
  (YN = (1/2) * XZ) ∧ (YN = 6.5) ∧ (area (Δ XYZ) = 30) := 
by
  sorry

end median_and_area_of_triangle_l383_383748


namespace joe_money_left_l383_383766

noncomputable def total_spent : ℝ :=
  let notebooks := 7 * 4
  let books := 2 * 12
  let pens := 5 * 2
  let stickers := 3 * 6
  let shoes := 40
  let tshirt := 18
  let total_cost_before_tax := notebooks + books + pens + stickers + shoes + tshirt
  let sales_tax := total_cost_before_tax * 0.05
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  let lunch_and_tip := 15 + 3
  let transportation := 8
  let charity := 10
  total_cost_after_tax + lunch_and_tip + transportation + charity

theorem joe_money_left (initial_amount : ℝ) (total_spent : ℝ) : initial_amount - total_spent = 19.10 :=
by
  let initial_amount := 200
  have total_spent_eq : total_spent = 144.90 + 36 := 
    by sorry
  rw total_spent_eq
  calc 
    200 - (144.90 + 36)
        = 200 - 180.90 : by rw add_comm
    ... = 19.10 : by norm_num

end joe_money_left_l383_383766


namespace tony_solving_puzzles_time_l383_383383

theorem tony_solving_puzzles_time : ∀ (warm_up_time long_puzzle_ratio num_long_puzzles : ℕ),
  warm_up_time = 10 →
  long_puzzle_ratio = 3 →
  num_long_puzzles = 2 →
  (warm_up_time + long_puzzle_ratio * warm_up_time * num_long_puzzles) = 70 :=
by
  intros
  sorry

end tony_solving_puzzles_time_l383_383383


namespace coin_flip_probability_l383_383818

theorem coin_flip_probability :
  let total_outcomes : ℕ := 2^6,
      fav_outcomes := 2,  -- HHH TTT or TTT HHH
      probability := fav_outcomes / total_outcomes in
  probability = (1:ℚ) / 32 :=
by
  sorry

end coin_flip_probability_l383_383818


namespace value_of_expression_l383_383232

theorem value_of_expression (a b : ℝ) (h1 : a^2 + 2012 * a + 1 = 0) (h2 : b^2 + 2012 * b + 1 = 0) :
  (2 + 2013 * a + a^2) * (2 + 2013 * b + b^2) = -2010 := 
  sorry

end value_of_expression_l383_383232


namespace find_common_ratio_and_sum_l383_383665

theorem find_common_ratio_and_sum (q : ℝ) (S : ℕ → ℝ) 
  (h1 : S 2 = 3) 
  (h2 : S 4 = 15) 
  (h3 : ∀ n, 0 < q) :
  q = 2 ∧ S 6 = 63 :=
by {
  sorry
}

end find_common_ratio_and_sum_l383_383665


namespace correct_answer_l383_383208

def A : Set ℝ := { x | x^2 + 2 * x - 3 > 0 }
def B : Set ℝ := { -1, 0, 1, 2 }

theorem correct_answer : A ∩ B = { 2 } :=
  sorry

end correct_answer_l383_383208


namespace problem1_Omega_A1_and_A2_problem2_max_and_min_Omega_A_problem3_min_Omega_A_l383_383631

-- Define Omega function
def Omega (A : Set ℝ) : Set ℝ := 
  {π ∏ X | X ∈ (Set.Subsets A), X ≠ ∅}

-- Problem 1: Given specific sets A1 and A2
def A1 : Set ℝ := {\frac{1}{2}, 1, 4}
def A2 : Set ℝ := {2, 3, 5}

theorem problem1_Omega_A1_and_A2 : |Omega A1| = 4 ∧ |Omega A2| = 7 := by sorry

-- Problem 2: For a set A of 5 positive integers
def A2_int (A : Set ℝ) : Prop := 
  ∃ (a1 a2 a3 a4 a5 : ℝ), A = {a1, a2, a3, a4, a5} ∧ (∀ i ∈ A, i ∈ ℤ ∧ i > 0)

theorem problem2_max_and_min_Omega_A :
  ∀ (A : Set ℝ), A2_int A → (|Omega A| ≤ 31 ∧ |Omega A| ≥ 11) := by sorry

-- Problem 3: For a set A of 7 positive real numbers
def A3_real (A : Set ℝ) : Prop := 
  ∃ (a1 a2 a3 a4 a5 a6 a7 : ℝ), A = {a1, a2, a3, a4, a5, a6, a7} ∧ (∀ i ∈ A, i > 0)

theorem problem3_min_Omega_A :
  ∀ (A : Set ℝ), A3_real A → |Omega A| ≥ 13 := by sorry

end problem1_Omega_A1_and_A2_problem2_max_and_min_Omega_A_problem3_min_Omega_A_l383_383631


namespace square_side_length_l383_383725

theorem square_side_length (x : ℝ) (h : x^2 = 12) : x = 2 * Real.sqrt 3 :=
sorry

end square_side_length_l383_383725


namespace AO_plus_AQ_AR_AS_eq_five_l383_383891

open Real

noncomputable def regularHexagon (s : ℝ) : Type :=
{A B C D E F : ℝ}

variables (s : ℝ)
variables (A B C D E F : ℝ) -- vertices of regular hexagon
variables (O : ℝ) (h : ℝ)
variables (AP AQ AR AS OP : ℝ)

def height_eq (s : ℝ) : ℝ := s * (sqrt 3) / 2

def area_eq (s : ℝ) : ℝ := 3 * s ^ 2 * (sqrt 3 / 2)

def AO (s : ℝ) : ℝ := 1 -- derived from equations given and conditions mentioned.

def AQ (s : ℝ) : ℝ -- to be calculated based on hexagonal geometry.

def AR (s : ℝ) : ℝ -- to be calculated based on hexagonal geometry.

def AS (s : ℝ) : ℝ -- to be calculated based on hexagonal geometry.

theorem AO_plus_AQ_AR_AS_eq_five (hex : regularHexagon s) (h_eq : h = height_eq s) (area_eq : area_eq s) (OP_eq : OP = 1) :
  let AO := AO s,
      AQ := AQ s,
      AR := AR s,
      AS := AS s
  in AO + AQ + AR + AS = 5 :=
sorry

end AO_plus_AQ_AR_AS_eq_five_l383_383891


namespace rotation_angle_is_270_l383_383453

noncomputable def angle_of_rotation (Z Q P : Point) (shaded_unshaded : Quadrilateral) : ℝ :=
  sorry

theorem rotation_angle_is_270 (Z Q P : Point) (shaded_unshaded : Quadrilateral) 
  (h1 : rotates_clockwise_around Z shaded_unshaded) 
  (h2 : bottom_edge_rotates_from_ZQ_to_ZP shaded_unshaded Q P) 
  (h3 : angle_PZQ_is_90_degrees Z Q P) : 
  angle_of_rotation Z Q P shaded_unshaded = 270 :=
sorry

end rotation_angle_is_270_l383_383453


namespace solution_set_k_l383_383280

def lines_divide_plane {R : Type} [linear_ordered_field R] (k : R) : Prop :=
  let l1 := λ (x y : R), x - 2 * y + 1 = 0
  let l2 := λ (x y : R), x - 1 = 0
  let l3 := λ (x y : R), x + k * y = 0
  ∃ (p1 p2 : R), l1 p1 p2 ∧ l2 p1 p2 → k = -1 ∨ k = 0 ∨ k = -2

theorem solution_set_k : {k : ℝ // lines_divide_plane k} = {0, -1, -2} :=
sorry

end solution_set_k_l383_383280


namespace ten_thousand_times_ten_thousand_l383_383824

theorem ten_thousand_times_ten_thousand :
  10000 * 10000 = 100000000 :=
by
  sorry

end ten_thousand_times_ten_thousand_l383_383824


namespace percentage_girl_scouts_with_permission_is_63_l383_383460

-- Definitions based on given conditions
def total_scouts : ℕ := 100
def percentage_with_permission : ℝ := 0.70
def percentage_boy_scouts : ℝ := 0.60
def percentage_boy_scouts_with_permission : ℝ := 0.75

-- Calculate values derived from the conditions
def total_with_permission : ℕ := (percentage_with_permission * total_scouts).to_nat
def total_boy_scouts : ℕ := (percentage_boy_scouts * total_scouts).to_nat
def total_boy_scouts_with_permission : ℕ := (percentage_boy_scouts_with_permission * total_boy_scouts).to_nat
def total_girl_scouts : ℕ := total_scouts - total_boy_scouts
def total_girl_scouts_with_permission : ℕ := total_with_permission - total_boy_scouts_with_permission

-- The required proof problem
theorem percentage_girl_scouts_with_permission_is_63 :
  (100 * total_girl_scouts_with_permission / total_girl_scouts).to_nat = 63 := 
by
  sorry

end percentage_girl_scouts_with_permission_is_63_l383_383460


namespace difference_in_dimes_l383_383803

theorem difference_in_dimes : 
  ∀ (a b c : ℕ), (a + b + c = 100) → (5 * a + 10 * b + 25 * c = 835) → 
  (∀ b_max b_min, (b_max = 67) ∧ (b_min = 3) → (b_max - b_min = 64)) :=
by
  intros a b c h1 h2 b_max b_min h_bounds
  sorry

end difference_in_dimes_l383_383803


namespace no_infinite_sequence_exists_l383_383961

theorem no_infinite_sequence_exists :
  ¬∃ (a : ℕ → ℕ), (∀ n, n > 0 → a n > 0) ∧ (∀ n, a (n+2) = a (n+1) + has_sqrt.sqrt (a (n+1) + a n)) :=
sorry

end no_infinite_sequence_exists_l383_383961


namespace polynomial_abs_sum_l383_383773

theorem polynomial_abs_sum (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) :
  (1 - (2:ℝ) * x) ^ 8 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 →
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| + |a_8| = (3:ℝ) ^ 8 :=
sorry

end polynomial_abs_sum_l383_383773


namespace corn_price_decrease_second_island_l383_383811

-- Define the initial price of pearls in gold on the first island
def initial_price_gold_first_island (x : ℝ) : ℝ := 0.95 * x

-- Define the initial price of pearls in corn on the first island
def initial_price_corn_first_island (x : ℝ) : ℝ := 0.93 * x

-- Define the price ratio on the first island
def price_ratio_first_island : ℝ := initial_price_gold_first_island 1 / initial_price_corn_first_island 1

-- Define the initial price of pearls in gold on the second island
def initial_price_gold_second_island (x : ℝ) : ℝ := 0.99 * x

-- Define the new price of pearls in corn on the second island
def new_price_corn_second_island (y : ℝ) : ℝ := initial_price_gold_second_island 1 / price_ratio_first_island

-- Theorem statement: proof that decrease in corn price on second island is 3.08%
theorem corn_price_decrease_second_island (x : ℝ) : 
  1 - (new_price_corn_second_island x / x) = 0.0308 := 
sorry

end corn_price_decrease_second_island_l383_383811


namespace real_and_equal_roots_of_quadratic_l383_383158

theorem real_and_equal_roots_of_quadratic (k: ℝ) :
  (-(k+2))^2 - 4 * 3 * 12 = 0 ↔ k = 10 ∨ k = -14 :=
by
  sorry

end real_and_equal_roots_of_quadratic_l383_383158


namespace min_t_complete_circle_l383_383836

theorem min_t_complete_circle :
  ∃ t : ℝ, (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → r = sin θ) ∧ complete_circle r t := 
sorry

end min_t_complete_circle_l383_383836


namespace hundred_days_from_friday_is_sunday_l383_383399

def days_from_friday (n : ℕ) : Nat :=
  (n + 5) % 7  -- 0 corresponds to Sunday, starting from Friday (5 + 0 % 7 = 5 which is Friday)

theorem hundred_days_from_friday_is_sunday :
  days_from_friday 100 = 0 := by
  sorry

end hundred_days_from_friday_is_sunday_l383_383399


namespace tangent_perpendicular_intersection_x_4_l383_383218

noncomputable def f (x : ℝ) := (x^2 / 4) - (4 * Real.log x)
noncomputable def f' (x : ℝ) := (1/2 : ℝ) * x - 4 / x

theorem tangent_perpendicular_intersection_x_4 :
  ∀ x : ℝ, (0 < x) → (f' x = 1) → (x = 4) :=
by {
  sorry
}

end tangent_perpendicular_intersection_x_4_l383_383218


namespace fraction_power_equality_l383_383151

theorem fraction_power_equality :
  (72000 ^ 4) / (24000 ^ 4) = 81 := 
by
  sorry

end fraction_power_equality_l383_383151


namespace triangle_solutions_l383_383264

variable {a b c : ℝ}
variable {A B : ℝ}

theorem triangle_solutions (h1 : a > b) (h2 : a = 5) (h3 : c = 6) (h4 : sin B = 3 / 5) :
  b = Real.sqrt 13 ∧ sin A = 3 * Real.sqrt 13 / 13 ∧ sin (2 * A + Real.pi / 4) = 7 * Real.sqrt 2 / 26 :=
sorry

end triangle_solutions_l383_383264


namespace least_trees_required_l383_383883

theorem least_trees_required : ∃ n : ℕ, n = Nat.lcm (Nat.lcm 4 5) 6 :=
by
  use Nat.lcm (Nat.lcm 4 5) 6
  sorry

end least_trees_required_l383_383883


namespace fraction_product_l383_383417

theorem fraction_product :
  (∏ n in Finset.range 6, (1 / 3^(n+1) * 3^(2*(n+1))) = 729) :=
sorry

end fraction_product_l383_383417


namespace convex_ngon_triangle_count_l383_383200

theorem convex_ngon_triangle_count (n : ℕ) (h₁ : n ≥ 3) (h₂ : convex_ngon n) (h₃ : ∀ i j, i ≠ j → ¬parallel (side i) (side j)) :
  distinct_triangle_count (extend_sides (ngon n)) ≥ n - 2 :=
sorry

end convex_ngon_triangle_count_l383_383200


namespace system_solution_exists_l383_383338

theorem system_solution_exists (x y: ℝ) :
    (y^2 = (x + 8) * (x^2 + 2) ∧ y^2 - (8 + 4 * x) * y + (16 + 16 * x - 5 * x^2) = 0) → 
    ((x = 0 ∧ (y = 4 ∨ y = -4)) ∨ (x = -2 ∧ (y = 6 ∨ y = -6)) ∨ (x = 19 ∧ (y = 99 ∨ y = -99))) :=
    sorry

end system_solution_exists_l383_383338


namespace amc_problem_l383_383790

theorem amc_problem (T : ℤ) (hT : T = ∑ n in {n | ∃ m : ℤ, n^2 + 12 * n - 2017 = m^2}, n) : 
  T % 1000 = 217 :=
sorry

end amc_problem_l383_383790


namespace right_triangle_hypotenuse_l383_383514

theorem right_triangle_hypotenuse (a b : ℕ) (ha : a = 15) (hb : b = 36) : 
  ∃ h : ℕ, h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  . exact rfl
  . rw [ha, hb]
    norm_num
    sorry

end right_triangle_hypotenuse_l383_383514


namespace minimum_pies_for_trick_l383_383092

-- Definitions from conditions
def num_fillings : ℕ := 10
def num_pastries := (num_fillings * (num_fillings - 1)) / 2
def min_pies_for_trick (n : ℕ) : Prop :=
  ∀ remaining_pies : ℕ, remaining_pies = num_pastries - n → remaining_pies ≤ 9

theorem minimum_pies_for_trick : ∃ n : ℕ, min_pies_for_trick n ∧ n = 36 :=
by
  -- We need to show that there exists n such that,
  -- min_pies_for_trick holds and n = 36
  existsi (36 : ℕ)
  -- remainder of the proof (step solution) skipped
  sorry

end minimum_pies_for_trick_l383_383092


namespace probability_two_heads_five_tosses_l383_383308

theorem probability_two_heads_five_tosses : 
  (∀ p : ℕ → ℕ → ℚ , p 1 2 = 1 / 2) → 
  (∑ k in finset.range 6, if k = 2 then ↑(nat.choose 5 k) else 0) * ((1 / 2) ^ 5) = 5 / 16 :=
by sorry

end probability_two_heads_five_tosses_l383_383308


namespace gcd_digits_bounded_by_lcm_l383_383731

theorem gcd_digits_bounded_by_lcm (a b : ℕ) (h_a : 10^6 ≤ a ∧ a < 10^7) (h_b : 10^6 ≤ b ∧ b < 10^7) (h_lcm : 10^10 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^11) : Nat.gcd a b < 10^4 :=
by
  sorry

end gcd_digits_bounded_by_lcm_l383_383731


namespace number_of_points_on_ellipse_with_given_conditions_l383_383763

noncomputable def point_on_ellipse (x y : ℝ) : Prop := 
  (x^2 / 4) + (y^2 / 3) = 1

def is_focus (x y : ℝ) : Prop :=
  (x = -√(1 - (1 / 3^0.5))) ∧ (y = 0) ∨ (x = √(1 - (1 / 3^0.5))) ∧ (y = 0)

def angle_condition (P F1 F2 : ℝ × ℝ) : Prop :=
  let θ := ∠ F1 P F2 in 
  θ = π/3

theorem number_of_points_on_ellipse_with_given_conditions : 
  ∀ (P F1 F2 : ℝ × ℝ), 
    point_on_ellipse P.1 P.2 ∧ is_focus F1.1 F1.2 ∧ is_focus F2.1 F2.2 ∧ angle_condition P F1 F2 → 
    P = (-1, 0) ∨ P = (1, 0) ∨ P = (0, -√3 /2) ∨ P = (0, √3 /2) := sorry

end number_of_points_on_ellipse_with_given_conditions_l383_383763


namespace right_triangle_hypotenuse_l383_383512

theorem right_triangle_hypotenuse (a b : ℕ) (ha : a = 15) (hb : b = 36) : 
  ∃ h : ℕ, h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  . exact rfl
  . rw [ha, hb]
    norm_num
    sorry

end right_triangle_hypotenuse_l383_383512


namespace right_triangle_hypotenuse_l383_383518

theorem right_triangle_hypotenuse (a b : ℕ) (ha : a = 15) (hb : b = 36) : 
  ∃ h : ℕ, h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  . exact rfl
  . rw [ha, hb]
    norm_num
    sorry

end right_triangle_hypotenuse_l383_383518


namespace smallest_largest_product_ratio_l383_383192

theorem smallest_largest_product_ratio : 
  let m := (-2020) * (-2019) * (-2018),
      n := 2017 * 2018 * 2019
  in m / n = (-2020:ℤ) / 2017 :=
by
  sorry

end smallest_largest_product_ratio_l383_383192


namespace age_when_hired_l383_383052

-- Define the conditions in Lean 4
def hired_year : ℕ := 1988
def retire_year : ℕ := 2007
def rule_of_70 : ℕ := 70

-- Calculate the number of years worked
def years_worked : ℕ := retire_year - hired_year

-- Given the conditions, prove that age when hired is 51
theorem age_when_hired :
  ∃ A : ℕ, A + years_worked = rule_of_70 ∧ A = 51 :=
by 
  have Y := years_worked
  have A := 51
  use A
  simp [Y, hired_year, retire_year, rule_of_70]
  sorry

end age_when_hired_l383_383052


namespace right_triangle_hypotenuse_length_l383_383507

theorem right_triangle_hypotenuse_length (a b : ℕ) (h1 : a = 15) (h2 : b = 36) : 
  ∃ c : ℕ, c * c = a * a + b * b ∧ c = 39 := 
by
  have hyp_square := 225 + 1296 
  have h_calculation : 15 * 15 + 36 * 36 = 1521 := by
    calc
      15 * 15 = 225 : rfl
      36 * 36 = 1296 : rfl
      225 + 1296 = 1521 : rfl
  use 39
  split
  exact h_calculation
  rfl

end right_triangle_hypotenuse_length_l383_383507


namespace horner_mult_count_l383_383143

def polynomial : (ℝ → ℝ) := λ x, 3 * x^4 + 3 * x^3 + 2 * x^2 + 6 * x + 1

def horner_evaluation (f : ℝ → ℝ) (x : ℝ) (n_multiplications : ℕ) : Prop :=
n_multiplications = 4

theorem horner_mult_count :
  horner_evaluation polynomial 0.5 4 :=
by
  sorry

end horner_mult_count_l383_383143


namespace matrix_operations_correct_l383_383938

-- Define matrices
def A : matrix (fin 3) (fin 2) ℤ :=
  ![![2, 4], ![3, -1], ![0, 5]]

def B : matrix (fin 3) (fin 2) ℤ :=
  ![![1, -3], ![-1, 2], ![4, 0]]

def C : matrix (fin 3) (fin 2) ℤ :=
  ![![3, 0], ![4, 1], ![1, 5]]

-- Expected result after performing the operations
def expected : matrix (fin 3) (fin 2) ℤ :=
  ![![0, 1], ![-2, 0], ![3, 0]]

-- Proof statement
theorem matrix_operations_correct :
  A + B - C = expected :=
by
  sorry

end matrix_operations_correct_l383_383938


namespace minimum_pies_for_trick_l383_383095

-- Definitions from conditions
def num_fillings : ℕ := 10
def num_pastries := (num_fillings * (num_fillings - 1)) / 2
def min_pies_for_trick (n : ℕ) : Prop :=
  ∀ remaining_pies : ℕ, remaining_pies = num_pastries - n → remaining_pies ≤ 9

theorem minimum_pies_for_trick : ∃ n : ℕ, min_pies_for_trick n ∧ n = 36 :=
by
  -- We need to show that there exists n such that,
  -- min_pies_for_trick holds and n = 36
  existsi (36 : ℕ)
  -- remainder of the proof (step solution) skipped
  sorry

end minimum_pies_for_trick_l383_383095


namespace smallest_number_satisfying_conditions_l383_383911

theorem smallest_number_satisfying_conditions :
  ∃ (x : ℤ), (x + 3) % 7 = 0 ∧ (x - 5) % 8 = 0 ∧ x = 53 :=
by
  use 53
  split
  . show (53 + 3) % 7 = 0
  calc
    (53 + 3) % 7 = 56 % 7 := by ring
    _ = 0 := by norm_num
  split
  . show (53 - 5) % 8 = 0
  calc
    (53 - 5) % 8 = 48 % 8 := by ring
    _ = 0 := by norm_num
  . rfl

end smallest_number_satisfying_conditions_l383_383911


namespace number_five_digit_integers_div_by_25_l383_383705

theorem number_five_digit_integers_div_by_25 : 
  ∃ n : ℕ, n = 90 ∧
    (∀ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → ($ab325 : ℕ) % 25 = 0 → n = 90) := by
  sorry

end number_five_digit_integers_div_by_25_l383_383705


namespace point_K_trace_path_l383_383449

-- Definitions based on the problem conditions
def radius_fixed : ℝ := 2
def radius_moving : ℝ := 1

-- Definition of rolling condition without slipping
def rolls_without_slipping (R_fixed R_moving : ℝ) : Prop :=
  2 * real.pi * R_moving = 2 * real.pi * R_fixed / 2

-- Hypothesis combining the given conditions
axiom rolls_condition : rolls_without_slipping radius_fixed radius_moving

-- The point K trace path theorem
theorem point_K_trace_path :
  (∀ (K : ℝ) (O O1 : ℝ), 
    radius_fixed = 2 * radius_moving → 
    rolls_without_slipping radius_fixed radius_moving → 
    K is a point on a circle with radius radius_moving at any moment →
    K traces a diameter of the fixed circle) :=
by sorry

end point_K_trace_path_l383_383449


namespace complex_point_quadrant_l383_383954

open Complex

theorem complex_point_quadrant :
  let z := cos (2 * Real.pi / 3) + sin (2 * Real.pi / 3) * I in
  z.re < 0 ∧ z.im > 0 :=
by
  let z := cos (2 * Real.pi / 3) + sin (2 * Real.pi / 3) * I
  -- Proof omitted
  sorry

end complex_point_quadrant_l383_383954


namespace range_of_m_l383_383986

variable (m : ℝ)
def p := ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → x^2 - 2*x - 4*m^2 + 8*m - 2 ≥ 0
def q := ∃ x : ℝ, x ∈ Set.Icc (1 : ℝ) 2 ∧ Real.log (x^2 - m*x + 1) / Real.log (1/2) < -1

theorem range_of_m (hp : p m) (hq : q m) (hl : (p m) ∨ (q m)) (hf : ¬ ((p m) ∧ (q m))) :
  m < 1/2 ∨ m = 3/2 := sorry

end range_of_m_l383_383986


namespace profit_percent_l383_383100

theorem profit_percent (marked_price : ℚ) (pens_bought : ℚ) (discount : ℚ) :
  pens_bought = 120 →
  marked_price = 100 →
  discount = 2 →
  (100 * (1 - discount / 100) * pens_bought / (marked_price * pens_bought / 100)) * 100 = 205.8 :=
by
  intros h1 h2 h3
  have cp := (marked_price * pens_bought) / 100 -- cost price
  have sp := 100 * (1 - discount / 100) -- selling price per pen
  have profit := sp - cp -- profit per pen
  have profit_percent := (profit / cp) * 100 -- profit percent
  sorry

end profit_percent_l383_383100


namespace total_earnings_in_september_l383_383698

theorem total_earnings_in_september (
  mowing_rate: ℕ := 6
  mowing_hours: ℕ := 63
  weeds_rate: ℕ := 11
  weeds_hours: ℕ := 9
  mulch_rate: ℕ := 9
  mulch_hours: ℕ := 10
): 
  mowing_rate * mowing_hours + weeds_rate * weeds_hours + mulch_rate * mulch_hours = 567 := 
by
  sorry

end total_earnings_in_september_l383_383698


namespace annual_concert_tickets_l383_383129

theorem annual_concert_tickets (S NS : ℕ) (h1 : S + NS = 150) (h2 : 5 * S + 8 * NS = 930) : NS = 60 :=
by
  sorry

end annual_concert_tickets_l383_383129


namespace hexagon_ratio_l383_383788

variables (A B C D E F : Type) [convex_hexagon A B C D E F]

def angle_sum_condition (angle_B angle_D angle_F : ℝ) : Prop :=
  angle_B + angle_D + angle_F = 360

def product_ratio_condition (AB BC CD DE EF FA : ℝ) : Prop :=
  (AB / BC) * (CD / DE) * (EF / FA) = 1

theorem hexagon_ratio (angle_B angle_D angle_F : ℝ) (AB BC CD DE EF FA CA FD AE DB : ℝ)
  (h1 : angle_sum_condition angle_B angle_D angle_F)
  (h2 : product_ratio_condition AB BC CD DE EF FA) :
  (BC / CA) * (AE / EF) * (FD / DB) = 1 :=
by
  sorry

end hexagon_ratio_l383_383788


namespace calculate_expression_l383_383939

theorem calculate_expression : ((-2)^2 - |(-5)| - Real.sqrt 144) = -13 :=
by
  have h1 : (-2)^2 = 4 := by norm_num
  have h2 : |(-5)| = 5 := by norm_num
  have h3 : Real.sqrt 144 = 12 := by norm_num
  
  calc
    ((-2)^2 - |(-5)| - Real.sqrt 144)
        = (4 - 5 - 12) : by rw [h1, h2, h3]
    ... = -1 - 12 : by norm_num
    ... = -13 : by norm_num

end calculate_expression_l383_383939


namespace a_10_value_l383_383843

-- Definitions for the initial conditions and recurrence relation.
def seq (a : ℕ → ℝ) : Prop :=
  a 0 = 0 ∧
  ∀ n, a (n + 1) = (8 / 5) * a n + (6 / 5) * (Real.sqrt (4 ^ n - a n ^ 2))

-- Statement that proves a_10 = 24576 / 25 given the conditions.
theorem a_10_value (a : ℕ → ℝ) (h : seq a) : a 10 = 24576 / 25 :=
by
  sorry

end a_10_value_l383_383843


namespace max_acute_triangles_from_4_points_l383_383578

theorem max_acute_triangles_from_4_points (A B C D : Point) : 
  ∃ (triangles : list (triangle Point)), triangles.length = 4 ∧ 
  ∀ Δ ∈ triangles, Δ.isAcute := 
sorry

end max_acute_triangles_from_4_points_l383_383578


namespace intersect_once_resolve_k_l383_383259

noncomputable def line (k : ℝ) := λ x : ℝ, k * x + 2
noncomputable def curve (x y : ℝ) := x^2 / 2 + y^2 - 1

theorem intersect_once_resolve_k
  (k : ℝ)
  (exists_one_point : ∃ (x y : ℝ), line k x = y ∧ curve x y = 0) :
  k = -real.sqrt 6 / 2 ∨ k = real.sqrt 6 / 2 :=
by
  sorry

end intersect_once_resolve_k_l383_383259


namespace right_triangle_hypotenuse_length_l383_383558

theorem right_triangle_hypotenuse_length (a b : ℝ) (h_triangle : a = 15 ∧ b = 36) :
  ∃ (h : ℝ), h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  · exact rfl
  · rw [h_triangle.1, h_triangle.2]
    norm_num

end right_triangle_hypotenuse_length_l383_383558


namespace a_3_value_general_term_l383_383692

noncomputable def a : ℕ → ℝ
| 1 := 1
| n := (7 * 3 ^ (n - 2) - a (n - 1)) / 2

theorem a_3_value : a 3 = 9 :=
by {
  sorry
}

theorem general_term (n : ℕ) : a n = 3 ^ (n - 1) :=
by {
  sorry
}

end a_3_value_general_term_l383_383692


namespace day_100_days_from_friday_l383_383396

-- Define the days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Define a function to get the day of the week after a given number of days
def dayOfWeekAfter (start : Day) (n : ℕ) : Day :=
  match start with
  | Sunday    => match n % 7 with
                  | 0 => Sunday
                  | 1 => Monday
                  | 2 => Tuesday
                  | 3 => Wednesday
                  | 4 => Thursday
                  | 5 => Friday
                  | 6 => Saturday
                  | _ => start
  | Monday    => match n % 7 with
                  | 0 => Monday
                  | 1 => Tuesday
                  | 2 => Wednesday
                  | 3 => Thursday
                  | 4 => Friday
                  | 5 => Saturday
                  | 6 => Sunday
                  | _ => start
  | Tuesday   => match n % 7 with
                  | 0 => Tuesday
                  | 1 => Wednesday
                  | 2 => Thursday
                  | 3 => Friday
                  | 4 => Saturday
                  | 5 => Sunday
                  | 6 => Monday
                  | _ => start
  | Wednesday => match n % 7 with
                  | 0 => Wednesday
                  | 1 => Thursday
                  | 2 => Friday
                  | 3 => Saturday
                  | 4 => Sunday
                  | 5 => Monday
                  | 6 => Tuesday
                  | _ => start
  | Thursday  => match n % 7 with
                  | 0 => Thursday
                  | 1 => Friday
                  | 2 => Saturday
                  | 3 => Sunday
                  | 4 => Monday
                  | 5 => Tuesday
                  | 6 => Wednesday
                  | _ => start
  | Friday    => match n % 7 with
                  | 0 => Friday
                  | 1 => Saturday
                  | 2 => Sunday
                  | 3 => Monday
                  | 4 => Tuesday
                  | 5 => Wednesday
                  | 6 => Thursday
                  | _ => start
  | Saturday  => match n % 7 with
                  | 0 => Saturday
                  | 1 => Sunday
                  | 2 => Monday
                  | 3 => Tuesday
                  | 4 => Wednesday
                  | 5 => Thursday
                  | 6 => Friday
                  | _ => start

-- The proof problem as a Lean theorem
theorem day_100_days_from_friday : dayOfWeekAfter Friday 100 = Sunday := by
  -- Proof will go here
  sorry

end day_100_days_from_friday_l383_383396


namespace distinct_sum_representation_max_l383_383967

noncomputable def A (n : ℕ) : ℕ :=
  (Real.sqrt (8 * n + 1) - 1) / 2 |> Int.floor

theorem distinct_sum_representation_max (n : ℕ) (h : n > 2) :
  ∃ m : ℕ, ∃ (x : Fin m → ℕ) (h_distinct : Function.Injective x),
    (∑ i in Finset.range m, x i = n) ∧ m = A n :=
sorry

end distinct_sum_representation_max_l383_383967


namespace find_omega_l383_383260

theorem find_omega (ω : ℝ) (h_omega_pos : ω > 0) (h_period : ∃ T > 0, ∀ x, x + T = x ∨ y x = y (x + T) ∧ T = 4 * Real.pi) :
  ω = 1 / 4 :=
by
  -- Proof goes here; skipping with sorry.
  sorry

end find_omega_l383_383260


namespace range_of_m_l383_383680

def f (m x : ℝ) : ℝ := 2 * m * x^2 - 2 * (4 - m) * x + 1
def g (m x : ℝ) : ℝ := m * x

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f m x > 0 ∨ g m x > 0) ↔ 0 < m ∧ m < 8 :=
by
  sorry

end range_of_m_l383_383680


namespace problem_b_m_value_l383_383895

theorem problem_b_m_value (m : ℤ) :
  m > 2 → (∃! 119 pairs (x y : ℝ) : 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧
                              x + (m:ℝ) * y ∈ ℤ ∧ (m:ℝ) * x + y ∈ ℤ) ∧ m = 11 :=
by
  sorry

end problem_b_m_value_l383_383895


namespace prime_mod_condition_l383_383652

theorem prime_mod_condition (p : ℕ) (hp : Prime p) :
  (∃ x₀ : ℤ, p ∣ (x₀^2 - x₀ + 3)) ↔ (∃ y₀ : ℤ, p ∣ (y₀^2 - y₀ + 25)) :=
by 
  sorry

end prime_mod_condition_l383_383652


namespace non_congruent_triangles_proof_l383_383703

noncomputable def non_congruent_triangles_count : ℕ :=
  let points := [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)]
  9

theorem non_congruent_triangles_proof :
  non_congruent_triangles_count = 9 :=
sorry

end non_congruent_triangles_proof_l383_383703


namespace find_quadratic_polynomial_l383_383180

noncomputable def polynomial_with_root (x : ℂ) : ℂ :=
  2 * (x - (3 + complex.I)) * (x - (3 - complex.I))

theorem find_quadratic_polynomial :
  (∀ x : ℝ, polynomial_with_root x = 2 * x^2 - 12 * x + 20) :=
begin
  intros x,
  dsimp [polynomial_with_root],
  rw [← complex.of_real_re (x - 3 : ℂ), ← complex.of_real_im (x - 3 : ℂ)],
  simp only [complex.conj_apply, complex.ext_iff, complex.add_im, complex.add_re, complex.coe_add, complex.sub_re, complex.sub_im, complex.of_real_re, complex.of_real_im, complex.I_re, complex.I_im, complex.sub_conj, complex.coe_sub, complex.re_add, complex.add_sub_assoc, complex.re_sub, complex.of_real_sub, complex.im_add, complex.add_assoc, complex.im_complex, complex.mul_re, complex.mul_im, complex.of_real_mul, complex.zero_sub, complex.one_mul, complex.sub_mul, complex.pow_two, number.zero_mul, one_zero, sub_one, of_nat_two],
  sorry,
end

end find_quadratic_polynomial_l383_383180


namespace einstein_birthday_l383_383348

theorem einstein_birthday : 
  (150th_anniversary_day : ∀ 2024-03-14, day_of_week 2024-03-14 = thursday) →
  (leap_year_rule_1 : ∀ n : ℕ, (n % 4 = 0) → leap_year n) →
  (leap_year_rule_2 : ∀ n : ℕ, (n % 100 = 0) ∧ (n % 400 ≠ 0) → ¬ leap_year n) →
  (leap_year_rule_3 : ∀ n : ℕ, (n % 400 = 0) → leap_year n) →
  (calculates_leap_years : ∀ (years : list ℕ), number_of_leap_years years 37) →
  (calculates_non_leap_years : ∀ (years : list ℕ), number_of_non_leap_years years 113) →
  (days_difference_mod_7 : ∀ total_days 187, total_days % 7 = 5) →
  ∃ t: string, t = "Saturday"
:= sorry

end einstein_birthday_l383_383348


namespace necessary_but_not_sufficient_l383_383432

theorem necessary_but_not_sufficient (a b x y : ℤ) (ha : 0 < a) (hb : 0 < b) (h1 : x - y > a + b) (h2 : x * y > a * b) : 
  (x > a ∧ y > b) := sorry

end necessary_but_not_sufficient_l383_383432


namespace inequality_transform_l383_383714

variable {x y : ℝ}

theorem inequality_transform (h : x < y) : - (x / 2) > - (y / 2) :=
sorry

end inequality_transform_l383_383714


namespace planted_area_fraction_l383_383968

theorem planted_area_fraction (a b : ℕ) (c : ℝ) (frac : ℚ) :
  a = 5 ∧ b = 12 ∧ c = 3 ∧ frac = 11405 / 12005 →
  let area_triangle := (a * b) / 2 in
  let area_square := (c * 3) / 2 in
  let planted_area := area_triangle - area_square in
  planted_area / area_triangle = frac := 
begin
  intros,
  sorry
end

end planted_area_fraction_l383_383968


namespace millimeters_of_78_74_inches_l383_383234

noncomputable def inchesToMillimeters (inches : ℝ) : ℝ :=
  inches * 25.4

theorem millimeters_of_78_74_inches :
  round (inchesToMillimeters 78.74) = 2000 :=
by
  -- This theorem should assert that converting 78.74 inches to millimeters and rounding to the nearest millimeter equals 2000
  sorry

end millimeters_of_78_74_inches_l383_383234


namespace complex_modulus_l383_383211

theorem complex_modulus (z : ℂ) (h : z - 2 * complex.I = z * complex.I) : complex.abs z = Real.sqrt 2 := 
  sorry

end complex_modulus_l383_383211


namespace digit_y_in_base_7_divisible_by_19_l383_383600

def base7_to_decimal (a b c d : ℕ) : ℕ := a * 7^3 + b * 7^2 + c * 7 + d

theorem digit_y_in_base_7_divisible_by_19 (y : ℕ) (hy : y < 7) :
  (∃ k : ℕ, base7_to_decimal 5 2 y 3 = 19 * k) ↔ y = 8 :=
by {
  sorry
}

end digit_y_in_base_7_divisible_by_19_l383_383600


namespace line_parallel_to_plane_l383_383996

theorem line_parallel_to_plane 
  {α β : Plane} {m : Line} 
  (h3 : m ⊆ α) (h5 : α ∥ β) : m ∥ β := 
by 
  sorry

end line_parallel_to_plane_l383_383996


namespace tony_total_puzzle_time_l383_383381

def warm_up_puzzle_time : ℕ := 10
def number_of_puzzles : ℕ := 2
def multiplier : ℕ := 3
def time_per_puzzle : ℕ := warm_up_puzzle_time * multiplier
def total_time : ℕ := warm_up_puzzle_time + number_of_puzzles * time_per_puzzle

theorem tony_total_puzzle_time : total_time = 70 := 
by
  sorry

end tony_total_puzzle_time_l383_383381


namespace sum_of_possible_values_sum_of_values_l383_383792

theorem sum_of_possible_values (x y : ℝ)
  (h : x * y - 2 * x / y^2 - 2 * y / x^2 = 4) :
  (x - 2) * (y - 2) = 2 ∨ (x - 2) * (y - 2) = (2 + Real.sqrt 2)^2 :=
begin
  -- Condition hypotheses
  sorry
end

theorem sum_of_values (x y : ℝ)
  (h : x * y - 2 * x / y^2 - 2 * y / x^2 = 4) :
  (x - 2) * (y - 2) = 2 + (2 + Real.sqrt 2)^2 :=
begin
  sorry
end

end sum_of_possible_values_sum_of_values_l383_383792


namespace possible_card_arrangement_l383_383058

theorem possible_card_arrangement (n : ℕ) (h : 1 ≤ n ∧ n ≤ 10) :
  (n = 3 ∨ n = 4 ∨ n = 7 ∨ n = 8) ↔ 
  (∃ (S : ℕ) (y_i : ℕ → ℕ), 
    (S = 2 * ∑ i in finset.range(n+1), i + ∑ i in finset.range(2*n + 1) \ {n+1}, i) ∧
    (S = ∑ i in finset.range(n), (3*i + 2*y_i i)) ∧
    (4 ∣ n * (n+1))) :=
sorry

end possible_card_arrangement_l383_383058


namespace find_fourth_point_l383_383753

-- Define the equations and given points
def curve (x y : ℝ) : Prop := x * y = 2

def given_points := [(4, 1/2), (-2, -1), (2/3, 3)]

-- Define the unknown fourth point
def fourth_point (p : ℝ × ℝ) : Prop :=
  let x := p.fst
  let y := p.snd in
  curve x y ∧
  (4 * -2 * (2/3) * x = 4) ∧
  y = 2 / x

theorem find_fourth_point : ∃ p : ℝ × ℝ, fourth_point p :=
by
  use (-1/2, -4)
  unfold fourth_point curve
  simp
  split
  case left => exact (by norm_num : (-1/2) * (-4) = 2)
  norm_num

end find_fourth_point_l383_383753


namespace hypotenuse_length_l383_383490

-- Definitions for the problem
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def leg1 : ℕ := 15
def leg2 : ℕ := 36
def hypotenuse : ℕ := 39

-- Lean 4 statement
theorem hypotenuse_length (a b c : ℕ) (h : is_right_triangle a b c) (ha : a = leg1) (hb : b = leg2) :
  c = hypotenuse :=
begin
  sorry
end

end hypotenuse_length_l383_383490


namespace cos_alpha_plus_pi_over_3_l383_383210

theorem cos_alpha_plus_pi_over_3 (α : ℝ) (h1 : sin α = (4 * real.sqrt 3) / 7) (h2 : 0 < α ∧ α < real.pi / 2) : 
  cos (α + real.pi / 3) = -11 / 14 :=
by
  sorry

end cos_alpha_plus_pi_over_3_l383_383210


namespace fraction_product_is_243_l383_383412

/- Define the sequence of fractions -/
def fraction_seq : list ℚ :=
  [1/3, 9, 1/27, 81, 1/243, 243, 1/729, 729, 1/2187, 6561]

/- Define the product of the sequence of fractions -/
def product_fractions : ℚ :=
  (fraction_seq.foldl (*) 1)

/- The theorem we want to prove -/
theorem fraction_product_is_243 : product_fractions = 243 := 
  sorry

end fraction_product_is_243_l383_383412


namespace interval_first_bell_l383_383141

theorem interval_first_bell (x : ℕ) : (Nat.lcm (Nat.lcm (Nat.lcm x 10) 14) 18 = 630) → x = 1 := by
  sorry

end interval_first_bell_l383_383141


namespace cost_of_south_american_stamps_before_90s_is_175_cents_l383_383892

noncomputable def cost_per_stamp : String → ℕ 
| "Brazil" => 7
| "Peru" => 5
| _ => 0

noncomputable def number_of_stamps (decade : String) (country : String) : ℕ 
| "50s", "Brazil" => 6
| "50s", "Peru" => 8
| "60s", "Brazil" => 9
| "60s", "Peru" => 6
| _, _ => 0

noncomputable def total_cost_before_90s : ℕ :=
  (number_of_stamps "50s" "Brazil" * cost_per_stamp "Brazil" +
   number_of_stamps "60s" "Brazil" * cost_per_stamp "Brazil" +
   number_of_stamps "50s" "Peru" * cost_per_stamp "Peru" +
   number_of_stamps "60s" "Peru" * cost_per_stamp "Peru") / 100

theorem cost_of_south_american_stamps_before_90s_is_175_cents :
  total_cost_before_90s = 175 :=
by
  sorry

end cost_of_south_american_stamps_before_90s_is_175_cents_l383_383892


namespace sandra_pencils_l383_383329

theorem sandra_pencils :
  ∀ (a1 a2 a3 a5 a4 : ℕ),
    a1 = 78 →
    a2 = 87 →
    a3 = 96 →
    a5 = 114 →
    (a2 - a1) = 9 →
    (a3 - a2) = 9 →
    (a5 - a3 - 18) = 0 →  -- Ensuring the pattern holds for a5
    a4 = a3 + 9 →
    a4 = 105 :=
by
  intros a1 a2 a3 a5 a4 h1 h2 h3 h5 pattern12 pattern23 pattern5to3 pattern34
  rw [h1, h2, h3, h5] at pattern12 pattern23 pattern5to3
  have h6 : 87 - 78 = 9 := by rfl
  have h7 : 96 - 87 = 9 := by rfl
  have h8 : 114 - 96 - 18 = 0 := by rfl
  have h9 : 105 = 96 + 9 := by rfl
  rw [←pattern34]
  exact h9

end sandra_pencils_l383_383329


namespace hypotenuse_length_l383_383482

-- Definitions for the problem
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def leg1 : ℕ := 15
def leg2 : ℕ := 36
def hypotenuse : ℕ := 39

-- Lean 4 statement
theorem hypotenuse_length (a b c : ℕ) (h : is_right_triangle a b c) (ha : a = leg1) (hb : b = leg2) :
  c = hypotenuse :=
begin
  sorry
end

end hypotenuse_length_l383_383482


namespace students_taking_art_l383_383447

def total_students : ℕ := 500
def students_taking_music : ℕ := 20
def students_taking_both : ℕ := 10
def students_taking_neither : ℕ := 470

theorem students_taking_art :
  ∃ (A : ℕ), A = 20 ∧ total_students = 
             (students_taking_music - students_taking_both) + (A - students_taking_both) + students_taking_both + students_taking_neither :=
by
  sorry

end students_taking_art_l383_383447


namespace average_speed_correct_l383_383444

noncomputable def total_distance : ℝ := 120 + 180 + 75

def speed_segment1 : ℝ := 80
def distance_segment1 : ℝ := 120
def time_segment1 : ℝ := distance_segment1 / speed_segment1

def stop_time1 : ℝ := 15 / 60

def speed_segment2 : ℝ := 100
def distance_segment2 : ℝ := 180
def time_segment2 : ℝ := distance_segment2 / speed_segment2
def total_time_segment2 : ℝ := time_segment2 + stop_time1

def stop_time2 : ℝ := 10 / 60

def speed_segment3 : ℝ := 120
def distance_segment3 : ℝ := 75
def time_segment3 : ℝ := distance_segment3 / speed_segment3
def total_time_segment3 : ℝ := time_segment3 + stop_time2

noncomputable def total_time : ℝ := time_segment1 + total_time_segment2 + total_time_segment3

noncomputable def average_speed : ℝ := total_distance / total_time

theorem average_speed_correct : abs (average_speed - 86.37) < 0.01 := 
by sorry

end average_speed_correct_l383_383444


namespace oranges_kilos_bought_l383_383724

-- Definitions based on the given conditions
variable (O A x : ℝ)

-- Definitions from conditions
def A_value : Prop := A = 29
def equation1 : Prop := x * O + 5 * A = 419
def equation2 : Prop := 5 * O + 7 * A = 488

-- The theorem we want to prove
theorem oranges_kilos_bought {O A x : ℝ} (A_value: A = 29) (h1: x * O + 5 * A = 419) (h2: 5 * O + 7 * A = 488) : x = 5 :=
by
  -- start of proof
  sorry  -- proof omitted

end oranges_kilos_bought_l383_383724


namespace remainder_when_divided_l383_383632
-- First, import the necessary library.

-- Define the problem conditions and the goal.
theorem remainder_when_divided (P Q Q' R R' S T D D' D'' : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = D' * Q' + D'' * R' + R')
  (h3 : S = D'' * T)
  (h4 : R' = S + T) :
  P % (D * D' * D'') = D * R' + R := by
  sorry

end remainder_when_divided_l383_383632


namespace faith_change_l383_383164

def flour_cost : ℕ := 5
def cake_stand_cost : ℕ := 28
def given_amount : ℕ := 20 * 2 + 3
def total_cost := flour_cost + cake_stand_cost
def change := given_amount - total_cost

theorem faith_change : change = 10 :=
by
  -- the proof goes here
  sorry

end faith_change_l383_383164


namespace distinguishable_large_triangles_l383_383346

theorem distinguishable_large_triangles (num_colors : ℕ) (red_color : bool)
  (distinct_arrangements : ℕ → ℕ → ℕ)
  (at_least_one_red : bool → Prop) : 
  num_colors = 8 → 
  red_color = true → 
  distinct_arrangements 3 7 = 21 →
  at_least_one_red true →
  let total_distinguishable (corner_red : ℕ) (corner_other : ℕ) 
    (distinct_complete : ℕ) (central_colors : ℕ) := 
    (corner_red + corner_other + distinct_complete) * central_colors
  in total_distinguishable 1 7 21 8 = 232 :=
by
  intros,
  let total_distinguishable := (1 + 7 + 21) * 8,
  have h : total_distinguishable = 232 := by norm_num,
  exact h

end distinguishable_large_triangles_l383_383346


namespace parabola_value_of_a_l383_383216

noncomputable def parabola_vertex_focus (a : ℝ) : Prop :=
  ∃ (p : ℝ), p > 0 ∧ ∀ x y : ℝ, (y^2 = -2 * p * x) → ∃ A : ℝ × ℝ, A = (-1, a) ∧
  (real.sq (A.1 + p/2) + real.sq A.2 = 4 ∧ real.sq (A.1 + 1 - (-1)) = 4)

theorem parabola_value_of_a (a : ℝ) (h : parabola_vertex_focus a) : a = 2 ∨ a = -2 :=
sorry

end parabola_value_of_a_l383_383216


namespace cinco_de_mayo_day_days_between_feb_14_and_may_5_l383_383962

theorem cinco_de_mayo_day {
  feb_14_is_tuesday : ∃ n : ℕ, n % 7 = 2
}: 
∃ n : ℕ, n % 7 = 5 := sorry

theorem days_between_feb_14_and_may_5: 
  ∃ d : ℕ, 
  d = 81 := sorry

end cinco_de_mayo_day_days_between_feb_14_and_may_5_l383_383962


namespace tiles_not_interchangeable_on_paved_floor_l383_383357

/-- A type to represent the two types of tiles -/
inductive TileType
| TwoByTwo
| OneByFour

/-- A function that represents whether tiles of the two types can be rearranged to replace a broken tile with a new tile of the other type -/
def tiles_not_interchangeable (floor: list TileType) : Prop :=
∀ brokenTile : TileType, 
  ∀ newTile : TileType, 
  brokenTile ≠ newTile → brokenTile ∈ floor → ¬ newTile ∈ floor

theorem tiles_not_interchangeable_on_paved_floor (floor: list TileType) :
  tiles_not_interchangeable floor :=
by
  sorry

end tiles_not_interchangeable_on_paved_floor_l383_383357


namespace Sn_lt_1_In_le_Sn_l383_383793

noncomputable def v : ℕ → ℕ 
| 1 := 2
| 2 := 3
| 3 := 7
| n := (List.prod (List.map v (List.range (n-1)))) + 1

noncomputable def S (n : ℕ) : ℝ := 
(List.sum (List.map (λ k, 1 / (v (k+1) : ℝ)) (List.range n)))

theorem Sn_lt_1 (n : ℕ) : S n < 1 :=
begin
  sorry
end

theorem In_le_Sn {i : ℕ} (I : ℝ) (vinv_set : set ℝ) (h : vinv_set = {x ∈ (λ k, 1 / (v (k+1) : ℝ))} ∧ I ∈ vinv_set ∧ I < 1) : 
I ≤ S i ∧ S i < 1 :=
begin
  sorry
end

end Sn_lt_1_In_le_Sn_l383_383793


namespace bus_stop_time_l383_383427

theorem bus_stop_time (speed_exclusive_stoppage : ℝ) (speed_inclusive_stoppage : ℝ) (h1: speed_exclusive_stoppage = 86) (h2 : speed_inclusive_stoppage = 76) : 
  let time_stopped := (10 / (speed_exclusive_stoppage / 60)) in 
  time_stopped ≈ 6.98 :=
by 
  have eq10 : 10 = speed_exclusive_stoppage - speed_inclusive_stoppage, from sorry,
  have time_expr : time_stopped = 10 * (60 / speed_exclusive_stoppage), from sorry,
  have approximation : time_stopped ≈ 6.98, from sorry,
  exact approximation

end bus_stop_time_l383_383427


namespace exam_students_count_l383_383888

theorem exam_students_count (n : ℕ) (T : ℕ) (h1 : T = 90 * n) 
                            (h2 : (T - 90) / (n - 2) = 95) : n = 20 :=
by {
  sorry
}

end exam_students_count_l383_383888


namespace pastry_problem_minimum_n_l383_383069

theorem pastry_problem_minimum_n (fillings : Finset ℕ) (n : ℕ) : 
    fillings.card = 10 →
    (∃ pairs : Finset (ℕ × ℕ), pairs.card = 45 ∧ ∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ≠ p.2 ∧ p.1 ∈ fillings ∧ p.2 ∈ fillings) →
    (∀ (remaining_pies : Finset (ℕ × ℕ)), remaining_pies.card = 45 - n → 
     ∃ f1 f2, (f1, f2) ∈ remaining_pies → (f1 ∈ fillings ∧ f2 ∈ fillings)) →
    n = 36 :=
by
  intros h_fillings h_pairs h_remaining_pies
  sorry

end pastry_problem_minimum_n_l383_383069


namespace sqrt_one_div_four_is_one_div_two_l383_383846

theorem sqrt_one_div_four_is_one_div_two : Real.sqrt (1 / 4) = 1 / 2 :=
by
  sorry

end sqrt_one_div_four_is_one_div_two_l383_383846


namespace intersecting_planes_l383_383241

variable (α β : Plane)
variable (m : Line)

theorem intersecting_planes (h_intersect : Intersect α β) :
  ( (m ⊥ α → ∃ ! l, l ∈ β ∧ l ⊥ m) ∧
    (m ∈ α → ¬∃ l, l ∈ β ∧ l ⊥ m) ∧
    (m ∈ α → ∃ l, l ∈ β ∧ l ⊥ m) ) :=
by sorry

end intersecting_planes_l383_383241


namespace log_sixteen_ninetytwo_l383_383610

theorem log_sixteen_ninetytwo : ∀ (a b : ℝ),
  a = (6: ℝ)^(1/2) →
  b = (6: ℝ)^4 →
  log a b = 8 :=
by
  sorry

end log_sixteen_ninetytwo_l383_383610


namespace right_triangle_hypotenuse_l383_383516

theorem right_triangle_hypotenuse (a b : ℕ) (ha : a = 15) (hb : b = 36) : 
  ∃ h : ℕ, h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  . exact rfl
  . rw [ha, hb]
    norm_num
    sorry

end right_triangle_hypotenuse_l383_383516


namespace find_abc_l383_383456

noncomputable def x (t : ℝ) := 3 * Real.cos t - 2 * Real.sin t
noncomputable def y (t : ℝ) := 3 * Real.sin t

theorem find_abc :
  ∃ a b c : ℝ, 
  (a = 1/9) ∧ 
  (b = 4/27) ∧ 
  (c = 5/27) ∧ 
  (∀ t : ℝ, a * (x t)^2 + b * (x t) * (y t) + c * (y t)^2 = 1) :=
by
  sorry

end find_abc_l383_383456


namespace maximize_f_l383_383577

open Nat

-- Define the combination function
def comb (n k : ℕ) : ℕ := choose n k

-- Define the probability function f(n)
def f (n : ℕ) : ℚ := 
  (comb n 2 * comb (100 - n) 8 : ℚ) / comb 100 10

-- Define the theorem to find the value of n that maximizes f(n)
theorem maximize_f : ∃ n : ℕ, 2 ≤ n ∧ n ≤ 92 ∧ (∀ m : ℕ, 2 ≤ m ∧ m ≤ 92 → f n ≥ f m) ∧ n = 20 :=
by
  sorry

end maximize_f_l383_383577


namespace probability_same_university_l383_383262

theorem probability_same_university :
  let universities := 5
  let total_ways := universities * universities
  let favorable_ways := universities
  (favorable_ways : ℚ) / total_ways = (1 / 5 : ℚ) := 
by
  sorry

end probability_same_university_l383_383262


namespace base_conversion_sum_l383_383163

def A := 10
def B := 11

def convert_base11_to_base10 (n : ℕ) : ℕ :=
  let d2 := n / 11^2
  let d1 := (n % 11^2) / 11
  let d0 := n % 11
  d2 * 11^2 + d1 * 11 + d0

def convert_base12_to_base10 (n : ℕ) : ℕ :=
  let d2 := n / 12^2
  let d1 := (n % 12^2) / 12
  let d0 := n % 12
  d2 * 12^2 + d1 * 12 + d0

def n1 := 2 * 11^2 + 4 * 11 + 9    -- = 249_11 in base 10
def n2 := 3 * 12^2 + A * 12 + B   -- = 3AB_12 in base 10

theorem base_conversion_sum :
  (convert_base11_to_base10 294 + convert_base12_to_base10 563 = 858) := by
  sorry

end base_conversion_sum_l383_383163


namespace Z_real_iff_Z_pure_imaginary_iff_Z_second_quadrant_iff_l383_383796

def Z (m : ℝ) : ℂ := ⟨m^2 - 2m - 3, m^2 + 3m + 2⟩

-- I: Z is real if and only if m = -1 or m = -2
theorem Z_real_iff (m : ℝ) : (Z m).im = 0 ↔ m = -1 ∨ m = -2 := by
  sorry

-- II: Z is pure imaginary if and only if m = 3
theorem Z_pure_imaginary_iff (m : ℝ) : (Z m).re = 0 ↔ m = 3 := by
  sorry

-- III: Z is in the second quadrant if and only if -1 < m < 3
theorem Z_second_quadrant_iff (m : ℝ) : (Z m).re < 0 ∧ (Z m).im > 0 ↔ -1 < m ∧ m < 3 := by
  sorry

end Z_real_iff_Z_pure_imaginary_iff_Z_second_quadrant_iff_l383_383796


namespace trick_proof_l383_383065

-- Defining the number of fillings and total pastries based on combinations
def num_fillings := 10

def total_pastries : ℕ := (num_fillings * (num_fillings - 1)) / 2

-- Definition stating that the smallest number of pastries n such that Vasya can always determine at least one filling of any remaining pastry
def min_n := 36

-- The theorem stating the proof problem
theorem trick_proof (n m: ℕ) (h1: n = 10) (h2: m = (n * (n - 1)) / 2) : min_n = 36 :=
by
  sorry

end trick_proof_l383_383065


namespace number_of_observations_l383_383840

/-- 
Given:
  (1) The initial mean of the observations is 36.
  (2) One observation was incorrectly recorded as 23 instead of 48.
  (3) The corrected mean of the observations is 36.5.
Prove:
  The number of observations is 50.
-/
theorem number_of_observations (n : ℝ) 
  (initial_mean : n * 36) 
  (error_correction : n * 36 + (48 - 23)) 
  (corrected_mean : n * 36.5) 
  (eq_condition : n * 36 + 25 = n * 36.5) 
  : n = 50 := 
  sorry

end number_of_observations_l383_383840


namespace ellipse_equation_l383_383126

theorem ellipse_equation (
  center_origin : ∃ O : ℝ × ℝ, O = (0, 0),
  foci_on_x_axis : ∃ F1 F2 : ℝ × ℝ, (F1 = (c, 0)) ∧ (F2 = (-c, 0)),
  point_on_ellipse : ∃ P : ℝ × ℝ, P = (2, real.sqrt 3) ∧ ((2 * 2 + real.sqrt 3 * real.sqrt 3) / (a * a) = 1),
  distances_arithmetic_sequence : ∃ PF1 PF2 F1F2 : ℝ, (PF1 + PF2 = 4 * c) ∧ (PF1 + PF2 = 2 * F1F2)
) : (∃ a b : ℝ, (a^2 = 8) ∧ (b^2 = 6) ∧ ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) :=
sorry

end ellipse_equation_l383_383126


namespace gecko_eggs_hatch_l383_383908

theorem gecko_eggs_hatch : 
    (total_eggs infertile_percent calcification_fraction : ℕ) (total_eggs = 30) 
    (infertile_percent = 20) (calcification_fraction = 3) → 
    (total_eggs * (100 - infertile_percent) / 100 * (calcification_fraction - 1) / calcification_fraction = 16) :=
by
    intros
    have infertile_eggs := total_eggs * infertile_percent / 100
    have fertile_eggs := total_eggs - infertile_eggs
    have unhatched_eggs := fertile_eggs / calcification_fraction
    have hatched_eggs := fertile_eggs - unhatched_eggs
    show hatched_eggs = 16
    sorry

end gecko_eggs_hatch_l383_383908


namespace missing_number_odd_l383_383694

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

def possible_sums (a : List ℕ) (x : ℕ) : List ℕ := a.map (λ n, n + x)

def count_even_odd (l : List ℕ) : ℕ × ℕ :=
  l.foldl (λ (p : ℕ × ℕ) (n : ℕ), if is_even n then (p.1 + 1, p.2) else (p.1, p.2 + 1)) (0, 0)

theorem missing_number_odd (x : ℕ) :
  let a := [11, 44, 55] in
  count_even_odd (possible_sums a x) = (1, 2) ∨ count_even_odd (possible_sums a x) = (2, 1) →
  is_odd x :=
by {
  let a := [11, 44, 55],
  assume h : count_even_odd (possible_sums a x) = (1, 2) ∨ count_even_odd (possible_sums a x) = (2, 1),
  sorry
}

end missing_number_odd_l383_383694


namespace trick_proof_l383_383063

-- Defining the number of fillings and total pastries based on combinations
def num_fillings := 10

def total_pastries : ℕ := (num_fillings * (num_fillings - 1)) / 2

-- Definition stating that the smallest number of pastries n such that Vasya can always determine at least one filling of any remaining pastry
def min_n := 36

-- The theorem stating the proof problem
theorem trick_proof (n m: ℕ) (h1: n = 10) (h2: m = (n * (n - 1)) / 2) : min_n = 36 :=
by
  sorry

end trick_proof_l383_383063


namespace right_triangle_hypotenuse_length_l383_383503

theorem right_triangle_hypotenuse_length (a b : ℕ) (h1 : a = 15) (h2 : b = 36) : 
  ∃ c : ℕ, c * c = a * a + b * b ∧ c = 39 := 
by
  have hyp_square := 225 + 1296 
  have h_calculation : 15 * 15 + 36 * 36 = 1521 := by
    calc
      15 * 15 = 225 : rfl
      36 * 36 = 1296 : rfl
      225 + 1296 = 1521 : rfl
  use 39
  split
  exact h_calculation
  rfl

end right_triangle_hypotenuse_length_l383_383503


namespace pencils_difference_l383_383769

theorem pencils_difference
  (pencils_total : ℕ)
  (pencils_given_manny : ℕ)
  (pencils_kept: ℕ)
  (pencils_total = 50)
  (pencils_given_manny = 10)
  (pencils_kept = 20) :
  (pencils_total - pencils_kept - pencils_given_manny) - pencils_given_manny = 10 :=
by
  sorry

end pencils_difference_l383_383769


namespace simplify_expression_l383_383196

theorem simplify_expression (a b : ℝ) (h1 : 2 * b - a < 3) (h2 : 2 * a - b < 5) : 
  -abs (2 * b - a - 7) - abs (b - 2 * a + 8) + abs (a + b - 9) = -6 :=
by
  sorry

end simplify_expression_l383_383196


namespace max_number_of_books_laughlin_can_buy_l383_383289

-- Definitions of costs and the budget constraint
def individual_book_cost : ℕ := 3
def four_book_bundle_cost : ℕ := 10
def seven_book_bundle_cost : ℕ := 15
def budget : ℕ := 20

-- Condition that Laughlin must buy at least one 4-book bundle
def minimum_required_four_book_bundles : ℕ := 1

-- Define the function to calculate the maximum number of books Laughlin can buy
def max_books (budget : ℕ) (individual_book_cost : ℕ) 
              (four_book_bundle_cost : ℕ) (seven_book_bundle_cost : ℕ) 
              (min_four_book_bundles : ℕ) : ℕ :=
  let remaining_budget_after_four_bundle := budget - (min_four_book_bundles * four_book_bundle_cost)
  if remaining_budget_after_four_bundle >= seven_book_bundle_cost then
    min_four_book_bundles * 4 + 7
  else if remaining_budget_after_four_bundle >= individual_book_cost then
    min_four_book_bundles * 4 + remaining_budget_after_four_bundle / individual_book_cost
  else
    min_four_book_bundles * 4

-- Proof statement: Laughlin can buy a maximum of 7 books
theorem max_number_of_books_laughlin_can_buy : 
  max_books budget individual_book_cost four_book_bundle_cost seven_book_bundle_cost minimum_required_four_book_bundles = 7 :=
by
  sorry

end max_number_of_books_laughlin_can_buy_l383_383289


namespace calculate_triangle_sides_l383_383856

open Real

noncomputable def triangle_sides (R r : ℝ) : ℝ × ℝ × ℝ :=
  if h : R > 0 ∧ r > 0 then
    let AB := 2 * sqrt (R * r)
    let BC := 2 * R * sqrt (r / (R + r))
    let AC := 2 * r * sqrt (R / (R + r))
    (AB, BC, AC)
  else
    (0, 0, 0)

theorem calculate_triangle_sides 
  (R r : ℝ) (hR : R > 0) (hr : r > 0) :
  triangle_sides R r =
  (2 * sqrt (R * r), 
   2 * R * sqrt (r / (R + r)), 
   2 * r * sqrt (R / (R + r))) :=
by
  rw [triangle_sides]
  split_ifs
  · rfl
  · exfalso
    apply h
    exact ⟨hR, hr⟩

end calculate_triangle_sides_l383_383856


namespace bacteria_growth_time_l383_383902

theorem bacteria_growth_time : 
  (∀ n : ℕ, 2 ^ n = 4096 → (n * 15) / 60 = 3) :=
by
  sorry

end bacteria_growth_time_l383_383902


namespace lg_values_count_l383_383635

def different_lg_values (s : Finset ℕ) : Finset ℤ :=
  (s.product s).filter (λ p, p.1 ≠ p.2).image (λ p, p.1 - p.2)

theorem lg_values_count (a b : Finset ℕ) (h : ∀ x ∈ a ∪ b, x ∈ {1, 3, 5, 7, 9}):
  different_lg_values (Finset.of_list [1, 3, 5, 7, 9]).card = 10 :=
by {
  sorry
}

end lg_values_count_l383_383635


namespace exists_arithmetic_sequence_l383_383323

theorem exists_arithmetic_sequence (k : ℕ) (hk : k > 0) :
  ∃ (a b : ℕ → ℕ), (∀ i, 1 ≤ i ∧ i ≤ k → a i < b i ∧ Nat.coprime (a i) (b i)) ∧
    (∀ i, 1 ≤ i ∧ i ≤ k → ∃ j, j ≤ k ∧ (i ≠ j → a i ≠ a j ∧ b i ≠ b j)) ∧
    (∀ i, 1 ≤ i ∧ i < k → a (i+1) - a i = b (i+1) - b i) :=
by
  sorry

end exists_arithmetic_sequence_l383_383323


namespace water_channel_area_l383_383352

-- Define the given conditions
def top_width := 14
def bottom_width := 8
def depth := 70

-- The area formula for a trapezium given the top width, bottom width, and height
def trapezium_area (a b h : ℕ) : ℕ :=
  (a + b) * h / 2

-- The main theorem stating the area of the trapezium
theorem water_channel_area : 
  trapezium_area top_width bottom_width depth = 770 := by
  -- Proof can be completed here
  sorry

end water_channel_area_l383_383352


namespace trick_proof_l383_383062

-- Defining the number of fillings and total pastries based on combinations
def num_fillings := 10

def total_pastries : ℕ := (num_fillings * (num_fillings - 1)) / 2

-- Definition stating that the smallest number of pastries n such that Vasya can always determine at least one filling of any remaining pastry
def min_n := 36

-- The theorem stating the proof problem
theorem trick_proof (n m: ℕ) (h1: n = 10) (h2: m = (n * (n - 1)) / 2) : min_n = 36 :=
by
  sorry

end trick_proof_l383_383062


namespace problem1_problem2_l383_383893

noncomputable section

variable {a b c : ℝ}

-- Proof for the first problem
theorem problem1 (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_neq : a ≠ b ∨ b ≠ c ∨ c ≠ a) :
  a * (b^2 + c^2) + b * (c^2 + a^2) + c * (a^2 + b^2) > 6 * a * b * c :=
by sorry

-- Proof for the second problem
theorem problem2 :
  sqrt 6 + sqrt 7 > 2 * sqrt 2 + sqrt 5 :=
by sorry

end problem1_problem2_l383_383893


namespace molecular_weight_of_8_moles_fe2o3_l383_383018

namespace Chem

def atomic_weight_fe : Float := 55.845
def atomic_weight_o : Float := 15.999

def molecular_weight_fe2o3 : Float :=
  2 * atomic_weight_fe + 3 * atomic_weight_o

def molecular_weight_8_moles_fe2o3 : Float :=
  8 * molecular_weight_fe2o3

theorem molecular_weight_of_8_moles_fe2o3 :
  molecular_weight_8_moles_fe2o3 = 1277.496 := by
  -- We acknowledge the necessity of conducting the computation:
  have h1 : molecular_weight_fe2o3 = 2 * 55.845 + 3 * 15.999 := by
    unfold atomic_weight_fe atomic_weight_o
    unfold molecular_weight_fe2o3
    norm_num1 /- 111.69 + 47.997 -/ sorry
  norm_num1 -- We assert:
  show 8 * (2 * 55.845 + 3 * 15.999) = 1277.496 from sorry

end Chem

end molecular_weight_of_8_moles_fe2o3_l383_383018


namespace hypotenuse_length_l383_383480

theorem hypotenuse_length (a b c : ℕ) (h₀ : a = 15) (h₁ : b = 36) (h₂ : a^2 + b^2 = c^2) : c = 39 :=
by
  -- Proof is omitted
  sorry

end hypotenuse_length_l383_383480


namespace right_triangle_hypotenuse_length_l383_383502

theorem right_triangle_hypotenuse_length (a b : ℕ) (h1 : a = 15) (h2 : b = 36) : 
  ∃ c : ℕ, c * c = a * a + b * b ∧ c = 39 := 
by
  have hyp_square := 225 + 1296 
  have h_calculation : 15 * 15 + 36 * 36 = 1521 := by
    calc
      15 * 15 = 225 : rfl
      36 * 36 = 1296 : rfl
      225 + 1296 = 1521 : rfl
  use 39
  split
  exact h_calculation
  rfl

end right_triangle_hypotenuse_length_l383_383502


namespace power_of_point_l383_383776

open EuclideanGeometry

theorem power_of_point (Γ : Circle) (P A B C D : Point) 
    (hP_outside : P ∉ Γ.points) 
    (hAB : lies_on A Γ ∧ lies_on B Γ) 
    (hCD : lies_on C Γ ∧ lies_on D Γ) 
    (hPA : collinear P A) 
    (hPB : collinear P B) 
    (hPC : collinear P C)
    (hPD : collinear P D) : 
    (dist P A) * (dist P B) = (dist P C) * (dist P D) :=
sorry

end power_of_point_l383_383776


namespace find_n_l383_383251

theorem find_n (n : ℤ) (h : (n + 1999) / 2 = -1) : n = -2001 := 
sorry

end find_n_l383_383251


namespace largest_among_four_numbers_l383_383928

theorem largest_among_four_numbers
  (a b : ℝ)
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : a + b = 1) :
  b > max (max (1/2) (2 * a * b)) (a^2 + b^2) := 
sorry

end largest_among_four_numbers_l383_383928


namespace six_digit_numbers_with_at_least_one_even_digit_l383_383240

def range_six_digit_numbers : List ℕ := List.range' 100000 900000

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def all_odd_digits (n : ℕ) : Prop := 
  (n.digits 10).all is_odd_digit

def six_digit_with_odd_digits : ℕ := range_six_digit_numbers.filter all_odd_digits |> List.length

theorem six_digit_numbers_with_at_least_one_even_digit :
  List.length range_six_digit_numbers - six_digit_with_odd_digits = 884375 :=
by
  sorry

end six_digit_numbers_with_at_least_one_even_digit_l383_383240


namespace inequality_l383_383433

theorem inequality
  (n : ℕ) (hn : n ≥ 4)
  (a : Fin n → ℝ) (hpos : ∀ i, 0 < a i)
  (hsum : ∑ i, (a i)^2 = 1) :
  (∑ i : Fin n, a i / ((a ((i + 1) % n))^2 + 1)) ≥ 
  (4 / 5) * (∑ i : Fin n, (a i) * Real.sqrt (a i))^2 := 
sorry

end inequality_l383_383433


namespace power_rule_example_l383_383140

variable {R : Type*} [Ring R] (a b : R)

theorem power_rule_example : (a * b^3) ^ 2 = a^2 * b^6 :=
sorry

end power_rule_example_l383_383140


namespace grace_earnings_in_september_l383_383696

theorem grace_earnings_in_september
  (hours_mowing : ℕ) (hours_pulling_weeds : ℕ) (hours_putting_mulch : ℕ)
  (rate_mowing : ℕ) (rate_pulling_weeds : ℕ) (rate_putting_mulch : ℕ)
  (total_hours_mowing : hours_mowing = 63) (total_hours_pulling_weeds : hours_pulling_weeds = 9) (total_hours_putting_mulch : hours_putting_mulch = 10)
  (rate_for_mowing : rate_mowing = 6) (rate_for_pulling_weeds : rate_pulling_weeds = 11) (rate_for_putting_mulch : rate_putting_mulch = 9) :
  hours_mowing * rate_mowing + hours_pulling_weeds * rate_pulling_weeds + hours_putting_mulch * rate_putting_mulch = 567 :=
by
  intros
  sorry

end grace_earnings_in_september_l383_383696


namespace hypotenuse_right_triangle_l383_383548

theorem hypotenuse_right_triangle (a b : ℕ) (h1 : a = 15) (h2 : b = 36) :
  ∃ c, c ^ 2 = a ^ 2 + b ^ 2 ∧ c = 39 :=
by
  sorry

end hypotenuse_right_triangle_l383_383548


namespace incorrect_statement_A_l383_383878

-- Definitions based on conditions
def equilibrium_shifts (condition: Type) : Prop := sorry
def value_K_changes (condition: Type) : Prop := sorry

-- The incorrect statement definition
def statement_A (condition: Type) : Prop := equilibrium_shifts condition → value_K_changes condition

-- The final theorem stating that 'statement_A' is incorrect
theorem incorrect_statement_A (condition: Type) : ¬ statement_A condition :=
sorry

end incorrect_statement_A_l383_383878


namespace age_ratio_in_two_years_l383_383909

variable (S M : ℕ)

-- Conditions
def sonCurrentAge : Prop := S = 18
def manCurrentAge : Prop := M = S + 20
def multipleCondition : Prop := ∃ k : ℕ, M + 2 = k * (S + 2)

-- Statement to prove
theorem age_ratio_in_two_years (h1 : sonCurrentAge S) (h2 : manCurrentAge S M) (h3 : multipleCondition S M) : 
  (M + 2) / (S + 2) = 2 := 
by
  sorry

end age_ratio_in_two_years_l383_383909


namespace polynomial_at_1_gcd_of_72_120_168_l383_383894

-- Define the polynomial function
def polynomial (x : ℤ) : ℤ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x - 6

-- Assertion that the polynomial evaluated at x = 1 gives 9
theorem polynomial_at_1 : polynomial 1 = 9 := by
  -- Usually, this is where the detailed Horner's method proof would go
  sorry

-- Define the gcd function for three numbers
def gcd3 (a b c : ℤ) : ℤ := Int.gcd (Int.gcd a b) c

-- Assertion that the GCD of 72, 120, and 168 is 24
theorem gcd_of_72_120_168 : gcd3 72 120 168 = 24 := by
  -- Usually, this is where the detailed Euclidean algorithm proof would go
  sorry

end polynomial_at_1_gcd_of_72_120_168_l383_383894


namespace same_surface_area_after_cube_removal_l383_383469

/-- Define the dimensions of the original rectangular solid -/
def length : ℝ := 5
def width : ℝ := 3
def height : ℝ := 4

/-- Calculate the surface area of the original rectangular solid -/
def surface_area_original (l w h : ℝ) : ℝ :=
  2 * (l * w + l * h + w * h)

-- original surface area calculation
def S_original := surface_area_original length width height

/-- Define the dimensions of the cube being removed -/
def cube_side : ℝ := 2

/-- The contribution of the cube's surface area to the external surface before removal -/
def cube_exposed_area (s : ℝ) : ℝ :=
  2 * (s * s)

def S_cube_exposed := cube_exposed_area cube_side

/-- The new faces exposed of the rectangular solid after removing the cube -/
def new_faces_exposed_area (s : ℝ) : ℝ :=
  2 * (s * s)

def S_new_faces := new_faces_exposed_area cube_side

/-- Net change in the surface area after removing the cube -/
def ΔS : ℝ := S_new_faces - S_cube_exposed

/-- Prove that the surface area of the new solid remains the same -/
theorem same_surface_area_after_cube_removal :
  S_original = S_original + ΔS := 
by
  -- Adding reasoning directly in the theorem to fit the criteria
  sorry

end same_surface_area_after_cube_removal_l383_383469


namespace min_time_to_cook_cakes_l383_383913

theorem min_time_to_cook_cakes (cakes : ℕ) (pot_capacity : ℕ) (time_per_side : ℕ) 
  (h1 : cakes = 3) (h2 : pot_capacity = 2) (h3 : time_per_side = 5) : 
  ∃ t, t = 15 := by
  sorry

end min_time_to_cook_cakes_l383_383913


namespace percentage_increase_of_x_compared_to_y_l383_383261

-- We are given that y = 0.5 * z and x = 0.6 * z
-- We need to prove that the percentage increase of x compared to y is 20%

theorem percentage_increase_of_x_compared_to_y (x y z : ℝ) 
  (h1 : y = 0.5 * z) 
  (h2 : x = 0.6 * z) : 
  (x / y - 1) * 100 = 20 :=
by 
  -- Placeholder for actual proof
  sorry

end percentage_increase_of_x_compared_to_y_l383_383261


namespace smallest_n_for_trick_l383_383086

theorem smallest_n_for_trick (fillings : Finset Fin 10)
  (pastries : Finset (Fin 45)) 
  (has_pairs : ∀ p ∈ pastries, ∃ f1 f2 ∈ fillings, f1 ≠ f2 ∧ p = pair f1 f2) : 
  ∃ n (tray : Finset (Fin 45)), 
    (tray.card = n ∧ n = 36 ∧ 
    ∀ remaining_p ∈ pastries \ tray, ∃ f ∈ fillings, f ∈ remaining_p) :=
by
  sorry

end smallest_n_for_trick_l383_383086


namespace average_weight_increase_l383_383827

theorem average_weight_increase 
    (A : ℝ) -- initial average weight
    (weight_person_out : ℝ) (weight_person_out = 47)
    (weight_person_in : ℝ) (weight_person_in = 68) : 
    let x : ℝ := (weight_person_in - weight_person_out) / 6 in
    x = 3.5 :=
by 
    obtain x : ℝ := (weight_person_in - weight_person_out) / 6
    have : x = 21 / 6 := by sorry
    rw this
    norm_num
    rfl

end average_weight_increase_l383_383827


namespace evaluate_4_over_04_eq_400_l383_383611

noncomputable def evaluate_fraction : Float :=
  (0.4)^4 / (0.04)^3

theorem evaluate_4_over_04_eq_400 : evaluate_fraction = 400 :=
by
  sorry

end evaluate_4_over_04_eq_400_l383_383611


namespace exponentiation_rule_l383_383138

variable {a b : ℝ}

theorem exponentiation_rule (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 := by
  sorry

end exponentiation_rule_l383_383138


namespace units_digit_2_pow_2130_l383_383313

theorem units_digit_2_pow_2130 : (Nat.pow 2 2130) % 10 = 4 :=
by sorry

end units_digit_2_pow_2130_l383_383313


namespace arithmetic_sequence_sum_l383_383650

noncomputable def a_n (a1 d : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d
noncomputable def S_n (a1 d : ℕ) (n : ℕ) : ℕ := n * a1 + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_sum (a1 d : ℕ) 
  (h1 : a1 + d = 6) 
  (h2 : (a1 + 2 * d)^2 = a1 * (a1 + 6 * d)) 
  (h3 : d ≠ 0) : 
  S_n a1 d 8 = 88 := 
by 
  sorry

end arithmetic_sequence_sum_l383_383650


namespace right_triangle_hypotenuse_l383_383540

theorem right_triangle_hypotenuse (a b : ℝ) (h : a^2 + b^2 = 39^2) : a = 15 ∧ b = 36 := by
  sorry

end right_triangle_hypotenuse_l383_383540


namespace find_m_l383_383228

theorem find_m (m : ℝ) (h : 3 ∈ {m + 2, 2 * m^2 + m}) : m = -3 / 2 := 
sorry

end find_m_l383_383228


namespace magnitude_of_diff_l383_383230

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 6)
def are_parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem magnitude_of_diff (x : ℝ) (h : are_parallel vector_a (vector_b x)) :
  ∥(vector_a.1 - vector_b x).1, (vector_a.2 - vector_b x).2∥ = 2 * Real.sqrt 5 :=
sorry

end magnitude_of_diff_l383_383230


namespace correct_answer_l383_383598

theorem correct_answer (x : ℝ) (h1 : 2 * x = 60) : x / 2 = 15 :=
by
  sorry

end correct_answer_l383_383598


namespace max_sums_in_interval_one_l383_383301

open Finset

def max_sums_in_interval (x : ℕ → ℝ) (n : ℕ) : ℕ := 
  (nat.choose n (n / 2))

theorem max_sums_in_interval_one 
  (x : ℕ → ℝ) 
  (n : ℕ)
  (hx : ∀ i, i < n → x i > 1) :
  ∃ T : finset (finset ℕ), 
  ∀ J ∈ T, ∃ k, T.card = max_sums_in_interval x n :=
sorry

end max_sums_in_interval_one_l383_383301


namespace number_eq_180_l383_383898

theorem number_eq_180 (x : ℝ) (h : 64 + 5 * 12 / (x / 3) = 65) : x = 180 :=
sorry

end number_eq_180_l383_383898


namespace greatest_five_digit_divisible_by_12_15_18_l383_383972

def is_divisible (n k : ℕ) : Prop := k ≠ 0 ∧ n % k = 0
def is_five_digit (n : ℕ) : Prop := n >= 10000 ∧ n < 100000

theorem greatest_five_digit_divisible_by_12_15_18 :
  ∃ n, is_five_digit n ∧ is_divisible n 12 ∧ is_divisible n 15 ∧ is_divisible n 18 ∧
  (∀ m, is_five_digit m ∧ is_divisible m 12 ∧ is_divisible m 15 ∧ is_divisible m 18 → m ≤ n) :=
begin
  use 99900,
  split, -- Check if 99900 is a 5-digit number
  { unfold is_five_digit, simp, },
  split, -- Check if 99900 is divisible by 12
  { unfold is_divisible, exact ⟨by norm_num, by norm_num⟩, },
  split, -- Check if 99900 is divisible by 15
  { unfold is_divisible, exact ⟨by norm_num, by norm_num⟩, },
  split, -- Check if 99900 is divisible by 18
  { unfold is_divisible, exact ⟨by norm_num, by norm_num⟩, },
  -- Check if 99900 is the largest such number
  sorry,
end

end greatest_five_digit_divisible_by_12_15_18_l383_383972


namespace domain_of_f_l383_383831

noncomputable def f (x : ℝ) : ℝ := Real.log (x - x^2)

theorem domain_of_f : {x : ℝ | 0 < x ∧ x < 1} = {x : ℝ | x - x^2 > 0} :=
by
  ext x
  exact ⟨λ ⟨h1, h2⟩ => by linarith, λ h => by linarith [h]⟩

end domain_of_f_l383_383831


namespace sum_of_adjacent_integers_l383_383366

/--
  The positive integer divisors of 245, except 1, are arranged around a circle.
  We aim to prove that the sum of the two integers adjacent to 49 is 280.
-/
theorem sum_of_adjacent_integers (h_divisors : ∀ (d : ℕ), d ∣ 245 → d ≠ 1 →
  d = 5 ∨ d = 7 ∨ d = 35 ∨ d = 49 ∨ d = 245) :
  let divisors := {5, 7, 35, 49, 245}
  ∃ (a b : ℕ), a ≠ b ∧ a ∈ divisors ∧ b ∈ divisors ∧
  (∃ common_factor : ℕ, common_factor > 1 ∧ common_factor ∣ a ∧ common_factor ∣ b ∧ (a = 49 ∨ b = 49)) ∧
  a + b = 280 :=
by
  sorry

end sum_of_adjacent_integers_l383_383366


namespace number_of_zeros_l383_383612

noncomputable def g (x : ℝ) : ℝ := Real.cos (Real.log x)

theorem number_of_zeros (n : ℕ) : (1 < x ∧ x < Real.exp Real.pi) → (∃! x : ℝ, g x = 0 ∧ 1 < x ∧ x < Real.exp Real.pi) → n = 1 :=
sorry

end number_of_zeros_l383_383612


namespace cos_theta_sum_of_fraction_l383_383267

theorem cos_theta_sum_of_fraction (r1 r2 r3: ℝ) (θ φ : ℝ) (h1: 5 = r1) (h2: 12 = r2) (h3: 13 = r3) (h4: θ + φ < real.pi) :
  let cosθ : ℚ := (2*(12/13:ℚ)^2 - 1) in
  cosθ.num + cosθ.denom = 288 :=
by {
  have h5 : cosθ = 119/169 := sorry,
  have sum : 119 + 169 = 288 := by norm_num,
  exact sum,
}

end cos_theta_sum_of_fraction_l383_383267


namespace angle_of_inclination_135_deg_l383_383369

noncomputable def angle_of_inclination_of_line (a b c : ℝ) (hx : ∀ x : ℝ, a * sin (Real.pi / 4 + x) - b * cos (Real.pi / 4 + x) = a * sin (Real.pi / 4 - x) - b * cos (Real.pi / 4 - x)) : ℝ :=
let K := a / b in
if K = -1 then (3 * Real.pi / 4).to_deg else sorry

theorem angle_of_inclination_135_deg (a b c : ℝ) (hx : ∀ x : ℝ, a * sin (Real.pi / 4 + x) - b * cos (Real.pi / 4 + x) = a * sin (Real.pi / 4 - x) - b * cos (Real.pi / 4 - x)) :
  angle_of_inclination_of_line a b c hx = 135 :=
sorry

end angle_of_inclination_135_deg_l383_383369


namespace c_geq_one_l383_383296

theorem c_geq_one {a b : ℕ} {c : ℝ} (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (h : (a + 1) / (b + c) = b / a) : 1 ≤ c :=
  sorry

end c_geq_one_l383_383296


namespace day_of_week_in_100_days_l383_383402

theorem day_of_week_in_100_days (start_day : ℕ) (h : start_day = 5) : 
  (start_day + 100) % 7 = 0 := 
by
  cases h with 
  | rfl => -- start_day is Friday, which is represented as 5
  sorry

end day_of_week_in_100_days_l383_383402


namespace angle_of_inclination_of_line_l383_383971

theorem angle_of_inclination_of_line :
  ∀ t : ℝ, let x := - t * Real.cos (20 * Real.pi / 180) in
  let y := 3 + t * Real.sin (20 * Real.pi / 180) in
  ∃ θ : ℝ, (theta = 160 * Real.pi / 180) := sorry

end angle_of_inclination_of_line_l383_383971


namespace avg_std_dev_shift_l383_383343

variables {n : ℕ} [nontrivial n]
variables (x : ℕ → ℝ) (a : ℝ)
noncomputable def average (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  (∑ i in finset.range n, x i) / n

noncomputable def std_dev (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  real.sqrt ((∑ i in finset.range n, (x i - average x n) ^ 2) / n)

theorem avg_std_dev_shift (x : ℕ → ℝ) (a : ℝ) (n : ℕ) [fact (0 < n)] :
  let avg := average x n,
      sd := std_dev x n in
  average (λ i, x i + a) n = avg + a ∧
  std_dev (λ i, x i + a) n = sd :=
by
  sorry

end avg_std_dev_shift_l383_383343


namespace imaginary_part_of_complex_l383_383617

theorem imaginary_part_of_complex : 
  (complex.imag (1 / (2 - complex.I)) = 1 / 5) :=
by {
  sorry
}

end imaginary_part_of_complex_l383_383617


namespace sphere_surface_area_l383_383641

theorem sphere_surface_area 
  (R : ℝ) 
  (h : ℝ := real.sqrt 3) 
  (r : ℝ := 1) 
  (h_cone : R^2 = (R - h)^2 + r^2) : 
  4 * π * R^2 = (16 * π) / 3 := by
sorry

end sphere_surface_area_l383_383641


namespace pastry_trick_l383_383082

theorem pastry_trick (fillings : Fin 10) (n : ℕ) :
  ∃ n, (n = 36 ∧ ∀ remaining_pastries, 
    (remaining_pastries.length = 45 - n) → 
    (∃ remaining_filling ∈ fillings, true)) := 
sorry

end pastry_trick_l383_383082


namespace fraction_product_l383_383416

theorem fraction_product :
  (∏ n in Finset.range 6, (1 / 3^(n+1) * 3^(2*(n+1))) = 729) :=
sorry

end fraction_product_l383_383416


namespace width_of_room_l383_383362

theorem width_of_room (length : ℝ) (cost_rate : ℝ) (total_cost : ℝ) (width : ℝ)
  (h1 : length = 5.5)
  (h2 : cost_rate = 800)
  (h3 : total_cost = 16500)
  (h4 : width = total_cost / cost_rate / length) : width = 3.75 :=
by
  sorry

end width_of_room_l383_383362


namespace min_distance_is_18_l383_383227

noncomputable def minimize_distance (a b c d : ℝ) : ℝ := (a - c) ^ 2 + (b - d) ^ 2

theorem min_distance_is_18 (a b c d : ℝ) (h1 : b = a - 2 * Real.exp a) (h2 : c + d = 4) :
  minimize_distance a b c d = 18 :=
sorry

end min_distance_is_18_l383_383227


namespace sub_three_five_l383_383590

theorem sub_three_five : 3 - 5 = -2 := 
by 
  sorry

end sub_three_five_l383_383590


namespace unexpected_grades_is_ten_l383_383333

def unexpected_grades (grades : List ℕ) : ℕ :=
  let count : ℕ → ℕ := fun g => grades.count (· = g)
  grades.foldl (fun (acc, seen) g => 
    let unexpected := if ∀ other in [2, 3, 4, 5], other ≠ g → count other > count g then 1 else 0
    (acc + unexpected, g :: seen)
  ) (0, []).fst

theorem unexpected_grades_is_ten (grades : List ℕ) :
  grades.count (· = 2) = 10 ∧ grades.count (· = 3) = 10 ∧ grades.count (· = 4) = 10 ∧ grades.count (· = 5) = 10 →
  unexpected_grades grades = 10 := by
  sorry

end unexpected_grades_is_ten_l383_383333


namespace isosceles_triangle_three_digit_numbers_count_l383_383303

theorem isosceles_triangle_three_digit_numbers_count :
  let n : Finset ℕ := (Finset.range 1000).filter (λ x, let a := x / 100, b := (x / 10) % 10, c := x % 10 in 
                                (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧ 
                                (a = b ∨ b = c ∨ c = a)) in
  n.card = 165 :=
by sorry

end isosceles_triangle_three_digit_numbers_count_l383_383303


namespace largest_x_eq_120_div_11_l383_383174

theorem largest_x_eq_120_div_11 (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 11 / 12) : x ≤ 120 / 11 :=
sorry

end largest_x_eq_120_div_11_l383_383174


namespace general_formula_sum_b_formula_l383_383204

-- Define the arithmetic sequence conditions
def sequence_condition (a : ℕ → ℕ) : Prop :=
  a 3 = 5 ∧ a 17 = 3 * a 6

-- Proof for the general formula of the arithmetic sequence
theorem general_formula (a : ℕ → ℕ) (h : sequence_condition a) : 
  ∀ n, a n = 2 * n - 1 :=
sorry

-- Define b_n sequence given the formula of a_n
def b (n : ℕ) (a : ℕ → ℕ) : ℝ := 2 / (n * (a n + 3))

-- Define the sum of the first n terms of b_n
def sum_b (n : ℕ) (a : ℕ → ℕ) : ℝ :=
  ∑ i in Finset.range n, b i a

-- Proof for the sum of the first n terms of b_n
theorem sum_b_formula (a : ℕ → ℕ) (h : sequence_condition a) : 
  ∀ n, sum_b n a = n / (n + 1) :=
sorry

end general_formula_sum_b_formula_l383_383204


namespace right_triangle_hypotenuse_length_l383_383529

theorem right_triangle_hypotenuse_length :
  ∀ (a b h : ℕ), a = 15 → b = 36 → h^2 = a^2 + b^2 → h = 39 :=
by
  intros a b h ha hb hyp
  -- In the proof, we would use ha, hb, and hyp to show h = 39
  sorry

end right_triangle_hypotenuse_length_l383_383529


namespace pascal_triangle_45th_number_l383_383008

theorem pascal_triangle_45th_number : nat.choose 46 44 = 1035 := 
by sorry

end pascal_triangle_45th_number_l383_383008


namespace cheaper_actual_call_rate_l383_383877

theorem cheaper_actual_call_rate :
  ∀ (effective_rate_mobile effective_rate_telecom : ℝ),
  (effective_rate_mobile = 0.26 * (10 / 13)) ∧ (effective_rate_telecom = 0.30 * (2 / 5)) →
  effective_rate_telecom < effective_rate_mobile ∧ (effective_rate_mobile - effective_rate_telecom = 0.08) :=
begin
  sorry
end

end cheaper_actual_call_rate_l383_383877


namespace distance_to_center_of_mass_l383_383375

noncomputable def geometric_radii (i : ℕ) : ℝ :=
  2 * (1 / 2) ^ (i - 1)

noncomputable def disk_mass (i : ℕ) (base_mass : ℝ) : ℝ :=
  base_mass * (1 / 2) ^ (2 * (i - 1))

noncomputable def vertical_center_of_mass (base_mass : ℝ) : ℝ :=
  let mass_sum := base_mass * (4 / 3) in
  let weighted_sum := base_mass * 2 * (8 / 7) in
  weighted_sum / mass_sum

theorem distance_to_center_of_mass {base_mass : ℝ} (h : base_mass > 0) :
  vertical_center_of_mass base_mass = 6 / 7 :=
by
  calc vertical_center_of_mass base_mass = _ : sorry

end distance_to_center_of_mass_l383_383375


namespace solve_for_x_l383_383169

theorem solve_for_x : ∃ x : ℝ, x^4 + 10 * x^3 + 9 * x^2 - 50 * x - 56 = 0 ↔ x = -2 :=
by
  sorry

end solve_for_x_l383_383169


namespace hypotenuse_length_l383_383478

theorem hypotenuse_length (a b c : ℕ) (h₀ : a = 15) (h₁ : b = 36) (h₂ : a^2 + b^2 = c^2) : c = 39 :=
by
  -- Proof is omitted
  sorry

end hypotenuse_length_l383_383478


namespace limit_tangent_logarithm_l383_383934

theorem limit_tangent_logarithm (a : ℝ) (hla : a ≠ 0) :
  (filter.tendsto (λ x, (tan x - tan a) / (log x - log a)) (nhds a) (nhds (a / cos a ^ 2))) :=
sorry

end limit_tangent_logarithm_l383_383934


namespace total_gain_was_l383_383430

variable (X T : ℝ)
variables (nandan_gain krishan_gain total_gain : ℝ)

-- Conditions
def condition1 : Prop := krishan_gain = 72000
def condition2 : Prop := nandan_gain = 6000
def condition3 : Prop := total_gain = nandan_gain + krishan_gain

-- The proof goal
theorem total_gain_was (h1 : condition1) (h2 : condition2) : total_gain = 78000 :=
by
  rw [condition3, h2, h1]
  rfl

# Print the Lean 4 statement
#print total_gain_was

end total_gain_was_l383_383430


namespace dilation_translation_correct_l383_383181

def transformation_matrix (d: ℝ) (tx: ℝ) (ty: ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![d, 0, tx],
    ![0, d, ty],
    ![0, 0, 1]
  ]

theorem dilation_translation_correct :
  transformation_matrix 4 2 3 =
  ![
    ![4, 0, 2],
    ![0, 4, 3],
    ![0, 0, 1]
  ] :=
by
  sorry

end dilation_translation_correct_l383_383181


namespace angle_610_third_quadrant_l383_383897

-- Define what a quadrant is in terms of degrees
def quadrant (angle : ℝ) : string :=
  let mod_angle := angle % 360
  if 0 ≤ mod_angle ∧ mod_angle < 90 then "first"
  else if 90 ≤ mod_angle ∧ mod_angle < 180 then "second"
  else if 180 ≤ mod_angle ∧ mod_angle < 270 then "third"
  else "fourth"

-- The theorem to prove that 610° is in the third quadrant
theorem angle_610_third_quadrant : quadrant 610 = "third" :=
  by
    sorry

end angle_610_third_quadrant_l383_383897


namespace number_of_pairs_l383_383704

theorem number_of_pairs (a b : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) :
  (∃ n, n = 3 ∧ 
         a * b + 100 = 15 * nat.lcm a b + 10 * nat.gcd a b) := 
begin
  use 3,
  sorry
end

end number_of_pairs_l383_383704


namespace number_of_not_appearing_numbers_eq_l383_383917

def sequence (n : Nat) : Nat :=
if n < 3 then [2, 1, 9].get! n
else ([2, 1, 9].get! (n - 3) + [2, 1, 9].get! (n - 2) + [2, 1, 9].get! (n - 1)) % 10

def four_digit_numbers := [1113, 2226, 2125, 2215]

def number_of_four_digit_exclusions := 1

theorem number_of_not_appearing_numbers_eq :
  (four_digit_numbers.filter (λ num =>
    let digits := [num / 1000 % 10, num / 100 % 10, num / 10 % 10, num % 10]
    ¬ ∃ n, List.take 4 (List.drop n (List.range 10000).map sequence) = digits
  )).length = number_of_four_digit_exclusions := by
sorry

end number_of_not_appearing_numbers_eq_l383_383917


namespace sum_placed_on_SI_is_1833_l383_383033

namespace SimpleInterestProblem

def compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * ((1 + r / 100) ^ t - 1)

def simple_interest_principal (SI : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  SI * 100 / (r * t)

theorem sum_placed_on_SI_is_1833.33 :
  let P := 8000
  let r_CI := 20
  let t_CI := 2
  let CI := 3520
  let r_SI := 16
  let t_SI := 6
  let SI := CI / 2
  simple_interest_principal SI r_SI t_SI = 1833.33 :=
by
  sorry
  
end SimpleInterestProblem

end sum_placed_on_SI_is_1833_l383_383033


namespace sum_S_i_l383_383194

noncomputable def binom_expansion (x : ℝ) (n : ℕ) : ℝ := (1 + x) ^ n

noncomputable def S_n (n : ℕ) : ℝ := n * (2 ^ (n - 1))

theorem sum_S_i (n : ℕ) (hn : 0 < n) : ∑ i in finset.range n, S_n (i+1) = (n - 1) * 2^n + 1 := by
  sorry

end sum_S_i_l383_383194


namespace student_score_after_study_and_tutoring_l383_383565

def score (hours : ℕ) : ℝ := 20 * hours

theorem student_score_after_study_and_tutoring
  (hours_studied : ℕ)
  (initial_score : ℝ)
  (tutor_boost : ℝ)
  (score_variation : ∀ (t : ℕ), score t = initial_score * (t / hours_studied))
  (final_hours : ℕ = 5)
  (initial_hours : ℕ = 4)
  (initial_points : ℝ = 80)
  (tutor_improvement : ℝ = 0.10)
  : initial_score = 20 ∧ score final_hours * (1 + tutor_improvement) = 110 := by
  sorry

end student_score_after_study_and_tutoring_l383_383565


namespace b_share_correct_l383_383426

-- Defining the work rates of a, b, c, and d
def work_rate_a := 1 / 6
def work_rate_b := 1 / 8
def work_rate_c := 1 / 12
def work_rate_d := 1 / 15

-- Total payment for the work
def total_payment : ℝ := 2400

-- Total work rate when they work together
def total_work_rate := work_rate_a + work_rate_b + work_rate_c + work_rate_d

-- b's share of the total payment
def b_share := (work_rate_b / total_work_rate) * total_payment

-- The assertion to prove
theorem b_share_correct : b_share ≈ 679.25 := sorry

end b_share_correct_l383_383426


namespace price_equation_l383_383448

variable (x : ℝ)

def first_discount (x : ℝ) : ℝ := x - 5

def second_discount (price_after_first_discount : ℝ) : ℝ := 0.8 * price_after_first_discount

theorem price_equation
  (hx : second_discount (first_discount x) = 60) :
  0.8 * (x - 5) = 60 := by
  sorry

end price_equation_l383_383448


namespace range_of_m_l383_383678

noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem range_of_m (m : ℝ) :
  (∃ t : ℝ, t = g x ∧ x ∈ Ioo 0 2 ∧ (|t|^2 + m * |t| + 2 * m + 3 = 0)) →
  m ∈ Set.Icc (-3/2 : ℝ) (-4:3) :=
sorry

end range_of_m_l383_383678


namespace sum_of_z_and_conjugate_l383_383787

theorem sum_of_z_and_conjugate (z : ℂ) (h : z = i / (1 + 2 * i)) : 
  z + conj(z) = 4 / 5 := 
sorry

end sum_of_z_and_conjugate_l383_383787


namespace number_of_paths_in_grid_l383_383605

theorem number_of_paths_in_grid :
 ∃ n : ℕ, (n = (nat.choose 16 8)) ∧ n = 12870 := 
by 
  let n := nat.choose 16 8
  use n
  constructor
  . 
    -- Prove that n = nat.choose 16 8
    refl
  .
    -- Prove that nat.choose 16 8 = 12870
    calc n = nat.choose 16 8 : by refl
       ... = 12870 : by sorry

end number_of_paths_in_grid_l383_383605


namespace find_n_l383_383977

noncomputable def n_value : ℕ :=
  let term1 := (1 / 2) * (3 / 4)
  let term2 := (5 / 6) * (7 / 8)
  let term3 := (9 / 10) * (11 / 12)
  let sum_of_terms := term1 + term2 + term3
  let lhs := sum_of_terms
  let rhs := (2315 : ℕ) / 1200
  2315

theorem find_n : ∃ n : ℕ, n > 0 ∧ (1 / 2) * (3 / 4) + (5 / 6) * (7 / 8) + (9 / 10) * (11 / 12) = n / 1200 :=
by
  use n_value
  split
  · exact Nat.zero_lt_succ _
  · have h : (1 / 2) * (3 / 4) + (5 / 6) * (7 / 8) + (9 / 10) * (11 / 12) = 2315 / 1200 := by sorry
    exact h

end find_n_l383_383977


namespace right_triangle_hypotenuse_length_l383_383553

theorem right_triangle_hypotenuse_length (a b : ℝ) (h_triangle : a = 15 ∧ b = 36) :
  ∃ (h : ℝ), h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  · exact rfl
  · rw [h_triangle.1, h_triangle.2]
    norm_num

end right_triangle_hypotenuse_length_l383_383553


namespace volume_of_inequality_region_correct_l383_383978

open Real

noncomputable def volume_of_region : ℝ :=
  let region := {p : ℝ × ℝ × ℝ | 
                  let x := p.1, y := p.2, z := p.3 in
                  |x - y + z| + |x - y - z| ≤ 10 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 } in
  let base_area := 1 / 2 * 5 * 5 in
  let height := 5 in
  base_area * height

theorem volume_of_inequality_region_correct :
  volume_of_region = 62.5 :=
by
  -- The actual detailed proof will go here, omitted for brevity.
  -- It will involve showing that this volume computation for the defined region is accurate
  -- according to the steps detailed in the solution.
  sorry

end volume_of_inequality_region_correct_l383_383978


namespace purity_of_first_solution_l383_383903

-- Given conditions
variables (purity_first_solution : ℚ) (A : ℚ)
variables (purity_second_solution : ℚ) (B : ℚ)
variables (final_amount : ℚ) (final_purity : ℚ)
variable (pure_salt_first_solution : ℚ)

--Conditions as lean statements
def cond1 : (purity_first_solution = pure_salt_first_solution / A) := sorry
def cond2 : (pure_salt_first_solution + 24 = 30) := sorry
def cond3 : (A + 40 = 60) := sorry

-- Let's assume the purity of the second solution is constant 60% and remaining conditions
def purity_second_solution : ℚ := 0.60
def B : ℚ := 40
def final_amount : ℚ := 60
def final_purity : ℚ := 0.50

-- The purity of the first solution is 30% is to be proven under these conditions
theorem purity_of_first_solution :
  purity_first_solution = 30 := by
  sorry

end purity_of_first_solution_l383_383903


namespace bus_patrons_correct_l383_383964

-- Definitions corresponding to conditions
def number_of_golf_carts : ℕ := 13
def patrons_per_cart : ℕ := 3
def car_patrons : ℕ := 12

-- Multiply to get total patrons transported by golf carts
def total_patrons := number_of_golf_carts * patrons_per_cart

-- Calculate bus patrons
def bus_patrons := total_patrons - car_patrons

-- The statement to prove
theorem bus_patrons_correct : bus_patrons = 27 :=
by
  sorry

end bus_patrons_correct_l383_383964


namespace new_savings_after_expense_increase_l383_383910

theorem new_savings_after_expense_increase
    (monthly_salary : ℝ)
    (initial_saving_percent : ℝ)
    (expense_increase_percent : ℝ)
    (initial_salary : monthly_salary = 20000)
    (saving_rate : initial_saving_percent = 0.1)
    (increase_rate : expense_increase_percent = 0.1) :
    monthly_salary - (monthly_salary * (1 - initial_saving_percent + (1 - initial_saving_percent) * expense_increase_percent)) = 200 :=
by
  sorry

end new_savings_after_expense_increase_l383_383910


namespace ArtisticHub_sales_l383_383825

theorem ArtisticHub_sales (brushes paints : ℕ) (total : ℕ) (h_brushes : brushes = 45) (h_paints : paints = 22) (h_total : total = 100) : (total - brushes - paints) = 33 :=
by
  rw [h_brushes, h_paints, h_total]
  -- calculation steps
  sorry

end ArtisticHub_sales_l383_383825


namespace mr_harris_time_to_store_l383_383029

variable (T : ℝ)
variable (you_speed_factor : ℝ := 2)
variable (destination_factor : ℝ := 3)
variable (your_time_to_destination : ℝ := 3)

theorem mr_harris_time_to_store (h : T / you_speed_factor = your_time_to_destination / destination_factor) : T = 2 :=
by
  simp only [you_speed_factor, destination_factor, your_time_to_destination] at h
  sorry

end mr_harris_time_to_store_l383_383029


namespace hypotenuse_length_l383_383476

theorem hypotenuse_length (a b c : ℕ) (h₀ : a = 15) (h₁ : b = 36) (h₂ : a^2 + b^2 = c^2) : c = 39 :=
by
  -- Proof is omitted
  sorry

end hypotenuse_length_l383_383476


namespace inequality_proof_l383_383989

variable {x y z : ℝ}

theorem inequality_proof 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hxyz : x + y + z = 1) : 
  (1 / x^2 + x) * (1 / y^2 + y) * (1 / z^2 + z) ≥ (28 / 3)^3 :=
sorry

end inequality_proof_l383_383989


namespace fourth_point_of_intersection_l383_383751

variable (a b r : ℝ)

def is_on_curve (p : ℝ × ℝ) : Prop := p.1 * p.2 = 2

def is_on_circle (p : ℝ × ℝ) : Prop := (p.1 - a) ^ 2 + (p.2 - b) ^ 2 = r ^ 2

theorem fourth_point_of_intersection (h1 : is_on_circle (4, 1/2))
                                     (h2 : is_on_circle (-2, -1))
                                     (h3 : is_on_circle (2/3, 3))
                                     (h_curve1 : is_on_curve (4, 1/2))
                                     (h_curve2 : is_on_curve (-2, -1))
                                     (h_curve3 : is_on_curve (2/3, 3)) :
  ∃ p : ℝ × ℝ, is_on_curve p ∧ is_on_circle p ∧ p ≠ (4, 1/2) ∧ p ≠ (-2, -1) ∧ p ≠ (2/3, 3) ∧ p = (-3/4, -8/3).

end fourth_point_of_intersection_l383_383751


namespace sum_of_x_coordinates_of_intersections_l383_383829

open Real

-- Definition of line segments as continuous piecewise functions
noncomputable def g (x : ℝ) : ℝ :=
  if h₁ : -4 ≤ x ∧ x ≤ -1 then (5 / 3 * x + 1 / 3)
  else if h₂ : -1 < x ∧ x < 1 then x
  else if h₃ : 1 ≤ x ∧ x ≤ 4 then (5 / 3 * x - 2 / 3)
  else 0

-- The main theorem to be proven
theorem sum_of_x_coordinates_of_intersections : 
  let f := λ x : ℝ, x^2 - 2 in 
  let intersections := {x : ℝ | g x = f x} in
  (∑ x in intersections, x) = 2 := 
  by
    sorry

end sum_of_x_coordinates_of_intersections_l383_383829


namespace equality_of_ha_l383_383810

theorem equality_of_ha 
  {p a b α β γ : ℝ} 
  (h1 : h_a = (2 * (p - a) * Real.cos (β / 2) * Real.cos (γ / 2)) / Real.cos (α / 2))
  (h2 : h_a = (2 * (p - b) * Real.sin (β / 2) * Real.cos (γ / 2)) / Real.sin (α / 2)) : 
  (2 * (p - a) * Real.cos (β / 2) * Real.cos (γ / 2)) / Real.cos (α / 2) = 
  (2 * (p - b) * Real.sin (β / 2) * Real.cos (γ / 2)) / Real.sin (α / 2) :=
by sorry

end equality_of_ha_l383_383810


namespace composition_of_even_is_even_l383_383299

variable {α : Type} [Inhabited α]

def even_function (g : α → α) : Prop :=
  ∀ x, g (-x) = g x

theorem composition_of_even_is_even (g : α → α) (h : even_function g) :
  even_function (λ x, g (g x)) :=
by
  unfold even_function at *
  assume x
  calc
    g (g (-x)) = g (g x) : sorry

end composition_of_even_is_even_l383_383299


namespace sum_of_integers_greater_than_2_and_less_than_11_l383_383409

theorem sum_of_integers_greater_than_2_and_less_than_11 : 
  (∑ n in finset.range 10, if n > 2 then n else 0) = 52 :=
by
  sorry

end sum_of_integers_greater_than_2_and_less_than_11_l383_383409


namespace number_of_permutations_l383_383353

theorem number_of_permutations (digits : List ℕ) (h_unique : digits.nodup) :
  digits = [1, 2, 3, 4, 5] →
  ∃ n, n = 60 ∧ (List.permutations digits).countp (λ l, l.indexOf 1 < l.indexOf 2) = n := by
  intro h_digits
  use 60
  split
  . rfl
  . have h_total := List.permutations_length digits
    rw h_digits at h_total
    have : 5! = 120 := rfl
    simp only [List.permutations_length_eq_factorial_of_nodup h_unique] at h_total
    rw [this] at h_total
    have h_half : 120 / 2 = 60 := rfl
    rw [List.countp_permutations digits (λ l, l.indexOf 1 < l.indexOf 2) h_unique] 
    rw mul_comm at h_total
    exact h_half
, sorry

end number_of_permutations_l383_383353


namespace find_Q_l383_383024

noncomputable def q : Vector := ⟨-18/13, 12/13⟩

def line_vector (a : ℝ) : Vector := ⟨a, (3/2) * a + 3⟩
def u_vector (d : ℝ) : Vector := ⟨-(3/2) * d, d⟩

def projection (v u : Vector) : Vector :=
  ((v.1 * u.1 + v.2 * u.2) / (u.1 * u.1 + u.2 * u.2)) • u

theorem find_Q (d : ℝ) (a : ℝ) :
  projection (line_vector a) (u_vector d) = q := sorry

end find_Q_l383_383024


namespace y_is_80_percent_less_than_x_l383_383103

theorem y_is_80_percent_less_than_x (x y : ℝ) (h : x = 5 * y) : ((x - y) / x) * 100 = 80 :=
by sorry

end y_is_80_percent_less_than_x_l383_383103


namespace gcd_digit_bound_l383_383733

theorem gcd_digit_bound (a b : ℕ) (h₁ : 10^6 ≤ a) (h₂ : a < 10^7) (h₃ : 10^6 ≤ b) (h₄ : b < 10^7) 
  (h₅ : 10^{10} ≤ lcm a b) (h₆ : lcm a b < 10^{11}) : 
  gcd a b < 10^4 :=
sorry

end gcd_digit_bound_l383_383733


namespace arithmetic_sequence_150th_term_l383_383738

theorem arithmetic_sequence_150th_term :
  let a1 := 2
  let d := 5
  let n := 150
  in a1 + (n - 1) * d = 747 :=
by
  -- sorry is a placeholder for the proof
  sorry

end arithmetic_sequence_150th_term_l383_383738


namespace percentage_increase_fall_is_approximately_39_91_l383_383119

theorem percentage_increase_fall_is_approximately_39_91 
    (x : ℝ) 
    (h1 : 0 ≤ x)  -- condition 1: Percentage increase in the fall
    (h2 : (100 + x) * 0.81 = 113.33)  -- condition 3: Total change as a function of percentage increase
    : x ≈ 39.91 := -- approximately 39.91%

-- Place holder to indicate incomplete proof.
sorry

end percentage_increase_fall_is_approximately_39_91_l383_383119


namespace percentage_less_C_D_l383_383134

-- Given conditions
def full_marks : ℝ := 500
def D_marks : ℝ := 0.80 * full_marks
def A_marks : ℝ := 360
def B_marks : ℝ := A_marks / 0.90
def C_marks : ℝ := B_marks / 1.25

-- Statement to prove
theorem percentage_less_C_D : ((D_marks - C_marks) / D_marks) * 100 = 20 := by
  sorry

end percentage_less_C_D_l383_383134


namespace cheaper_store_difference_l383_383670

theorem cheaper_store_difference :
  let list_price := 52 : ℝ
  let discount_super_deals := 12 : ℝ
  let discount_penny_save := 15 : ℝ
  let price_super_deals := list_price - discount_super_deals
  let price_penny_save := list_price - discount_penny_save
  let difference_dollars := price_super_deals - price_penny_save
  let difference_cents := difference_dollars * 100
  difference_cents = 300 := 
by
  sorry

end cheaper_store_difference_l383_383670


namespace remove_terms_to_make_sum_l383_383424

theorem remove_terms_to_make_sum (a b c d e f : ℚ) (h₁ : a = 1/3) (h₂ : b = 1/5) (h₃ : c = 1/7) (h₄ : d = 1/9) (h₅ : e = 1/11) (h₆ : f = 1/13) :
  a + b + c + d + e + f - e - f = 3/2 :=
by
  sorry

end remove_terms_to_make_sum_l383_383424


namespace isosceles_triangle_base_length_l383_383387

theorem isosceles_triangle_base_length (a b : ℕ) (h1 : a = 8) (h2 : 2 * a + b = 30) : b = 14 := by
  rw [←h1, mul_add, two_mul, add_assoc, add_left_eq_add, add_zero] at h2
  exact (eq_sub_of_add_eq h2).symm

end isosceles_triangle_base_length_l383_383387


namespace probability_sum_five_correct_l383_383193

noncomputable def set_of_numbers := {1, 2, 3, 4, 5}

def valid_pairs := [(1,2), (1,3), (1,4), (1,5), (2,3), 
                    (2,4), (2,5), (3,4), (3,5), (4,5)]

def pairs_with_sum_five := if (1,4) ∈ valid_pairs ∧ (2,3) ∈ valid_pairs
                           then 2
                           else 0

def total_pairs := valid_pairs.length

def probability_of_sum_five := pairs_with_sum_five / total_pairs

theorem probability_sum_five_correct :
  probability_of_sum_five = 1 / 5 := 
by
  sorry

end probability_sum_five_correct_l383_383193


namespace lending_period_C_l383_383098

theorem lending_period_C (R : ℝ) (P_B P_C T_B I : ℝ) (h1 : R = 13.75) (h2 : P_B = 4000) (h3 : P_C = 2000) (h4 : T_B = 2) (h5 : I = 2200) : 
  ∃ T_C : ℝ, T_C = 4 :=
by
  -- Definitions and known facts
  let I_B := (P_B * R * T_B) / 100
  let I_C := I - I_B
  let T_C := I_C / ((P_C * R) / 100)
  -- Prove the target
  use T_C
  sorry

end lending_period_C_l383_383098


namespace minimum_value_of_f_l383_383867

def f (x : ℝ) : ℝ := 5 * x^2 - 20 * x + 1357

theorem minimum_value_of_f : ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x₀ : ℝ, f x₀ = m) := 
by 
  use 1337
  sorry

end minimum_value_of_f_l383_383867


namespace min_expression_l383_383657

theorem min_expression (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1/a + 1/b = 1) : 
  (∃ x : ℝ, x = min ((1 / (a - 1)) + (4 / (b - 1))) 4) :=
sorry

end min_expression_l383_383657


namespace problem1_problem2_l383_383780

theorem problem1 (a b : ℤ) (h : Even (5 * b + a)) : Even (a - 3 * b) :=
sorry

theorem problem2 (a b : ℤ) (h : Odd (5 * b + a)) : Odd (a - 3 * b) :=
sorry

end problem1_problem2_l383_383780


namespace hundred_days_from_friday_is_sunday_l383_383397

def days_from_friday (n : ℕ) : Nat :=
  (n + 5) % 7  -- 0 corresponds to Sunday, starting from Friday (5 + 0 % 7 = 5 which is Friday)

theorem hundred_days_from_friday_is_sunday :
  days_from_friday 100 = 0 := by
  sorry

end hundred_days_from_friday_is_sunday_l383_383397


namespace right_triangle_hypotenuse_l383_383521

theorem right_triangle_hypotenuse (a b : ℕ) (ha : a = 15) (hb : b = 36) : 
  ∃ h : ℕ, h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  . exact rfl
  . rw [ha, hb]
    norm_num
    sorry

end right_triangle_hypotenuse_l383_383521


namespace function_is_linear_l383_383359

noncomputable def linear_or_not (f : ℝ → ℝ) :=
  ∀ x y : ℝ, f(x + y) = f(x) + f(y)

theorem function_is_linear (f : ℝ → ℝ) (h : linear_or_not f) : ∃ k : ℝ, ∀ x : ℝ, f(x) = k * x :=
  sorry

end function_is_linear_l383_383359


namespace triangle_inequality_cosecant_l383_383647

theorem triangle_inequality_cosecant (α β γ : ℝ) (h_triangle: α + β + γ = π) :
  (real.csc (α / 2))^2 + (real.csc (β / 2))^2 + (real.csc (γ / 2))^2 ≥ 12 ∧ 
  ((real.csc (α / 2))^2 + (real.csc (β / 2))^2 + (real.csc (γ / 2))^2 = 12 → α = β ∧ β = γ) :=
by
  sorry

end triangle_inequality_cosecant_l383_383647


namespace hypotenuse_length_l383_383492

theorem hypotenuse_length (a b : ℤ) (h₀ : a = 15) (h₁ : b = 36) : 
  ∃ c : ℤ, c^2 = a^2 + b^2 ∧ c = 39 := 
by {
  sorry
}

end hypotenuse_length_l383_383492


namespace sum_of_remaining_six_numbers_l383_383345

theorem sum_of_remaining_six_numbers :
  ∀ (S T U : ℕ), 
    S = 20 * 500 → T = 14 * 390 → U = S - T → U = 4540 :=
by
  intros S T U hS hT hU
  sorry

end sum_of_remaining_six_numbers_l383_383345


namespace number_of_functions_satisfying_property_l383_383294

-- Define the set A
def A : finset ℕ := finset.range 1994  -- {1, 2, ..., 1993}

-- Define the function f and its iterations
def f (x : ℕ) := x  -- Placeholder, real definition would depend on later formal proof
def f_iter : ℕ → (ℕ → ℕ) → ℕ → ℕ
| 1, f, x := f x
| (n+1), f, x := f (f_iter n f x)

-- Definition of the property we are interested in
def f_property (f : ℕ → ℕ) : Prop :=
∀ x ∈ A, f_iter 1993 f x = f x

-- Main theorem
theorem number_of_functions_satisfying_property : finset.card {f // f_property f} = 3985 :=
sorry

end number_of_functions_satisfying_property_l383_383294


namespace max_x_2y_l383_383785

noncomputable def max_value (x y : ℝ) : ℝ :=
√(5 / 18) + 1 / 2

theorem max_x_2y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : x + 2 * y ≤ max_value x y := by
  sorry

end max_x_2y_l383_383785


namespace longest_pole_in_room_l383_383975

theorem longest_pole_in_room :
  ∀ (l w h : ℕ), l = 12 → w = 8 → h = 9 → sqrt (l^2 + w^2 + h^2) = 17 :=
by
  intros l w h hl hw hh
  rw [hl, hw, hh]
  norm_num
  sorry

end longest_pole_in_room_l383_383975


namespace pentagon_angle_Q_l383_383700

theorem pentagon_angle_Q :
  ∀ (angle_B angle_C angle_D angle_E : ℕ)
    (sum_interior_angles pentagon : ℕ),
    pentagon = 5 →
    sum_interior_angles = 540 →
    angle_B = 120 →
    angle_C = 95 →
    angle_D = 118 →
    angle_E = 105 →
    (∃ angle_Q : ℕ, angle_Q + angle_B + angle_C + angle_D + angle_E = 540 ∧ angle_Q = 102) :=
by
  intros angle_B angle_C angle_D angle_E sum_interior_angles pentagon hpent hsum hB hC hD hE
  use 102
  split
  · calc
      102 + angle_B + angle_C + angle_D + angle_E
      = 102 + 120 + 95 + 118 + 105 : by rw [hB, hC, hD, hE]
      ... = 102 + 438 : by norm_num
      ... = 540 : by norm_num
  · rfl

end pentagon_angle_Q_l383_383700


namespace max_f_value_l383_383642
-- Begin by importing the entire Mathlib to ensure all necessary tools are available

-- Define n as a natural number
def n : ℕ

-- Define convex n-gon S with vertices labeled sequentially from 1 to n.
structure ConvexNGon :=
  (vertices : Fin n → ℝ × ℝ)
  (isConvex : Prop) -- We should have a convexity definition, but assuming a property here.

-- Define function f(S) to denote the maximum possible number of distinct n-term sequences
-- Assuming observation point is outside and forms an angle < 180 degrees, ensuring not collinear with any two vertices
def f (S : ConvexNGon) : ℕ := sorry -- Placeholder as defining the exact function requires more elaborate setup

-- Theorem statement to show the maximum possible value of f(S)
theorem max_f_value (S : ConvexNGon) : f(S) ≤ 2 * (Nat.choose n 2 + Nat.choose n 4) :=
sorry

end max_f_value_l383_383642


namespace converse_false_l383_383711

theorem converse_false (a : ℝ) :
  (a = 4 → (∀ x y : ℝ, x^2 / a^2 + y^2 / 4 = 1 → a > 2 ∧ a ≠ 4)) := 
by {
  intros h1 h2 x y h3,
  have h4 : a^2 > 4, -- we know that if it represents an ellipse with foci on the x-axis, then it must be a standard ellipse equation
  -- Further proof steps (omitted here) would show that a > 2 follows from this.
  sorry
}

end converse_false_l383_383711


namespace completing_the_square_l383_383025

theorem completing_the_square (x : ℝ) :
  4 * x^2 - 2 * x - 1 = 0 → (x - 1/4)^2 = 5/16 := 
by
  sorry

end completing_the_square_l383_383025


namespace trick_proof_l383_383060

-- Defining the number of fillings and total pastries based on combinations
def num_fillings := 10

def total_pastries : ℕ := (num_fillings * (num_fillings - 1)) / 2

-- Definition stating that the smallest number of pastries n such that Vasya can always determine at least one filling of any remaining pastry
def min_n := 36

-- The theorem stating the proof problem
theorem trick_proof (n m: ℕ) (h1: n = 10) (h2: m = (n * (n - 1)) / 2) : min_n = 36 :=
by
  sorry

end trick_proof_l383_383060


namespace right_triangle_hypotenuse_length_l383_383522

theorem right_triangle_hypotenuse_length :
  ∀ (a b h : ℕ), a = 15 → b = 36 → h^2 = a^2 + b^2 → h = 39 :=
by
  intros a b h ha hb hyp
  -- In the proof, we would use ha, hb, and hyp to show h = 39
  sorry

end right_triangle_hypotenuse_length_l383_383522


namespace problem_l383_383655

noncomputable def point_on_curve (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  y = a * x^3 + b * x

noncomputable def tangent_slope_at_point (a b : ℝ) (P : ℝ × ℝ) (slope : ℝ) : Prop :=
  let (x, _) := P
  3 * a * x^2 + b = slope

theorem problem (a b : ℝ) (P : ℝ × ℝ) (h1 : point_on_curve a b P) (h2 : tangent_slope_at_point a b P 9) :
  a * b = -3 ∧ (let f (x : ℝ) := a * x^3 + b * x in 
               ∀ x ∈ (Set.Icc (-3/2) 3), f x ∈ (Set.Icc (-2 : ℝ) 18)) :=
by
  sorry

end problem_l383_383655


namespace max_neg_of_equation_l383_383823

theorem max_neg_of_equation (a b c d : ℤ) (h : 2^a + 2^b = 5^c + 5^d) : 
  a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0 → False :=
sorry

end max_neg_of_equation_l383_383823


namespace distance_to_center_of_mass_l383_383374

noncomputable def geometric_radii (i : ℕ) : ℝ :=
  2 * (1 / 2) ^ (i - 1)

noncomputable def disk_mass (i : ℕ) (base_mass : ℝ) : ℝ :=
  base_mass * (1 / 2) ^ (2 * (i - 1))

noncomputable def vertical_center_of_mass (base_mass : ℝ) : ℝ :=
  let mass_sum := base_mass * (4 / 3) in
  let weighted_sum := base_mass * 2 * (8 / 7) in
  weighted_sum / mass_sum

theorem distance_to_center_of_mass {base_mass : ℝ} (h : base_mass > 0) :
  vertical_center_of_mass base_mass = 6 / 7 :=
by
  calc vertical_center_of_mass base_mass = _ : sorry

end distance_to_center_of_mass_l383_383374


namespace right_triangle_hypotenuse_length_l383_383504

theorem right_triangle_hypotenuse_length (a b : ℕ) (h1 : a = 15) (h2 : b = 36) : 
  ∃ c : ℕ, c * c = a * a + b * b ∧ c = 39 := 
by
  have hyp_square := 225 + 1296 
  have h_calculation : 15 * 15 + 36 * 36 = 1521 := by
    calc
      15 * 15 = 225 : rfl
      36 * 36 = 1296 : rfl
      225 + 1296 = 1521 : rfl
  use 39
  split
  exact h_calculation
  rfl

end right_triangle_hypotenuse_length_l383_383504


namespace profit_from_sales_l383_383857

theorem profit_from_sales
  (item1_price item2_price : ℝ)
  (h_item1_price : 84 = 1.4 * item1_price)
  (h_item2_price : 84 = 0.8 * item2_price)
  (h_total_selling_price : 2 * 84 = 168)
  (h1 : 84 = 1.4 * item1_price → item1_price = 60)
  (h2 : 84 = 0.8 * item2_price → item2_price = 105)
  (h3 : 84 * 2 = 168) :
  (168 - (item1_price + item2_price)) = 3 := by
  have item1_price := 60
  have item2_price := 105
  have total_cost := item1_price + item2_price
  have total_selling_price := 84 * 2
  rw [h1, h2] at total_cost
  rw h3
  sorry

end profit_from_sales_l383_383857


namespace wheel_radius_approx_l383_383452

noncomputable def radius_of_wheel 
  (distance : ℝ) (revolutions : ℝ) (pi : ℝ) : ℝ :=
  distance / (revolutions * 2 * pi)

theorem wheel_radius_approx 
  (distance : ℝ) (revolutions : ℝ) (pi : ℝ) 
  (h_distance : distance = 11000) 
  (h_revolutions : revolutions = 1000.4024994347707) 
  (h_pi : pi = real.pi) :
  radius_of_wheel distance revolutions pi ≈ 1.749 :=
by {
  sorry
}

end wheel_radius_approx_l383_383452


namespace sum_first_8_log_a_eq_4_l383_383279

noncomputable def a (n : ℕ) : ℝ := 2 * (5 / 2)^(n - 4)
def log_a_sum (n : ℕ) := ∑ k in (Finset.range n).map Nat.succ, Real.log (a k)

theorem sum_first_8_log_a_eq_4 :
  log_a_sum 8 = 4 := by
  sorry

end sum_first_8_log_a_eq_4_l383_383279


namespace sum_of_underlined_numbers_positive_l383_383044

theorem sum_of_underlined_numbers_positive (L : List ℤ) (h : L.length = 100) :
  let underlined := L.filter (λ x, x > 0 ∨ ∃ y ∈ L.tail, x + y > 0)
  ∑ x in underlined, x > 0 := sorry

end sum_of_underlined_numbers_positive_l383_383044


namespace hypotenuse_length_l383_383479

theorem hypotenuse_length (a b c : ℕ) (h₀ : a = 15) (h₁ : b = 36) (h₂ : a^2 + b^2 = c^2) : c = 39 :=
by
  -- Proof is omitted
  sorry

end hypotenuse_length_l383_383479


namespace janice_initial_sentences_l383_383286

theorem janice_initial_sentences:
  ∀ (r t1 t2 t3 t4: ℕ), 
  r = 6 → 
  t1 = 20 → 
  t2 = 15 → 
  t3 = 40 → 
  t4 = 18 → 
  (t1 * r + t2 * r + t4 * r - t3 = 536 - 258) → 
  536 - (t1 * r + t2 * r + t4 * r - t3) = 258 := by
  intros
  sorry

end janice_initial_sentences_l383_383286


namespace domino_coloring_possible_l383_383812

def valid_coloring (rect : Type) (color : rect → ℕ) : Prop :=
  ∀ (p1 p2 : rect), (distance p1 p2 = 1) →
    if (on_shared_edge p1 p2) then (color p1 ≠ color p2)
    else (color p1 = color p2)

theorem domino_coloring_possible (rect : Type) (unit_square : rect) :
  (divided_into_dominoes rect unit_square) →
  ∃ (color : rect → ℕ), valid_coloring rect color :=
by
  sorry

end domino_coloring_possible_l383_383812


namespace trapezoid_area_sum_l383_383118

/-
 A trapezoid has four side lengths 5, 6, 8, and 9. 
 Prove that the sum of all possible areas of the trapezoid is 28√3 + 42√2.
-/

noncomputable def height_config1 : ℝ := real.sqrt 48
noncomputable def area_config1 : ℝ := (1 / 2) * (5 + 9) * height_config1

noncomputable def height_config2 : ℝ := real.sqrt 72
noncomputable def area_config2 : ℝ := (1 / 2) * (6 + 8) * height_config2

theorem trapezoid_area_sum : area_config1 + area_config2 = 28 * real.sqrt 3 + 42 * real.sqrt 2 :=
by sorry

end trapezoid_area_sum_l383_383118


namespace correct_statements_in_triangle_l383_383282

theorem correct_statements_in_triangle (a b c : ℝ) (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π) :
  (c = a * Real.cos B + b * Real.cos A) ∧ 
  (a^3 + b^3 = c^3 → a^2 + b^2 > c^2) :=
by
  sorry

end correct_statements_in_triangle_l383_383282


namespace pascal_triangle_45th_number_l383_383013

theorem pascal_triangle_45th_number :
  let row := List.range (46 + 1) in
  row.nth 44 = some 1035 :=
by
  let row := List.range (46 + 1)
  have binom_46_2 : nat.binom 46 2 = 1035 := by
    -- Calculations for binomials can be validated here
    calc
      nat.binom 46 2 = 46 * 45 / (2 * 1) : by norm_num
      _ = 1035 : by norm_num
  show row.nth 44 = some (nat.binom 46 2) from by
    rw binom_46_2
    simp only [List.nth_range, option.some_eq_coe, nat.lt_succ_iff, nat.le_refl]
  sorry -- Additional reasoning if necessary

end pascal_triangle_45th_number_l383_383013


namespace polynomial_expression_l383_383028

theorem polynomial_expression :
  (2 * x^2 + 3 * x + 7) * (x + 1) - (x + 1) * (x^2 + 4 * x - 63) + (3 * x - 14) * (x + 1) * (x + 5) = 4 * x^3 + 4 * x^2 :=
by
  sorry

end polynomial_expression_l383_383028


namespace pastry_trick_l383_383083

theorem pastry_trick (fillings : Fin 10) (n : ℕ) :
  ∃ n, (n = 36 ∧ ∀ remaining_pastries, 
    (remaining_pastries.length = 45 - n) → 
    (∃ remaining_filling ∈ fillings, true)) := 
sorry

end pastry_trick_l383_383083


namespace output_is_78_l383_383756

def function_machine (input : ℕ) : ℕ :=
  let result := input * 3 in
  if result <= 25 then result - 7
  else (result + 3) * 2

theorem output_is_78 : function_machine 12 = 78 := sorry

end output_is_78_l383_383756


namespace find_f_of_neg3_l383_383342

noncomputable def f : ℚ → ℚ := sorry 

theorem find_f_of_neg3 (h : ∀ (x : ℚ) (hx : x ≠ 0), 5 * f (x⁻¹) + 3 * (f x) * x⁻¹ = 2 * x^2) :
  f (-3) = -891 / 22 :=
sorry

end find_f_of_neg3_l383_383342


namespace highest_seat_number_in_sample_l383_383828

theorem highest_seat_number_in_sample (total_students sample_size : ℕ)
  (sample_interval student_in_sample highest_seat : ℕ)
  (h_total_students : total_students = 56)
  (h_sample_size : sample_size = 4)
  (h_sample_interval : sample_interval = total_students / sample_size)
  (h_student_2_in_sample : student_in_sample = 2)
  (h_highest_seat : highest_seat = student_in_sample + 3 * sample_interval) :
  highest_seat = 44 :=
by
  rw [h_total_students, h_sample_size, h_sample_interval, h_student_2_in_sample, h_highest_seat]
  sorry

end highest_seat_number_in_sample_l383_383828


namespace min_value_function_l383_383952

noncomputable def function_to_minimize (x y : ℝ) : ℝ :=
  (x * y + x) / (x^2 + y^2 + 2 * y)

theorem min_value_function : 
  ∃ (x y : ℝ), (1/4 ≤ x ∧ x ≤ 3/4) ∧ (1/5 ≤ y ∧ y ≤ 2/5) ∧ function_to_minimize x y = 21/20 := 
by
  use 3/5, 2/5
  split; norm_num
  split; norm_num
  unfold function_to_minimize
  norm_num
  sorry

end min_value_function_l383_383952


namespace root_on_unit_circle_iff_mod_l383_383431

theorem root_on_unit_circle_iff_mod (n : ℕ) (hn : 0 < n) :
  (∃ z : ℂ, z^(n+1) - z^n - 1 = 0 ∧ |z| = 1) ↔ n % 6 = 4 :=
by
  sorry

end root_on_unit_circle_iff_mod_l383_383431


namespace right_triangle_hypotenuse_length_l383_383557

theorem right_triangle_hypotenuse_length (a b : ℝ) (h_triangle : a = 15 ∧ b = 36) :
  ∃ (h : ℝ), h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  · exact rfl
  · rw [h_triangle.1, h_triangle.2]
    norm_num

end right_triangle_hypotenuse_length_l383_383557


namespace hypotenuse_length_l383_383501

theorem hypotenuse_length (a b : ℤ) (h₀ : a = 15) (h₁ : b = 36) : 
  ∃ c : ℤ, c^2 = a^2 + b^2 ∧ c = 39 := 
by {
  sorry
}

end hypotenuse_length_l383_383501


namespace floor_sqrt_15_eq_3_eval_expr_l383_383609

theorem floor_sqrt_15_eq_3 : ⌊real.sqrt 15⌋ = 3 := by
  sorry

theorem eval_expr : (⌊real.sqrt 15⌋ + 1) ^ 2 = 16 := by
  have h : ⌊real.sqrt 15⌋ = 3 := floor_sqrt_15_eq_3
  rw [h]
  norm_num

end floor_sqrt_15_eq_3_eval_expr_l383_383609


namespace smallest_subtraction_divisible_l383_383869

theorem smallest_subtraction_divisible (n : ℕ) (h : n = 427751) : 
  ∃ x, (427751 - x) % 101 = 0 ∧ x = 66 :=
by
  use 66
  split
  exact sorry
  exact sorry

end smallest_subtraction_divisible_l383_383869


namespace john_drove_hours_l383_383287

theorem john_drove_hours (h : ℕ) (rate : ℕ) (lunch_hours : ℕ) (total_distance : ℕ)
  (H1 : rate = 55)
  (H2 : lunch_hours = 3)
  (H3 : total_distance = 275)
  (H4 : rate * h + rate * lunch_hours = total_distance) :
  h = 2 :=
begin
  -- proof would go here
  sorry
end

end john_drove_hours_l383_383287


namespace units_digit_sum_factorials_l383_383021

theorem units_digit_sum_factorials :
  let factorial (n : ℕ) := if n = 0 then 1 else List.product (List.range (n+1)).tail
  units_digit (1! + 2! + 3! + 4! + Σ (n >= 5), n! + (2! * 4! + 3! * 7!)) = 1 :=
by
sorry

end units_digit_sum_factorials_l383_383021


namespace max_arithmetic_sequences_from_20_terms_l383_383778

theorem max_arithmetic_sequences_from_20_terms :
  let a : ℕ → ℤ := λ n, a0 + n * d  -- define the arithmetic sequence
  let seq := [a 0, a 1, a 2, ..., a 19],  -- list of the first 20 terms of the sequence
  ∃ A B C ∈ seq, (A, B, C) are in arithmetic progression →
    (count_max_arithmetic_seq seq = 180) :=
sorry

end max_arithmetic_sequences_from_20_terms_l383_383778


namespace hypotenuse_length_l383_383491

-- Definitions for the problem
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def leg1 : ℕ := 15
def leg2 : ℕ := 36
def hypotenuse : ℕ := 39

-- Lean 4 statement
theorem hypotenuse_length (a b c : ℕ) (h : is_right_triangle a b c) (ha : a = leg1) (hb : b = leg2) :
  c = hypotenuse :=
begin
  sorry
end

end hypotenuse_length_l383_383491


namespace hypotenuse_length_l383_383487

-- Definitions for the problem
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def leg1 : ℕ := 15
def leg2 : ℕ := 36
def hypotenuse : ℕ := 39

-- Lean 4 statement
theorem hypotenuse_length (a b c : ℕ) (h : is_right_triangle a b c) (ha : a = leg1) (hb : b = leg2) :
  c = hypotenuse :=
begin
  sorry
end

end hypotenuse_length_l383_383487


namespace function_range_quadratic_function_maximum_l383_383437

-- Problem 1: Function range
theorem function_range (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 3) :
  let f := λ x, (1 / 2) ^ (-x^2 + 4 * x + 1) in
  ∃ y, y ∈ [1/32, 1/2] ∧ y = f x := sorry

-- Problem 2: Quadratic function maximum value
theorem quadratic_function_maximum (a : ℝ) :
  let f := λ x, -x^2 + 2 * a * x + 1 - a in
  (∃ x ∈ [0, 1], f x ≤ f y ∀ y ∈ [0, 1] ∧ f x = 2) → (a = -1 ∨ a = 2) := sorry

end function_range_quadratic_function_maximum_l383_383437


namespace max_value_of_expression_l383_383043

theorem max_value_of_expression (a b c : ℝ) (h : a + b + c = 0) : 
  max (abc * (\frac{1}{a} + \frac{1}{b} + \frac{1}{c})^3) = \frac{27}{8} :=
by
  sorry

end max_value_of_expression_l383_383043


namespace vasya_password_combinations_l383_383002

theorem vasya_password_combinations :
  let count := (λ (A B C : ℕ), 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    A ≠ 2 ∧ B ≠ 2 ∧ C ≠ 2 ∧ 
    (A ∈ [0, 1, 3, 4, 5, 6, 7, 8, 9]) ∧ 
    (B ∈ [0, 1, 3, 4, 5, 6, 7, 8, 9]) ∧ 
    (C ∈ [0, 1, 3, 4, 5, 6, 7, 8, 9])) 
  in ∃! n, n = 9 * 8 * 7 ∧
  (∃ A B C, count A B C) := 
by
  sorry

end vasya_password_combinations_l383_383002


namespace area_between_curves_l383_383440

-- Defining the two curves
def curve1 (x : ℝ) : ℝ := x^3 - x
def curve2 (x : ℝ) : ℝ := x^2 - (22 / 27)

-- Defining the integral bounds based on the intersection points derived
def lower_bound1 : ℝ := (1 / 6) - (sqrt 5 / 2)
def upper_bound1 : ℝ := 2 / 3
def lower_bound2 : ℝ := 2 / 3
def upper_bound2 : ℝ := (1 / 6) + (sqrt 5 / 2)

-- Define the integrals for the area between the curves
def integral1 : ℝ :=
  ∫ x in lower_bound1..upper_bound1, (curve1 x - curve2 x)
def integral2 : ℝ :=
  ∫ x in lower_bound2..upper_bound2, (curve2 x - curve1 x)

-- Prove that the total area is 13/12
theorem area_between_curves : 
  integral1 + integral2 = 13 / 12 := 
sorry

end area_between_curves_l383_383440


namespace ratio_of_area_l383_383034
   
   noncomputable def area_of_square (side : ℝ) : ℝ := side * side
   noncomputable def area_of_circle (radius : ℝ) : ℝ := Real.pi * radius * radius
   def radius_of_inscribed_circle (side : ℝ) : ℝ := side / 2
   
   theorem ratio_of_area (side : ℝ) (h : side = 6) : area_of_circle (radius_of_inscribed_circle side) / area_of_square side = Real.pi / 4 :=
   by 
     -- Use the given condition side = 6
     have h1 : radius_of_inscribed_circle side = 3 := by rw [radius_of_inscribed_circle, h]; norm_num
     have h2 : area_of_square side = 36 := by rw [area_of_square, h]; norm_num
     have h3 : area_of_circle 3 = Real.pi * 9 := by rw area_of_circle; norm_num
     -- Calculate the ratio
     rw [h1, h2, h3]
     norm_num -- This simplifies 9 * Real.pi / 36 to Real.pi / 4
   
   
end ratio_of_area_l383_383034


namespace largest_AC_value_l383_383292

theorem largest_AC_value : ∃ (a b c d : ℕ), 
  a < 20 ∧ b < 20 ∧ c < 20 ∧ d < 20 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (∃ (AC BD : ℝ), AC * BD = a * c + b * d ∧
  AC ^ 2 + BD ^ 2 = a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2 ∧
  AC = Real.sqrt 458) :=
sorry

end largest_AC_value_l383_383292


namespace correct_value_calculation_l383_383709

theorem correct_value_calculation (x : ℤ) (h : 2 * (x + 6) = 28) : 6 * x = 48 :=
by
  -- Proof steps would be here
  sorry

end correct_value_calculation_l383_383709


namespace right_triangle_hypotenuse_length_l383_383525

theorem right_triangle_hypotenuse_length :
  ∀ (a b h : ℕ), a = 15 → b = 36 → h^2 = a^2 + b^2 → h = 39 :=
by
  intros a b h ha hb hyp
  -- In the proof, we would use ha, hb, and hyp to show h = 39
  sorry

end right_triangle_hypotenuse_length_l383_383525


namespace greatest_three_digit_divisible_by_3_6_5_l383_383406

/-- Define a three-digit number and conditions for divisibility by 3, 6, and 5 -/
def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000
def is_divisible_by (n : ℕ) (d : ℕ) : Prop := d ∣ n

/-- Greatest three-digit number divisible by 3, 6, and 5 is 990 -/
theorem greatest_three_digit_divisible_by_3_6_5 : ∃ n : ℕ, is_three_digit n ∧ is_divisible_by n 3 ∧ is_divisible_by n 6 ∧ is_divisible_by n 5 ∧ n = 990 :=
sorry

end greatest_three_digit_divisible_by_3_6_5_l383_383406


namespace find_pass_percentage_set1_l383_383046

variable (P : ℕ) -- pass percentage for the first set

def num_students_set1 := 40
def num_students_set2 := 50
def num_students_set3 := 60
def num_students_total := 150

def pass_percentage_set2 := 90
def pass_percentage_set3 := 80
def overall_pass_percentage := 88.66666666666667

def students_passed_set2 := (pass_percentage_set2 / 100) * num_students_set2
def students_passed_set3 := (pass_percentage_set3 / 100) * num_students_set3
def students_passed_total := (overall_pass_percentage / 100) * num_students_total

theorem find_pass_percentage_set1 :
  let students_passed_set1 := students_passed_total - students_passed_set2 - students_passed_set3 in
  let pass_percentage_set1 := (students_passed_set1 / num_students_set1) * 100 in
  pass_percentage_set1 = 100 := sorry

end find_pass_percentage_set1_l383_383046


namespace area_of_smaller_circle_l383_383855

-- Define the conditions for the problem
structure CircleTangents (R1 R2 : ℝ) :=
(radius_ratio : R2 = 2 * R1)
(tangent_length : ∀ {A B}, ∃ P, PA = 3 ∧ AB = 3)

-- The main theorem to prove
theorem area_of_smaller_circle (R1 R2 : ℝ) (h : CircleTangents R1 R2) : 
    (π * R1^2 = 3 * π) :=
by
  -- use the conditions to solve for the area of the smaller circle
  sorry

end area_of_smaller_circle_l383_383855


namespace selling_price_ratio_l383_383464

theorem selling_price_ratio (CP SP1 SP2 : ℝ) (h1 : SP1 = CP + 0.5 * CP) (h2 : SP2 = CP + 3 * CP) :
  SP2 / SP1 = 8 / 3 :=
by
  sorry

end selling_price_ratio_l383_383464


namespace right_triangle_hypotenuse_length_l383_383530

theorem right_triangle_hypotenuse_length :
  ∀ (a b h : ℕ), a = 15 → b = 36 → h^2 = a^2 + b^2 → h = 39 :=
by
  intros a b h ha hb hyp
  -- In the proof, we would use ha, hb, and hyp to show h = 39
  sorry

end right_triangle_hypotenuse_length_l383_383530


namespace concave_number_probability_l383_383568

def is_concave (a b c : ℕ) : Prop := (a > b) ∧ (b < c)

def is_valid_digit (n : ℕ) : Prop := n ∈ {1, 2, 3, 4}

def distinct_digits (a b c : ℕ) : Prop := (a ≠ b) ∧ (a ≠ c) ∧ (b ≠ c)

theorem concave_number_probability : 
  (finset.filter (λ t, let ⟨a, b, c⟩ := t in is_concave a b c)
    (finset.univ : finset (ℕ × ℕ × ℕ)).filter (λ t, 
      let ⟨a, b, c⟩ := t in is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ distinct_digits a b c)).card
  =
  1 / 3 *
    (finset.filter (λ t, 
      let ⟨a, b, c⟩ := t in is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ distinct_digits a b c)).card :=
begin
  sorry
end

end concave_number_probability_l383_383568


namespace median_inequality_l383_383322

variables {α : Type*} [LinearOrderedField α]

def is_triangle (a b c : α) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def median (a b c : α) : α :=
  Real.sqrt ((2 * a^2 + 2 * b^2 - c^2) / 4)

theorem median_inequality {a b c : α} (h : is_triangle a b c) :
  let m_c := median a b c in (a + b - c) / 2 < m_c ∧ m_c < (a + b) / 2 :=
by
  sorry

end median_inequality_l383_383322


namespace sum_of_squares_neq_fourth_powers_l383_383592

theorem sum_of_squares_neq_fourth_powers (m n : ℕ) : 
  m^2 + (m + 1)^2 ≠ n^4 + (n + 1)^4 :=
by 
  sorry

end sum_of_squares_neq_fourth_powers_l383_383592


namespace length_of_plot_l383_383032

-- Definitions used as conditions
def cost_per_meter : ℝ := 26.50
def total_cost : ℝ := 5300
def additional_length : ℝ := 32

-- Theorem statement
theorem length_of_plot (b : ℝ) (P : ℝ) :
  P = 2 * (b + (b + additional_length)) →
  P = total_cost / cost_per_meter →
  length_of_plot = b + additional_length →
  b = 34 →
  length_of_plot = 66 :=
by
  sorry

end length_of_plot_l383_383032


namespace bezdikov_population_l383_383265

variable (W M : ℕ) -- original number of women and men
variable (W_current M_current : ℕ) -- current number of women and men

theorem bezdikov_population (h1 : W = M + 30)
                          (h2 : W_current = W / 4)
                          (h3 : M_current = M - 196)
                          (h4 : W_current = M_current + 10) : W_current + M_current = 134 :=
by
  sorry

end bezdikov_population_l383_383265


namespace find_monic_polynomial_l383_383781

noncomputable def polynomial_transformed (f : Polynomial ℝ) : Polynomial ℝ :=
  f.comp (Polynomial.X - 1)

theorem find_monic_polynomial :
  let f : Polynomial ℝ := Polynomial.X^4 - 3*Polynomial.X^3 - 4*Polynomial.X^2 + 12*Polynomial.X - 9 in
  let g : Polynomial ℝ := Polynomial.X^4 - 7*Polynomial.X^3 + 11*Polynomial.X^2 + 11*Polynomial.X - 21 in
  polynomial_transformed f = g :=
by
  sorry

end find_monic_polynomial_l383_383781


namespace minimum_pies_for_trick_l383_383093

-- Definitions from conditions
def num_fillings : ℕ := 10
def num_pastries := (num_fillings * (num_fillings - 1)) / 2
def min_pies_for_trick (n : ℕ) : Prop :=
  ∀ remaining_pies : ℕ, remaining_pies = num_pastries - n → remaining_pies ≤ 9

theorem minimum_pies_for_trick : ∃ n : ℕ, min_pies_for_trick n ∧ n = 36 :=
by
  -- We need to show that there exists n such that,
  -- min_pies_for_trick holds and n = 36
  existsi (36 : ℕ)
  -- remainder of the proof (step solution) skipped
  sorry

end minimum_pies_for_trick_l383_383093


namespace second_hand_bisects_angle_l383_383361

theorem second_hand_bisects_angle (t k : ℕ) : 
  t = (43200 * k) / 1427 → 
  let hour_hand_pos := t / 720,
      minute_hand_pos := t / 60 in 
  (let bisector_initial := (hour_hand_pos + minute_hand_pos) / 2,
       new_hour_pos := hour_hand_pos + t / 720,
       new_minute_pos := minute_hand_pos + t / 60,
       new_bisector_pos := (new_hour_pos + new_minute_pos) / 2 in
    true) := 
sorry

end second_hand_bisects_angle_l383_383361


namespace num_students_above_90_is_12_l383_383271

noncomputable def num_students_scoring_above_90 (σ : ℝ) (hσ : σ > 0) (prob : ℝ) (total_students : ℕ) :=
  if prob = 0.48 ∧ total_students = 600 then 0.02 * 600 else 0

theorem num_students_above_90_is_12 (σ : ℝ) (hσ : σ > 0) (h1 : ∀ x : ℝ, x ~ N(70, σ^2)) (h2 : prob = 0.48) (h3 : total_students = 600) : 
  num_students_scoring_above_90 σ hσ prob total_students = 12 := by
  sorry

end num_students_above_90_is_12_l383_383271


namespace monomial_count_is_4_l383_383579

def is_monomial (exp : String) : Bool :=
  match exp with
  | "a" => true
  | "-2ab" => true
  | "x+y" => false
  | "x^2+y^2" => false
  | "-1" => true
  | "1/2ab^2c^3" => true
  | _ => false

theorem monomial_count_is_4 : 
  (is_monomial "a" ?= true ∧
  is_monomial "-2ab" ?= true ∧
  is_monomial "x+y" ?= false ∧
  is_monomial "x^2+y^2" ?= false ∧
  is_monomial "-1" ?= true ∧
  is_monomial "1/2ab^2c^3" ?= true) → 
  list.count is_monomial ["a", "-2ab", "x+y", "x^2+y^2", "-1", "1/2ab^2c^3"] = 4 := 
by {
  -- Each check here could be verified individually.
  sorry
}

end monomial_count_is_4_l383_383579


namespace cotangent_cosine_relation_l383_383663

theorem cotangent_cosine_relation (A : ℝ) (hA : 0 < A) (hA_π : A < π) (cotA : Real.cot A = -12/5) : Real.cos A = -12/13 :=
by
  sorry

end cotangent_cosine_relation_l383_383663


namespace males_listen_to_station_l383_383358

theorem males_listen_to_station (
  males_dont_listen : ℕ,
  females_listen : ℕ,
  total_listen : ℕ,
  total_dont_listen : ℕ,
  total_surveyed : ℕ
) : males_listen = 84 :=
by
  have total_people := total_listen + total_dont_listen,
  have females_dont_listen := total_dont_listen - males_dont_listen,
  have total_females := females_listen + females_dont_listen,
  have total_males := total_surveyed - total_females,
  have males_listen := total_males - males_dont_listen,
  sorry

end males_listen_to_station_l383_383358


namespace transformed_g_properties_l383_383224

noncomputable def g (x : ℝ) : ℝ :=
  if x ∈ set.Icc (-3) 0 then -x - 2
  else if x ∈ set.Ioc 0 2 then real.sqrt (4 - (x - 2) ^ 2) - 2
  else if x ∈ set.Ioc 2 3 then 2 * (x - 2)
  else 0

noncomputable def transformed_g (x : ℝ) : ℝ := (1 / 3) * g(x) + 2

theorem transformed_g_properties :
  (∀ x ∈ set.Icc (-3) 0, transformed_g x = -(1 / 3) * x + 4 / 3) ∧
  (∀ x ∈ set.Ioc 0 2, transformed_g x = (1 / 3) * real.sqrt (4 - (x - 2) ^ 2) + 4 / 3) ∧
  (∀ x ∈ set.Ioc 2 3, transformed_g x = (2 / 3) * x + 2 / 3) :=
sorry

end transformed_g_properties_l383_383224


namespace area_square_ABCD_l383_383130

-- Given the conditions
def side_length_original_square : ℝ := 6
def radius_semicircle : ℝ := side_length_original_square / 2
def side_length_ABCD : ℝ := side_length_original_square + 2 * radius_semicircle
def area_ABCD : ℝ := side_length_ABCD * side_length_ABCD

-- Proof statement
theorem area_square_ABCD : 
  side_length_original_square = 6 → 
  (∀ sem : radius_semicircle = side_length_original_square / 2, 
  side_length_ABCD = side_length_original_square + 2 * radius_semicircle) → 
  area_ABCD = 144 := 
by {
  intros,
  sorry
}

end area_square_ABCD_l383_383130


namespace mean_and_variance_transformed_l383_383255

variables {X : Type*} [Nonempty X] [Fintype X]
variables (x : X → ℝ) (y : X → ℝ)
variables (n : ℕ) (S : ℝ) (mean_x : ℝ)
variables (h_mean_x : mean x = mean_x)
variables (h_var_x : var x = S^2)
variables (h_y_transformed : ∀ i, y i = 3 * x i + 5)
variables (h_card : Fintype.card X = n)

theorem mean_and_variance_transformed :
  mean y = 3 * mean_x + 5 ∧ var y = 9 * S^2 :=
by
  sorry

end mean_and_variance_transformed_l383_383255


namespace product_of_real_parts_l383_383603

theorem product_of_real_parts (x : ℂ) (h : x^2 + 2*x = 1 + complex.I*real.sqrt 3) :
  (x.re - 1) * (-1 - x.re) = -1 / 2 := 
sorry

end product_of_real_parts_l383_383603


namespace average_of_multiples_of_6_l383_383863

def first_n_multiples_sum (n : ℕ) : ℕ :=
  (n * (6 + 6 * n)) / 2

def first_n_multiples_avg (n : ℕ) : ℕ :=
  (first_n_multiples_sum n) / n

theorem average_of_multiples_of_6 (n : ℕ) : first_n_multiples_avg n = 66 → n = 11 := by
  sorry

end average_of_multiples_of_6_l383_383863


namespace shortest_distance_from_vertex_to_path_l383_383645

theorem shortest_distance_from_vertex_to_path
  (r l : ℝ)
  (hr : r = 1)
  (hl : l = 3) :
  ∃ d : ℝ, d = 1.5 :=
by
  -- Given a cone with a base radius of 1 cm and a slant height of 3 cm
  -- We need to prove the shortest distance from the vertex to the path P back to P is 1.5 cm
  sorry

end shortest_distance_from_vertex_to_path_l383_383645


namespace maximize_product_972_l383_383859

-- Define the conditions and the question
def maxProductThreeDigits (a b c d e : ℕ) : Prop :=
  a ∈ {3, 5, 7, 8, 9} ∧ b ∈ {3, 5, 7, 8, 9} ∧ c ∈ {3, 5, 7, 8, 9} ∧
  d ∈ {3, 5, 7, 8, 9} ∧ e ∈ {3, 5, 7, 8, 9} ∧
  list.nodup [a, b, c, d, e] ∧
  (odd (100 * a + 10 * b + c) ∧ even (10 * d + e) ∨
  even (100 * a + 10 * b + c) ∧ odd (10 * d + e)) ∧ 
  (∀ p q r s t : ℕ,
   p ∈ {3, 5, 7, 8, 9} ∧ q ∈ {3, 5, 7, 8, 9} ∧ r ∈ {3, 5, 7, 8, 9} ∧
   s ∈ {3, 5, 7, 8, 9} ∧ t ∈ {3, 5, 7, 8, 9} ∧ 
   list.nodup [p, q, r, s, t] ∧ 
   (odd (100 * p + 10 * q + r) ∧ even (10 * s + t) ∨ 
   even (100 * p + 10 * q + r) ∧ odd (10 * s + t)) → 
   (100 * a + 10 * b + c) * (10 * d + e) ≥ (100 * p + 10 * q + r) * (10 * s + t))

theorem maximize_product_972 :
  maxProductThreeDigits 9 7 2 8 5 :=
sorry

end maximize_product_972_l383_383859


namespace smallest_value_of_y_l383_383870

open Real

theorem smallest_value_of_y : 
  ∃ (y : ℝ), 6 * y^2 - 29 * y + 24 = 0 ∧ (∀ z : ℝ, 6 * z^2 - 29 * z + 24 = 0 → y ≤ z) ∧ y = 4 / 3 := 
sorry

end smallest_value_of_y_l383_383870


namespace fraction_product_is_243_l383_383411

/- Define the sequence of fractions -/
def fraction_seq : list ℚ :=
  [1/3, 9, 1/27, 81, 1/243, 243, 1/729, 729, 1/2187, 6561]

/- Define the product of the sequence of fractions -/
def product_fractions : ℚ :=
  (fraction_seq.foldl (*) 1)

/- The theorem we want to prove -/
theorem fraction_product_is_243 : product_fractions = 243 := 
  sorry

end fraction_product_is_243_l383_383411


namespace train_platform_length_l383_383117

theorem train_platform_length 
  (speed_train_kmph : ℕ) 
  (time_cross_platform : ℕ) 
  (time_cross_man : ℕ) 
  (L_platform : ℕ) :
  speed_train_kmph = 72 ∧ 
  time_cross_platform = 34 ∧ 
  time_cross_man = 18 ∧ 
  L_platform = 320 :=
by
  sorry

end train_platform_length_l383_383117


namespace right_triangle_hypotenuse_length_l383_383509

theorem right_triangle_hypotenuse_length (a b : ℕ) (h1 : a = 15) (h2 : b = 36) : 
  ∃ c : ℕ, c * c = a * a + b * b ∧ c = 39 := 
by
  have hyp_square := 225 + 1296 
  have h_calculation : 15 * 15 + 36 * 36 = 1521 := by
    calc
      15 * 15 = 225 : rfl
      36 * 36 = 1296 : rfl
      225 + 1296 = 1521 : rfl
  use 39
  split
  exact h_calculation
  rfl

end right_triangle_hypotenuse_length_l383_383509


namespace hundred_days_from_friday_is_sunday_l383_383389

/-- Given that today is Friday, determine that 100 days from now is Sunday. -/
theorem hundred_days_from_friday_is_sunday (today : ℕ) (days_in_week : ℕ := 7) 
(friday : ℕ := 0) (sunday : ℕ := 2) : (((today + 100) % days_in_week) = sunday) :=
sorry

end hundred_days_from_friday_is_sunday_l383_383389


namespace tg_equation_solution_l383_383030

noncomputable def tg (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

theorem tg_equation_solution (x : ℝ) (n : ℤ) 
    (h₀ : Real.cos x ≠ 0) :
    (tg x + tg (50 * Real.pi / 180) + tg (70 * Real.pi / 180) = 
    tg x * tg (50 * Real.pi / 180) * tg (70 * Real.pi / 180)) →
    (∃ n : ℤ, x = 60 * Real.pi / 180 + n * Real.pi) :=
sorry

end tg_equation_solution_l383_383030


namespace smallest_number_of_neighbors_l383_383804

noncomputable def number_of_points : ℕ := 2005
noncomputable def angle_limit : ℝ := 10

def neighbors (P Q : ℕ) (angle_subtended : ℝ) : Prop :=
  angle_subtended ≤ angle_limit

theorem smallest_number_of_neighbors :
  ∀ (points : finset ℕ)
    (angle : ℕ → ℕ → ℝ),
    points.card = number_of_points →
    (∀ P Q ∈ points, neighbors P Q (angle P Q)) →
    ∃ minimal_pairs : ℕ,
    minimal_pairs = (25 * (nat.choose 57 2)) + (10 * (nat.choose 58 2)) :=
begin
  intros points angle h_points h_neighbors,
  sorry -- Proof goes here
end

end smallest_number_of_neighbors_l383_383804


namespace line_equation_through_point_slope_l383_383172

theorem line_equation_through_point_slope :
  ∃ (a b c : ℝ), (a, b) ≠ (0, 0) ∧ (a * 1 + b * 3 + c = 0) ∧ (y = -4 * x → k = -4 / 9) ∧ (∀ (x y : ℝ), y - 3 = k * (x - 1) → 4 * x + 3 * y - 13 = 0) :=
sorry

end line_equation_through_point_slope_l383_383172


namespace geometric_sequence_a3_value_l383_383278

theorem geometric_sequence_a3_value
  {a : ℕ → ℝ}
  (h1 : a 1 + a 5 = 82)
  (h2 : a 2 * a 4 = 81)
  (h3 : ∀ n : ℕ, a (n + 1) = a n * a 3 / a 2) :
  a 3 = 9 :=
sorry

end geometric_sequence_a3_value_l383_383278


namespace median_length_of_right_triangle_l383_383681

noncomputable def length_of_median (a b c : ℕ) : ℝ := 
  if a * a + b * b = c * c then c / 2 else 0

theorem median_length_of_right_triangle :
  length_of_median 9 12 15 = 7.5 :=
by
  -- Insert the proof here
  sorry

end median_length_of_right_triangle_l383_383681


namespace no_such_n_exists_l383_383830

theorem no_such_n_exists 
  (h : ∀ n : ℕ, ¬ all_digits_appear_same_number_of_times (concat_sequence n)) 
: true :=
by {
    sorry
}

end no_such_n_exists_l383_383830


namespace wrapping_paper_length_l383_383120

-- Definitions based on conditions
def width_of_paper : ℝ := 4 -- cm
def tube_diameter_final : ℝ := 15 -- cm
def num_wraps : ℝ := 450
def tube_diameter_initial : ℝ := 1 -- cm
def additional_length_factor : ℝ := 1.10

-- Proof statement to be verified in Lean
theorem wrapping_paper_length :
  (sum (λ k, (tube_diameter_initial + width_of_paper * k) * π) (range (num_wraps.to_nat))) * additional_length_factor / 100 = 2227.5 * π :=
sorry

end wrapping_paper_length_l383_383120


namespace sum_of_a_for_quadratic_has_one_solution_l383_383957

noncomputable def discriminant (a : ℝ) : ℝ := (a + 12)^2 - 4 * 3 * 16

theorem sum_of_a_for_quadratic_has_one_solution : 
  (∀ a : ℝ, discriminant a = 0) → 
  (-12 + 8 * Real.sqrt 3) + (-12 - 8 * Real.sqrt 3) = -24 :=
by
  intros h
  simp [discriminant] at h
  sorry

end sum_of_a_for_quadratic_has_one_solution_l383_383957


namespace find_X_eq_A_l383_383624

variable {α : Type*}
variable (A X : Set α)

theorem find_X_eq_A (h : X ∩ A = X ∪ A) : X = A := by
  sorry

end find_X_eq_A_l383_383624


namespace midpoint_distance_from_school_l383_383922

def distance_school_kindergarten_km := 1
def distance_school_kindergarten_m := 700
def distance_kindergarten_house_m := 900

theorem midpoint_distance_from_school : 
  (1000 * distance_school_kindergarten_km + distance_school_kindergarten_m + distance_kindergarten_house_m) / 2 = 1300 := 
by
  sorry

end midpoint_distance_from_school_l383_383922


namespace find_m_value_l383_383662

theorem find_m_value (m : ℝ) (h1 : ∀ x : ℝ, y = (m + 1)x^{3 - |m|} + 2 → y = a * x + b) (h2 : ∀ x : ℝ, ∀ a b : ℝ, y = (m + 1) * x + b → y < (m + 1) * (x + 1) + b) : 
  m = -2 :=
sorry

end find_m_value_l383_383662


namespace fraction_value_l383_383300

theorem fraction_value (p q r s : ℝ) (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -4 / 7 :=
by
  sorry

end fraction_value_l383_383300


namespace count_numbers_containing_zero_l383_383235

def contains_zero (n : ℕ) : Prop :=
  ∃ d : ℕ, (d < 10) ∧ (n % 10^(d+1) / 10^d = 0)

theorem count_numbers_containing_zero : 
  finset.card (finset.filter contains_zero (finset.range 3201)) = 993 :=
by {
  -- Proof would go here
  sorry
}

end count_numbers_containing_zero_l383_383235


namespace evaluate_f_g_3_l383_383248

def g (x : ℝ) := x^3
def f (x : ℝ) := 3 * x - 2

theorem evaluate_f_g_3 : f (g 3) = 79 := by
  sorry

end evaluate_f_g_3_l383_383248


namespace solve_system_of_equations_l383_383794

open Complex

noncomputable section

def system_of_equations (n : ℕ) (x : Fin n → ℂ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → ∑ i : Fin n, (i.1 + 1 : ℂ) * x i ^ k = 0

theorem solve_system_of_equations (n : ℕ) (x : Fin n → ℂ) :
  system_of_equations n x → ∀ i, x i = 0 :=
  by
  sorry

end solve_system_of_equations_l383_383794


namespace trigonometric_sum_identites_proof_l383_383588

noncomputable def problem_statement : Prop :=
  sin (63 * Real.pi / 180) * cos (18 * Real.pi / 180) + 
  cos (63 * Real.pi / 180) * cos (108 * Real.pi / 180) = 
  Real.sqrt 2 / 2

theorem trigonometric_sum_identites_proof : problem_statement := 
by 
  sorry

end trigonometric_sum_identites_proof_l383_383588


namespace right_triangle_hypotenuse_length_l383_383510

theorem right_triangle_hypotenuse_length (a b : ℕ) (h1 : a = 15) (h2 : b = 36) : 
  ∃ c : ℕ, c * c = a * a + b * b ∧ c = 39 := 
by
  have hyp_square := 225 + 1296 
  have h_calculation : 15 * 15 + 36 * 36 = 1521 := by
    calc
      15 * 15 = 225 : rfl
      36 * 36 = 1296 : rfl
      225 + 1296 = 1521 : rfl
  use 39
  split
  exact h_calculation
  rfl

end right_triangle_hypotenuse_length_l383_383510


namespace right_triangle_hypotenuse_l383_383517

theorem right_triangle_hypotenuse (a b : ℕ) (ha : a = 15) (hb : b = 36) : 
  ∃ h : ℕ, h = 39 ∧ h^2 = a^2 + b^2 :=
by
  use 39
  split
  . exact rfl
  . rw [ha, hb]
    norm_num
    sorry

end right_triangle_hypotenuse_l383_383517


namespace camila_weeks_to_goal_l383_383144

open Nat

noncomputable def camila_hikes : ℕ := 7
noncomputable def amanda_hikes : ℕ := 8 * camila_hikes
noncomputable def steven_hikes : ℕ := amanda_hikes + 15
noncomputable def additional_hikes_needed : ℕ := steven_hikes - camila_hikes
noncomputable def hikes_per_week : ℕ := 4
noncomputable def weeks_to_goal : ℕ := additional_hikes_needed / hikes_per_week

theorem camila_weeks_to_goal : weeks_to_goal = 16 :=
  by sorry

end camila_weeks_to_goal_l383_383144


namespace distinct_patterns_4x3_grid_l383_383233

theorem distinct_patterns_4x3_grid : 
  ∀ (patterns : set (set (nat × nat))),
  (∀ p ∈ patterns, ∃ a b c, 
    p = {(a div 3, a mod 3), (b div 3, b mod 3), (c div 3, c mod 3)} ∧ ∀ i j ∈ p, i ≠ j ∧ j ≠ c ∧ c ≠ i) →
  (∀ p1 p2 ∈ patterns, 
    (p1 = p2 ∨ p1 = rotate p2 ∨ p1 = flip p2) ↔ p1 = p2) → 
  (∀ s1 s2 ∈ patterns, s1 ≠ s2)  
  → ∃ (count : ℕ), count = 7 :=
begin
  sorry
end

end distinct_patterns_4x3_grid_l383_383233


namespace businessman_l383_383443

theorem businessman's_total_income :
  (I : ℝ) →
  (I * 0.20) + (I * 0.22) + (I * 0.15) + (I * 0.10) + (I * 0.05) + ((I * 0.28) * 0.08) = I - 25000 →
  I = 97049.68 :=
begin
  sorry
end

end businessman_l383_383443


namespace minimum_value_l383_383178

noncomputable def polynomial_expr (x y : ℝ) : ℝ :=
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 5

theorem minimum_value : ∃ x y : ℝ, (polynomial_expr x y = 8) := 
sorry

end minimum_value_l383_383178


namespace hypotenuse_right_triangle_l383_383543

theorem hypotenuse_right_triangle (a b : ℕ) (h1 : a = 15) (h2 : b = 36) :
  ∃ c, c ^ 2 = a ^ 2 + b ^ 2 ∧ c = 39 :=
by
  sorry

end hypotenuse_right_triangle_l383_383543


namespace right_triangle_hypotenuse_l383_383541

theorem right_triangle_hypotenuse (a b : ℝ) (h : a^2 + b^2 = 39^2) : a = 15 ∧ b = 36 := by
  sorry

end right_triangle_hypotenuse_l383_383541


namespace camila_weeks_to_goal_l383_383145

open Nat

noncomputable def camila_hikes : ℕ := 7
noncomputable def amanda_hikes : ℕ := 8 * camila_hikes
noncomputable def steven_hikes : ℕ := amanda_hikes + 15
noncomputable def additional_hikes_needed : ℕ := steven_hikes - camila_hikes
noncomputable def hikes_per_week : ℕ := 4
noncomputable def weeks_to_goal : ℕ := additional_hikes_needed / hikes_per_week

theorem camila_weeks_to_goal : weeks_to_goal = 16 :=
  by sorry

end camila_weeks_to_goal_l383_383145


namespace hundred_days_from_friday_is_sunday_l383_383392

/-- Given that today is Friday, determine that 100 days from now is Sunday. -/
theorem hundred_days_from_friday_is_sunday (today : ℕ) (days_in_week : ℕ := 7) 
(friday : ℕ := 0) (sunday : ℕ := 2) : (((today + 100) % days_in_week) = sunday) :=
sorry

end hundred_days_from_friday_is_sunday_l383_383392


namespace john_completes_fourth_task_at_12_20_pm_l383_383288

noncomputable def task_time (start_time end_time : ℕ) : ℕ := end_time - start_time

theorem john_completes_fourth_task_at_12_20_pm
    (start_time : ℕ := 9 * 60)  -- 9:00 AM in minutes from midnight
    (end_time_third_task : ℕ := 11 * 60 + 30)  -- 11:30 AM in minutes from midnight
    (tasks_count : ℕ := 4)
    (equal_time_consuming_tasks : Prop :=
        let total_time := task_time start_time end_time_third_task in
        total_time % (tasks_count - 1) = 0) : Prop :=
    let total_time := task_time start_time end_time_third_task,
        task_duration := total_time / (tasks_count - 1)
    in end_time_third_task + task_duration = 12 * 60 + 20  -- 12:20 PM

end john_completes_fourth_task_at_12_20_pm_l383_383288


namespace domain_length_of_f_l383_383951

noncomputable def log8 : ℝ → ℝ := λ x, Real.log x / Real.log 8
noncomputable def log_quarter_base : ℝ → ℝ := λ x, Real.log x / Real.log (1/4)
noncomputable def log2 : ℝ → ℝ := λ x, Real.log x / Real.log 2
noncomputable def log_half_base : ℝ → ℝ := λ x, Real.log x / Real.log (1/2)
noncomputable def log4 : ℝ → ℝ := λ x, Real.log x / Real.log 4

noncomputable def f (x : ℝ) : ℝ :=
  log4 (log_half_base (log2 (log_quarter_base (log8 x))))

theorem domain_length_of_f :
  ∃ (p q : ℕ), p.gcd q = 1 ∧ (2 - 1 = 1 ∧ p + q = 2) :=
by
  use 1
  use 1
  split
  . exact Nat.gcd_one_right 1
  split
  . norm_num
  . norm_num

end domain_length_of_f_l383_383951


namespace limit_cos_series_eq_l383_383177

noncomputable def limit_of_cos_series (x : ℝ) : ℝ :=
  lim (λ n : ℕ, (finset.range n).sum (λ k, (1 : ℝ) / (2 : ℝ) ^ k * real.cos ((k : ℝ) * x) + 1))

theorem limit_cos_series_eq (x : ℝ) :
  limit_of_cos_series x = (2 * (2 - real.cos x)) / (5 - 4 * real.cos x) :=
sorry

end limit_cos_series_eq_l383_383177


namespace marla_colors_blue_rows_l383_383309

theorem marla_colors_blue_rows :
  ∃ blue_rows : ℕ,
  let total_squares := 10 * 15,
      red_squares := 4 * 6,
      green_squares := 66,
      blue_squares := total_squares - red_squares - green_squares,
      rows_with_blue := blue_squares / 15 in
  blue_squares % 15 = 0 ∧ rows_with_blue = 4 := 
by
  sorry

end marla_colors_blue_rows_l383_383309


namespace largest_real_x_is_120_over_11_l383_383175

noncomputable def largest_real_x (x : ℝ) : Prop :=
  floor x / x = 11 / 12

theorem largest_real_x_is_120_over_11 :
  ∃ x, largest_real_x x ∧ x ≤ 120 / 11 :=
sorry

end largest_real_x_is_120_over_11_l383_383175


namespace limit_tangent_logarithm_l383_383936

noncomputable def limit_function (a : ℝ) : ℝ :=
  lim (λ x : ℝ, (tan x - tan a) / (log x - log a)) a

theorem limit_tangent_logarithm (a : ℝ) (h : cos a ≠ 0) :
  limit_function a = a / (cos a)^2 :=
by sorry

end limit_tangent_logarithm_l383_383936


namespace matthew_crackers_l383_383798

theorem matthew_crackers (initial_crackers given_away_crackers crackers_per_friend : ℕ) 
  (h_initial : initial_crackers = 23)
  (h_remaining : initial_crackers - given_away_crackers = 11)
  (h_each_ate : crackers_per_friend = 6) :
  given_away_crackers / crackers_per_friend = 2 :=
by
  suffices h_given_away : given_away_crackers = 12 by
    rw [h_given_away, h_each_ate]
    exact Nat.div_self (by norm_num : 6 ≠ 0)
  calc
    given_away_crackers = initial_crackers - 11 : by rw h_remaining
    ... = 12 : by rw h_initial; norm_num

end matthew_crackers_l383_383798


namespace limit_tangent_logarithm_l383_383937

noncomputable def limit_function (a : ℝ) : ℝ :=
  lim (λ x : ℝ, (tan x - tan a) / (log x - log a)) a

theorem limit_tangent_logarithm (a : ℝ) (h : cos a ≠ 0) :
  limit_function a = a / (cos a)^2 :=
by sorry

end limit_tangent_logarithm_l383_383937


namespace day_of_week_in_100_days_l383_383401

theorem day_of_week_in_100_days (start_day : ℕ) (h : start_day = 5) : 
  (start_day + 100) % 7 = 0 := 
by
  cases h with 
  | rfl => -- start_day is Friday, which is represented as 5
  sorry

end day_of_week_in_100_days_l383_383401


namespace solve_for_x_l383_383715

theorem solve_for_x (x : ℝ) (h : 0.20 * x = 0.15 * 1500 - 15) : x = 1050 := 
by
  sorry

end solve_for_x_l383_383715


namespace Part1_Part2_l383_383673

section Part1

def f_a2 (x : ℝ) : ℝ := |x + 2| + |x - 3|

theorem Part1 (x : ℝ) : f_a2 x ≥ 2 * x ↔ x ∈ set.Iic (5 / 2) :=
by
  sorry

end Part1

section Part2

variable (a : ℝ)
def f_a (x : ℝ) : ℝ := |x + a| + |x - 3|

theorem Part2 : (∃ x : ℝ, f_a a x ≤ (1 / 2) * a + 5) ↔ a ∈ set.Icc (-16 / 3) 4 :=
by
  sorry

end Part2

end Part1_Part2_l383_383673


namespace sum_of_first_2011_terms_is_3028_l383_383833

def sequence : ℕ → ℕ
| 0       := 1 
| 1       := 7
| 2       := 8
| (n + 3) := (sequence(n) + sequence(n + 1) + sequence(n + 2)) % 4

def sum_seq_first_2011_terms : ℕ :=
  (Finset.range 2011).sum sequence

theorem sum_of_first_2011_terms_is_3028 : sum_seq_first_2011_terms = 3028 :=
  by
    sorry

end sum_of_first_2011_terms_is_3028_l383_383833


namespace smallest_n_for_trick_l383_383085

theorem smallest_n_for_trick (fillings : Finset Fin 10)
  (pastries : Finset (Fin 45)) 
  (has_pairs : ∀ p ∈ pastries, ∃ f1 f2 ∈ fillings, f1 ≠ f2 ∧ p = pair f1 f2) : 
  ∃ n (tray : Finset (Fin 45)), 
    (tray.card = n ∧ n = 36 ∧ 
    ∀ remaining_p ∈ pastries \ tray, ∃ f ∈ fillings, f ∈ remaining_p) :=
by
  sorry

end smallest_n_for_trick_l383_383085


namespace ravi_overall_profit_l383_383326

variable (cost_refrigerator : ℕ) (cost_mobile : ℕ)
variable (loss_percent_refrigerator : ℕ) (profit_percent_mobile : ℕ)

def selling_price_refrigerator (cost_refrigerator : ℕ) (loss_percent_refrigerator : ℕ) : ℕ :=
  cost_refrigerator - cost_refrigerator * loss_percent_refrigerator / 100

def selling_price_mobile (cost_mobile : ℕ) (profit_percent_mobile : ℕ) : ℕ :=
  cost_mobile + cost_mobile * profit_percent_mobile / 100

def total_cost_price (cost_refrigerator : ℕ) (cost_mobile : ℕ) : ℕ :=
  cost_refrigerator + cost_mobile

def total_selling_price (selling_price_refrigerator : ℕ) (selling_price_mobile : ℕ) : ℕ :=
  selling_price_refrigerator + selling_price_mobile

def overall_profit (total_selling_price : ℕ) (total_cost_price : ℕ) : ℕ :=
  total_selling_price - total_cost_price

theorem ravi_overall_profit 
  (cost_refrigerator = 15000)  
  (cost_mobile = 8000) 
  (loss_percent_refrigerator = 3) 
  (profit_percent_mobile = 10) :
  overall_profit 
    (total_selling_price 
      (selling_price_refrigerator cost_refrigerator loss_percent_refrigerator) 
      (selling_price_mobile cost_mobile profit_percent_mobile))
    (total_cost_price cost_refrigerator cost_mobile) = 350 := 
sorry

end ravi_overall_profit_l383_383326


namespace gcd_digit_bound_l383_383727

theorem gcd_digit_bound (a b : ℕ) (h1 : a < 10^7) (h2 : b < 10^7) (h3 : 10^10 ≤ Nat.lcm a b) :
  Nat.gcd a b < 10^4 :=
by
  sorry

end gcd_digit_bound_l383_383727


namespace triangle_side_range_l383_383760

theorem triangle_side_range (A B C : Type) [has_lt A] [has_sub A] [has_add A] [has_of_nat A] [has_to String A]
  (AB AC BC : A) (hAB : AB = 3) (hAC : AC = 5) :
  2 < BC ∧ BC < 8 := 
sorry

end triangle_side_range_l383_383760


namespace option_d_is_true_l383_383217

theorem option_d_is_true (x y : ℝ) (h : x > y) : 2^x + 2^(-y) > 2 :=
sorry

end option_d_is_true_l383_383217


namespace rectangle_area_l383_383904

theorem rectangle_area (r : ℝ) (L W : ℝ) (h₀ : r = 7) (h₁ : 2 * r = W) (h₂ : L / W = 3) : 
  L * W = 588 :=
by sorry

end rectangle_area_l383_383904


namespace exponents_multiplication_exponents_power_exponents_distributive_l383_383591

variables (x y m : ℝ)

theorem exponents_multiplication (x : ℝ) : (x^5) * (x^2) = x^7 :=
by sorry

theorem exponents_power (m : ℝ) : (m^2)^4 = m^8 :=
by sorry

theorem exponents_distributive (x y : ℝ) : (-2 * x * y^2)^3 = -8 * x^3 * y^6 :=
by sorry

end exponents_multiplication_exponents_power_exponents_distributive_l383_383591


namespace calc_difference_l383_383707

theorem calc_difference :
  let a := (7/12 : ℚ) * 450
  let b := (3/5 : ℚ) * 320
  let c := (5/9 : ℚ) * 540
  let d := b + c
  d - a = 229.5 := by
  -- declare the variables and provide their values
  sorry

end calc_difference_l383_383707


namespace find_area_triangle_ABC_l383_383038

open Real
open Geometry

noncomputable def triangle_ABC_area (A B C D D1 D2 : Point) : ℝ :=
if h : IsAcuteTriangle A B C ∧
        OnLineSegment A B D ∧
        Angle D C A = 45 ∧
        SymmetricWithRespectToLine D1 D BC ∧
        SymmetricWithRespectToLine D2 D1 AC ∧
        OnExtension B C D2 ∧
        Distance B C = sqrt 3 * Distance C D2 ∧
        Distance A B = 4
then
  area_of_triangle A B C
else
  0

theorem find_area_triangle_ABC
  (A B C D D1 D2 : Point)
  (h_acute : IsAcuteTriangle A B C)
  (h_D_on_AB : OnLineSegment A B D)
  (h_angle_DCA : Angle D C A = 45)
  (h_sym_D1_D : SymmetricWithRespectToLine D1 D BC)
  (h_sym_D2_D1 : SymmetricWithRespectToLine D2 D1 AC)
  (h_D2_on_extension : OnExtension B C D2)
  (h_BC_CDstretch : Distance B C = sqrt 3 * Distance C D2)
  (h_AB_length : Distance A B = 4)
  : triangle_ABC_area A B C D D1 D2 = 4 := by
  sorry


end find_area_triangle_ABC_l383_383038


namespace hypotenuse_length_l383_383494

theorem hypotenuse_length (a b : ℤ) (h₀ : a = 15) (h₁ : b = 36) : 
  ∃ c : ℤ, c^2 = a^2 + b^2 ∧ c = 39 := 
by {
  sorry
}

end hypotenuse_length_l383_383494


namespace problem_statement_l383_383242

theorem problem_statement (a b : ℝ) (h : (a - 1)^2 + |b + 2| = 0) : (a + b)^2023 = -1 :=
begin
  sorry
end

end problem_statement_l383_383242


namespace hyperbola_problem_ellipse_problem_l383_383997

noncomputable def hyperbola := λ a b : ℝ, (λ (x y : ℝ), (x ^ 2) / (a ^ 2) - (y ^ 2) / (b ^ 2) = 1)
noncomputable def ellipse := λ a b : ℝ, (λ (x y : ℝ), (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1)

-- Hyperbola problem
theorem hyperbola_problem (a b : ℝ)
  (ha : 0 < a) (hb : a < b)
  (p1 p2 : ℝ × ℝ)
  (cond1 : hyperbola a b p1.fst p1.snd)
  (cond2 : hyperbola a b p2.fst p2.snd)
  (orthogonal : p1.fst * p2.fst + p1.snd * p2.snd = 0) :
  (1 / (p1.fst ^ 2 + p1.snd ^ 2)) + (1 / (p2.fst ^ 2 + p2.snd ^ 2)) = 1 / (a ^ 2) - 1 / (b ^ 2) :=
sorry

-- Ellipse problem
theorem ellipse_problem (a b : ℝ)
  (ha : 0 < a) (hb : a < b)
  (p1 p2 : ℝ × ℝ)
  (cond1 : ellipse a b p1.fst p1.snd)
  (cond2 : ellipse a b p2.fst p2.snd)
  (orthogonal : p1.fst * p2.fst + p1.snd * p2.snd = 0) :
  (1 / (p1.fst ^ 2 + p1.snd ^ 2)) + (1 / (p2.fst ^ 2 + p2.snd ^ 2)) = 1 / (a ^ 2) + 1 / (b ^ 2) :=
sorry

end hyperbola_problem_ellipse_problem_l383_383997


namespace hypotenuse_right_triangle_l383_383542

theorem hypotenuse_right_triangle (a b : ℕ) (h1 : a = 15) (h2 : b = 36) :
  ∃ c, c ^ 2 = a ^ 2 + b ^ 2 ∧ c = 39 :=
by
  sorry

end hypotenuse_right_triangle_l383_383542


namespace right_triangle_acute_angles_l383_383842

theorem right_triangle_acute_angles (A B C : Point)
  (h : Triangle ABC)
  (angle_C_90 : angle A C B = π / 2)
  (R r : ℝ) 
  (circumcircle : Circle) 
  (incircle : Circle)
  (ratio_radii : circumcircle.radius / incircle.radius = 5 / 2)
  (circumcircle_def : circumcircle.center = midpoint A B)
  (incircle_def : incircle.center = incenter A B C) :
  angle A B C = arcsin (3 / 5) ∧ angle B A C = arcsin (4 / 5) := 
sorry

end right_triangle_acute_angles_l383_383842


namespace f_50_value_l383_383298

def f : ℝ → ℝ := sorry

axiom f_condition : ∀ x : ℝ, f (x^2 + x) + 2 * f (x^2 - 3*x + 2) = 9 * x^2 - 15 * x

theorem f_50_value : f 50 = 146 :=
by
  sorry

end f_50_value_l383_383298


namespace lines_through_single_point_or_parallel_l383_383648

theorem lines_through_single_point_or_parallel
  (O A : Point)
  {angle : Angle}
  (A_inside : A ∈ interior angle)
  {M N : Point}
  (M_side : M ∈ side_1 angle)
  (N_side : N ∈ side_2 angle)
  (angles_equal : ∠M A O = ∠O A N) :
  ∃ R : Point, ∀ M' N' : Point, M' ∈ side_1 angle → N' ∈ side_2 angle → ∠M' A O = ∠O A N' → line_through M' N' = line_through R ∨ parallel (line_through M' N') (line_through M N).
sorry

end lines_through_single_point_or_parallel_l383_383648


namespace tax_rate_calculation_l383_383429

theorem tax_rate_calculation (price_before_tax total_price : ℝ) 
  (h_price_before_tax : price_before_tax = 92) 
  (h_total_price : total_price = 98.90) : 
  (total_price - price_before_tax) / price_before_tax * 100 = 7.5 := 
by 
  -- Proof will be provided here.
  sorry

end tax_rate_calculation_l383_383429


namespace right_triangle_hypotenuse_l383_383532

theorem right_triangle_hypotenuse (a b : ℝ) (h : a^2 + b^2 = 39^2) : a = 15 ∧ b = 36 := by
  sorry

end right_triangle_hypotenuse_l383_383532


namespace Q_trajectory_circle_l383_383651

open Set

variable {F1 F2 P Q : Point}
variable (a : ℝ)

-- Assuming P is on the ellipse with foci F1, F2 and major axis length 2a
axiom on_ellipse (P : Point) : dist P F1 + dist P F2 = 2 * a

-- Condition: F1P extended to Q such that |PQ| = |PF2|
axiom PQ_eq_PF2 : dist P Q = dist P F2

-- Given conditions and facts, prove that Q forms a circle with center F1 and radius 2a
theorem Q_trajectory_circle : is_circle F1 Q (2 * a) :=
by
  sorry

end Q_trajectory_circle_l383_383651


namespace fraction_product_is_729_l383_383414

-- Define the sequence of fractions
noncomputable def fraction_sequence : List ℚ :=
  [1/3, 9, 1/27, 81, 1/243, 729, 1/729, 6561, 1/2187, 59049, 1/6561, 59049/81]

-- Define the condition that each supposed pair product is 3
def pair_product_condition (seq : List ℚ) (i : ℕ) : Prop :=
  seq[2 * i] * seq[2 * i + 1] = 3

-- State the main theorem
theorem fraction_product_is_729 :
  fraction_sequence.prod = 729 := by
  sorry

end fraction_product_is_729_l383_383414


namespace correct_quotient_is_approx_l383_383186

theorem correct_quotient_is_approx (q r : ℝ) (h₁ : q = 21) (h₂ : r = 2) : 
  let number := q * 1.8 + r in 
  (number / 18) ≈ 2.21 :=
by
  have := ((21 * 1.8 + 2) / 18) 
  have := 2.21 -- approx statement to two decimal places
  sorry -- will need a more precise approximation statement

end correct_quotient_is_approx_l383_383186


namespace stanley_ran_distance_l383_383340

variable walk_distance run_distance : ℝ

-- Stanley walked 0.2 miles.
def walk_distance := 0.2

-- Stanley ran 0.2 miles farther than he walked.
def run_distance := walk_distance + 0.2

-- Proof statement: How far did Stanley run?
theorem stanley_ran_distance : run_distance = 0.4 := by
  -- proof steps would go here
  sorry

end stanley_ran_distance_l383_383340


namespace ratio_of_wealth_l383_383946

variable (c d e f p w : ℝ) -- these are all real numbers representing percentages and total amounts
variables (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0) -- positive percentages

-- Definition of population and wealth calculations
def population_A := p * (c / 100)
def wealth_A := w * (d / 100)
def population_B := p * (e / 100)
def wealth_B := w * (f / 100)

-- Definition of per capita wealth
def per_capita_wealth_A := wealth_A / population_A
def per_capita_wealth_B := wealth_B / population_B

-- Main theorem to be proven
theorem ratio_of_wealth (h_positive_p : p > 0) (h_positive_w : w > 0) :
    (per_capita_wealth_A c d p w) / (per_capita_wealth_B e f p w) = (d * e) / (c * f) :=
by
  sorry

end ratio_of_wealth_l383_383946


namespace butter_per_gallon_l383_383586

-- Define all conditions as hypotheses
def price_per_gallon : ℝ := 3
def price_per_stick : ℝ := 1.5
def total_cows : ℕ := 12
def milk_per_cow : ℝ := 4
def total_customers : ℕ := 6
def milk_per_customer : ℝ := 6
def total_earnings : ℝ := 144

-- Total milk produced by all cows
def total_milk : ℝ := total_cows * milk_per_cow

-- Total milk sold to all customers
def milk_sold : ℝ := total_customers * milk_per_customer

-- Revenue from milk sales
def revenue_from_milk : ℝ := milk_sold * price_per_gallon

-- Revenue from butter sales
def revenue_from_butter : ℝ := total_earnings - revenue_from_milk

-- Butter sales in sticks
def sticks_of_butter_sold : ℝ := revenue_from_butter / price_per_stick

-- Milk left for butter
def milk_for_butter : ℝ := total_milk - milk_sold

-- The theorem statement
theorem butter_per_gallon (price_per_gallon price_per_stick : ℝ)
    (total_cows milk_per_cow total_customers milk_per_customer : ℕ)
    (total_earnings : ℝ)
    (h1 : price_per_gallon = 3)
    (h2 : price_per_stick = 1.5)
    (h3 : total_cows = 12)
    (h4 : milk_per_cow = 4)
    (h5 : total_customers = 6)
    (h6 : milk_per_customer = 6)
    (h7 : total_earnings = 144) :
    let total_milk := total_cows * milk_per_cow,
        milk_sold := total_customers * milk_per_customer,
        revenue_from_milk := milk_sold * price_per_gallon,
        revenue_from_butter := total_earnings - revenue_from_milk,
        sticks_of_butter_sold := revenue_from_butter / price_per_stick,
        milk_for_butter := total_milk - milk_sold in
    sticks_of_butter_sold / milk_for_butter = 2 :=
by
  -- Proof goes here
  sorry

end butter_per_gallon_l383_383586


namespace trigonometric_expression_value_l383_383183

theorem trigonometric_expression_value :
  let c := real.cos (real.pi / 18)
  let s := real.sin (17 * real.pi / 18)
  c = real.sin (4 * real.pi / 9) → s = -real.sin (real.pi / 18) →
  (sqrt 3 / c - 1 / s) = -4 :=
begin
  intros c_eq_s s_eq_neg_s,
  sorry,
end

end trigonometric_expression_value_l383_383183


namespace find_certain_number_l383_383047

theorem find_certain_number (x : ℝ) (h : 0.7 * x = 28) : x = 40 := 
by
  sorry

end find_certain_number_l383_383047


namespace problem1_problem2_l383_383436

/-- Problem 1: Calculate the expression. -/
theorem problem1 : abs (sqrt 5 - 1) - (pi - 2)^0 + (-1/2)^(-2) = sqrt 5 + 2 :=
by sorry

/-- Problem 2: Simplify the given expression. -/
theorem problem2 (x : ℝ) : (3/(x + 1) - x + 1) / ((x^2 + 4*x + 4) / (x + 1)) = (2 - x) / (x + 2) :=
by sorry

end problem1_problem2_l383_383436


namespace profit_percentage_correct_l383_383471

-- Definitions from conditions
def market_price_per_pen : ℝ := 1
def discount : ℝ := 0.01
def cost_price : ℝ := 36
def selling_price_per_pen := market_price_per_pen * (1 - discount)
def total_selling_price := 60 * selling_price_per_pen
def profit := total_selling_price - cost_price
def profit_percentage := (profit / cost_price) * 100

-- Theorem to prove
theorem profit_percentage_correct : profit_percentage = 65 :=
by
  sorry

end profit_percentage_correct_l383_383471


namespace intersection_not_contains_neg1_l383_383723

open Set

variable (A : Set ℤ)
variable (B : Set ℕ)

theorem intersection_not_contains_neg1 (hA : A = { x : ℤ | x < 3 }) (hB : B ⊆ { x : ℕ | true }) : ¬ (SetOf (-1) ⊆ A ∩ B) :=
sorry

end intersection_not_contains_neg1_l383_383723


namespace arithmetic_sequence_a20_l383_383667

theorem arithmetic_sequence_a20 (a : ℕ → ℝ) (d : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 1 + a 3 + a 5 = 18)
  (h3 : a 2 + a 4 + a 6 = 24) :
  a 20 = 40 :=
sorry

end arithmetic_sequence_a20_l383_383667


namespace find_p_from_roots_l383_383666

-- Defining the polynomial and its roots
variables (m n p : ℝ) (x_3 : ℝ) (hx3_pos : 0 < x_3)
variables (hpoly : ∀ x, (3 * x^3 + m * x^2 + n * x + p = 0) ↔ 
  (x = 4 + 3 * complex.I ∨ x = 4 - 3 * complex.I ∨ x = x_3))

theorem find_p_from_roots (hroots : p = -75 * x_3) : p = -75 * x_3 :=
by
  exact hroots

end find_p_from_roots_l383_383666


namespace vasya_pastry_trick_l383_383073

theorem vasya_pastry_trick :
  ∀ (pastries : Finset (Finset Nat))
    (filling_set : Finset Nat),
    (filling_set.card = 10) →
    (pastries.card = 45) →
    (∀ p ∈ pastries, p.card = 2 ∧ p ⊆ filling_set) →
    ∃ n, n = 36 ∧
    ∀ remain_p ∈ (pastries \ pastries.sort (λ x y, x < y)).take (45 - n), 
      ∃ f ∈ filling_set, f ∈ remain_p :=
begin
  sorry

end vasya_pastry_trick_l383_383073


namespace speed_of_stream_l383_383871

theorem speed_of_stream (c v : ℝ) (h1 : c - v = 8) (h2 : c + v = 12) : v = 2 :=
by {
  -- proof will go here
  sorry
}

end speed_of_stream_l383_383871
