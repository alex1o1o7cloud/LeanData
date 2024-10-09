import Mathlib

namespace min_rice_proof_l247_24799

noncomputable def minRicePounds : ℕ := 2

theorem min_rice_proof (o r : ℕ) (h1 : o ≥ 8 + 3 * r / 4) (h2 : o ≤ 5 * r) :
  r ≥ 2 :=
by
  sorry

end min_rice_proof_l247_24799


namespace sets_of_bleachers_l247_24733

def totalFans : ℕ := 2436
def fansPerSet : ℕ := 812

theorem sets_of_bleachers (n : ℕ) (h : totalFans = n * fansPerSet) : n = 3 :=
by {
    sorry
}

end sets_of_bleachers_l247_24733


namespace find_length_of_rectangular_playground_l247_24756

def perimeter (L B : ℕ) : ℕ := 2 * (L + B)

theorem find_length_of_rectangular_playground (P B : ℕ) (hP : P = 1200) (hB : B = 500) : ∃ L, perimeter L B = P ∧ L = 100 :=
by
  sorry

end find_length_of_rectangular_playground_l247_24756


namespace additional_rate_of_interest_l247_24797

variable (P A A' : ℝ) (T : ℕ) (R : ℝ)

-- Conditions
def principal_amount := (P = 8000)
def original_amount := (A = 9200)
def time_period := (T = 3)
def new_amount := (A' = 9440)

-- The Lean statement to prove the additional percentage of interest
theorem additional_rate_of_interest  (P A A' : ℝ) (T : ℕ) (R : ℝ)
    (h1 : principal_amount P)
    (h2 : original_amount A)
    (h3 : time_period T)
    (h4 : new_amount A') :
    (A' - P) / (P * T) * 100 - (A - P) / (P * T) * 100 = 1 :=
by
  sorry

end additional_rate_of_interest_l247_24797


namespace complete_contingency_table_chi_square_test_certainty_l247_24744

-- Defining the initial conditions given in the problem
def total_students : ℕ := 100
def boys_dislike : ℕ := 10
def girls_like : ℕ := 20
def dislike_probability : ℚ := 0.4

-- Completed contingency table values based on given and inferred values
def boys_total : ℕ := 50
def girls_total : ℕ := 50
def boys_like : ℕ := boys_total - boys_dislike
def girls_dislike : ℕ := 30
def total_like : ℕ := boys_like + girls_like
def total_dislike : ℕ := boys_dislike + girls_dislike

-- Chi-square value from the solution
def K_squared : ℚ := 50 / 3

-- Declaring the proof problem for the completed contingency table
theorem complete_contingency_table :
  boys_total + girls_total = total_students ∧ 
  total_like + total_dislike = total_students ∧ 
  dislike_probability * total_students = total_dislike ∧ 
  boys_like = 40 ∧ 
  girls_dislike = 30 :=
sorry

-- Declaring the proof problem for the chi-square test
theorem chi_square_test_certainty :
  K_squared > 10.828 :=
sorry

end complete_contingency_table_chi_square_test_certainty_l247_24744


namespace Mr_Mayer_purchase_price_l247_24772

theorem Mr_Mayer_purchase_price 
  (P : ℝ) 
  (H1 : (1.30 * 2) * P = 2600) : 
  P = 1000 := 
by
  sorry

end Mr_Mayer_purchase_price_l247_24772


namespace total_passengers_landed_l247_24734

theorem total_passengers_landed (on_time late : ℕ) (h_on_time : on_time = 14507) (h_late : late = 213) :
  on_time + late = 14720 :=
by
  sorry

end total_passengers_landed_l247_24734


namespace sector_central_angle_l247_24757

-- Definitions and constants
def arc_length := 4 -- arc length of the sector in cm
def area := 2       -- area of the sector in cm²

-- The central angle of the sector we want to prove
def theta := 4      -- radian measure of the central angle

-- Main statement to prove
theorem sector_central_angle : 
  ∃ (r : ℝ), (1 / 2) * theta * r^2 = area ∧ theta * r = arc_length :=
by
  -- No proof is required as per the instruction
  sorry

end sector_central_angle_l247_24757


namespace ellipse_distance_pf2_l247_24789

noncomputable def ellipse_focal_length := 2 * Real.sqrt 2
noncomputable def ellipse_equation (a : ℝ) (a_gt_one : a > 1)
  (P : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  (x^2 / a) + y^2 = 1

theorem ellipse_distance_pf2
  (a : ℝ) (a_gt_one : a > 1)
  (focus_distance : 2 * Real.sqrt (a - 1) = 2 * Real.sqrt 2)
  (F1 F2 P : ℝ × ℝ)
  (on_ellipse : ellipse_equation a a_gt_one P)
  (PF1_eq_two : dist P F1 = 2)
  (a_eq : a = 3) :
  dist P F2 = 2 * Real.sqrt 3 - 2 := 
sorry

end ellipse_distance_pf2_l247_24789


namespace clock_angle_solution_l247_24751

theorem clock_angle_solution (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 360) :
    (θ = 15) ∨ (θ = 165) :=
by
  sorry

end clock_angle_solution_l247_24751


namespace elena_alex_total_dollars_l247_24729

theorem elena_alex_total_dollars :
  (5 / 6 : ℚ) + (7 / 15 : ℚ) = (13 / 10 : ℚ) :=
by
    sorry

end elena_alex_total_dollars_l247_24729


namespace values_of_x_l247_24782

def P (x : ℝ) : ℝ := x^3 - 5 * x^2 + 8 * x

theorem values_of_x (x : ℝ) :
  P x = P (x + 1) ↔ (x = 1 ∨ x = 4 / 3) :=
by sorry

end values_of_x_l247_24782


namespace maximum_area_of_garden_l247_24784

theorem maximum_area_of_garden (w l : ℝ) 
  (h_perimeter : 2 * w + l = 400) : 
  ∃ (A : ℝ), A = 20000 ∧ A = w * l ∧ l = 400 - 2 * w ∧ ∀ (w' : ℝ) (l' : ℝ),
    2 * w' + l' = 400 → w' * l' ≤ 20000 :=
by
  sorry

end maximum_area_of_garden_l247_24784


namespace intersection_height_correct_l247_24777

noncomputable def height_of_intersection (height1 height2 distance : ℝ) : ℝ :=
  let line1 (x : ℝ) := - (height1 / distance) * x + height1
  let line2 (x : ℝ) := - (height2 / distance) * x
  let x_intersect := - (height2 * distance) / (height1 - height2)
  line1 x_intersect

theorem intersection_height_correct :
  height_of_intersection 40 60 120 = 120 :=
by
  sorry

end intersection_height_correct_l247_24777


namespace age_difference_between_brother_and_cousin_l247_24753

-- Define the ages used in the problem 
def Lexie_age : ℕ := 8
def Grandma_age : ℕ := 68
def Brother_age : ℕ := Lexie_age - 6
def Sister_age : ℕ := 2 * Lexie_age
def Uncle_age : ℕ := Grandma_age - 12
def Cousin_age : ℕ := Brother_age + 5

-- The proof problem statement in Lean 4
theorem age_difference_between_brother_and_cousin : 
  Brother_age < Cousin_age ∧ Cousin_age - Brother_age = 5 :=
by
  -- Definitions and imports are done above. The statement below should prove the age difference.
  sorry

end age_difference_between_brother_and_cousin_l247_24753


namespace odd_periodic_function_l247_24755

theorem odd_periodic_function (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_period : ∀ x : ℝ, f (x + 5) = f x)
  (h_f1 : f 1 = 1)
  (h_f2 : f 2 = 2) :
  f 3 - f 4 = -1 :=
sorry

end odd_periodic_function_l247_24755


namespace trigonometric_identity_l247_24713

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180))
  = (4 * Real.sin (10 * Real.pi / 180) / Real.sin (40 * Real.pi / 180)) :=
by sorry

end trigonometric_identity_l247_24713


namespace product_prices_determined_max_product_A_pieces_l247_24728

theorem product_prices_determined (a b : ℕ) :
  (20 * a + 15 * b = 380) →
  (15 * a + 10 * b = 280) →
  a = 16 ∧ b = 4 :=
by sorry

theorem max_product_A_pieces (x : ℕ) :
  (16 * x + 4 * (100 - x) ≤ 900) →
  x ≤ 41 :=
by sorry

end product_prices_determined_max_product_A_pieces_l247_24728


namespace remainder_12401_163_l247_24732

theorem remainder_12401_163 :
  let original_number := 12401
  let divisor := 163
  let quotient := 76
  let remainder := 13
  original_number = divisor * quotient + remainder :=
by
  sorry

end remainder_12401_163_l247_24732


namespace xy_sum_143_l247_24726

theorem xy_sum_143 (x y : ℕ) (h1 : x < 30) (h2 : y < 30) (h3 : x + y + x * y = 143) (h4 : 0 < x) (h5 : 0 < y) :
  x + y = 22 ∨ x + y = 23 ∨ x + y = 24 :=
by
  sorry

end xy_sum_143_l247_24726


namespace smallest_multiple_of_40_gt_100_l247_24719

theorem smallest_multiple_of_40_gt_100 :
  ∃ x : ℕ, 0 < x ∧ 40 * x > 100 ∧ ∀ y : ℕ, 0 < y ∧ 40 * y > 100 → x ≤ y → 40 * x = 120 :=
by
  sorry

end smallest_multiple_of_40_gt_100_l247_24719


namespace jackson_meat_left_l247_24731

theorem jackson_meat_left (total_meat : ℕ) (meatballs_fraction : ℚ) (spring_rolls_meat : ℕ) :
  total_meat = 20 →
  meatballs_fraction = 1/4 →
  spring_rolls_meat = 3 →
  total_meat - (meatballs_fraction * total_meat + spring_rolls_meat) = 12 := by
  intros ht hm hs
  sorry

end jackson_meat_left_l247_24731


namespace prime_number_p_squared_divides_sum_or_cube_divides_sum_of_cubes_l247_24765

variable {p a b : ℤ}

theorem prime_number_p_squared_divides_sum_or_cube_divides_sum_of_cubes
  (hp : Prime p) (hp_ne_3 : p ≠ 3)
  (h1 : p ∣ (a + b)) (h2 : p^2 ∣ (a^3 + b^3)) :
  p^2 ∣ (a + b) ∨ p^3 ∣ (a^3 + b^3) :=
sorry

end prime_number_p_squared_divides_sum_or_cube_divides_sum_of_cubes_l247_24765


namespace polynomial_no_real_roots_l247_24730

def f (x : ℝ) : ℝ := 4 * x ^ 8 - 2 * x ^ 7 + x ^ 6 - 3 * x ^ 4 + x ^ 2 - x + 1

theorem polynomial_no_real_roots : ∀ x : ℝ, f x > 0 := by
  sorry

end polynomial_no_real_roots_l247_24730


namespace x_varies_z_pow_l247_24712

variable (k j : ℝ)
variable (y z : ℝ)

-- Given conditions
def x_varies_y_squared (x : ℝ) := x = k * y^2
def y_varies_z_cuberoot_squared := y = j * z^(2/3)

-- To prove: 
theorem x_varies_z_pow (x : ℝ) (h1 : x_varies_y_squared k y x) (h2 : y_varies_z_cuberoot_squared j z y) : ∃ m : ℝ, x = m * z^(4/3) :=
by
  sorry

end x_varies_z_pow_l247_24712


namespace bamboo_volume_l247_24795

theorem bamboo_volume :
  ∃ (a₁ d a₅ : ℚ), 
  (4 * a₁ + 6 * d = 5) ∧ 
  (3 * a₁ + 21 * d = 4) ∧ 
  (a₅ = a₁ + 4 * d) ∧ 
  (a₅ = 85 / 66) :=
sorry

end bamboo_volume_l247_24795


namespace identity_solution_l247_24794

theorem identity_solution (x : ℝ) :
  ∃ a b : ℝ, (2 * x + a) ^ 3 = 5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x ∧
             a = -1 ∧ b = 1 :=
by
  -- we can skip the proof as this is just a statement
  sorry

end identity_solution_l247_24794


namespace find_a_of_perpendicular_lines_l247_24776

theorem find_a_of_perpendicular_lines (a : ℝ) :
  let line1 : ℝ := a * x + y - 1
  let line2 : ℝ := 4 * x + (a - 3) * y - 2
  (∀ x y : ℝ, (line1 = 0 → line2 ≠ 0 → line1 * line2 = -1)) → a = 3 / 5 :=
by
  sorry

end find_a_of_perpendicular_lines_l247_24776


namespace miles_hiked_first_day_l247_24764

theorem miles_hiked_first_day (total_distance remaining_distance : ℕ)
  (h1 : total_distance = 36)
  (h2 : remaining_distance = 27) :
  total_distance - remaining_distance = 9 :=
by
  sorry

end miles_hiked_first_day_l247_24764


namespace larger_integer_is_21_l247_24738

theorem larger_integer_is_21 (a b : ℕ) (h₀ : 0 < b) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : a = 21 ∨ b = 21 :=
by
  sorry

end larger_integer_is_21_l247_24738


namespace marbles_exchange_l247_24705

-- Define the initial number of marbles for Drew and Marcus
variables {D M x : ℕ}

-- Conditions
axiom Drew_initial (D M : ℕ) : D = M + 24
axiom Drew_after_give (D x : ℕ) : D - x = 25
axiom Marcus_after_receive (M x : ℕ) : M + x = 25

-- The goal is to prove: x = 12
theorem marbles_exchange : ∀ {D M x : ℕ}, D = M + 24 ∧ D - x = 25 ∧ M + x = 25 → x = 12 :=
by 
    sorry

end marbles_exchange_l247_24705


namespace select_pairs_eq_l247_24707

open Set

-- Definitions for sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

-- Statement of the theorem
theorem select_pairs_eq :
  {p | p.1 ∈ A ∧ p.2 ∈ B} = {(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)} :=
by sorry

end select_pairs_eq_l247_24707


namespace parabola_focus_coordinates_l247_24727

theorem parabola_focus_coordinates :
  ∀ (x y : ℝ), x^2 = 8 * y → ∃ F : ℝ × ℝ, F = (0, 2) :=
  sorry

end parabola_focus_coordinates_l247_24727


namespace correct_option_l247_24769

variable (p q : Prop)

/-- If only one of p and q is true, then p or q is a true proposition. -/
theorem correct_option (h : (p ∧ ¬ q) ∨ (¬ p ∧ q)) : p ∨ q :=
by sorry

end correct_option_l247_24769


namespace inequality_xyz_equality_condition_l247_24766

theorem inequality_xyz (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : 
  x + y + z ≤ 2 + x * y * z :=
sorry

theorem equality_condition (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) :
  (x + y + z = 2 + x * y * z) ↔ (x = 0 ∧ y = 1 ∧ z = 1) ∨ (x = 1 ∧ y = 0 ∧ z = 1) ∨ (x = 1 ∧ y = 1 ∧ z = 0) ∨
                                                  (x = 0 ∧ y = -1 ∧ z = -1) ∨ (x = -1 ∧ y = 0 ∧ z = 1) ∨
                                                  (x = -1 ∧ y = 1 ∧ z = 0) :=
sorry

end inequality_xyz_equality_condition_l247_24766


namespace rectangle_perimeter_eq_26_l247_24775

theorem rectangle_perimeter_eq_26 (a b c W : ℕ) (h_tri : a = 5 ∧ b = 12 ∧ c = 13)
  (h_right_tri : a^2 + b^2 = c^2) (h_W : W = 3) (h_area_eq : 1/2 * (a * b) = (W * L))
  (A L : ℕ) (hA : A = 30) (hL : L = A / W) :
  2 * (L + W) = 26 :=
by
  sorry

end rectangle_perimeter_eq_26_l247_24775


namespace pepperoni_crust_ratio_l247_24780

-- Define the conditions as Lean 4 statements
def L : ℕ := 50
def C : ℕ := 2 * L
def D : ℕ := 210
def S : ℕ := L + C + D
def S_E : ℕ := S / 4
def CR : ℕ := 600
def CH : ℕ := 400
def PizzaTotal (P : ℕ) : ℕ := CR + P + CH
def PizzaEats (P : ℕ) : ℕ := (PizzaTotal P) / 5
def JacksonEats : ℕ := 330

theorem pepperoni_crust_ratio (P : ℕ) (h1 : S_E + PizzaEats P = JacksonEats) : P / CR = 1 / 3 :=
by sorry

end pepperoni_crust_ratio_l247_24780


namespace represent_1947_as_squares_any_integer_as_squares_l247_24701

theorem represent_1947_as_squares :
  ∃ (a b c : ℤ), 1947 = a * a - b * b - c * c :=
by
  use 488, 486, 1
  sorry

theorem any_integer_as_squares (n : ℤ) :
  ∃ (a b c d : ℤ), n = a * a + b * b + c * c + d * d :=
by
  sorry

end represent_1947_as_squares_any_integer_as_squares_l247_24701


namespace crabapple_recipients_sequences_l247_24746

-- Define the number of students in Mrs. Crabapple's class
def num_students : ℕ := 12

-- Define the number of class meetings per week
def num_meetings : ℕ := 5

-- Define the total number of different sequences
def total_sequences : ℕ := num_students ^ num_meetings

-- The target theorem to prove
theorem crabapple_recipients_sequences :
  total_sequences = 248832 := by
  sorry

end crabapple_recipients_sequences_l247_24746


namespace christmas_gift_count_l247_24759

theorem christmas_gift_count (initial_gifts : ℕ) (additional_gifts : ℕ) (gifts_to_orphanage : ℕ)
  (h1 : initial_gifts = 77)
  (h2 : additional_gifts = 33)
  (h3 : gifts_to_orphanage = 66) :
  (initial_gifts + additional_gifts - gifts_to_orphanage = 44) :=
by
  sorry

end christmas_gift_count_l247_24759


namespace initial_pipes_count_l247_24704

theorem initial_pipes_count (n r : ℝ) 
  (h1 : n * r = 1 / 12) 
  (h2 : (n + 10) * r = 1 / 4) : 
  n = 5 := 
by 
  sorry

end initial_pipes_count_l247_24704


namespace polynomial_root_multiplicity_l247_24758

theorem polynomial_root_multiplicity (A B n : ℤ) (h1 : A + B + 1 = 0) (h2 : (n + 1) * A + n * B = 0) :
  A = n ∧ B = -(n + 1) :=
sorry

end polynomial_root_multiplicity_l247_24758


namespace intersecting_lines_l247_24792

theorem intersecting_lines (m b : ℝ)
  (h1 : ∀ x, (9 : ℝ) = 2 * m * x + 3 → x = 3)
  (h2 : ∀ x, (9 : ℝ) = 4 * x + b → x = 3) :
  b + 2 * m = -1 :=
sorry

end intersecting_lines_l247_24792


namespace division_and_multiplication_l247_24767

theorem division_and_multiplication (a b c d : ℝ) : (a / b / c * d) = 30 :=
by 
  let a := 120
  let b := 6
  let c := 2
  let d := 3
  sorry

end division_and_multiplication_l247_24767


namespace color_theorem_l247_24773

/-- The only integers \( k \geq 1 \) such that if each integer is colored in one of these \( k \)
colors, there must exist integers \( a_1 < a_2 < \cdots < a_{2023} \) of the same color where the
differences \( a_2 - a_1, a_3 - a_2, \cdots, a_{2023} - a_{2022} \) are all powers of 2 are
\( k = 1 \) and \( k = 2 \). -/
theorem color_theorem : ∀ (k : ℕ), (k ≥ 1) →
  (∀ f : ℕ → Fin k,
    ∃ a : Fin 2023 → ℕ,
    (∀ i : Fin (2023 - 1), ∃ n : ℕ, 2^n = (a i.succ - a i)) ∧
    (∀ i j : Fin 2023, i < j → f (a i) = f (a j)))
  ↔ k = 1 ∨ k = 2 := by
  sorry

end color_theorem_l247_24773


namespace product_of_fractions_is_27_l247_24739

theorem product_of_fractions_is_27 :
  (1/3) * (9/1) * (1/27) * (81/1) * (1/243) * (729/1) = 27 :=
by
  sorry

end product_of_fractions_is_27_l247_24739


namespace unique_prime_pair_l247_24700

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem unique_prime_pair :
  ∀ p : ℕ, is_prime p ∧ is_prime (p + 1) → p = 2 := by
  sorry

end unique_prime_pair_l247_24700


namespace equation_of_parabola_l247_24736

def parabola_passes_through_point (a h : ℝ) : Prop :=
  2 = a * (8^2) + h

def focus_x_coordinate (a h : ℝ) : Prop :=
  h + (1 / (4 * a)) = 3

theorem equation_of_parabola :
  ∃ (a h : ℝ), parabola_passes_through_point a h ∧ focus_x_coordinate a h ∧
    (∀ x y : ℝ, x = (15 / 256) * y^2 - (381 / 128)) :=
sorry

end equation_of_parabola_l247_24736


namespace least_number_of_coins_l247_24743

theorem least_number_of_coins (n : ℕ) : 
  (n % 7 = 3) ∧ (n % 5 = 4) ∧ (∀ m : ℕ, (m % 7 = 3) ∧ (m % 5 = 4) → n ≤ m) → n = 24 :=
by
  sorry

end least_number_of_coins_l247_24743


namespace sum_first_six_terms_l247_24785

noncomputable def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_six_terms :
  geometric_series_sum (1/4) (1/4) 6 = 4095 / 12288 :=
by 
  sorry

end sum_first_six_terms_l247_24785


namespace find_number_l247_24747

theorem find_number 
  (n : ℤ)
  (h1 : n % 7 = 2)
  (h2 : n % 8 = 4)
  (quot_7 : ℤ)
  (quot_8 : ℤ)
  (h3 : n = 7 * quot_7 + 2)
  (h4 : n = 8 * quot_8 + 4)
  (h5 : quot_7 = quot_8 + 7) :
  n = 380 := by
  sorry

end find_number_l247_24747


namespace correct_statements_l247_24724

theorem correct_statements (f : ℝ → ℝ)
  (h_add : ∀ x y : ℝ, f (x + y) = f (x) + f (y))
  (h_pos : ∀ x : ℝ, x > 0 → f (x) > 0) :
  (f 0 ≠ 1) ∧
  (∀ x : ℝ, f (-x) = -f (x)) ∧
  ¬ (∀ x : ℝ, |f (x)| = |f (-x)|) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f (x₁) < f (x₂)) ∧
  ¬ (∀ x : ℝ, f (x) + 1 < f (x + 1)) :=
by
  sorry

end correct_statements_l247_24724


namespace area_product_is_2_l247_24778

open Real

-- Definitions for parabola, points, and the condition of dot product
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def dot_product_condition (A B : ℝ × ℝ) : Prop :=
  (A.1 * B.1 + A.2 * B.2) = -4

def area (O F P : ℝ × ℝ) : ℝ :=
  0.5 * abs (O.1 * (F.2 - P.2) + F.1 * (P.2 - O.2) + P.1 * (O.2 - F.2))

-- Points A and B are on the parabola and the dot product condition holds
variables (A B : ℝ × ℝ)
variable (H_A_on_parabola : parabola A.1 A.2)
variable (H_B_on_parabola : parabola B.1 B.2)
variable (H_dot_product : dot_product_condition A B)

-- Focus of the parabola
def F : ℝ × ℝ := (1, 0)

-- Origin
def O : ℝ × ℝ := (0, 0)

-- Prove that the product of areas is 2
theorem area_product_is_2 : 
  area O F A * area O F B = 2 :=
sorry

end area_product_is_2_l247_24778


namespace problem1_problem2_l247_24787

-- Problem 1
theorem problem1 (f : ℝ → ℝ) (x : ℝ) (h : ∀ x, f x = abs (x - 1)) :
  f x ≥ (1/2) * (x + 1) ↔ (x ≤ 1/3) ∨ (x ≥ 3) :=
sorry

-- Problem 2
theorem problem2 (g : ℝ → ℝ) (A : Set ℝ) (a : ℝ) 
  (h1 : ∀ x, g x = abs (x - a) - abs (x - 2))
  (h2 : A ⊆ Set.Icc (-1 : ℝ) 3) :
  (1 ≤ a ∧ a < 2) ∨ (2 ≤ a ∧ a ≤ 3) :=
sorry

end problem1_problem2_l247_24787


namespace arithmetic_sequence_problem_l247_24722

theorem arithmetic_sequence_problem (q a₁ a₂ a₃ : ℕ) (a : ℕ → ℕ) (c : ℕ → ℕ) (S T : ℕ → ℕ)
  (h1 : q > 1)
  (h2 : a₁ + a₂ + a₃ = 7)
  (h3 : a₁ + 3 + a₃ + 4 = 6 * a₂) :
  (∀ n : ℕ, a n = 2^(n-1)) ∧ (∀ n : ℕ, T n = (3 * n - 5) * 2^n + 5) :=
by
  sorry

end arithmetic_sequence_problem_l247_24722


namespace original_average_l247_24714

theorem original_average (A : ℝ) (h : (2 * (12 * A)) / 12 = 100) : A = 50 :=
by
  sorry

end original_average_l247_24714


namespace find_c_plus_d_l247_24708

def is_smallest_two_digit_multiple_of_5 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ ∃ k : ℕ, n = 5 * k ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ ∃ k', m = 5 * k') → n ≤ m

def is_smallest_three_digit_multiple_of_7 (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = 7 * k ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ ∃ k', m = 7 * k') → n ≤ m

theorem find_c_plus_d :
  ∃ c d : ℕ, is_smallest_two_digit_multiple_of_5 c ∧ is_smallest_three_digit_multiple_of_7 d ∧ c + d = 115 :=
by
  sorry

end find_c_plus_d_l247_24708


namespace unguarded_area_eq_225_l247_24742

-- Define the basic conditions of the problem in Lean
structure Room where
  side_length : ℕ
  unguarded_fraction : ℚ
  deriving Repr

-- Define the specific room used in the problem
def problemRoom : Room :=
  { side_length := 10,
    unguarded_fraction := 9/4 }

-- Define the expected unguarded area in square meters
def expected_unguarded_area (r : Room) : ℚ :=
  r.unguarded_fraction * (r.side_length ^ 2)

-- Prove that the unguarded area is 225 square meters
theorem unguarded_area_eq_225 (r : Room) (h : r = problemRoom) : expected_unguarded_area r = 225 := by
  -- The proof in this case is omitted.
  sorry

end unguarded_area_eq_225_l247_24742


namespace parabola_tangent_y_intercept_correct_l247_24710

noncomputable def parabola_tangent_y_intercept (a : ℝ) : Prop :=
  let C := fun x : ℝ => x^2
  let slope := 2 * a
  let tangent_line := fun x : ℝ => slope * (x - a) + C a
  let Q := (0, tangent_line 0)
  Q = (0, -a^2)

-- Statement of the problem as a Lean theorem
theorem parabola_tangent_y_intercept_correct (a : ℝ) (h : a > 0) :
  parabola_tangent_y_intercept a := 
by 
  sorry

end parabola_tangent_y_intercept_correct_l247_24710


namespace Jamir_swims_more_l247_24779

def Julien_distance_per_day : ℕ := 50
def Sarah_distance_per_day (J : ℕ) : ℕ := 2 * J
def combined_distance_per_week (J S M : ℕ) : ℕ := 7 * (J + S + M)

theorem Jamir_swims_more :
  let J := Julien_distance_per_day
  let S := Sarah_distance_per_day J
  ∃ M, combined_distance_per_week J S M = 1890 ∧ (M - S = 20) := by
    let J := Julien_distance_per_day
    let S := Sarah_distance_per_day J
    use 120
    sorry

end Jamir_swims_more_l247_24779


namespace jack_years_after_son_death_l247_24735

noncomputable def jackAdolescenceTime (L : Real) : Real := (1 / 6) * L
noncomputable def jackFacialHairTime (L : Real) : Real := (1 / 12) * L
noncomputable def jackMarriageTime (L : Real) : Real := (1 / 7) * L
noncomputable def jackSonBornTime (L : Real) (marriageTime : Real) : Real := marriageTime + 5
noncomputable def jackSonLifetime (L : Real) : Real := (1 / 2) * L
noncomputable def jackSonDeathTime (bornTime : Real) (sonLifetime : Real) : Real := bornTime + sonLifetime
noncomputable def yearsAfterSonDeath (L : Real) (sonDeathTime : Real) : Real := L - sonDeathTime

theorem jack_years_after_son_death : 
  yearsAfterSonDeath 84 
    (jackSonDeathTime (jackSonBornTime 84 (jackMarriageTime 84)) (jackSonLifetime 84)) = 4 :=
by
  sorry

end jack_years_after_son_death_l247_24735


namespace central_angle_of_sector_l247_24718

theorem central_angle_of_sector (r α : ℝ) (h_arc_length : α * r = 5) (h_area : 0.5 * α * r^2 = 5): α = 5 / 2 := by
  sorry

end central_angle_of_sector_l247_24718


namespace probability_two_cards_diff_suits_l247_24771

def prob_two_cards_diff_suits {deck_size suits cards_per_suit : ℕ} (h1 : deck_size = 40) (h2 : suits = 4) (h3 : cards_per_suit = 10) : ℚ :=
  let total_cards := deck_size
  let cards_same_suit := cards_per_suit - 1
  let cards_diff_suit := total_cards - 1 - cards_same_suit 
  cards_diff_suit / (total_cards - 1)

theorem probability_two_cards_diff_suits (h1 : 40 = 40) (h2 : 4 = 4) (h3 : 10 = 10) :
  prob_two_cards_diff_suits h1 h2 h3 = 10 / 13 :=
by
  sorry

end probability_two_cards_diff_suits_l247_24771


namespace price_first_oil_l247_24798

theorem price_first_oil (P : ℝ) (h1 : 10 * P + 5 * 66 = 15 * 58.67) : P = 55.005 :=
sorry

end price_first_oil_l247_24798


namespace sum_f_sequence_l247_24754

noncomputable def f (x : ℝ) : ℝ := 1 / (4^x + 2)

theorem sum_f_sequence :
  f (1/10) + f (2/10) + f (3/10) + f (4/10) + f (5/10) + f (6/10) + f (7/10) + f (8/10) + f (9/10) = 9 / 4 :=
by {
  sorry
}

end sum_f_sequence_l247_24754


namespace find_age_difference_l247_24706

variable (a b c : ℕ)

theorem find_age_difference (h : a + b = b + c + 20) : c = a - 20 :=
by
  sorry

end find_age_difference_l247_24706


namespace lines_intersect_at_l247_24761

theorem lines_intersect_at :
  ∃ t u : ℝ, (∃ (x y : ℝ),
    (x = 2 + 3 * t ∧ y = 4 - 2 * t) ∧
    (x = -1 + 6 * u ∧ y = 5 + u) ∧
    (x = 1/5 ∧ y = 26/5)) :=
by
  sorry

end lines_intersect_at_l247_24761


namespace marys_next_birthday_l247_24741

noncomputable def calculate_marys_age (d j s m TotalAge : ℝ) (H1 : j = 1.15 * d) (H2 : s = 1.30 * d) (H3 : m = 1.25 * s) (H4 : j + d + s + m = TotalAge) : ℝ :=
  m + 1

theorem marys_next_birthday (d j s m TotalAge : ℝ) 
  (H1 : j = 1.15 * d)
  (H2 : s = 1.30 * d)
  (H3 : m = 1.25 * s)
  (H4 : j + d + s + m = TotalAge)
  (H5 : TotalAge = 80) :
  calculate_marys_age d j s m TotalAge H1 H2 H3 H4 = 26 :=
sorry

end marys_next_birthday_l247_24741


namespace min_value_x_plus_y_l247_24702

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : (2 * x + Real.sqrt (4 * x^2 + 1)) * (Real.sqrt (y^2 + 4) - 2) ≥ y) : 
  x + y >= 2 := 
by
  sorry

end min_value_x_plus_y_l247_24702


namespace vector_dot_product_proof_l247_24796

variable (a b : ℝ × ℝ)

def dot_product (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2

theorem vector_dot_product_proof
  (h1 : a = (1, -3))
  (h2 : b = (3, 7)) :
  dot_product a b = -18 :=
by 
  sorry

end vector_dot_product_proof_l247_24796


namespace hyperbola_condition_l247_24752

theorem hyperbola_condition (m n : ℝ) : 
  (mn < 0) ↔ (∀ x y : ℝ, ∃ k ∈ {a : ℝ | a ≠ 0}, (x^2 / m + y^2 / n = 1)) := sorry

end hyperbola_condition_l247_24752


namespace jerry_boxes_l247_24781

theorem jerry_boxes (boxes_sold boxes_left : ℕ) (h₁ : boxes_sold = 5) (h₂ : boxes_left = 5) : (boxes_sold + boxes_left = 10) :=
by
  sorry

end jerry_boxes_l247_24781


namespace minimal_divisors_at_kth_place_l247_24749

open Nat

theorem minimal_divisors_at_kth_place (n k : ℕ) (hnk : n ≥ k) (S : ℕ) (hS : ∃ d : ℕ, d ≥ n ∧ d = S ∧ ∀ i, i ≤ d → exists m, m = d):
  ∃ (min_div : ℕ), min_div = ⌈ (n : ℝ) / k ⌉ :=
by
  sorry

end minimal_divisors_at_kth_place_l247_24749


namespace geometric_series_first_term_l247_24709

theorem geometric_series_first_term 
  (S : ℝ) (r : ℝ) (a : ℝ)
  (h_sum : S = 40) (h_ratio : r = 1/4) :
  S = a / (1 - r) → a = 30 := by
  sorry

end geometric_series_first_term_l247_24709


namespace monthly_income_of_A_l247_24763

theorem monthly_income_of_A (A B C : ℝ)
  (h1 : (A + B) / 2 = 5050)
  (h2 : (B + C) / 2 = 6250)
  (h3 : (A + C) / 2 = 5200) :
  A = 4000 :=
sorry

end monthly_income_of_A_l247_24763


namespace find_p_q_l247_24711

noncomputable def f (p q : ℝ) (x : ℝ) : ℝ :=
if x < -1 then p * x + q else 5 * x - 10

theorem find_p_q (p q : ℝ) (h : ∀ x, f p q (f p q x) = x) : p + q = 11 :=
sorry

end find_p_q_l247_24711


namespace ratio_of_second_to_first_l247_24748

theorem ratio_of_second_to_first (A1 A2 A3 : ℕ) (h1 : A1 = 600) (h2 : A3 = A1 + A2 - 400) (h3 : A1 + A2 + A3 = 3200) : A2 / A1 = 2 :=
by
  sorry

end ratio_of_second_to_first_l247_24748


namespace perpendicular_plane_line_sum_l247_24768

theorem perpendicular_plane_line_sum (x y : ℝ)
  (h1 : ∃ k : ℝ, (2, -4 * x, 1) = (6 * k, 12 * k, -3 * k * y))
  : x + y = -2 :=
sorry

end perpendicular_plane_line_sum_l247_24768


namespace Lin_peels_15_potatoes_l247_24723

-- Define the conditions
def total_potatoes : Nat := 60
def homer_rate : Nat := 2 -- potatoes per minute
def christen_rate : Nat := 3 -- potatoes per minute
def lin_rate : Nat := 4 -- potatoes per minute
def christen_join_time : Nat := 6 -- minutes
def lin_join_time : Nat := 9 -- minutes

-- Prove that Lin peels 15 potatoes
theorem Lin_peels_15_potatoes :
  ∃ (lin_potatoes : Nat), lin_potatoes = 15 :=
by
  sorry

end Lin_peels_15_potatoes_l247_24723


namespace directrix_of_parabola_l247_24715

theorem directrix_of_parabola (p : ℝ) (hp : 0 < p) (h_point : ∃ (x y : ℝ), y^2 = 2 * p * x ∧ (x = 2 ∧ y = 2)) :
  x = -1/2 :=
sorry

end directrix_of_parabola_l247_24715


namespace problem1_problem2_l247_24716

-- Problem 1
theorem problem1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  a * b + b * c + c * a ≤ 1 / 3 :=
sorry

-- Problem 2
theorem problem2 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≥ b) :
  2 * a ^ 3 - b ^ 3 ≥ 2 * a * b ^ 2 - a ^ 2 * b :=
sorry

end problem1_problem2_l247_24716


namespace Marge_savings_l247_24786

theorem Marge_savings
  (lottery_winnings : ℝ)
  (taxes_paid : ℝ)
  (student_loan_payment : ℝ)
  (amount_after_taxes : ℝ)
  (amount_after_student_loans : ℝ)
  (fun_money : ℝ)
  (investment : ℝ)
  (savings : ℝ)
  (h_win : lottery_winnings = 12006)
  (h_tax : taxes_paid = lottery_winnings / 2)
  (h_after_tax : amount_after_taxes = lottery_winnings - taxes_paid)
  (h_loans : student_loan_payment = amount_after_taxes / 3)
  (h_after_loans : amount_after_student_loans = amount_after_taxes - student_loan_payment)
  (h_fun : fun_money = 2802)
  (h_savings_investment : amount_after_student_loans - fun_money = savings + investment)
  (h_investment : investment = savings / 5)
  (h_left : amount_after_student_loans - fun_money = 1200) :
  savings = 1000 :=
by
  sorry

end Marge_savings_l247_24786


namespace sqrt_eq_pm_4_l247_24750

theorem sqrt_eq_pm_4 : {x : ℝ | x * x = 16} = {4, -4} :=
by sorry

end sqrt_eq_pm_4_l247_24750


namespace complement_A_possible_set_l247_24791

variable (U A B : Set ℕ)

theorem complement_A_possible_set (hU : U = {1, 2, 3, 4, 5, 6})
  (h_union : A ∪ B = {1, 2, 3, 4, 5}) 
  (h_inter : A ∩ B = {3, 4, 5}) :
  ∃ C, C = U \ A ∧ C = {6} :=
by
  sorry

end complement_A_possible_set_l247_24791


namespace find_angle_C_find_area_of_triangle_l247_24717

-- Given triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively
-- And given conditions: c * cos B = (2a - b) * cos C

variable (a b c : ℝ) (A B C : ℝ)
variable (h1 : c * Real.cos B = (2 * a - b) * Real.cos C)
variable (h2 : c = 2)
variable (h3 : a + b + c = 2 * Real.sqrt 3 + 2)

-- Prove that angle C = π / 3
theorem find_angle_C : C = Real.pi / 3 :=
by sorry

-- Given angle C, side c, and perimeter, prove the area of triangle ABC
theorem find_area_of_triangle (h4 : C = Real.pi / 3) : 
  1 / 2 * a * b * Real.sin C = 2 * Real.sqrt 3 / 3 :=
by sorry

end find_angle_C_find_area_of_triangle_l247_24717


namespace estimate_expr_range_l247_24762

theorem estimate_expr_range :
  5 < (2 * Real.sqrt 5 + 5 * Real.sqrt 2) * Real.sqrt (1 / 5) ∧
  (2 * Real.sqrt 5 + 5 * Real.sqrt 2) * Real.sqrt (1 / 5) < 6 :=
  sorry

end estimate_expr_range_l247_24762


namespace extreme_points_l247_24774

noncomputable def f (a x : ℝ) : ℝ := Real.log (1 / (2 * x)) - a * x^2 + x

theorem extreme_points (
  a : ℝ
) (h : 0 < a ∧ a < (1 : ℝ) / 8) :
  ∃ x1 x2 : ℝ, f a x1 + f a x2 > 3 - 4 * Real.log 2 :=
sorry

end extreme_points_l247_24774


namespace parking_lot_perimeter_l247_24783

theorem parking_lot_perimeter (x y: ℝ) 
  (h1: x = (2 / 3) * y)
  (h2: x^2 + y^2 = 400)
  (h3: x * y = 120) :
  2 * (x + y) = 20 * Real.sqrt 5 :=
by
  sorry

end parking_lot_perimeter_l247_24783


namespace find_angle_D_l247_24737

theorem find_angle_D 
  (A B C D : ℝ)
  (h1 : A + B = 180)
  (h2 : C = D)
  (h3 : A = 50) :
  D = 25 := 
by
  sorry

end find_angle_D_l247_24737


namespace toy_position_from_left_l247_24790

/-- Define the total number of toys -/
def total_toys : ℕ := 19

/-- Define the position of toy (A) from the right -/
def position_from_right : ℕ := 8

/-- Prove the main statement: The position of toy (A) from the left is 12 given the conditions -/
theorem toy_position_from_left : total_toys - position_from_right + 1 = 12 := by
  sorry

end toy_position_from_left_l247_24790


namespace determine_k_l247_24703

theorem determine_k (k : ℝ) :
  (∀ x : ℝ, (x - 3) * (x - 5) = k - 4 * x) ↔ k = 11 :=
by
  sorry

end determine_k_l247_24703


namespace quadratic_equation_original_eq_l247_24720

theorem quadratic_equation_original_eq :
  ∃ (α β : ℝ), (α + β = 3) ∧ (α * β = -6) ∧ (∀ (x : ℝ), x^2 - 3 * x - 6 = 0 → (x = α ∨ x = β)) :=
sorry

end quadratic_equation_original_eq_l247_24720


namespace flower_garden_mystery_value_l247_24788

/-- Prove the value of "花园探秘" given the arithmetic sum conditions and unique digit mapping. -/
theorem flower_garden_mystery_value :
  ∀ (shu_hua_hua_yuan : ℕ) (wo_ai_tan_mi : ℕ),
  shu_hua_hua_yuan + 2011 = wo_ai_tan_mi →
  (∃ (hua yuan tan mi : ℕ),
    0 ≤ hua ∧ hua < 10 ∧
    0 ≤ yuan ∧ yuan < 10 ∧
    0 ≤ tan ∧ tan < 10 ∧
    0 ≤ mi ∧ mi < 10 ∧
    hua ≠ yuan ∧ hua ≠ tan ∧ hua ≠ mi ∧
    yuan ≠ tan ∧ yuan ≠ mi ∧ tan ≠ mi ∧
    shu_hua_hua_yuan = hua * 1000 + yuan * 100 + tan * 10 + mi ∧
    wo_ai_tan_mi = 9713) := sorry

end flower_garden_mystery_value_l247_24788


namespace pigeon_problem_l247_24725

theorem pigeon_problem (x y : ℕ) :
  (1 / 6 : ℝ) * (x + y) = y - 1 ∧ x - 1 = y + 1 → x = 4 ∧ y = 2 :=
by
  sorry

end pigeon_problem_l247_24725


namespace minimal_abs_diff_l247_24721

theorem minimal_abs_diff (a b : ℤ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : a * b - 3 * a + 7 * b = 222) : |a - b| = 54 :=
by
  sorry

end minimal_abs_diff_l247_24721


namespace part_I_part_II_l247_24770

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| + |2 * x + 5|
def g (x : ℝ) : ℝ := |x - 1| - |2 * x|

-- Part I
theorem part_I : ∀ x : ℝ, g x > -4 → -5 < x ∧ x < -3 :=
by
  sorry

-- Part II
theorem part_II : 
  (∃ x1 x2 : ℝ, f x1 a = g x2) → -6 ≤ a ∧ a ≤ -4 :=
by
  sorry

end part_I_part_II_l247_24770


namespace factor_100_minus_16y2_l247_24793

theorem factor_100_minus_16y2 (y : ℝ) : 100 - 16 * y^2 = 4 * (5 - 2 * y) * (5 + 2 * y) := 
by sorry

end factor_100_minus_16y2_l247_24793


namespace grain_distance_l247_24740

theorem grain_distance
    (d : ℝ) (v_church : ℝ) (v_cathedral : ℝ)
    (h_d : d = 400) (h_v_church : v_church = 20) (h_v_cathedral : v_cathedral = 25) :
    ∃ x : ℝ, x = 1600 / 9 ∧ v_church * x = v_cathedral * (d - x) :=
by
  sorry

end grain_distance_l247_24740


namespace union_complement_eq_complement_intersection_eq_l247_24760

-- Define the universal set U and sets A, B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 3, 5, 7}

-- Theorem 1: A ∪ (U \ B) = {2, 4, 5, 6}
theorem union_complement_eq : A ∪ (U \ B) = {2, 4, 5, 6} := by
  sorry

-- Theorem 2: U \ (A ∩ B) = {1, 2, 3, 4, 6, 7}
theorem complement_intersection_eq : U \ (A ∩ B) = {1, 2, 3, 4, 6, 7} := by
  sorry

end union_complement_eq_complement_intersection_eq_l247_24760


namespace cos_squared_plus_twice_sin_double_alpha_l247_24745

theorem cos_squared_plus_twice_sin_double_alpha (α : ℝ) (h : Real.tan α = 3 / 4) :
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 :=
by
  sorry

end cos_squared_plus_twice_sin_double_alpha_l247_24745
