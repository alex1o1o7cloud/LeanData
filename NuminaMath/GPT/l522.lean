import Mathlib

namespace smallest_positive_cube_ends_in_112_l522_522520

theorem smallest_positive_cube_ends_in_112 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 112 ∧ (∀ m : ℕ, (m > 0 ∧ m^3 % 1000 = 112) → n ≤ m) :=
by
  sorry

end smallest_positive_cube_ends_in_112_l522_522520


namespace distinct_x_intercepts_l522_522932

theorem distinct_x_intercepts : 
  let f := λ x : ℝ, (x - 5) * (x^2 + 5 * x + 6) in
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ f a = 0 ∧ f b = 0 ∧ f c = 0 :=
by
  let f := λ x : ℝ, (x - 5) * (x^2 + 5 * x + 6)
  have h1 : f 5 = 0 := sorry
  have h2 : f (-2) = 0 := sorry
  have h3 : f (-3) = 0 := sorry
  have ha : 5 ≠ -2 := by linarith
  have hb : -2 ≠ -3 := by linarith
  have hc : 5 ≠ -3 := by linarith
  use [5, -2, -3]
  exact ⟨ha, hb, hc, h1, h2, h3⟩

end distinct_x_intercepts_l522_522932


namespace sum_of_midpoints_l522_522724

theorem sum_of_midpoints (a b c : ℝ) (h : a + b + c = 12) : 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 12 :=
by
  sorry

end sum_of_midpoints_l522_522724


namespace coconut_grove_yield_l522_522608

theorem coconut_grove_yield (x Y : ℕ) (h1 : x = 10)
  (h2 : (x + 2) * 30 + x * Y + (x - 2) * 180 = 3 * x * 100) : Y = 120 :=
by
  -- Proof to be provided
  sorry

end coconut_grove_yield_l522_522608


namespace niko_total_profit_l522_522331

def pairs_of_socks : Nat := 9
def cost_per_pair : ℝ := 2
def profit_percentage_first_four : ℝ := 0.25
def profit_per_pair_remaining_five : ℝ := 0.2

theorem niko_total_profit :
  let total_profit_first_four := 4 * (cost_per_pair * profit_percentage_first_four)
  let total_profit_remaining_five := 5 * profit_per_pair_remaining_five
  let total_profit := total_profit_first_four + total_profit_remaining_five
  total_profit = 3 := by
  sorry

end niko_total_profit_l522_522331


namespace ways_to_take_out_beer_l522_522015

theorem ways_to_take_out_beer :
  (∃ (x y z : ℕ), x + y + z = 37 ∧ (∃ (a b c : ℕ), a * 4 + b * 3 = 24 ∧ a + b = 7 ∧ z = nat.choose 7 3)) :=
begin
  sorry
end

end ways_to_take_out_beer_l522_522015


namespace convex_pentagon_angle_greater_than_36_l522_522342

theorem convex_pentagon_angle_greater_than_36
  (α γ : ℝ)
  (h_sum : 5 * α + 10 * γ = 3 * Real.pi)
  (h_convex : ∀ i : Fin 5, (α + i.val * γ < Real.pi)) :
  α > Real.pi / 5 :=
sorry

end convex_pentagon_angle_greater_than_36_l522_522342


namespace croatian_math_olympiad_2016_l522_522511

theorem croatian_math_olympiad_2016 (f : ℝ → ℝ) (h : ∀ x y : ℝ, f(x * y + 1) = f(x) * f(y) - f(y) - x + 2) :
  f = λ x, x + 1 :=
by
  sorry

end croatian_math_olympiad_2016_l522_522511


namespace coloring_satisfies_conditions_l522_522979

/-- Define what it means for a point to be a lattice point -/
def is_lattice_point (x y : ℤ) : Prop := true

/-- Define the coloring function based on coordinates -/
def color (x y : ℤ) : Prop :=
  (x % 2 = 1 ∧ y % 2 = 1) ∨   -- white
  (x % 2 = 1 ∧ y % 2 = 0) ∨   -- black
  (x % 2 = 0)                 -- red (both (even even) and (even odd) are included)

/-- Proving the method of coloring lattice points satisfies the given conditions -/
theorem coloring_satisfies_conditions :
  (∀ x y : ℤ, is_lattice_point x y → 
    color x y ∧ 
    ∃ (A B C : ℤ × ℤ), 
      (is_lattice_point A.fst A.snd ∧ 
       is_lattice_point B.fst B.snd ∧ 
       is_lattice_point C.fst C.snd ∧ 
       color A.fst A.snd ∧ 
       color B.fst B.snd ∧ 
       color C.fst C.snd ∧
       ∃ D : ℤ × ℤ, 
         (is_lattice_point D.fst D.snd ∧ 
          color D.fst D.snd ∧ 
          D.fst = A.fst + C.fst - B.fst ∧ 
          D.snd = A.snd + C.snd - B.snd))) :=
sorry

end coloring_satisfies_conditions_l522_522979


namespace smallest_number_divisible_condition_l522_522406

theorem smallest_number_divisible_condition (x : ℕ) :
  (∃ k : ℕ, k ≠ 0 ∧ x + 3 = 35 * k) →
  (∃ l : ℕ, l ≠ 0 ∧ x + 3 = 25 * l) →
  (∃ m : ℕ, m ≠ 0 ∧ x + 3 = 21 * m) →
  x + 3 = 4728 * Nat.lcm (Nat.lcm 35 25) 21 →
  x = 2482197 :=
begin
  sorry
end

end smallest_number_divisible_condition_l522_522406


namespace common_divisors_9240_13860_l522_522940

def num_divisors (n : ℕ) : ℕ :=
  -- function to calculate the number of divisors (implementation is not provided here)
  sorry

theorem common_divisors_9240_13860 :
  let d := Nat.gcd 9240 13860
  d = 924 → num_divisors d = 24 := by
  intros d gcd_eq
  rw [gcd_eq]
  sorry

end common_divisors_9240_13860_l522_522940


namespace gcd_Sm_Sn_l522_522928

theorem gcd_Sm_Sn {m n : ℕ} (hmn_coprime : Nat.coprime m n) : 
  Nat.gcd (5^m + 7^m) (5^n + 7^n) = 12 :=
sorry

end gcd_Sm_Sn_l522_522928


namespace common_divisors_count_l522_522943

-- Define the numbers
def num1 := 9240
def num2 := 13860

-- Define the gcd of the numbers
def gcdNum := Nat.gcd num1 num2

-- Prove the number of divisors of the gcd is 48
theorem common_divisors_count : (Nat.divisors gcdNum).card = 48 :=
by
  -- Normally we would provide a detailed proof here
  sorry

end common_divisors_count_l522_522943


namespace infinite_slips_have_repeated_numbers_l522_522974

theorem infinite_slips_have_repeated_numbers
  (slips : Set ℕ) (h_inf_slips : slips.Infinite)
  (h_sub_infinite_imp_repeats : ∀ s : Set ℕ, s.Infinite → ∃ x ∈ s, ∃ y ∈ s, x ≠ y ∧ x = y) :
  ∃ n : ℕ, {x ∈ slips | x = n}.Infinite :=
by sorry

end infinite_slips_have_repeated_numbers_l522_522974


namespace max_arithmetic_sequence_sum_l522_522193

theorem max_arithmetic_sequence_sum :
  let a := 5
  let d := - (5 / 7)
  ∃ (n : ℕ), (n = 7 ∨ n = 8) ∧ 
            ((n / 2) * (2 * a + (n - 1) * d) = 20) :=
by
  let a := 5
  let d := - (5 / 7)
  -- The proof is omitted.
  sorry

end max_arithmetic_sequence_sum_l522_522193


namespace prime_and_composite_l522_522430

theorem prime_and_composite {n : ℕ} (h : n > 2) 
  : (nat.prime (2^n - 1) ∨ nat.prime (2^n + 1)) → (¬ nat.prime (2^n - 1) ∨ ¬ nat.prime (2^n + 1)) := 
by 
  sorry

end prime_and_composite_l522_522430


namespace value_of_f_neg_3_over_2_l522_522599

def f (x : ℝ) : ℝ :=
if x < 1 then f (x + 1) else 2 * x - 1

theorem value_of_f_neg_3_over_2 : f (-3/2) = 2 := by
  sorry

end value_of_f_neg_3_over_2_l522_522599


namespace y_paid_per_week_l522_522733

variable (x y z : ℝ)

-- Conditions
axiom h1 : x + y + z = 900
axiom h2 : x = 1.2 * y
axiom h3 : z = 0.8 * y

-- Theorem to prove
theorem y_paid_per_week : y = 300 := by
  sorry

end y_paid_per_week_l522_522733


namespace initial_average_of_observations_l522_522691

theorem initial_average_of_observations {A : ℝ} (h1 : (6 * A + 7) / 7 = A - 1) : A = 14 :=
begin
  sorry
end

end initial_average_of_observations_l522_522691


namespace largest_prime_factor_of_sum_l522_522039

def valid_sequence (seq : Vector ℕ 4) : Prop :=
  ∀ (i : Fin 4), 
    let next_i := (i + 1) % 4 in
    (seq[i] % 100) / 10 = (seq[next_i] / 100) ∧
    (seq[i] % 10) = ((seq[next_i] % 100) / 10) 

noncomputable def sum_of_sequence (seq : Vector ℕ 4) : ℕ :=
  Vector.foldl (· + ·) 0 seq

theorem largest_prime_factor_of_sum (seq : Vector ℕ 4) (h : valid_sequence seq) :
  37 ∣ sum_of_sequence seq :=
sorry

end largest_prime_factor_of_sum_l522_522039


namespace find_x_y_z_l522_522393

theorem find_x_y_z :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 
  4 * real.sqrt (real.cbrt 7 - real.cbrt 6) = real.cbrt x + real.cbrt y - real.cbrt z ∧
  x + y + z = 79 :=
by 
  sorry

end find_x_y_z_l522_522393


namespace sum_volumes_of_spheres_sum_volumes_of_tetrahedrons_l522_522057

noncomputable def volume_of_spheres (V : ℝ) : ℝ :=
  V * (27 / 26)

noncomputable def volume_of_tetrahedrons (V : ℝ) : ℝ :=
  (3 * V * Real.sqrt 3) / (13 * Real.pi)

theorem sum_volumes_of_spheres (V : ℝ) : 
  (∑' n : ℕ, (V * (1/27)^n)) = volume_of_spheres V :=
sorry

theorem sum_volumes_of_tetrahedrons (V : ℝ) (r : ℝ) : 
  (∑' n : ℕ, (8/9 / Real.sqrt 3 * (r^3) * (1/27)^n * (1/26))) = volume_of_tetrahedrons V :=
sorry

end sum_volumes_of_spheres_sum_volumes_of_tetrahedrons_l522_522057


namespace angle_between_bisector_and_altitude_l522_522118

-- Given triangle ABC with certain properties and drawing relevant lines and angles
variables {A B C H D : Type} -- points A, B, C, H, D
variables {angle BAC ABC ACB BAD HAH: Type} -- angles defined

-- Definitions of angles of the triangle ABC
def α : Type := angle BAC -- α is ∠BAC
def β : Type := angle ABC -- β is ∠ABC
def γ : Type := angle ACB -- γ is ∠ACB

-- Definitions of altitude AH and angle bisector AD
def AH_altitude := ∀ (A B C H :Type), H = foot_of_perpendicular A B C -- AH is the altitude
def AD_bisector := ∀ (A B C D : Type), D = angle_bisector A B C -- AD is the angle bisector

-- The key theorem to be proven
theorem angle_between_bisector_and_altitude 
  (α β γ : Type) 
  (H : Type) 
  (AH_altitude : AH_altitude) 
  (AD_bisector : AD_bisector) : 
  angle_between_bisector_and_altitude AD AH = (β - γ) / 2 := 
  sorry

end angle_between_bisector_and_altitude_l522_522118


namespace f_f_2_eq_2_l522_522872

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else -2 * x + 3

theorem f_f_2_eq_2 : f (f 2) = 2 :=
by 
  -- hint for proofs can be provided here when necessary
  sorry

end f_f_2_eq_2_l522_522872


namespace p_lt_q_l522_522297

theorem p_lt_q (n : ℕ) (x : Fin n → ℝ) (a : ℝ) (h : a ≠ (∑ i, x i) / n) :
  (∑ i, (x i - (∑ i, x i) / n)^2) < (∑ i, (x i - a)^2) :=
sorry

end p_lt_q_l522_522297


namespace geometric_series_sum_l522_522485

noncomputable def sum_geometric_series : ℝ :=
  ∑' n from 2, ∑ k from 1 to n-1, k / 3^(n + k)

theorem geometric_series_sum :
  sum_geometric_series = 9 / 128 :=
by
  sorry

end geometric_series_sum_l522_522485


namespace josh_money_left_l522_522288

def initial_amount : ℝ := 9
def spent_on_drink : ℝ := 1.75
def spent_on_item : ℝ := 1.25

theorem josh_money_left : initial_amount - (spent_on_drink + spent_on_item) = 6 := by
  sorry

end josh_money_left_l522_522288


namespace inequality_am_gm_l522_522640

theorem inequality_am_gm (n : ℕ) (h₁ : 3 ≤ n) 
(a : Fin n → ℝ) (h₂ : ∀ i, 1 < i ∧ i ∈ (Finset.range n).erase 0 → a i > 0)
(h₃ : (∏ i in (Finset.range n).erase 0, a i) = 1) : 
(∏ i in Finset.filter (λ j, 1 < j) (Finset.range n), (1 + a i)^i) > n^n :=
by
  sorry

end inequality_am_gm_l522_522640


namespace tan_11pi_over_6_l522_522007

theorem tan_11pi_over_6 : Real.tan (11 * Real.pi / 6) = - (Real.sqrt 3 / 3) :=
by
  sorry

end tan_11pi_over_6_l522_522007


namespace tank_capacity_l522_522440

-- Definitions from conditions
def initial_fraction := (1 : ℚ) / 4  -- The tank is 1/4 full initially
def added_amount := 5  -- Adding 5 liters

-- The proof problem to show that the tank's total capacity c equals 60 liters
theorem tank_capacity
  (c : ℚ)  -- The total capacity of the tank in liters
  (h1 : c / 4 + added_amount = c / 3)  -- Adding 5 liters makes the tank 1/3 full
  : c = 60 := 
sorry

end tank_capacity_l522_522440


namespace johns_average_speed_l522_522281

def start_time : ℕ := 8 * 60 + 15 -- convert 8:15 a.m. to minutes
def end_time : ℕ := 16 * 60 + 45 -- convert 4:45 p.m. to minutes
def total_time_hours : ℝ := (end_time - start_time : ℕ) / 60.0 -- total time in hours

def first_distance : ℝ := 100 -- distance in miles
def second_distance : ℝ := 80 -- distance in miles
def total_distance : ℝ := first_distance + second_distance -- total distance in miles

def average_speed : ℝ := total_distance / total_time_hours -- average speed in mph

theorem johns_average_speed :
  average_speed = 360 / 17 :=
by
  -- proof goes here
  sorry

end johns_average_speed_l522_522281


namespace bernardo_always_larger_l522_522811

def bernardo_set : set ℕ := {2, 4, 6, 8, 10}
def silvia_set : set ℕ := {1, 3, 5, 7, 9}

theorem bernardo_always_larger : 
  ∀ b1 b2 b3 s1 s2 s3 : ℕ,
    b1 ∈ bernardo_set → b2 ∈ bernardo_set → b3 ∈ bernardo_set →
    b1 ≠ b2 → b1 ≠ b3 → b2 ≠ b3 →
    s1 ∈ silvia_set → s2 ∈ silvia_set → s3 ∈ silvia_set →
    s1 ≠ s2 → s1 ≠ s3 → s2 ≠ s3 →
    100 * b1 + 10 * b2 + b3 > 100 * s1 + 10 * s2 + s3 :=
by sorry

end bernardo_always_larger_l522_522811


namespace exists_equally_dividing_line_l522_522874

-- Definitions based on the conditions
def PointsOnPlane := { points: set (ℝ × ℝ) // points.card = 2000 }
def NoFourCollinear (P: PointsOnPlane) := ¬ ∃ (a b c d: P.val), collinear {a, b, c} ∧ collinear {a, b, d}
def RedPoints (P: PointsOnPlane) := {p ∈ P.val | /* p is red */}
def BluePoints (P: PointsOnPlane) := {p ∈ P.val | /* p is blue */}

-- Given Problem Conditions
variables (P: PointsOnPlane) [decidable_pred (NoFourCollinear P)] 
  (red_points: set (ℝ × ℝ)) 
  (blue_points: set (ℝ × ℝ))
  (h_red: red_points ⊆ P.val ∧ red_points.card = 1600)
  (h_blue: blue_points ⊆ P.val ∧ blue_points.card = 400)

-- The required statement
theorem exists_equally_dividing_line (P: PointsOnPlane) :
  NoFourCollinear P →
  ∃ (l: ℝ × ℝ → Prop), 
    ∃ (n_red_left n_red_right n_blue_left n_blue_right: ℕ),
      l ∩ P.val = {points | points ∈ P.val ∧ (points.val.snd < 0)} ∨ (points.val.snd = 0) ∨ (points.val.snd > 0) ∧
      red_points.inter {points | points ∈ P.val ∧ (points.val.snd < 0)}.card = 800 ∧
      red_points.inter {points | points ∈ P.val ∧ (points.val.snd > 0)}.card = 800 ∧
      blue_points.inter {points | points ∈ P.val ∧ (points.val.snd < 0)}.card = 200 ∧
      blue_points.inter {points | points ∈ P.val ∧ (points.val.snd > 0)}.card = 200
  | _, _ => sorry

end exists_equally_dividing_line_l522_522874


namespace arithmetic_prog_triangle_l522_522603

theorem arithmetic_prog_triangle (a b c : ℝ) (h : a < b ∧ b < c ∧ 2 * b = a + c)
    (hα : ∀ t, t = a ↔ t = min a (min b c))
    (hγ : ∀ t, t = c ↔ t = max a (max b c)) :
    3 * (Real.tan (α / 2)) * (Real.tan (γ / 2)) = 1 := sorry

end arithmetic_prog_triangle_l522_522603


namespace coral_must_read_pages_to_finish_book_l522_522103

theorem coral_must_read_pages_to_finish_book
  (total_pages first_week_read second_week_percentage pages_remaining first_week_left second_week_read : ℕ)
  (initial_pages_read : ℕ := total_pages / 2)
  (remaining_after_first_week : ℕ := total_pages - initial_pages_read)
  (read_second_week : ℕ := remaining_after_first_week * second_week_percentage / 100)
  (remaining_after_second_week : ℕ := remaining_after_first_week - read_second_week)
  (final_pages_to_read : ℕ := remaining_after_second_week):
  total_pages = 600 → first_week_read = 300 → second_week_percentage = 30 →
  pages_remaining = 300 → first_week_left = 300 → second_week_read = 90 →
  remaining_after_first_week = 300 - 300 →
  remaining_after_second_week = remaining_after_first_week - second_week_read →
  third_week_read = remaining_after_second_week →
  third_week_read = 210 := by
  sorry

end coral_must_read_pages_to_finish_book_l522_522103


namespace eccentricity_of_ellipse_equation_of_ellipse_l522_522538

noncomputable theory
open_locale real

variables {a b c : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : a = sqrt 2 * b)

def ellipse_equation (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

theorem eccentricity_of_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (h3 : a = sqrt 2 * b) : (sqrt (a^2 - b^2)) / a = sqrt 2 / 2 :=
sorry

theorem equation_of_ellipse (c : ℝ) (h3 : a = sqrt 2) 
    (h4 : circle (0, b) (2 * b, -b) (c / 2, -c / 2) 
    ⟨(3 * (c / 2) + (c / 2) + 3)/ sqrt 10 = sqrt 10 / 2 * c⟩)
    : ellipse_equation 1 :=
sorry

end eccentricity_of_ellipse_equation_of_ellipse_l522_522538


namespace find_number_l522_522858

-- Let x be the unknown number
variable (x : ℝ)

-- Define the given condition
def condition (x : ℝ) : Prop :=
  (0.10 * x) + (0.08 * 24) = 5.92

-- State the theorem we want to prove
theorem find_number : condition 40 :=
by
  unfold condition
  rw [show 0.10 * 40 = 4.00, by norm_num]
  rw [show 0.08 * 24 = 1.92, by norm_num]
  rw [show 4.00 + 1.92 = 5.92, by norm_num]
  sorry

end find_number_l522_522858


namespace problem1_problem2_l522_522883

open EuclideanGeometry

variables (A B C H E F D G M N : Point)
variables (O₁ O₂ : Point)
variables (triangle_ABC : Triangle)
variables (H : isOrthocenter H triangle_ABC)
variables (E : isOnSegment E H C)
variables (F : extendsTo F H C (distance E C))
variables (D : perpendicular F D B C)
variables (G : perpendicular E G H B)
variables (M : isMidpoint M C F)
variables (O₁ : isCircumcenter O₁ (triangle.make A B G))
variables (O₂ : isCircumcenter O₂ (triangle.make B C H))
variables (N : isOtherIntersectionPoint N (circumcircle O₁) (circumcircle O₂))

theorem problem1 : cyclic A B D G := by sorry

theorem problem2 : cyclic O₁ O₂ M N := by sorry

end problem1_problem2_l522_522883


namespace find_smolest_number_l522_522259

def A := 12345678987654321
def sum60 (l : List ℕ) := l.sum = 60
def natural_number_form (A : ℕ) (remove_digits : List ℕ) := 
  List.perm (A.digits 10) (remove_digits ++ (A - remove_digits.sum).digits 10)

theorem find_smolest_number (B : ℕ) (remove_digits : List ℕ) :
  A = 12345678987654321 ∧ 
  sum60 remove_digits ∧ 
  natural_number_form A remove_digits ∧ 
  B = (Digits remaining_digits).eval 10 →
  B = 489 := 
sorry

end find_smolest_number_l522_522259


namespace largest_even_of_sum_140_l522_522388

theorem largest_even_of_sum_140 :
  ∃ (n : ℕ), 2 * n + 2 * (n + 1) + 2 * (n + 2) + 2 * (n + 3) = 140 ∧ 2 * (n + 3) = 38 :=
by
  sorry

end largest_even_of_sum_140_l522_522388


namespace problem_statement_l522_522201

noncomputable def A := 1
noncomputable def ϕ := π / 2

def f (x : ℝ) : ℝ := A * Real.sin (x + ϕ)

theorem problem_statement :
  (∀ x, x ∈ ℝ → f x ≤ 1) ∧ f (π / 3) = 1 / 2 
  → f (3 * π / 4) = -Real.sqrt 2 / 2 := 
by
  -- The proof is omitted here.
  sorry

end problem_statement_l522_522201


namespace log_property_implies_y_eq_9_l522_522592

theorem log_property_implies_y_eq_9 (y : ℝ) (h : log 3 y * log y 9 = 2) : y = 9 :=
sorry

end log_property_implies_y_eq_9_l522_522592


namespace find_side_length_of_triangle_l522_522604

variable (A B C a b c : ℝ)

theorem find_side_length_of_triangle
  (h1 : B = π / 3)
  (h2 : b = 6)
  (h3 : sin A - 2 * sin C = 0) :
  a = 4 * sqrt 3 := by
  sorry

end find_side_length_of_triangle_l522_522604


namespace possible_values_expression_l522_522173

-- Defining the main expression 
def main_expression (a b c d : ℝ) : ℝ :=
  (a / |a|) + (b / |b|) + (c / |c|) + (d / |d|) + (abcd / |abcd|)

-- The theorem that we need to prove
theorem possible_values_expression (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) : 
  main_expression a b c d ∈ {5, 1, -3} :=
sorry

end possible_values_expression_l522_522173


namespace group_of_three_possible_l522_522609

theorem group_of_three_possible (n_people : ℕ) (n_quarrels : ℕ) (n_groups_of_three : ℕ) 
(h1 : n_people = 10) 
(h2 : n_quarrels = 14) 
(h3 : n_groups_of_three = 8) : 
    ∃ (group : Finset (Fin n_people)), group.card = 3 ∧ (group.pairwise (λ x y, (x, y) ∉ n_quarrels)) :=
sorry

end group_of_three_possible_l522_522609


namespace problem_l522_522530

noncomputable theory

def m (x : ℝ) := (2 * Real.sin x, 2 * Real.cos x)
def n := (Real.cos (Real.pi / 3), -Real.sin (Real.pi / 3))
def f (x : ℝ) := (m x).1 * n.1 + (m x).2 * n.2 + 1
def g (x : ℝ) := f (Real.pi / 2 * x)
def h (x : ℝ) := (Real.sin x * (f (x + Real.pi / 3))^2 - 8) / (1 + (Real.cos x)^2)

theorem problem (x : ℝ) :
  f (Real.pi / 2) = 2 ∧
  (∀ x, f x ≤ 3) ∧
  (finset.range 2015).sum (λ k, g (k + 1)) = 2015 + Real.sqrt 3 ∧
  (∃ M m, (∀ x ∈ set.Icc (-5 * Real.pi / 4) (5 * Real.pi / 4), m ≤ h x ∧ h x ≤ M) ∧
    M + m = -8) :=
by
  sorry

end problem_l522_522530


namespace length_of_first_train_is_140_l522_522741

theorem length_of_first_train_is_140 
  (speed1 : ℝ) (speed2 : ℝ) (time_to_cross : ℝ) (length2 : ℝ) 
  (h1 : speed1 = 60) 
  (h2 : speed2 = 40) 
  (h3 : time_to_cross = 12.239020878329734) 
  (h4 : length2 = 200) : 
  ∃ (length1 : ℝ), length1 = 140 := 
by
  sorry

end length_of_first_train_is_140_l522_522741


namespace molecular_weight_3_moles_l522_522110

theorem molecular_weight_3_moles
  (C_weight : ℝ)
  (H_weight : ℝ)
  (N_weight : ℝ)
  (O_weight : ℝ)
  (Molecular_formula : ℕ → ℕ → ℕ → ℕ → Prop)
  (molecular_weight : ℝ)
  (moles : ℝ) :
  C_weight = 12.01 →
  H_weight = 1.008 →
  N_weight = 14.01 →
  O_weight = 16.00 →
  Molecular_formula 13 9 5 7 →
  molecular_weight = 156.13 + 9.072 + 70.05 + 112.00 →
  moles = 3 →
  3 * molecular_weight = 1041.756 :=
by
  sorry

end molecular_weight_3_moles_l522_522110


namespace find_phi_find_max_min_l522_522199

namespace MyProblem

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := sqrt 5 * sin (2 * x + φ)

theorem find_phi (φ : ℝ) (h0 : 0 < φ ∧ φ < π) (h1 : ∀ x, f (π/3 - x) φ = f (π/3 + x) φ) : φ = 5 * π / 6 :=
by
  sorry

theorem find_max_min (x : ℝ) (h0 : x ∈ Icc (-(π/12)) (π/2)) : 
  let φ := 5 * π / 6
  in f x φ ≤ sqrt 5 ∧ f x φ ≥ -sqrt 5 ∧ 
     (f (-π/12) φ = sqrt 15 / 2) ∧ (f (π/3) φ = -sqrt 5) :=
by
  sorry

end MyProblem

end find_phi_find_max_min_l522_522199


namespace convex_quadrilateral_diagonal_division_l522_522037

theorem convex_quadrilateral_diagonal_division (quadrilateral : Type) (is_convex : convex quadrilateral)
    (alyosha_claim : can_be_divided_into_acute_triangles quadrilateral)
    (borya_claim : can_be_divided_into_right_triangles quadrilateral)
    (vasya_claim : can_be_divided_into_obtuse_triangles quadrilateral)
    (exactly_one_wrong : exactly_one_wrong_claim quadrilateral alyosha_claim borya_claim vasya_claim) :
  vasya_claim :=
sorry

end convex_quadrilateral_diagonal_division_l522_522037


namespace closest_approximation_l522_522423

-- Define all the conditions
def a : ℝ := 69.28
def b : ℝ := 0.004
def c : ℝ := 0.03
def result := (a * b) / c

-- Prove that the result is approximately equal to 9.24 when rounded to two decimal places.
theorem closest_approximation : Real.round (result * 100) / 100 = 9.24 := by
  sorry

end closest_approximation_l522_522423


namespace possible_values_expression_l522_522175

-- Defining the main expression 
def main_expression (a b c d : ℝ) : ℝ :=
  (a / |a|) + (b / |b|) + (c / |c|) + (d / |d|) + (abcd / |abcd|)

-- The theorem that we need to prove
theorem possible_values_expression (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) : 
  main_expression a b c d ∈ {5, 1, -3} :=
sorry

end possible_values_expression_l522_522175


namespace area_of_triangle_arithmetic_sequence_l522_522890

theorem area_of_triangle_arithmetic_sequence (a b c : ℝ) (h : a < b ∧ b < c)
  (h_seq : b - a = 4 ∧ c - b = 4) 
  (h120 : ∃ A B C : ℝ, A + B + C = π ∧ A = (2 / 3) * π ∧ A + B + C = a + b + c) :
  let s := (a + b + c) / 2 in
  let area := sqrt (s * (s - a) * (s - b) * (s - c)) in
  area = 15 * sqrt 3 :=
by
  -- Let x be the middle side
  have x_eq : b = 10 := by
    -- Derived from the given conditions and cosine rule simplification
    sorry
  -- Calculate side lengths
  have sides_eq : a = 6 ∧ b = 10 ∧ c = 14 := by
    -- Derived from arithmetic sequence with common difference of 4
    sorry
  -- The area of the triangle using Heron's formula
  have area_eq : area = 15 * sqrt 3 := by
    -- Calculation using sides 6, 10, 14 and angle 120 degrees
    sorry
  exact area_eq

end area_of_triangle_arithmetic_sequence_l522_522890


namespace angle_between_a_and_c_l522_522886

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

def angle_between_vectors (u v : V) : ℝ := real.arccos (inner_product_space.inner u v / (‖u‖ * ‖v‖))

theorem angle_between_a_and_c
  (h1 : a + b + c = 0)
  (h2 : ‖a‖ = 1)
  (h3 : ‖b‖ = 1)
  (angle_ab : angle_between_vectors a b = real.pi / 3) :
  angle_between_vectors a c = 5 * real.pi / 6 :=
sorry

end angle_between_a_and_c_l522_522886


namespace possible_values_of_expression_l522_522164

theorem possible_values_of_expression (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (v : ℤ), v ∈ ({5, 1, -3, -5} : Set ℤ) ∧ v = (Int.sign a + Int.sign b + Int.sign c + Int.sign d + Int.sign (a * b * c * d)) :=
by
  sorry

end possible_values_of_expression_l522_522164


namespace four_digit_div_by_99_then_sum_div_by_18_l522_522341

/-- 
If a whole number with at most four digits is divisible by 99, then 
the sum of its digits is divisible by 18. 
-/
theorem four_digit_div_by_99_then_sum_div_by_18 (n : ℕ) (h1 : n < 10000) (h2 : 99 ∣ n) : 
  18 ∣ (n.digits 10).sum := 
sorry

end four_digit_div_by_99_then_sum_div_by_18_l522_522341


namespace max_value_ab_bc_cd_l522_522301

theorem max_value_ab_bc_cd (a b c d : ℝ) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0)
  (h_sum : a + b + c + d = 120) : ab + bc + cd ≤ 3600 :=
by {
  sorry
}

end max_value_ab_bc_cd_l522_522301


namespace academy_topics_selection_l522_522617

-- Defining the main statement based on the problem conditions
theorem academy_topics_selection :
  ∃ (selected_topics : Finset Topic), 
    (selected_topics.card = 250) ∧ 
    (∀ t ∈ selected_topics, ∀ (a1 a2 a3 : Academician), 
      (t.interests a1 a2 a3) → 
      (∀ t' ∈ selected_topics, (t' ≠ t → ¬ t'.interests a1 a2 a3))) := 
sorry

end academy_topics_selection_l522_522617


namespace possible_values_expression_l522_522168

theorem possible_values_expression (a b c d : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) :
  ∃ (x : ℝ), x ∈ {5, 1, -3} ∧ x = (a / |a| + b / |b| + c / |c| + d / |d| + (abcd / |abcd|)) :=
by
  sorry

end possible_values_expression_l522_522168


namespace triangle_area_l522_522893

theorem triangle_area (A B C D E F : Type) 
  [Triangle A B C]
  (D_mid : midpoint D A C)
  (E_point : point_on_line E C B 2 1) 
  (AE_BD_intersect : intersects (line_through A E) (line_through B D) F)
  (area_AFB : triangle_area (triangle A F B) = 1) :
  triangle_area (triangle A B C) = 4 := 
sorry

end triangle_area_l522_522893


namespace mary_initial_nickels_l522_522323

variable {x : ℕ}

theorem mary_initial_nickels (h : x + 5 = 12) : x = 7 := by
  sorry

end mary_initial_nickels_l522_522323


namespace values_of_d_divisible_by_13_l522_522143

def base8to10 (d : ℕ) : ℕ := 3 * 8^3 + d * 8^2 + d * 8 + 7

theorem values_of_d_divisible_by_13 (d : ℕ) (h : d ≥ 0 ∧ d < 8) :
  (1543 + 72 * d) % 13 = 0 ↔ d = 1 ∨ d = 2 :=
by sorry

end values_of_d_divisible_by_13_l522_522143


namespace max_p_value_and_sequences_l522_522117

def is_valid_rectangle (a : ℕ) : Prop :=
  a * 2 ≤ 64 ∧ a ≥ 1

def sequence_sum (seq : List ℕ) : Prop :=
  seq.sum = 32

def increasing_sequence (seq : List ℕ) : Prop :=
  ∀ (i j : ℕ), i < j → i < seq.length → j < seq.length → seq.nth_le i sorry < seq.nth_le j sorry

theorem max_p_value_and_sequences :
  ∃ (seqs : List (List ℕ)),
    (∀ seq ∈ seqs, is_valid_rectangle seq.length ∧ sequence_sum seq ∧ increasing_sequence seq) ∧
    7 = seqs.length ∧
    seqs = [
      [1, 2, 3, 4, 5, 6, 11],
      [1, 2, 3, 4, 5, 7, 10],
      [1, 2, 3, 4, 5, 8, 9],
      [1, 2, 3, 4, 6, 7, 9],
      [1, 2, 3, 5, 6, 7, 8]] :=
sorry

end max_p_value_and_sequences_l522_522117


namespace evaluate_integral_l522_522845

open Real

theorem evaluate_integral : 
  ∫ x in 0..1, sqrt (1 - x^2) + (1/2) * x = (π / 4) + (1 / 4) :=
by
  sorry

end evaluate_integral_l522_522845


namespace solution_count_l522_522589

theorem solution_count (a : ℝ) : 
  (a = 0 ∧ (∃ x, x * |x - a| = a ↔ x = 0)) ∧
  (a ≠ 0 → 
    ((-4 < a ∧ a < 4) →
      ∃! x, x * |x - a| = a) ∧
    ((a = 4 ∨ a = -4) →
      ∃ x1 x2, x1 ≠ x2 ∧ x1 * |x1 - a| = a ∧ x2 * |x2 - a| = a ∧ 
        ∀ x3, x3 * |x3 - a| = a → (x3 = x1 ∨ x3 = x2))) ∧
    ((a > 4 ∨ a < -4) →
        ∃ x1 x2 x3, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 * |x1 - a| = a ∧ x2 * |x2 - a| = a ∧ x3 * |x3 - a| = a ∧
        ∀ x4, x4 * |x4 - a| = a → (x4 = x1 ∨ x4 = x2 ∨ x4 = x3))) :=
sorry

end solution_count_l522_522589


namespace prob_of_nine_correct_is_zero_l522_522458

-- Define the necessary components and properties of the problem
def is_correct_placement (letter: ℕ) (envelope: ℕ) : Prop := letter = envelope

def is_random_distribution (letters : Fin 10 → Fin 10) : Prop := true

-- State the theorem formally
theorem prob_of_nine_correct_is_zero (f : Fin 10 → Fin 10) :
  is_random_distribution f →
  (∃ (count : ℕ), count = 9 ∧ (∀ i : Fin 10, is_correct_placement i (f i) ↔ i = count)) → false :=
by
  sorry

end prob_of_nine_correct_is_zero_l522_522458


namespace runners_meet_again_after_1200_seconds_l522_522526

-- Definitions for runner speeds and track length
def v1 := 4.0 -- meters per second
def v2 := 5.0 -- meters per second
def v3 := 6.0 -- meters per second
def v4 := 7.0 -- meters per second
def track_length := 400 -- meters

-- Main theorem statement
theorem runners_meet_again_after_1200_seconds (t : ℝ) :
  (t > 0) →
  (∃ k : ℤ, t = k * 400 / v1) ∧
  (∃ k : ℤ, t = k * 400 / v2) ∧
  (∃ k : ℤ, t = k * 400 / v3) ∧
  (∃ k : ℤ, t = k * 400 / v4) ↔
  t = 1200 :=
by
  sorry

end runners_meet_again_after_1200_seconds_l522_522526


namespace range_of_a_l522_522205

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then a * x - 1 else x^2 - 2 * a * x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ a ∈ set.Ici (2/3) :=
by sorry

end range_of_a_l522_522205


namespace tyler_cd_count_l522_522400

theorem tyler_cd_count :
  let initial_CDs := 100
  let cds_given_away := initial_CDs * 35 / 100
  let after_giving_away := initial_CDs - cds_given_away
  let birthday_gift := 30
  let after_birthday := after_giving_away + birthday_gift
  let damaged_cds := after_birthday * 1 / 5
  let after_cleanup := after_birthday - damaged_cds
  let jeff_cd_collection := 80
  let jeff_gift := jeff_cd_collection * 15 / 100
  let final_cds := after_cleanup + jeff_gift
  in final_cds = 88 :=
by
  let initial_CDs := 100
  let cds_given_away := initial_CDs * 35 / 100
  let after_giving_away := initial_CDs - cds_given_away
  let birthday_gift := 30
  let after_birthday := after_giving_away + birthday_gift
  let damaged_cds := after_birthday * 1 / 5
  let after_cleanup := after_birthday - damaged_cds
  let jeff_cd_collection := 80
  let jeff_gift := jeff_cd_collection * 15 / 100
  let final_cds := after_cleanup + jeff_gift
  show final_cds = 88 from sorry

end tyler_cd_count_l522_522400


namespace train_passes_man_in_approx_4_15_secs_l522_522797

def length_of_train : ℝ := 150
def speed_of_train_kmh : ℝ := 120
def speed_of_man_kmh : ℝ := 10

def kmh_to_ms (kmh : ℝ) : ℝ := kmh * 1000 / 3600

def speed_of_train_ms : ℝ := kmh_to_ms speed_of_train_kmh
def speed_of_man_ms : ℝ := kmh_to_ms speed_of_man_kmh

def relative_speed_ms : ℝ := speed_of_train_ms + speed_of_man_ms

noncomputable def time_to_pass_man : ℝ := length_of_train / relative_speed_ms

theorem train_passes_man_in_approx_4_15_secs :
  abs (time_to_pass_man - 4.15) < 0.01 :=
by
  sorry

end train_passes_man_in_approx_4_15_secs_l522_522797


namespace iterative_average_difference_l522_522095

def iterative_average (seq : List ℝ) : ℝ :=
  seq.foldl (λ acc x, (acc + x) / 2) 0

def increasing_sequence : List ℝ := [1, 2, 3, 4, 5]
def decreasing_sequence : List ℝ := [5, 4, 3, 2, 1]

theorem iterative_average_difference :
  (iterative_average increasing_sequence - iterative_average decreasing_sequence) = 2.125 :=
by
  sorry

end iterative_average_difference_l522_522095


namespace altitudes_geometric_progression_l522_522343

theorem altitudes_geometric_progression 
  {a q : ℝ} (h₁ : a > 0) (h₂ : q > 0) 
  (h₃ : ∃ S : ℝ, S = (1 / 2) * a * (2 * S / a) ∧ S = (1 / 2) * (aq) * (2 * S / (aq)) ∧ S = (1 / 2) * (aq^2) * (2 * S / (aq^2))) :
  ∃ r : ℝ, 
    ∃ hₐ hₐq hₐq2 : ℝ, 
      hₐ = 2 * (classical.some h₃) / a ∧ 
      hₐq = 2 * (classical.some h₃) / (a * q) ∧ 
      hₐq2 = 2 * (classical.some h₃) / (a * q^2) ∧ 
      (hₐq = hₐ / q) ∧
      (hₐq2 = hₐ / q^2) := 
sorry

end altitudes_geometric_progression_l522_522343


namespace g_values_product_l522_522644

-- Define the function g and its properties
variable (g : ℝ → ℝ)
variable (h_fun_eq : ∀ x y : ℝ, g(x^2 + y^2) = (x + y) * (g(x) - g(y)))
variable (h_g1 : g 1 = 3)

theorem g_values_product : ∃ (n s : ℝ), (∀ x y : ℝ, g(2) = x * s) ∧ n * s = 6 := by
  sorry

end g_values_product_l522_522644


namespace part1_l522_522316

def centrally_symmetric_function (y1 y2 : ℝ → ℝ) (p : ℝ × ℝ) : Prop := 
  ∀ (x : ℝ), y2 (2 * p.1 - x) = 2 * p.2 - y1 x

theorem part1 : centrally_symmetric_function (λ x, 2 * x + 3) (λ x, 2 * x - 1) (1, 1) :=
sorry

end part1_l522_522316


namespace ratio_surface_areas_l522_522906

-- Given definitions
def surface_area_cube (side_length : ℝ) : ℝ := 6 * (side_length ^ 2)

def radius_circumscribed_sphere (side_length : ℝ) : ℝ := (side_length * Real.sqrt 3) / 2

def surface_area_sphere (radius : ℝ) : ℝ := 4 * Real.pi * (radius ^ 2)

-- Main statement to prove
theorem ratio_surface_areas (side_length : ℝ) (h : side_length = 1) :
  surface_area_cube side_length / surface_area_sphere (radius_circumscribed_sphere side_length) = 2 / Real.pi := 
by
  -- Placeholder for actual proof
  sorry

end ratio_surface_areas_l522_522906


namespace rate_of_current_l522_522765

noncomputable def speed_in_still_water : ℝ := 3.3
noncomputable def downstream_time (c t : ℝ) : ℝ := (speed_in_still_water + c) * t
noncomputable def upstream_time (c t : ℝ) : ℝ := (speed_in_still_water - c) * (2 * t)

theorem rate_of_current (c t : ℝ) (h : downstream_time c t = upstream_time c t) : c = 1.1 :=
by
  have h1 : (speed_in_still_water + c) * t = (speed_in_still_water - c) * (2 * t), from h
  sorry

end rate_of_current_l522_522765


namespace intersection_points_count_l522_522820

theorem intersection_points_count
  : (∀ n : ℤ, ∃ (x y : ℝ), (x - ⌊x⌋) ^ 2 + y ^ 2 = 2 * (x - ⌊x⌋) ∨ y = 1 / 3 * x) →
    (∃ count : ℕ, count = 12) :=
by
  sorry

end intersection_points_count_l522_522820


namespace infinite_geometric_series_sum_l522_522844

theorem infinite_geometric_series_sum :
  let a := (5 : ℚ) / 4
  let r := (-2 : ℚ) / 3
  S = a / (1 - r) →
  (S : ℚ) = 3 / 4 :=
by
  intros a r S ha hr
  sorry

end infinite_geometric_series_sum_l522_522844


namespace eleven_sided_convex_polygon_diagonals_l522_522821

theorem eleven_sided_convex_polygon_diagonals (n : ℕ) (hn : n = 11) :
  ∀ (polygon : Polygon), is_convex polygon → has_obtuse_angle polygon →
  number_of_diagonals polygon = 44 :=
by
  sorry

end eleven_sided_convex_polygon_diagonals_l522_522821


namespace train_speed_second_part_l522_522455

variables (x v : ℝ)

theorem train_speed_second_part
  (h1 : ∀ t1 : ℝ, t1 = x / 30)
  (h2 : ∀ t2 : ℝ, t2 = 2 * x / v)
  (h3 : ∀ t : ℝ, t = 3 * x / 22.5) :
  (x / 30) + (2 * x / v) = (3 * x / 22.5) → v = 20 :=
by
  intros h4
  sorry

end train_speed_second_part_l522_522455


namespace angle_A_correct_triangle_area_correct_l522_522969

noncomputable def triangle_angle_measure (AC BC : ℝ) (cos_B : ℝ) : ℝ :=
  if AC = 8 ∧ BC = 7 ∧ cos_B = -1/7 then π/3 else 0

noncomputable def triangle_area (AC BC : ℝ) (cos_B : ℝ) (A : ℝ) : ℝ :=
  if AC = 8 ∧ BC = 7 ∧ cos_B = -1/7 ∧ A = π/3 then 6 * real.sqrt 3 else 0

theorem angle_A_correct :
  triangle_angle_measure 8 7 (-1/7) = π/3 :=
by sorry

theorem triangle_area_correct :
  triangle_area 8 7 (-1/7) (π/3) = 6 * real.sqrt 3 :=
by sorry

end angle_A_correct_triangle_area_correct_l522_522969


namespace johns_uncommon_cards_l522_522632

def packs_bought : ℕ := 10
def cards_per_pack : ℕ := 20
def uncommon_fraction : ℚ := 1 / 4

theorem johns_uncommon_cards : packs_bought * (cards_per_pack * uncommon_fraction) = (50 : ℚ) := 
by 
  sorry

end johns_uncommon_cards_l522_522632


namespace number_of_sets_satisfying_conditions_l522_522525

open Set Finset

theorem number_of_sets_satisfying_conditions :
  {x : Finset ℕ // {1, 2} ⊆ x ∧ x ⊆ {1, 2, 3, 4, 5}}.card = 8 :=
by
  have subset_1_2_3_4_5 : {1, 2} ⊆ {1, 2, 3, 4, 5} := by simp
  let base_set := {1, 2, 3, 4, 5}
  let must_include := {1, 2}
  let optional_elements := base_set \ must_include
  let power_set_optional := optional_elements.powerset
  let all_valid_sets := power_set_optional.image (λ s, must_include ∪ s)
  have card_image : all_valid_sets.card = power_set_optional.card := by sorry
  have power_set_card : power_set_optional.card = 2 ^ optional_elements.card := by simp
  have optional_card : optional_elements.card = 3 := by simp
  rw [card_image, power_set_card, optional_card]
  simp

end number_of_sets_satisfying_conditions_l522_522525


namespace repeat_decimals_subtraction_l522_522507

-- Define repeating decimal 0.4 repeating as a fraction
def repr_decimal_4 : ℚ := 4 / 9

-- Define repeating decimal 0.6 repeating as a fraction
def repr_decimal_6 : ℚ := 2 / 3

-- Theorem stating the equivalence of subtraction of these repeating decimals
theorem repeat_decimals_subtraction :
  repr_decimal_4 - repr_decimal_6 = -2 / 9 :=
sorry

end repeat_decimals_subtraction_l522_522507


namespace length_of_CK_l522_522570

theorem length_of_CK
  (A B C D E K : Type)
  [metric_space A] [metric_space B] [metric_space C]
  [metric_space D] [metric_space E] [metric_space K]
  (AB : dist A B = √3)
  (intersects_at_K : ∀ (AD BE : Type), AD ∩ BE = K)
  (concyclic_points : ∀ (K D C E : Type), ∃ (O : Type), is_circumcircle O {K, D, C, E})
  : dist C K = 1 :=
by sorry

end length_of_CK_l522_522570


namespace count_adjacent_even_pairs_correct_l522_522857

def is_even (n : ℕ) : Prop := n % 2 = 0

def has_adjacent_even_digits (n : ℕ) : Prop :=
  let digits := nat.digits 10 n
  ∃ i, i + 1 < digits.length ∧ is_even (digits.get i) ∧ is_even (digits.get (i + 1))

def count_valid_numbers : ℕ :=
  (Finset.range 2018).count (λ n, has_adjacent_even_digits n)

theorem count_adjacent_even_pairs_correct : 
  count_valid_numbers = 738 :=
by
  sorry

end count_adjacent_even_pairs_correct_l522_522857


namespace length_parallel_line_l522_522692

-- Definitions for the isosceles triangle and conditions
def is_isosceles_triangle (A B C : Type*) (AB AC BC : ℝ) : Prop :=
  AB = AC

-- Definitions and conditions 
def base_length (BC : ℝ) : Prop := BC = 20
def parallel_line (DE BC : Type*) : Prop

-- Areas conditions
def area_ratio (area_ABC area_ADE : ℝ) : Prop :=
  area_ADE = (1 / 4) * area_ABC

-- Main theorem to prove length of DE
theorem length_parallel_line
  {A B C : Type*}
  {AB AC BC : ℝ}
  (h_isosceles : is_isosceles_triangle A B C AB AC BC)
  (h_base : base_length BC)
  {DE : ℝ}
  (h_parallel : parallel_line DE BC)
  {area_ABC area_ADE : ℝ}
  (h_area : area_ratio area_ABC area_ADE) :
  DE = 10 :=
sorry

end length_parallel_line_l522_522692


namespace molecular_weight_correct_l522_522405

-- Define the atomic weights
def atomic_weight_Cu : ℝ := 63.546
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999

-- Define the number of atoms in the compound
def num_atoms_Cu : ℕ := 1
def num_atoms_C : ℕ := 1
def num_atoms_O : ℕ := 3

-- Define the molecular weight calculation
def molecular_weight : ℝ :=
  num_atoms_Cu * atomic_weight_Cu + 
  num_atoms_C * atomic_weight_C + 
  num_atoms_O * atomic_weight_O

-- Prove the molecular weight of the compound
theorem molecular_weight_correct : molecular_weight = 123.554 :=
by
  sorry

end molecular_weight_correct_l522_522405


namespace general_term_formula_l522_522273

noncomputable def a_sequence (a : ℝ) (n : ℕ) : ℝ :=
  nat.rec_on n a (λ n an, sqrt ((1 - sqrt (1 - an^2)) / 2))

theorem general_term_formula (a : ℝ) (n : ℕ) (h : 0 < a ∧ a < 1) :
  a_sequence a n = sin ((arcsin a) / 2^(n-1)) :=
sorry

end general_term_formula_l522_522273


namespace evaluate_expression_l522_522755

theorem evaluate_expression (a b : ℝ) (h : (1/2 * a * (1:ℝ)^3 - 3 * b * 1 + 4 = 9)) :
  (1/2 * a * (-1:ℝ)^3 - 3 * b * (-1) + 4 = -1) := by
sorry

end evaluate_expression_l522_522755


namespace distinct_4_digit_integers_count_l522_522586

theorem distinct_4_digit_integers_count : 
  nat.factorial 4 / (nat.factorial 2 * nat.factorial 2) = 6 :=
by sorry

end distinct_4_digit_integers_count_l522_522586


namespace cost_four_bottles_eq_two_l522_522284

namespace Josette

-- Define the given conditions
def cost_three_bottles : ℝ := 1.50
def number_of_bottles_three : ℕ := 3
def number_of_bottles_four : ℕ := 4

-- Calculate the cost of one bottle
def cost_one_bottle : ℝ := cost_three_bottles / number_of_bottles_three

-- State the theorem to prove
theorem cost_four_bottles_eq_two : cost_one_bottle * number_of_bottles_four = 2 :=
by
  sorry

end Josette

end cost_four_bottles_eq_two_l522_522284


namespace edward_dunk_a_clown_tickets_l522_522122

-- Definitions for conditions
def total_tickets : ℕ := 79
def rides : ℕ := 8
def tickets_per_ride : ℕ := 7

-- Theorem statement
theorem edward_dunk_a_clown_tickets :
  let tickets_spent_on_rides := rides * tickets_per_ride
  let tickets_remaining := total_tickets - tickets_spent_on_rides
  tickets_remaining = 23 :=
by
  sorry

end edward_dunk_a_clown_tickets_l522_522122


namespace plane_dividing_ratio_l522_522098

-- Define points and their properties
variables {A B C D M N K L : Type}
variables [linear_order M] [linear_order N]
variables (midpoint_AC : M = (A + C) / 2)
variables (midpoint_BD : N = (B + D) / 2)
variables (CK_ratio_KD : CK / KD = 1 / 2)
variables (plane_divides_AB : AB.divides L in_ratio 1:2)

-- Goal statement
theorem plane_dividing_ratio
  (mid_AC : M = (A + C) / 2)
  (mid_BD : N = (B + D) / 2)
  (point_K : CK / KD = 1 / 2)
  : divides AB L in_ratio 1:2 :=
begin
  sorry
end

end plane_dividing_ratio_l522_522098


namespace find_angle_B_l522_522997

def triangle_angles (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * real.cos B - b * real.cos A = c ∧ C = real.pi / 5

theorem find_angle_B (A B C a b c : ℝ) 
    (h : triangle_angles A B C a b c) : B = 3 * real.pi / 10 :=
by sorry

end find_angle_B_l522_522997


namespace calculate_sum_l522_522431

theorem calculate_sum :
  ∃ (x y : ℝ), x = 300 * 0.70 ∧ y = x * 1.30 ∧ x + y = 483 :=
by
  let x := 300 * 0.70
  let y := x * 1.30
  use x, y
  simp [x, y]
  sorry

end calculate_sum_l522_522431


namespace find_angle_B_l522_522996

def triangle_angles (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * real.cos B - b * real.cos A = c ∧ C = real.pi / 5

theorem find_angle_B (A B C a b c : ℝ) 
    (h : triangle_angles A B C a b c) : B = 3 * real.pi / 10 :=
by sorry

end find_angle_B_l522_522996


namespace common_divisors_9240_13860_l522_522942

def num_divisors (n : ℕ) : ℕ :=
  -- function to calculate the number of divisors (implementation is not provided here)
  sorry

theorem common_divisors_9240_13860 :
  let d := Nat.gcd 9240 13860
  d = 924 → num_divisors d = 24 := by
  intros d gcd_eq
  rw [gcd_eq]
  sorry

end common_divisors_9240_13860_l522_522942


namespace total_items_count_l522_522582

-- Define the costs and total budget
def total_money : ℝ := 50.00
def pie_cost : ℝ := 6.00
def juice_cost : ℝ := 1.50

-- Define number of pies and juices bought
def max_pies : ℕ := Nat.floor (total_money / pie_cost)
def remaining_money : ℝ := total_money - (max_pies * pie_cost)
def max_juices : ℕ := Nat.floor (remaining_money / juice_cost)

-- Theorem statement
theorem total_items_count : (max_pies + max_juices) = 9 := by
  sorry

end total_items_count_l522_522582


namespace log_less_x_squared_l522_522357

theorem log_less_x_squared (x : ℝ) (h : 0 < x) : log (1 + x^2) < x^2 := 
sorry

end log_less_x_squared_l522_522357


namespace arcsin_sqrt3_div_2_eq_pi_div_3_l522_522484

theorem arcsin_sqrt3_div_2_eq_pi_div_3 : Real.arcsin (Real.sqrt 3 / 2) = Real.pi / 3 :=
by
  sorry

end arcsin_sqrt3_div_2_eq_pi_div_3_l522_522484


namespace smallest_positive_period_max_value_min_value_l522_522563

noncomputable def f (x : ℝ) : ℝ := (Math.sqrt 3 / 2) * Real.sin (2 * x) - (1 / 2) * Real.cos (2 * x)

theorem smallest_positive_period : ∃ T > 0, T = Real.pi ∧ ∀ x : ℝ, f (x + T) = f x := by
  sorry

theorem max_value : ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 1 := by
  sorry

theorem min_value : ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = -1 / 2 := by
  sorry

end smallest_positive_period_max_value_min_value_l522_522563


namespace sum_of_a_is_21_l522_522069

theorem sum_of_a_is_21
    (a b c d : ℤ)
    (h1 : a > b)
    (h2 : b > c)
    (h3 : c > d)
    (h4 : a + b + c + d = 60)
    (h5 : {a - b, b - c, c - d, a - c, a - d, b - d} = {2, 3, 5, 7, 8, 11}) :
    a = 21 := by
  sorry

end sum_of_a_is_21_l522_522069


namespace trajectory_equation_minimum_area_l522_522260

noncomputable theory
open_locale classical

-- Define the existence of points and the given distance ratio
def condition_P (x y : ℝ) : Prop := 
  let F := (-1:ℝ, 0:ℝ) in
  let distance_PF := real.sqrt ((x + 1)^2 + y^2) in
  let distance_PLine := abs (x + 2) in
  distance_PF / distance_PLine = real.sqrt 2 / 2

-- Define the trajectory equation as a proposition in Lean 4
theorem trajectory_equation (x y : ℝ) : condition_P x y → (x^2 / 2 + y^2 = 1) :=
sorry

-- Define the quadratic relationship of y1 and y2
def quadratic_y (m y1 y2 : ℝ) : Prop :=
  (m^2 + 2) * y1 * y2 - 2 * m * (y1 + y2) + 2 = 0

-- Define the midpoint M of the chord AB
def midpoint_AB (x1 y1 x2 y2 : ℝ) : ℝ × ℝ := 
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Main theorem for the minimum area S_min
theorem minimum_area (m : ℝ) : ∃ S_min : ℝ, S_min = 2 :=
sorry

end trajectory_equation_minimum_area_l522_522260


namespace sum_of_squares_of_medians_and_area_l522_522752

theorem sum_of_squares_of_medians_and_area (a b c : ℝ) (ha : a = 8) (hb : b = 15) (hc : c = 17) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let medians_sum := 3 / 4 * (a^2 + b^2 + c^2)
  medians_sum + area^2 = 4033.5 :=
by
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let medians_sum := 3 / 4 * (a^2 + b^2 + c^2)
  have ha : a = 8 := rfl
  have hb : b = 15 := rfl
  have hc : c = 17 := rfl
  sorry

end sum_of_squares_of_medians_and_area_l522_522752


namespace find_spherical_gamma_l522_522719

noncomputable def semiperimeter (a b c : ℝ) := (a + b + c) / 2
noncomputable def area (a b c s : ℝ) := real.sqrt (s * (s - a) * (s - b) * (s - c))
noncomputable def gamma_opposite_angle (a b c area : ℝ) := 2 * real.arcsin((2 * area) / (a * b))
noncomputable def spherical_excess (area r : ℝ) := ((180 * area) / (r ^ 2 * real.pi))
noncomputable def spherical_gamma (gamma epsilon : ℝ) := gamma + epsilon

theorem find_spherical_gamma (a b c r : ℝ) (gamma alpha beta : ℝ) (t t1 : ℝ)
  (h1 : a = 21) (h2 : b = 20) (h3 : c = 17) (h4 : r = 10) (h5 : t1 = t) (h6 : α₁ = α) (h7 : β₁ = β) (h8 : γ₁ = spherical_gamma (gamma_opposite_angle a b c t) (spherical_excess t r)) :
  γ₁ = 139 + 37 / 60 + 12 / 3600 := 
  sorry

end find_spherical_gamma_l522_522719


namespace aunt_may_milk_left_l522_522808

def morningMilkProduction (numCows numGoats numSheep : ℕ) (cowMilk goatMilk sheepMilk : ℝ) : ℝ :=
  numCows * cowMilk + numGoats * goatMilk + numSheep * sheepMilk

def eveningMilkProduction (numCows numGoats numSheep : ℕ) (cowMilk goatMilk sheepMilk : ℝ) : ℝ :=
  numCows * cowMilk + numGoats * goatMilk + numSheep * sheepMilk

def spoiledMilk (milkProduction : ℝ) (spoilageRate : ℝ) : ℝ :=
  milkProduction * spoilageRate

def freshMilk (totalMilk spoiledMilk : ℝ) : ℝ :=
  totalMilk - spoiledMilk

def soldMilk (freshMilk : ℝ) (saleRate : ℝ) : ℝ :=
  freshMilk * saleRate

def milkLeft (freshMilk soldMilk : ℝ) : ℝ :=
  freshMilk - soldMilk

noncomputable def totalMilkLeft (previousLeftover : ℝ) (morningLeft eveningLeft : ℝ) : ℝ :=
  previousLeftover + morningLeft + eveningLeft

theorem aunt_may_milk_left :
  let numCows := 5
  let numGoats := 4
  let numSheep := 10
  let cowMilkMorning := 13
  let goatMilkMorning := 0.5
  let sheepMilkMorning := 0.25
  let cowMilkEvening := 14
  let goatMilkEvening := 0.6
  let sheepMilkEvening := 0.2
  let morningSpoilageRate := 0.10
  let eveningSpoilageRate := 0.05
  let iceCreamSaleRate := 0.70
  let cheeseShopSaleRate := 0.80
  let previousLeftover := 15
  let morningMilk := morningMilkProduction numCows numGoats numSheep cowMilkMorning goatMilkMorning sheepMilkMorning
  let eveningMilk := eveningMilkProduction numCows numGoats numSheep cowMilkEvening goatMilkEvening sheepMilkEvening
  let morningSpoiled := spoiledMilk morningMilk morningSpoilageRate
  let eveningSpoiled := spoiledMilk eveningMilk eveningSpoilageRate
  let freshMorningMilk := freshMilk morningMilk morningSpoiled
  let freshEveningMilk := freshMilk eveningMilk eveningSpoiled
  let morningSold := soldMilk freshMorningMilk iceCreamSaleRate
  let eveningSold := soldMilk freshEveningMilk cheeseShopSaleRate
  let morningLeft := milkLeft freshMorningMilk morningSold
  let eveningLeft := milkLeft freshEveningMilk eveningSold
  totalMilkLeft previousLeftover morningLeft eveningLeft = 47.901 :=
by
  sorry

end aunt_may_milk_left_l522_522808


namespace train_length_is_250_l522_522063

noncomputable def train_length (speed_kmh : ℕ) (time_sec : ℕ) (station_length : ℕ) : ℕ :=
  (speed_kmh * 1000 / 3600 * time_sec) - station_length

theorem train_length_is_250 :
  train_length 36 45 200 = 250 :=
by
  sorry

end train_length_is_250_l522_522063


namespace six_people_sitting_in_seven_chairs_with_one_vacant_l522_522984

def num_ways_to_sit (N_chairs : Nat) (vacant_pos : Nat) (N_people : Nat) : Nat :=
  (N_people!).toNat

theorem six_people_sitting_in_seven_chairs_with_one_vacant :
  num_ways_to_sit 7 3 6 = 720 :=
by
  -- This is where the proof would go, sorry is used to skip it.
  sorry

end six_people_sitting_in_seven_chairs_with_one_vacant_l522_522984


namespace nine_point_spheres_coincide_orthocentric_tetrahedron_nine_point_sphere_nine_point_spheres_intersect_in_circle_l522_522418

-- Definitions
def triangle := (ℝ × ℝ × ℝ) -- representation of a triangle
def orthocentric_tetrahedron := Type -- abstract definition of an orthocentric tetrahedron

variable {A B C D : ℝ × ℝ × ℝ} -- vertices of the triangles

-- Nine-point circle details (abstractly defined for context)
def nine_point_circle (Δ : triangle) : Type := Type

-- Part (a)
theorem nine_point_spheres_coincide (h1 : nine_point_circle (A, B, C) = nine_point_circle (D, B, C)) : 
  (B - C) ⊥ (A - D) ↔ 
  sphere_containing_nine_point_circles A B C = sphere_containing_nine_point_circles D B C := 
sorry

-- Part (b)
theorem orthocentric_tetrahedron_nine_point_sphere (O : orthocentric_tetrahedron) : 
  sphere_containing_nine_point_circles_of_faces O :=
sorry

-- Part (c)
theorem nine_point_spheres_intersect_in_circle (h2 : (A - D) ⊥ (B - C)) : 
  let s1 := sphere_containing_nine_point_circles A B C D,
      s2 := sphere_containing_nine_point_circles B C D A
  in intersect_in_plane s1 s2 (perpendicular_bisecting_plane (B - C) (A - D)) :=
sorry

end nine_point_spheres_coincide_orthocentric_tetrahedron_nine_point_sphere_nine_point_spheres_intersect_in_circle_l522_522418


namespace min_a_plus_b_l522_522300

variable (a b : ℝ)
variable (ha_pos : a > 0)
variable (hb_pos : b > 0)
variable (h1 : a^2 - 12 * b ≥ 0)
variable (h2 : 9 * b^2 - 4 * a ≥ 0)

theorem min_a_plus_b (a b : ℝ) (ha_pos : a > 0) (hb_pos : b > 0)
  (h1 : a^2 - 12 * b ≥ 0) (h2 : 9 * b^2 - 4 * a ≥ 0) :
  a + b = 3.3442 := 
sorry

end min_a_plus_b_l522_522300


namespace smallest_p_q_r_s_l522_522957

theorem smallest_p_q_r_s :
  ∃ (p q r s : ℕ), a = 3^p ∧ b = 3^q ∧ c = 3^r ∧ d = 3^s ∧ p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧
    a^2 + b^3 + c^5 = d^7 ∧ p + q + r + s = 106 :=
begin
  sorry,
end

end smallest_p_q_r_s_l522_522957


namespace solve_x_l522_522897

-- Define the custom multiplication operation *
def custom_mul (a b : ℕ) : ℕ := 4 * a * b

-- Given that x * x + 2 * x - 2 * 4 = 0
def equation (x : ℕ) : Prop := custom_mul x x + 2 * x - 2 * 4 = 0

theorem solve_x (x : ℕ) (h : equation x) : x = 2 ∨ x = -4 := 
by 
  -- proof steps go here
  sorry

end solve_x_l522_522897


namespace expression_A_min_value_expression_A_minimum_at_expression_C_min_value_expression_C_minimum_at_l522_522008

theorem expression_A_min_value (x : ℝ) : 2^x + 2^(-x) ≥ 2 :=
by
  sorry

theorem expression_A_minimum_at (x : ℝ) : 2^x + 2^(-x) = 2 ↔ x = 0 :=
by
  sorry

theorem expression_C_min_value (x : ℝ) (hx : x ≠ 0) : (x^2 + 1) / |x| ≥ 2 :=
by
  sorry

theorem expression_C_minimum_at (x : ℝ) : (x^2 + 1) / |x| = 2 ↔ x = 1 ∨ x = -1 :=
by
  sorry

end expression_A_min_value_expression_A_minimum_at_expression_C_min_value_expression_C_minimum_at_l522_522008


namespace a5_value_l522_522155

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Assume the sequence is arithmetic
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
axiom sum_S6 : S 6 = 12
axiom term_a2 : a 2 = 5
axiom sum_formula (n : ℕ) : S n = n * (a 1 + a n) / 2

-- Prove a5 is -1
theorem a5_value (h_arith : arithmetic_sequence a)
  (h_S6 : S 6 = 12) (h_a2 : a 2 = 5) (h_sum_formula : ∀ n, S n = n * (a 1 + a n) / 2) :
  a 5 = -1 :=
sorry

end a5_value_l522_522155


namespace third_function_symmetry_l522_522391

variables {X Y : Type} [linear_order X] [linear_order Y]

noncomputable def φ : X → Y := sorry
noncomputable def φ_inv : Y → X := φ ⁻¹'

-- Definition of the third function whose graph is symmetric to φ_inv about the line x + y = 0
def symmetric_reflection (f : X → Y) : X → Y := λ x, -f (-x)

-- The statement we need to prove
theorem third_function_symmetry : 
  (symmetric_reflection φ_inv = λ x, -φ_inv (-x)) := 
sorry

end third_function_symmetry_l522_522391


namespace compute_P_2_l522_522308

def Q (x : ℝ) : ℝ := x^2 + 2*x + 3

def P (y : ℝ) : ℝ :=
  if h : ∃ x, Q x = y then
    let y := Classical.choose h in x^6 + 6*x^5 + 18*x^4 + 32*x^3 + 35*x^2 + 22*x + 8
  else 0

theorem compute_P_2 : P 2 = 2 :=
by
  -- Provide the relevant steps for Q and P as calculated
  have Q_minus_1 : Q (-1) = 2 := by
    simp [Q]
  have P_val : P (Q (-1)) = 2 := by
    rw [Q_minus_1]
    simp [P, Classical.choose_spec]
  exact P_val

end compute_P_2_l522_522308


namespace sine_amplitude_l522_522810

theorem sine_amplitude (a b c d : ℝ) (h : ∀ x : ℝ, -3 ≤ a * sin (b * x + c) + d ∧ a * sin (b * x + c) + d ≤ 5) :
  a = 4 :=
sorry

end sine_amplitude_l522_522810


namespace range_f_on_interval_one_zero_of_F_on_interval_l522_522913

-- Problem 1: Range of f(x) on [-1, 1]
theorem range_f_on_interval :
  let f (x : ℝ) := -4^x + 2^(x + 1) + 2
  in ∀ x, x ∈ set.Icc (-1 : ℝ) (1 : ℝ) → f x ∈ set.Icc (2 : ℝ) (3 : ℝ) := by
    sorry

-- Problem 2: Range of m for exactly one zero of F(x) = f(x) - m on [-1, 1]
theorem one_zero_of_F_on_interval :
  let f (x : ℝ) := -4^x + 2^(x + 1) + 2
  in ∀ m, (∃! x, x ∈ set.Icc (-1 : ℝ) (1 : ℝ) ∧ f x = m) ↔ (m = 3 ∨ (2 ≤ m ∧ m < 11 / 4)) := by
    sorry

end range_f_on_interval_one_zero_of_F_on_interval_l522_522913


namespace aqua_park_earnings_l522_522467

def admission_cost : ℕ := 12
def tour_cost : ℕ := 6
def group1_size : ℕ := 10
def group2_size : ℕ := 5

theorem aqua_park_earnings :
  (group1_size * admission_cost + group1_size * tour_cost) + (group2_size * admission_cost) = 240 :=
by
  sorry

end aqua_park_earnings_l522_522467


namespace probability_alice_after_bob_given_bob_before_45_is_9_over_16_l522_522800

noncomputable def probability_alice_after_bob_and_bob_before_45 : ℝ :=
  let total_area : ℝ := 0.5 * 60 * 60
  let restricted_area : ℝ := 0.5 * 45 * 45
  restricted_area / total_area

theorem probability_alice_after_bob_given_bob_before_45_is_9_over_16 :
  probability_alice_after_bob_and_bob_before_45 = 9 / 16 :=
sorry

end probability_alice_after_bob_given_bob_before_45_is_9_over_16_l522_522800


namespace factorize_expression_l522_522127

theorem factorize_expression (a b : ℝ) : ab^2 - 2ab + a = a * (b - 1)^2 := by
  sorry

end factorize_expression_l522_522127


namespace train_speed_is_5400432_kmh_l522_522456

noncomputable def train_speed_kmh (time_to_pass_platform : ℝ) (time_to_pass_man : ℝ) (length_platform : ℝ) : ℝ :=
  let speed_m_per_s := length_platform / (time_to_pass_platform - time_to_pass_man)
  speed_m_per_s * 3.6

theorem train_speed_is_5400432_kmh :
  train_speed_kmh 35 20 225.018 = 54.00432 :=
by
  sorry

end train_speed_is_5400432_kmh_l522_522456


namespace max_xy_l522_522239

theorem max_xy : 
  ∃ x y : ℕ, 5 * x + 3 * y = 100 ∧ x > 0 ∧ y > 0 ∧ x * y = 165 :=
by
  sorry

end max_xy_l522_522239


namespace non_zero_int_satisfy_eq_a_pow_a_n_l522_522846

theorem non_zero_int_satisfy_eq_a_pow_a_n (a : ℤ) (h1 : a ≠ 0)
  (h2 : ∀ n : ℕ, a ^ (a ^ n) = a) : a = 1 ∨ a = -1 :=
sorry

end non_zero_int_satisfy_eq_a_pow_a_n_l522_522846


namespace smallest_k_l522_522716

noncomputable def a : ℕ → ℝ
| 0     := 1
| 1     := real.cbrt 3
| (n+2) := a (n+1) * (a n)^2

def product (k : ℕ) : ℝ := ∏ i in finset.range k, a (i + 1)

theorem smallest_k :
  (∃ k : ℕ,  (product k).denom = 1) ∧
  (∀ k' : ℕ,  (k' < 8) → (product k').denom ≠ 1) :=
begin
  split,
  { use 8, sorry },
  { intros k' hk',
    sorry
  }

end smallest_k_l522_522716


namespace bug_moves_l522_522806

/-- In an 8 × 8 square grid, there are two bugs at points A and B.
    Each bug moves 1 square per step in one of the four directions: up, down, left, or right.
    Bug b moves two steps and bug a moves three steps.
    Prove that the number of ways for bug b to have a distance from point A that is less than
    or equal to the distance of bug a from point A is 748. -/
theorem bug_moves (A B : ℕ×ℕ) (grid_size : ℕ := 8) :
  (∃ a_moves b_moves : list (ℕ × ℕ), 
    a_moves.length = 3 ∧ b_moves.length = 2 ∧ 
    calc_distance_from A a_moves.last ≤ calc_distance_from A b_moves.last) →
  (number_of_valid_ways A B grid_size = 748) := 
sorry

/-- Function to calculate the Manhattan distance from point A to the final point after a series of moves -/
def calc_distance_from (start : ℕ × ℕ) (end : ℕ × ℕ) : ℕ :=
  abs (start.fst - end.fst) + abs (start.snd - end.snd)

/-- Function to count the number of valid ways for bug movements -/
def number_of_valid_ways (A B : ℕ × ℕ) (grid_size : ℕ) : ℕ :=
  -- logic to calculate the number of valid ways here
sorry

end bug_moves_l522_522806


namespace find_number_of_satisfying_integers_l522_522232

theorem find_number_of_satisfying_integers :
  let S := {n : ℤ | 100 ≤ n ∧ n < 1000 ∧ n % 7 = 3 ∧ n % 8 = 4 ∧ n % 10 = 6} 
  in S.card = 4 := 
sorry

end find_number_of_satisfying_integers_l522_522232


namespace total_gas_consumed_l522_522837

def highway_consumption_rate : ℕ := 3
def city_consumption_rate : ℕ := 5

-- Distances driven each day
def day_1_highway_miles : ℕ := 200
def day_1_city_miles : ℕ := 300

def day_2_highway_miles : ℕ := 300
def day_2_city_miles : ℕ := 500

def day_3_highway_miles : ℕ := 150
def day_3_city_miles : ℕ := 350

-- Function to calculate the total consumption for a given day
def daily_consumption (highway_miles city_miles : ℕ) : ℕ :=
  (highway_miles * highway_consumption_rate) + (city_miles * city_consumption_rate)

-- Total consumption over three days
def total_consumption : ℕ :=
  (daily_consumption day_1_highway_miles day_1_city_miles) +
  (daily_consumption day_2_highway_miles day_2_city_miles) +
  (daily_consumption day_3_highway_miles day_3_city_miles)

-- Theorem stating the total consumption over the three days
theorem total_gas_consumed : total_consumption = 7700 := by
  sorry

end total_gas_consumed_l522_522837


namespace turnover_five_days_eq_504_monthly_growth_rate_eq_20_percent_l522_522840

-- Definitions based on conditions
def turnover_first_four_days : ℝ := 450
def turnover_fifth_day : ℝ := 0.12 * turnover_first_four_days
def total_turnover_five_days : ℝ := turnover_first_four_days + turnover_fifth_day

-- Proof statement for part 1
theorem turnover_five_days_eq_504 :
  total_turnover_five_days = 504 := 
sorry

-- Definitions and conditions for part 2
def turnover_february : ℝ := 350
def turnover_april : ℝ := total_turnover_five_days
def growth_rate (x : ℝ) : Prop := (1 + x)^2 * turnover_february = turnover_april

-- Proof statement for part 2
theorem monthly_growth_rate_eq_20_percent :
  ∃ x : ℝ, growth_rate x ∧ x = 0.2 := 
sorry

end turnover_five_days_eq_504_monthly_growth_rate_eq_20_percent_l522_522840


namespace smallest_solution_of_quadratic_eq_l522_522006

theorem smallest_solution_of_quadratic_eq : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁ < x₂) ∧ (x₁^2 + 10 * x₁ - 40 = 0) ∧ (x₂^2 + 10 * x₂ - 40 = 0) ∧ x₁ = -8 :=
by {
  sorry
}

end smallest_solution_of_quadratic_eq_l522_522006


namespace Sn_geometric_sequence_l522_522880

-- Definition of the sequence and the conditions
def a_n (n : ℕ) : ℕ :=
  if n = 0 then 4 else 3 * (∑ i in range n, a_n i)

def S_n (n : ℕ) : ℕ := ∑ i in range n, a_n i

theorem Sn_geometric_sequence (n : ℕ) : S_n n = 4^n := by
  sorry

end Sn_geometric_sequence_l522_522880


namespace number_is_divisible_by_1980_l522_522729

theorem number_is_divisible_by_1980 :
  let num := (List.range' 19 62).map (λ n, n.toString).reduce (λ acc s, acc ++ s) in
  nat_digits (num.toNat) % 1980 = 0 :=
by
  sorry

end number_is_divisible_by_1980_l522_522729


namespace value_of_m_l522_522190

theorem value_of_m :
  ∀ m : ℝ, (x : ℝ) → (x^2 - 5 * x + m = (x - 3) * (x - 2)) → m = 6 :=
by
  sorry

end value_of_m_l522_522190


namespace trajectory_of_point_Q_l522_522186

theorem trajectory_of_point_Q :
  (∀ P : ℝ × ℝ, (P.1 * 2 - P.2 + 3 = 0) → (¬ is_fixed P) → 
  ∃ Q : ℝ × ℝ, Q.1 = -2 - P.1 ∧ Q.2 = 4 - P.2 ∧ 2 * Q.1 - Q.2 + 5 = 0) :=
by {
  sorry
}

end trajectory_of_point_Q_l522_522186


namespace num_standard_pairs_parity_l522_522817

-- Definitions based on conditions
def m : ℕ := _ -- m ≥ 3
def n : ℕ := _ -- n ≥ 3
axiom m_ge_3 : m ≥ 3
axiom n_ge_3 : n ≥ 3

def is_red_or_blue : fin m → fin n → Prop := _ -- Each cell is either red or blue
def is_standard_pair (i j : fin m) (k l : fin n) : Prop :=
  adjacent i j k l ∧ (is_red_or_blue i j ≠ is_red_or_blue k l) -- Condition for a standard pair

def number_of_standard_pairs : ℕ := 
  nat.count (λ (i j k l : fin m × fin n), is_standard_pair i j k l)

def number_of_blue_border_cells : ℕ := 
  nat.count (λ (i : fin m), is_blue (border_cell i))

-- Theorem to be proved
theorem num_standard_pairs_parity :
  (number_of_standard_pairs % 2 = 1 ↔ number_of_blue_border_cells % 2 = 1) ∧
  (number_of_standard_pairs % 2 = 0 ↔ number_of_blue_border_cells % 2 = 0) := 
sorry

end num_standard_pairs_parity_l522_522817


namespace abe_bob_same_color_prob_l522_522459

-- Definition of the given problem's conditions
def abe_jelly_beans := [(2, "green"), (1, "red")]
def bob_jelly_beans := [(2, "green"), (1, "yellow"), (2, "red"), (1, "blue")]

def prob_showing_green :=
  (2 / 3) * (2 / 6)

def prob_showing_red :=
  (1 / 3) * (2 / 6)

def prob_matching_colors :=
  prob_showing_green + prob_showing_red

theorem abe_bob_same_color_prob :
  prob_matching_colors = 1 / 3 :=
begin
  sorry
end

end abe_bob_same_color_prob_l522_522459


namespace decimal_to_binary_2008_l522_522492

theorem decimal_to_binary_2008 : (nat_to_bin 2008) = "11111011000" :=
by
  sorry

end decimal_to_binary_2008_l522_522492


namespace possible_values_of_expression_l522_522176

noncomputable def sign (x : ℝ) : ℝ :=
if x > 0 then 1 else -1

theorem possible_values_of_expression
  (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := (sign a) + (sign b) + (sign c) + (sign d) + (sign (a * b * c * d))
  in expr ∈ {5, 1, -1, -5} :=
by
  let expr := (sign a) + (sign b) + (sign c) + (sign d) + (sign (a * b * c * d))
  sorry

end possible_values_of_expression_l522_522176


namespace probability_rain_both_and_neither_l522_522714

variables (P_Monday P_Tuesday : ℝ)

def independent_events (P_A P_B : ℝ) := P_A * P_B

theorem probability_rain_both_and_neither
  (h_Monday : P_Monday = 0.4)
  (h_Tuesday : P_Tuesday = 0.3)
  (h_independent : independent_events P_Monday P_Tuesday = P_Monday * P_Tuesday):
  (P_both_days : independent_events P_Monday P_Tuesday = 0.12)
  ∧ (P_neither_days : independent_events (1 - P_Monday) (1 - P_Tuesday) = 0.42) :=
by {
  have P_both_days : independent_events P_Monday P_Tuesday = (0.4 * 0.3) := by rw [h_Monday, h_Tuesday];
  have h_both : P_both_days = 0.12 := by norm_num,
  have P_neither_days : independent_events (1 - P_Monday) (1 - P_Tuesday) = (0.6 * 0.7) := by
    { rw [h_Monday, h_Tuesday], norm_num },
  have h_neither : P_neither_days = 0.42 := by norm_num,
  exact ⟨h_both, h_neither⟩,
}

end probability_rain_both_and_neither_l522_522714


namespace solve_for_a_l522_522566

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x^2 - 2 * x - 1

theorem solve_for_a : ∃ a : ℝ, (∀ x : ℝ, ∀ f' = λ x, Real.exp x - 2*a*x - 2,
  let s := (Real.exp 1 - a*1^2 - 2*1 - 1) in -- f(1)
  let tangent_line := Real.exp 1 - 2*a - 2 in -- slope at (1, f(1))
  ∃ b : ℝ, b = -2 ∧ tangent_line = ((Real.exp 1 - a - 3 + 2) / (1 - 0)) ∧ a = -1) :=
sorry

end solve_for_a_l522_522566


namespace Shelby_fog_time_l522_522676

variable (x y : ℕ)

-- Conditions
def speed_sun := 7/12
def speed_rain := 5/12
def speed_fog := 1/4
def total_time := 60
def total_distance := 20

theorem Shelby_fog_time :
  ((speed_sun * (total_time - x - y)) + (speed_rain * x) + (speed_fog * y) = total_distance) → y = 45 :=
by
  sorry

end Shelby_fog_time_l522_522676


namespace cosine_relationship_l522_522889

open Real

noncomputable def functional_relationship (x y : ℝ) : Prop :=
  y = -(4 / 5) * sqrt (1 - x ^ 2) + (3 / 5) * x

theorem cosine_relationship (α β : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : 0 < β) (h4 : β < π / 2)
  (h5 : cos (α + β) = - 4 / 5) (h6 : sin β = x) (h7 : cos α = y) (h8 : 4 / 5 < x) (h9 : x < 1) :
  functional_relationship x y :=
sorry

end cosine_relationship_l522_522889


namespace min_value_of_a_l522_522894

-- Define the necessary conditions
variables {m n : ℝ} (h1 : m ≠ 0) (h2 : n = 1) (h3 : m ≠ n)
          (h4 : real.angle m (m - n) = 60) (h5 : 0 < |m| ∧ |m| ≤ a)

-- The expected solution
theorem min_value_of_a {a : ℝ} : a = (2 * real.sqrt 3 / 3) :=
sorry

end min_value_of_a_l522_522894


namespace octahedron_cut_area_l522_522449

theorem octahedron_cut_area (a b c : ℕ) (side_length : ℝ) 
  (prime_a_c: Nat.gcd a c = 1) (not_divisible_prime_square : ∀ p : ℕ, Prime p → ¬ p ^ 2 ∣ b)
  (hex_area : ∀ s : ℝ, side_length = 1 → 
    s = (1 / 2) * (Real.sqrt 3 / 2) → 
    let A := (3 * Real.sqrt 3 / 2) * s^2
    in A = (3 * Real.sqrt 3 / 8)) :
  a = 3 ∧ b = 3 ∧ c = 8 → a + b + c = 14 :=
by 
  sorry

end octahedron_cut_area_l522_522449


namespace prime_p_q_r_condition_l522_522513

theorem prime_p_q_r_condition (p q r : ℕ) (hp : Nat.Prime p) (hq_pos : 0 < q) (hr_pos : 0 < r)
    (hp_not_dvd_q : ¬ (p ∣ q)) (h3_not_dvd_q : ¬ (3 ∣ q)) (eqn : p^3 = r^3 - q^2) : 
    p = 7 := sorry

end prime_p_q_r_condition_l522_522513


namespace max_value_expression_l522_522209

variable (n : ℕ)
variable (x : Fin n → ℝ)
variable (h : ∀ i, 0 ≤ x i ∧ x i ≤ Real.pi / 2)

theorem max_value_expression :
  (∑ i in Finset.univ, Real.sqrt (Real.sin (x i))) * 
  (∑ i in Finset.univ, Real.sqrt (Real.cos (x i))) ≤ n^2 / Real.sqrt 2 := sorry

end max_value_expression_l522_522209


namespace intersection_point_l522_522443

variable (t u : ℝ)

def line1 (t : ℝ) := (2 + 3 * t, 2 - 4 * t)
def line2 (u : ℝ) := (4 + 5 * u, -6 + 3 * u)

theorem intersection_point :
  ∃ t u, line1 t = (160 / 29 : ℝ, -160 / 29 : ℝ) ∧ line1 t = line2 u :=
begin
  use [(46 / 29 : ℝ), (48 / 87 : ℝ)],
  simp [line1, line2],
  split,
  {
    simp,
    ring,
  },
  {
    simp,
    split,
    { norm_num, ring},
    { norm_num, ring }
  }
end

end intersection_point_l522_522443


namespace total_time_is_correct_l522_522322

-- Definitions based on conditions
def timeouts_for_running : ℕ := 5
def timeouts_for_throwing_food : ℕ := 5 * timeouts_for_running - 1
def timeouts_for_swearing : ℕ := timeouts_for_throwing_food / 3

-- Definition for total time-outs
def total_timeouts : ℕ := timeouts_for_running + timeouts_for_throwing_food + timeouts_for_swearing
-- Each time-out is 5 minutes
def timeout_duration : ℕ := 5

-- Total time in minutes
def total_time_in_minutes : ℕ := total_timeouts * timeout_duration

-- The proof statement
theorem total_time_is_correct : total_time_in_minutes = 185 := by
  sorry

end total_time_is_correct_l522_522322


namespace reflection_correct_l522_522519

-- Define vector addition
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
-- Define scalar multiplication
def scalar_mul (a : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (a * v.1, a * v.2)
-- Compute projection
def proj (v u : ℝ × ℝ) : ℝ × ℝ := scalar_mul (dot_product v u / dot_product u u) u
-- Compute reflection
def reflection (v u : ℝ × ℝ) : ℝ × ℝ := vec_add (scalar_mul 2 (proj v u)) (scalar_mul (-1) v)

-- Given specific vectors
def v : ℝ × ℝ := (2, -1)
def t : ℝ × ℝ := (1, 2)
def u : ℝ × ℝ := (-2, 1)
def translated_v : ℝ × ℝ := vec_add v t

-- The lemma to prove
theorem reflection_correct : reflection translated_v u = (1, -3) :=
by
  sorry

end reflection_correct_l522_522519


namespace smallest_product_of_set_l522_522521

noncomputable def smallest_product (s : Set ℤ) : ℤ :=
  let products := {x * y | x in s, y in s}
  Set.min' products sorry

theorem smallest_product_of_set :
  smallest_product ({-10, -8, -3, 0, 4, 6} : Set ℤ) = -60 :=
by {
  sorry
}

end smallest_product_of_set_l522_522521


namespace sum_of_exterior_angles_ge_360_exists_triangle_divided_into_3_similar_exists_triangle_divided_into_5_similar_l522_522344

-- Statement for Problem 1
theorem sum_of_exterior_angles_ge_360 (n : ℕ) (interior_angles : ℕ → ℝ) (h_inter : ∀ i, i < n → interior_angles i < 180) :
  ∑ i in (finset.range n), (180 - interior_angles i) ≥ 360 :=
sorry

-- Statement for Problem 2
theorem exists_triangle_divided_into_3_similar :
  ∃ (Δ : Type) [isTriangle Δ], (∃ (Δ1 Δ2 Δ3 : Δ), isSimilar Δ1 Δ ∧ isSimilar Δ2 Δ ∧ isSimilar Δ3 Δ ∧ decompose Δ Δ1 Δ2 Δ3) :=
sorry

-- Statement for Problem 3
theorem exists_triangle_divided_into_5_similar :
  ∃ (Δ : Type) [isTriangle Δ], (∃ (Δ1 Δ2 Δ3 Δ4 Δ5 : Δ), isSimilar Δ1 Δ ∧ isSimilar Δ2 Δ ∧ isSimilar Δ3 Δ ∧ isSimilar Δ4 Δ ∧ isSimilar Δ5 Δ ∧ decompose Δ Δ1 Δ2 Δ3 Δ4 Δ5) :=
sorry

end sum_of_exterior_angles_ge_360_exists_triangle_divided_into_3_similar_exists_triangle_divided_into_5_similar_l522_522344


namespace mo_rainy_days_last_week_l522_522420

theorem mo_rainy_days_last_week (R NR n : ℕ) (h1 : n * R + 4 * NR = 26) (h2 : 4 * NR - n * R = 14) (h3 : R + NR = 7) : R = 2 :=
sorry

end mo_rainy_days_last_week_l522_522420


namespace haley_cans_bagged_l522_522577

noncomputable def numberOfBags (totalCans: ℕ) (percentBagged: ℝ) (cansPerBag: ℕ) : ℕ :=
  Nat.floor ((percentBagged / 100) * totalCans) / cansPerBag

theorem haley_cans_bagged:
  numberOfBags 560 80 35 = 12 :=
by
  sorry

end haley_cans_bagged_l522_522577


namespace no_solution_natural_gt_one_l522_522836

theorem no_solution_natural_gt_one (x : ℕ) (h : 1 < x) : 
  (∑ k in (range (x^2 + 1)), (1 / (x + k : ℝ)) ≠ 1) :=
sorry

end no_solution_natural_gt_one_l522_522836


namespace selling_price_ratio_l522_522796

theorem selling_price_ratio (C_1 C_2 : ℝ) 
  (gain_first_car : C_1 * 1.10)
  (loss_second_car : C_2 * 0.90)
  (overall_profit : 0.01 * (C_1 + C_2) = (1.10 * C_1) + (0.90 * C_2) - (C_1 + C_2)) :
  C_2 = (9/11) * C_1 :=
by
  sorry

end selling_price_ratio_l522_522796


namespace parabola_y_intercepts_l522_522584

theorem parabola_y_intercepts : 
  let a := 3
  let b := -5
  let c := 1
  ∆ := b*b - 4*a*c 
  ∆ > 0 → ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ (3 * y1^2 - 5 * y1 + 1 = 0) ∧ (3 * y2^2 - 5 * y2 + 1 = 0) := 
by
  sorry

end parabola_y_intercepts_l522_522584


namespace distance_between_points_l522_522096

theorem distance_between_points 
  (a b c d : ℝ) 
  (h1 : a = 5) 
  (h2 : c = 10) 
  (h3 : b = 2 * a + 3) 
  (h4 : d = 2 * c + 3) 
  : (Real.sqrt ((c - a)^2 + (d - b)^2)) = 5 * Real.sqrt 5 :=
by
  sorry

end distance_between_points_l522_522096


namespace maximum_value_f_range_g_l522_522226

-- Define the initial vectors and function f(x)
def m (x : ℝ) : ℝ×ℝ := (Real.sin x, 1)
def n (A x : ℝ) : ℝ×ℝ := (Real.sqrt 3 * A * Real.cos x, A / 2 * Real.cos (2 * x))

def f (A x : ℝ) : ℝ := (m x).1 * (n A x).1 + (m x).2 * (n A x).2

-- Condition 1: Maximum value of f(x) is 6 when A = 6
theorem maximum_value_f : ∀ x : ℝ, f 6 x ≤ 6 :=
sorry

-- Define g(x) after the translation and compression of f(x)
def g (x : ℝ) : ℝ := 6 * Real.sin (4 * x + Real.pi / 3)

-- Range of g(x) on the interval [0, 5π / 24] is [-3, 6]
theorem range_g : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 * Real.pi / 24 → -3 ≤ g x ∧ g x ≤ 6 :=
sorry

end maximum_value_f_range_g_l522_522226


namespace combined_avg_age_l522_522364

-- Define the conditions
def avg_X : ℕ := 35
def n_X : ℕ := 5

def avg_Y : ℕ := 30
def n_Y : ℕ := 3

def avg_Z : ℕ := 45
def n_Z : ℕ := 2

-- Define the theorem to be proved
theorem combined_avg_age : 
  (avg_X * n_X + avg_Y * n_Y + avg_Z * n_Z) / (n_X + n_Y + n_Z) = 35.5 := 
by
  sorry

end combined_avg_age_l522_522364


namespace fA_is_even_fC_is_even_l522_522760

namespace EvenFunctions

-- Conditions: Define the functions
def fA (x : ℝ) : ℝ := (Real.sin x) ^ 2
def fB (x : ℝ) : ℝ := - (Real.sin x)
def fC (x : ℝ) : ℝ := Real.sin (abs x)
def fD (x : ℝ) : ℝ := (Real.sin x) + 1

-- Statement: Prove that fA and fC are even functions
theorem fA_is_even : ∀ x : ℝ, fA (-x) = fA x := 
by sorry

theorem fC_is_even : ∀ x : ℝ, fC (-x) = fC x := 
by sorry

end EvenFunctions

end fA_is_even_fC_is_even_l522_522760


namespace count_valid_pairs_l522_522517

section QuadraticDecomposition

-- Define the conditions
def is_valid_pair (a b : ℤ) : Prop :=
  10 ≤ a ∧ a ≤ 50 ∧ 0 ≤ b ∧ b ≤ 200 ∧
  ∃ r s : ℤ, r + s = a ∧ r * s = b

-- Define the theorem
theorem count_valid_pairs : 
  ∃ n : ℕ, n = ∑ a in Finset.Icc 10 50, 
    Finset.card {b | is_valid_pair a b} :=
sorry

end QuadraticDecomposition

end count_valid_pairs_l522_522517


namespace smallest_positive_period_of_sine_function_l522_522720

theorem smallest_positive_period_of_sine_function :
  ∃ T > 0, (∀ x, f x + T = f x) ∧ (∀ T' > 0, (∀ x, f x + T' = f x) → T' ≥ T) :=
by
  let f : ℝ → ℝ := λ x, Real.sin ( (π / 3) * x + 1 / 3 )
  use 6
  sorry

end smallest_positive_period_of_sine_function_l522_522720


namespace log_div_pow_simplification_l522_522078

theorem log_div_pow_simplification :
  (\log 10 (1 / 4) - \log 10 25) / 100^(-1/2) = -20 := 
  by sorry

end log_div_pow_simplification_l522_522078


namespace sin_ratio_property_l522_522262

noncomputable def vertices_and_ellipse :=
  ∃ (A B C : ℝ × ℝ), A = (0, 4) ∧ C = (0, -4) ∧ 
  (B.1^2 / 9 + B.2^2 / 25 = 1)

theorem sin_ratio_property : 
  vertices_and_ellipse → (∀ (A C : ℝ), A = 4 ∧ C = -4 → 
  (∃ (r : ℝ), r = 4 / 5)) :=
by
  intros h A hA B hB C hC
  have hSin := sorry,  -- This would be replaced by actual proof steps.
  exact ⟨(4 / 5), hSin⟩

end sin_ratio_property_l522_522262


namespace probability_of_rolling_2_or_4_l522_522409

theorem probability_of_rolling_2_or_4 (fair : ℕ) (sides : fin 6) : 
  (2/6 : ℚ) = (1/3 : ℚ) := 
by 
  sorry

end probability_of_rolling_2_or_4_l522_522409


namespace find_value_of_a_plus_b_l522_522871

-- Define the variables and conditions
variables (a b c d : ℝ)
hypothesis (h1 : ac + bd + bc + ad = 48)
hypothesis (h2 : c + d = 8)

-- Statement to prove
theorem find_value_of_a_plus_b : a + b = 6 := by
  sorry

end find_value_of_a_plus_b_l522_522871


namespace expression_value_l522_522645

-- Define the variables and the main statement
variable (w x y z : ℕ)

theorem expression_value :
  2^w * 3^x * 5^y * 11^z = 825 → w + 2 * x + 3 * y + 4 * z = 12 :=
by
  sorry -- Proof omitted

end expression_value_l522_522645


namespace compute_expression_l522_522086

theorem compute_expression : 1004^2 - 996^2 - 1000^2 + 1000^2 = 16000 := 
by sorry

end compute_expression_l522_522086


namespace number_of_primes_in_range_l522_522522

theorem number_of_primes_in_range {n : ℕ} (h : 1 < n) : 
  (∃ p, p.prime ∧ n! - n < p ∧ p < n! - 1) → (∃! p, p.prime ∧ n! - n < p ∧ p < n! - 1) ∨ 
  ¬ ∃ p, p.prime ∧ n! - n < p ∧ p < n! - 1 :=
by sorry

end number_of_primes_in_range_l522_522522


namespace division_of_fractions_l522_522935

theorem division_of_fractions : (2 / 3) / (1 / 4) = (8 / 3) := by
  sorry

end division_of_fractions_l522_522935


namespace distance_between_points_l522_522004

-- Define the two points
def point1 : ℝ × ℝ := (-3, 4)
def point2 : ℝ × ℝ := (5, -6)

-- Define the distance formula
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Prove that the distance between the two specific points is 2 * sqrt 41
theorem distance_between_points :
  distance point1 point2 = 2 * real.sqrt 41 := by
sorry

end distance_between_points_l522_522004


namespace fairfield_middle_school_geography_players_l522_522076

/-- At Fairfield Middle School, there are 24 players on the football team.
All players are enrolled in at least one of the subjects: history or geography.
There are 10 players taking history and 6 players taking both subjects.
We need to prove that the number of players taking geography is 20. -/
theorem fairfield_middle_school_geography_players
  (total_players : ℕ)
  (history_players : ℕ)
  (both_subjects_players : ℕ)
  (h1 : total_players = 24)
  (h2 : history_players = 10)
  (h3 : both_subjects_players = 6) :
  total_players - (history_players - both_subjects_players) = 20 :=
by {
  sorry
}

end fairfield_middle_school_geography_players_l522_522076


namespace probability_of_prime_l522_522376

open Finset

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def card_set : Finset ℕ := {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}

def prime_numbers := card_set.filter is_prime

theorem probability_of_prime :
  (prime_numbers.card : ℚ) / (card_set.card : ℚ) = 5 / 11 :=
  sorry

end probability_of_prime_l522_522376


namespace train_pass_platform_time_l522_522798

theorem train_pass_platform_time (L_t L_p t_p : ℕ) (h1 : L_t = 240) (h2 : t_p = 24) (h3 : L_p = 650) : 
  (L_t + L_p) / (L_t / t_p) = 89 :=
by
  rw [h1, h2, h3]
  calc (240 + 650) / (240 / 24) = 890 / 10 : by norm_num
                        ... = 89 : by norm_num

-- with sorry as simplified proof might not use calc and be more abstract

end train_pass_platform_time_l522_522798


namespace domain_of_f_minus_g_f_minus_g_is_odd_range_of_x_for_positive_f_minus_g_l522_522922

section
variables {a : ℝ} (f g : ℝ → ℝ)
def f (x : ℝ) : ℝ := log (1 + x) / log a
def g (x : ℝ) : ℝ := log (1 - x) / log a

/-- The domain of the function f(x) - g(x) is (-1, 1). -/
theorem domain_of_f_minus_g (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) : 
  ∀ x, (0 < 1 + x) ∧ (0 < 1 - x) ↔ -1 < x ∧ x < 1 :=
sorry
  
/-- The function f(x) - g(x) is an odd function. -/
theorem f_minus_g_is_odd (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) : 
  ∀ x, f (-x) - g (-x) = -(f x - g x) :=
sorry

/-- The range of x for which f(x) - g(x) > 0. -/
theorem range_of_x_for_positive_f_minus_g (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) :
  (a > 1 → (∀ x, 0 < x ∧ x < 1 ↔ 0 < f x - g x)) ∧ 
  (0 < a ∧ a < 1 → (∀ x, -1 < x ∧ x < 0 ↔ 0 < f x - g x)) :=
sorry
end

end domain_of_f_minus_g_f_minus_g_is_odd_range_of_x_for_positive_f_minus_g_l522_522922


namespace determine_angles_of_triangle_KLM_l522_522666

noncomputable def triangle_KLM (P Q : Point) (K L M S : Point) : Prop :=
  MidpointArcCircumcircle K L P ∧
  MidpointArcCircumcircle L M Q ∧
  AngleBisectorLS L S K M ∧
  Angle K L M = 2 * Angle K M L ∧
  Angle P S Q = 90

theorem determine_angles_of_triangle_KLM (P Q K L M S : Point)
  (h : triangle_KLM P Q K L M S) :
  Angle K L M = 45 ∧ Angle L K M = 90 ∧ Angle M L K = 45 := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  sorry

end determine_angles_of_triangle_KLM_l522_522666


namespace megan_water_consumption_l522_522659

theorem megan_water_consumption (t g : ℕ) (h1 : t = 3 * 60 + 40) (h2 : g = t / 20) : g = 11 :=
by
  -- Convert time into minutes
  have h_minutes : t = 220 := by
    rw [h1]
    norm_num
  -- Calculate glasses of water
  have h_glasses : g = 220 / 20 := by
    rw [h2, h_minutes]
  norm_num
  -- Final answer
  rw [h_glasses]
  norm_num

end megan_water_consumption_l522_522659


namespace find_interest_rate_l522_522318

def interest_rate_borrowed (p_borrowed: ℝ) (p_lent: ℝ) (time: ℝ) (rate_lent: ℝ) (gain: ℝ) (r: ℝ) : Prop :=
  let interest_from_ramu := p_lent * rate_lent * time / 100
  let interest_to_anwar := p_borrowed * r * time / 100
  gain = interest_from_ramu - interest_to_anwar

theorem find_interest_rate :
  interest_rate_borrowed 3900 5655 3 9 824.85 5.95 := sorry

end find_interest_rate_l522_522318


namespace real_part_of_z_l522_522312

noncomputable def z : ℂ := (3 : ℂ) * Complex.I + 1

theorem real_part_of_z :
  (∀ z : ℂ, Complex.add (Complex.mul Complex.I z) Complex.I = -3 + 2 * Complex.I → z.re) = 1 := 
by
  sorry

end real_part_of_z_l522_522312


namespace sequence_is_geometric_and_general_formula_l522_522881

theorem sequence_is_geometric_and_general_formula (a : ℕ → ℝ) (h0 : a 1 = 2 / 3)
  (h1 : ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) / (a (n + 1) + 1)) :
  ∃ r : ℝ, (0 < r ∧ r < 1 ∧ (∀ n : ℕ, a (n + 1) = (2:ℝ)^n / (1 + (2:ℝ)^n)) ∧
  ∀ n : ℕ, (1 / a (n + 1) - 1) = (1 / 2) * (1 / a n - 1)) := sorry

end sequence_is_geometric_and_general_formula_l522_522881


namespace number_from_division_l522_522035

theorem number_from_division (number : ℝ) (h : number / 2000 = 0.012625) : number = 25.25 :=
by
  sorry

end number_from_division_l522_522035


namespace equal_angle_ratios_l522_522278

theorem equal_angle_ratios (A B C P : Type*) 
  [Preorder A] [Preorder B] [Preorder C] [Preorder P]
  (angle : A → B → C → P → Type)
  (x : ℝ)
  (h1 : angle A B P P = angle A C P P)
  (h2 : angle C A P P = angle C B P P)
  (h3 : angle B C P P = angle B A P P) :
  x = 1 :=
sorry

end equal_angle_ratios_l522_522278


namespace possible_values_expression_l522_522169

theorem possible_values_expression (a b c d : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) :
  ∃ (x : ℝ), x ∈ {5, 1, -3} ∧ x = (a / |a| + b / |b| + c / |c| + d / |d| + (abcd / |abcd|)) :=
by
  sorry

end possible_values_expression_l522_522169


namespace triangle_right_angle_l522_522605

open Real

variables {A B C a b c : ℝ}

theorem triangle_right_angle
  (h1: a ≠ 0)
  (h2: b ≠ 0)
  (h3: c ≠ 0)
  (h4 : b * cos B + c * cos C = a * cos A)
  (h5 : A + B + C = π) :
  is_right_triangle A B C :=
sorry

end triangle_right_angle_l522_522605


namespace triangle_base_value_l522_522726

variable (L R B : ℕ)

theorem triangle_base_value
    (h1 : L = 12)
    (h2 : R = L + 2)
    (h3 : L + R + B = 50) :
    B = 24 := 
sorry

end triangle_base_value_l522_522726


namespace symmetric_point_l522_522541

-- Define the Point type and necessary structures
structure Point where
  x : ℝ
  y : ℝ

-- Define the given point P
def P : Point := ⟨1, 3⟩

-- Define the line as a function
def line (p : Point) : Prop := p.x + 2 * p.y - 2 = 0

-- Define the midpoint condition
def is_midpoint (p q m : Point) : Prop :=
  m.x = (p.x + q.x) / 2 ∧ m.y = (p.y + q.y) / 2

-- Define the perpendicular slope condition
def perpendicular_slope (p q : Point) : Prop :=
  (q.y - p.y) / (q.x - p.x) * (-1 / 2) = -1

-- Prove that the coordinates of point Q
theorem symmetric_point :
  ∃ q : Point, line ⟨(P.x + q.x) / 2, (P.y + q.y) / 2⟩ ∧ perpendicular_slope P q ∧ q = ⟨-1, -1⟩ :=
by
  exists ⟨-1, -1⟩
  split
  · -- Midpoint on the line condition
    unfold line is_midpoint
    simp
    linarith
  · split
    · -- Perpendicular slope condition
      unfold perpendicular_slope
      simp
      field_simp
      ring
    · -- q = (-1, -1)
      refl

end symmetric_point_l522_522541


namespace distinct_arith_prog_not_perfect_square_l522_522636

theorem distinct_arith_prog_not_perfect_square (a r : ℕ) (h_positive: 0 < a ∧ 0 < r) :
  let b := a + r,
      c := a + 2 * r,
      d := a + 3 * r in
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ¬ is_square (a * b * c * d) :=
by {
  simp only,
  sorry
}

end distinct_arith_prog_not_perfect_square_l522_522636


namespace tangent_triangle_angles_l522_522362

-- Definitions:
variables {α β γ : ℝ} (h_angles : α + β + γ = 180)

-- theorem to state and prove the properties for acute and obtuse cases
theorem tangent_triangle_angles 
    (h_acute : (α < 90) ∧ (β < 90) ∧ (γ < 90)) 
    (h_obtuse : (90 < α) ∨ (90 < β) ∨ (90 < γ)) :
    (α < 90 ∧ β < 90 ∧ γ < 90 → (set.eq (180 - 2 * α) ∧ set.eq (180 - 2 * β) ∧ set.eq (180 - 2 * γ)))
    ∧ ((90 < α) → (set.eq (2 * α - 180) ∧ set.eq (2 * γ) ∧ set.eq (2 * β))) :=
by 
  sorry

end tangent_triangle_angles_l522_522362


namespace niko_total_profit_l522_522332

def pairs_of_socks : Nat := 9
def cost_per_pair : ℝ := 2
def profit_percentage_first_four : ℝ := 0.25
def profit_per_pair_remaining_five : ℝ := 0.2

theorem niko_total_profit :
  let total_profit_first_four := 4 * (cost_per_pair * profit_percentage_first_four)
  let total_profit_remaining_five := 5 * profit_per_pair_remaining_five
  let total_profit := total_profit_first_four + total_profit_remaining_five
  total_profit = 3 := by
  sorry

end niko_total_profit_l522_522332


namespace bobs_password_probability_l522_522475

-- Define the set of two-digit even numbers
def even_two_digit_numbers : Finset ℕ := { n | 10 ≤ n ∧ n ≤ 98 ∧ n % 2 = 0 }.to_finset 

-- Define the set of vowels
def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}.to_finset 

-- Define the set of two-digit prime numbers
def prime_two_digit_numbers : Finset ℕ := { 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
                                           53, 59, 61, 67, 71, 73, 79, 83, 89, 97}.to_finset 

-- Define the calculation of probabilities
theorem bobs_password_probability :
  let even_prob := (even_two_digit_numbers.card : ℚ) / 90 in
  let vowel_prob := (vowels.card : ℚ) / 26 in
  let prime_prob := (prime_two_digit_numbers.card : ℚ) / 90 in
  even_prob * vowel_prob * prime_prob = 7 / 312 := by
       sorry

end bobs_password_probability_l522_522475


namespace arrangements_count_correct_l522_522867

noncomputable def count_arrangements : ℕ :=
  -- The total number of different arrangements of students A, B, C, D in 3 communities
  -- such that each community has at least one student, and A and B are not in the same community.
  sorry

theorem arrangements_count_correct : count_arrangements = 30 := by
  sorry

end arrangements_count_correct_l522_522867


namespace total_project_cost_l522_522688

def area : ℝ := 12544
def side_length : ℝ := Real.sqrt area
def perimeter : ℝ := 4 * side_length
def gate_width : ℝ := 2
def num_gates : ℕ := 3
def total_gate_width : ℝ := num_gates * gate_width
def length_of_barbed_wire : ℝ := perimeter - total_gate_width + 5
def cost_per_meter : ℝ := 2.25
def total_cost : ℝ := length_of_barbed_wire * cost_per_meter

theorem total_project_cost : total_cost = 1005.75 := by
  sorry

end total_project_cost_l522_522688


namespace total_profit_correct_l522_522064

-- Defining the given conditions
def A_investment : ℝ := 12000
def B_investment : ℝ := 16000
def C_investment : ℝ := 20000
def C_profit : ℝ := 36000

-- Defining the relationship and final proof goal
theorem total_profit_correct :
  ∃ (total_profit : ℝ), (total_profit = 86400) ∧ 
  (let total_parts := (A_investment / 4000 + B_investment / 4000 + C_investment / 4000) in
   let C_parts := C_investment / 4000 in
   total_profit = (C_profit / C_parts) * total_parts
  ) :=
begin
  sorry
end

end total_profit_correct_l522_522064


namespace solve_problem_l522_522336

-- Definitions based on conditions
def salty_cookies_eaten : ℕ := 28
def sweet_cookies_eaten : ℕ := 15

-- Problem statement
theorem solve_problem : salty_cookies_eaten - sweet_cookies_eaten = 13 := by
  sorry

end solve_problem_l522_522336


namespace min_fraction_ineq_l522_522136

theorem min_fraction_ineq (x y : ℝ) (hx : 0.4 ≤ x ∧ x ≤ 0.6) (hy : 0.3 ≤ y ∧ y ≤ 0.5) :
  ∃ z, (z = x * y / (x^2 + 2 * y^2)) ∧ z = 1 / 3 := sorry

end min_fraction_ineq_l522_522136


namespace penny_purchase_exceeded_minimum_spend_l522_522270

theorem penny_purchase_exceeded_minimum_spend :
  let bulk_price_per_pound := 5
  let minimum_spend := 40
  let tax_per_pound := 1
  let total_paid := 240
  let total_cost_per_pound := bulk_price_per_pound + tax_per_pound
  let pounds_purchased := total_paid / total_cost_per_pound
  let minimum_pounds_to_spend := minimum_spend / bulk_price_per_pound
  pounds_purchased - minimum_pounds_to_spend = 32 :=
by
  -- The proof is omitted here as per the instructions.
  sorry

end penny_purchase_exceeded_minimum_spend_l522_522270


namespace binary_addition_is_correct_l522_522461

-- Definitions for the binary numbers
def bin1 := "10101"
def bin2 := "11"
def bin3 := "1010"
def bin4 := "11100"
def bin5 := "1101"

-- Function to convert binary string to nat (using built-in functionality)
def binStringToNat (s : String) : Nat :=
  String.foldl (fun n c => 2 * n + if c = '1' then 1 else 0) 0 s

-- Binary numbers converted to nat
def n1 := binStringToNat bin1
def n2 := binStringToNat bin2
def n3 := binStringToNat bin3
def n4 := binStringToNat bin4
def n5 := binStringToNat bin5

-- The expected result in nat
def expectedSum := binStringToNat "11101101"

-- Proof statement
theorem binary_addition_is_correct : n1 + n2 + n3 + n4 + n5 = expectedSum :=
  sorry

end binary_addition_is_correct_l522_522461


namespace spiral_wire_length_l522_522447

noncomputable def wire_length (turns : ℕ) (height : ℝ) (circumference : ℝ) : ℝ :=
  Real.sqrt (height^2 + (turns * circumference)^2)

theorem spiral_wire_length
  (turns : ℕ) (height : ℝ) (circumference : ℝ)
  (turns_eq : turns = 10)
  (height_eq : height = 9)
  (circumference_eq : circumference = 4) :
  wire_length turns height circumference = 41 := 
by
  rw [turns_eq, height_eq, circumference_eq]
  simp [wire_length]
  norm_num
  rw [Real.sqrt_eq_rpow]
  norm_num
  sorry

end spiral_wire_length_l522_522447


namespace problem_solution_l522_522917

open Real

theorem problem_solution 
  (a : ℝ)
  (f : ℝ → ℝ := λ x, 2 * sin ( (3/2) * π - x) * cos (π + x) + sin (2 * (π / 2 - x)) + a) 
  (h_max : ∀ x ∈ Icc (0 : ℝ) (π / 6), f x ≤ 2)
  (h_max_val : ∃ x ∈ Icc (0 : ℝ) (π / 6), f x = 2) :
  ∃ T x_0 k ∈ ℤ, 
  (∀ x, f (x + T) = f x) ∧ T = π 
  ∧ (∀ k ∈ ℤ, 
    ∀ x ∈ Icc (-3 * π / 8 + k * π) (π / 8 + k * π), 
    (deriv f) x > 0)
  ∧ a = 1 - sqrt 2
  ∧ (∀ k ∈ ℤ, f (x_0 + k * π) = f (x_0 + π/2 + k * π)) 
  ∧ x_0 = π/8 := 
by
  sorry

end problem_solution_l522_522917


namespace inequality_for_positive_a_b_n_l522_522184

theorem inequality_for_positive_a_b_n (a b : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) : 
  (a + b) ^ n - a ^ n - b ^ n ≥ 2 ^ (2 * n) - 2 ^ (n + 1) :=
sorry

end inequality_for_positive_a_b_n_l522_522184


namespace maximize_savings_l522_522034

-- Definitions for the conditions
def initial_amount : ℝ := 15000

def discount_option1 (amount : ℝ) : ℝ := 
  let after_first : ℝ := amount * 0.75
  let after_second : ℝ := after_first * 0.90
  after_second * 0.95

def discount_option2 (amount : ℝ) : ℝ := 
  let after_first : ℝ := amount * 0.70
  let after_second : ℝ := after_first * 0.90
  after_second * 0.90

-- Theorem to compare the final amounts
theorem maximize_savings : discount_option2 initial_amount < discount_option1 initial_amount := 
  sorry

end maximize_savings_l522_522034


namespace number_of_common_tangents_l522_522501

theorem number_of_common_tangents 
  (circle1 : ∀ x y : ℝ, x^2 + y^2 = 1)
  (circle2 : ∀ x y : ℝ, 2 * y^2 - 6 * x - 8 * y + 9 = 0) : 
  ∃ n : ℕ, n = 3 :=
by
  -- Proof is skipped
  sorry

end number_of_common_tangents_l522_522501


namespace difference_in_soda_bottles_l522_522787

def diet_soda_bottles : ℕ := 4
def regular_soda_bottles : ℕ := 83

theorem difference_in_soda_bottles :
  regular_soda_bottles - diet_soda_bottles = 79 :=
by
  sorry

end difference_in_soda_bottles_l522_522787


namespace segment_coverage_l522_522333

theorem segment_coverage (segments : list (ℝ × ℝ)) :
  ∀ (x y : ℝ), (x, y) ∈ segments → 0 ≤ x ∧ y ≤ 1 →
  (∃ subs : list (ℝ × ℝ), (∀ (x y : ℝ), (x, y) ∈ subs → (x, y) ∈ segments) ∧
    (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ (x y : ℝ), (x, y) ∈ subs ∧ x ≤ t ∧ t ≤ y) ∧
    list.sum (list.map (λ (s : ℝ × ℝ), s.snd - s.fst) subs) ≤ 2) := by
  sorry

end segment_coverage_l522_522333


namespace determine_range_of_a_l522_522916

noncomputable def f (x a : ℝ) : ℝ :=
  if x > a then x + 2 else x^2 + 5*x + 2

noncomputable def g (x a : ℝ) : ℝ := f x a - 2*x

theorem determine_range_of_a (a : ℝ) :
  (∀ x, g x a = 0 → (x = 2 ∨ x = -1 ∨ x = -2)) →
  (-1 ≤ a ∧ a < 2) :=
by
  intro h
  sorry

end determine_range_of_a_l522_522916


namespace range_of_log2_sqrt_sin_l522_522750

noncomputable def sin_range (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ real.to_radians 180 → 0 ≤ real.sin x ∧ real.sin x ≤ 1

noncomputable def sqrt_sin_range (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ real.to_radians 180 → 0 ≤ real.sqrt (real.sin x) ∧ real.sqrt (real.sin x) ≤ 1

noncomputable def log2_sqrt_sin_range : Set ℝ :=
  {y : ℝ | ∃ x : ℝ, 0 ≤ x ∧ x ≤ real.to_radians 180 ∧ y = real.log2 (real.sqrt (real.sin x))}

theorem range_of_log2_sqrt_sin : log2_sqrt_sin_range = {y : ℝ | y ≤ 0} :=
sorry

end range_of_log2_sqrt_sin_l522_522750


namespace add_pure_acid_to_obtain_final_concentration_l522_522588

   variable (x : ℝ)

   def initial_solution_volume : ℝ := 60
   def initial_acid_concentration : ℝ := 0.10
   def final_acid_concentration : ℝ := 0.15

   axiom calculate_pure_acid (x : ℝ) :
     initial_acid_concentration * initial_solution_volume + x = final_acid_concentration * (initial_solution_volume + x)

   noncomputable def pure_acid_solution : ℝ := 3/0.85

   theorem add_pure_acid_to_obtain_final_concentration :
     x = pure_acid_solution := by
     sorry
   
end add_pure_acid_to_obtain_final_concentration_l522_522588


namespace probability_of_specific_roll_l522_522327

noncomputable def probability_event : ℚ :=
  let favorable_outcomes_first_die := 3 -- 1, 2, 3
  let total_outcomes_die := 8
  let probability_first_die := favorable_outcomes_first_die / total_outcomes_die
  
  let favorable_outcomes_second_die := 4 -- 5, 6, 7, 8
  let probability_second_die := favorable_outcomes_second_die / total_outcomes_die
  
  probability_first_die * probability_second_die

theorem probability_of_specific_roll :
  probability_event = 3 / 16 := 
  by
    sorry

end probability_of_specific_roll_l522_522327


namespace D_K_value_l522_522707

variables {A B C H K D : Type} [linear_ordered_semiring A] 

-- Assume some geometry definition, not strictly necessary for this purpose
variables (AC : segment A C) (BD : height B D)
variables (AD DC DK : A)

def conditions (AD DC BD : A) : Prop :=
AD = 2 ∧ DC = 3 

theorem D_K_value (h : conditions AD DC BD) : DK = sqrt 6 := by
  sorry

end D_K_value_l522_522707


namespace hall_volume_correct_l522_522018

noncomputable def hall_volume (length breadth height : ℝ) : ℝ := length * breadth * height

theorem hall_volume_correct :
    (∀ (l b : ℝ), l = 15 → b = 12 →
    ∀ (h : ℝ), 2 * (l * b) = 54 * h → hall_volume l b h = 8004) :=
by
    intros l b h l_eq b_eq areas_eq
    rw [l_eq, b_eq, hall_volume]
    sorry

end hall_volume_correct_l522_522018


namespace common_positive_divisors_count_l522_522937

-- To use noncomputable functions
noncomputable theory

open Nat

-- Define the two numbers
def num1 : ℕ := 9240
def num2 : ℕ := 13860

-- Define their greatest common divisor
def gcd_val : ℕ := gcd num1 num2

-- State the prime factorization of the gcd (this can be proven or assumed as a given condition for cleaner code)
def prime_factors_gcd := [(2, 2), (3, 1), (7, 1), (11, 1)]

-- Given the prime factorization, calculate the number of divisors
def number_of_divisors : ℕ := 
  prime_factors_gcd.foldr (λ (factor : ℕ × ℕ) acc, acc * (factor.snd + 1)) 1

-- The final theorem stating the number of common positive divisors of num1 and num2
theorem common_positive_divisors_count : number_of_divisors = 24 := by {
  -- Here would go the proof, which is not required in this task
  sorry
}

end common_positive_divisors_count_l522_522937


namespace clipped_convex_ngon_area_l522_522736

theorem clipped_convex_ngon_area (n : ℕ) (hn: n ≥ 6) :
  ∀ (P : Polygon) (init_hex : RegularHexagon P) (clips : (Polygon -> Polygon) -> Polygon),
  area P init_hex = 1 →
  (∀ (k : ℕ) (P_k : Polygon), 6 ≤ k → k ≤ n → clips P_k P → area P_k > (1/3)) :=
sorry

end clipped_convex_ngon_area_l522_522736


namespace number_of_pairs_l522_522518

def pair_count_satisfying_condition : Prop :=
  ∃ (x y : ℤ), x^2 + 6 * x * y + 5 * y^2 = 10 ^ 100

theorem number_of_pairs :
  (finset.card ((finset.univ : finset (ℤ × ℤ)).filter (λ ⟨x, y⟩, x^2 + 6 * x * y + 5 * y^2 = 10 ^ 100)) = 19594) := 
sorry

end number_of_pairs_l522_522518


namespace okml_is_parallelogram_l522_522638

-- Define the circle, chords, midpoints, and intersections with the given conditions.
structure Circle (α : Type _) :=
(center : α)
(radius : ℝ)

variables {α : Type _} [metric_space α]

structure Chord (C : Circle α) :=
(start finish : α)

def midpoint {C : Circle α} (ch : Chord C) : α := sorry

def intersect_at_right_angle {C : Circle α} (ch1 ch2 : Chord C) (pt : α) : Prop := sorry

-- Main theorem statement
theorem okml_is_parallelogram (C : Circle α) (AC BD : Chord C)
  (M : α) (K L : α)
  (h1 : intersect_at_right_angle AC BD M)
  (h2 : K = midpoint AC)
  (h3 : L = midpoint BD) :
  parallelogram C.center K M L :=
sorry

end okml_is_parallelogram_l522_522638


namespace perimeter_remaining_shape_l522_522472

theorem perimeter_remaining_shape (length width square1 square2 : ℝ) 
  (H_len : length = 50) (H_width : width = 20) 
  (H_sq1 : square1 = 12) (H_sq2 : square2 = 4) : 
  2 * (length + width) + 4 * (square1 + square2) = 204 :=
by 
  rw [H_len, H_width, H_sq1, H_sq2]
  sorry

end perimeter_remaining_shape_l522_522472


namespace exists_set_with_perfect_power_means_l522_522677

open BigOperators

def is_perfect_power (n : ℕ) : Prop :=
∃ (a b : ℕ), b ≥ 2 ∧ n = a ^ b

theorem exists_set_with_perfect_power_means :
  ∃ (A : Finset ℕ), A.card = 2022 ∧ (∀ B ⊆ A, is_perfect_power (B.sum / B.card)) :=
begin
  sorry
end

end exists_set_with_perfect_power_means_l522_522677


namespace aqua_park_earnings_l522_522465

/-- Define the costs and groups of visitors. --/
def admission_fee : ℕ := 12
def tour_fee : ℕ := 6
def group1_size : ℕ := 10
def group2_size : ℕ := 5

/-- Define the total earnings of the aqua park. --/
def total_earnings : ℕ := (admission_fee + tour_fee) * group1_size + admission_fee * group2_size

/-- Prove that the total earnings are $240. --/
theorem aqua_park_earnings : total_earnings = 240 :=
by
  -- proof steps would go here
  sorry

end aqua_park_earnings_l522_522465


namespace hyperbola_eq_ellipse_eq_x_major_ellipse_eq_y_major_l522_522851

-- Problem 1: Hyperbola equation
theorem hyperbola_eq (a b c : ℝ) (h1 : 2 * a = 6) (h2 : c = (5/3) * a) (h3 : c^2 = a^2 + b^2) :
  (x^2 / a^2) - (y^2 / b^2) = 1 :=
sorry

-- Problem 2: Ellipse equations
theorem ellipse_eq_x_major (a b : ℝ) (h1 : 3 * b = a) (h2 : (3 : ℝ, 0) ∈ set_of (λ (p : ℝ × ℝ), (p.1^2 / a^2) + (p.2^2 / b^2) = 1)) :
  (x^2 / a^2) + (y^2 / b^2) = 1 :=
sorry

theorem ellipse_eq_y_major (a b : ℝ) (h1 : 3 * a = b) (h2 : (3 : ℝ, 0) ∈ set_of (λ (p : ℝ × ℝ), (p.2^2 / a^2) + (p.1^2 / b^2) = 1)) :
  (y^2 / a^2) + (x^2 / b^2) = 1 :=
sorry

end hyperbola_eq_ellipse_eq_x_major_ellipse_eq_y_major_l522_522851


namespace arcsin_sqrt3_div_2_eq_pi_div_3_l522_522483

theorem arcsin_sqrt3_div_2_eq_pi_div_3 : Real.arcsin (Real.sqrt 3 / 2) = Real.pi / 3 :=
by
  sorry

end arcsin_sqrt3_div_2_eq_pi_div_3_l522_522483


namespace cost_four_bottles_eq_two_l522_522285

namespace Josette

-- Define the given conditions
def cost_three_bottles : ℝ := 1.50
def number_of_bottles_three : ℕ := 3
def number_of_bottles_four : ℕ := 4

-- Calculate the cost of one bottle
def cost_one_bottle : ℝ := cost_three_bottles / number_of_bottles_three

-- State the theorem to prove
theorem cost_four_bottles_eq_two : cost_one_bottle * number_of_bottles_four = 2 :=
by
  sorry

end Josette

end cost_four_bottles_eq_two_l522_522285


namespace percentage_decrease_last_year_l522_522713

-- Define the percentage decrease last year
variable (x : ℝ)

-- Define the condition that expresses the stock price this year
def final_price_change (x : ℝ) : Prop :=
  (1 - x / 100) * 1.10 = 1 + 4.499999999999993 / 100

-- Theorem stating the percentage decrease
theorem percentage_decrease_last_year : final_price_change 5 := by
  sorry

end percentage_decrease_last_year_l522_522713


namespace find_x_l522_522265

-- Definition of the problem conditions
def angle_ABC : ℝ := 85
def angle_BAC : ℝ := 55
def sum_angles_triangle (a b c : ℝ) : Prop := a + b + c = 180
def corresponding_angle (a b : ℝ) : Prop := a = b
def right_triangle_sum (a b : ℝ) : Prop := a + b = 90

-- The theorem to prove
theorem find_x :
  ∀ (x BCA : ℝ), sum_angles_triangle angle_ABC angle_BAC BCA ∧ corresponding_angle BCA 40 ∧ right_triangle_sum BCA x → x = 50 :=
by
  intros x BCA h
  sorry

end find_x_l522_522265


namespace cost_for_four_bottles_l522_522282

variable (cost_3_bottles : ℝ) (n_bottles : ℝ)

def cost_per_bottle (cost_3_bottles : ℝ) (n_bottles : ℝ) : ℝ :=
  cost_3_bottles / n_bottles

def cost_four_bottles (cost_per_bottle : ℝ) : ℝ :=
  cost_per_bottle * 4

theorem cost_for_four_bottles
  (hc3 : cost_3_bottles = 1.50)
  (hn : n_bottles = 3) :
  cost_four_bottles (cost_per_bottle cost_3_bottles n_bottles) = 2.00 :=
by
  sorry

end cost_for_four_bottles_l522_522282


namespace penny_exceeded_by_32_l522_522267

def bulk_price : ℤ := 5
def min_spend_before_tax : ℤ := 40
def tax_per_pound : ℤ := 1
def penny_payment : ℤ := 240

def total_cost_per_pound : ℤ := bulk_price + tax_per_pound

def min_pounds_for_min_spend : ℤ := min_spend_before_tax / bulk_price

def total_pounds_penny_bought : ℤ := penny_payment / total_cost_per_pound

def pounds_exceeded : ℤ := total_pounds_penny_bought - min_pounds_for_min_spend

theorem penny_exceeded_by_32 : pounds_exceeded = 32 := by
  sorry

end penny_exceeded_by_32_l522_522267


namespace problem_l522_522554

noncomputable def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

noncomputable def quadratic_roots (a₃ a₁₀ : ℝ) : Prop :=
a₃^2 - 3 * a₃ - 5 = 0 ∧ a₁₀^2 - 3 * a₁₀ - 5 = 0

theorem problem (a : ℕ → ℝ) (h1 : is_arithmetic_seq a)
  (h2 : quadratic_roots (a 3) (a 10)) :
  a 5 + a 8 = 3 :=
sorry

end problem_l522_522554


namespace max_possible_median_l522_522686

theorem max_possible_median (total_cups : ℕ) (total_customers : ℕ) (min_cups_per_customer : ℕ)
  (h1 : total_cups = 310) (h2 : total_customers = 120) (h3 : min_cups_per_customer = 1) :
  ∃ median : ℕ, median = 4 :=
by {
  sorry
}

end max_possible_median_l522_522686


namespace b_n_inequality_l522_522865

def largest_odd_divisor (n : ℕ) : ℕ :=
if h : n > 0 then (Finset.filter (λ d, d % 2 = 1) (Finset.divisors n)).max' (by {
  cases n with n h₀,
  { exfalso, exact nat.lt_asymm (nat.zero_lt_one) h },
  { simp only [Finset.nonempty_divisors, Finset.nonempty_filter], exact ⟨1, ⟨1, nat.one_dvd _⟩, by norm_num⟩ }
}) else 0

noncomputable def b (n : ℕ) : ℕ :=
(Finset.range n).sum (λ i, largest_odd_divisor (i + 1))

theorem b_n_inequality (n : ℕ) (h : n ≥ 1) : 
b n ≥ (n^2 + 2) / 3 ∧ (b n = (n^2 + 2) / 3 ↔ ∃ k, n = 2^k) :=
sorry

end b_n_inequality_l522_522865


namespace max_distinct_sums_l522_522441

theorem max_distinct_sums : 
  let values : List ℝ := [0.01, 0.05, 0.10, 0.50] in
  ∃ sums : List ℝ, 
    (∀ x y : ℝ, x ∈ values → y ∈ values → x ≤ y → (x, y) ∈ sums.product sums ↔ (x = y ∨ x < y)) ∧
    sums.length = 8 :=
by
  let values : List ℝ := [0.01, 0.05, 0.10, 0.50]
  let sums := [0.06, 0.11, 0.51, 0.10, 0.15, 0.55, 0.20, 0.60]
  use sums
  sorry

end max_distinct_sums_l522_522441


namespace two_digit_perfect_squares_divisible_by_7_l522_522233

theorem two_digit_perfect_squares_divisible_by_7 : 
  {n : ℕ | ∃ k : ℕ, k^2 = n ∧ 10 ≤ n ∧ n < 100 ∧ 7 ∣ n}.card = 1 := by
  sorry

end two_digit_perfect_squares_divisible_by_7_l522_522233


namespace equation_of_ellipse_l522_522902

noncomputable def foci1 : (ℝ × ℝ) := (-1, 0)
noncomputable def foci2 : (ℝ × ℝ) := (1, 0)

def intersection_points_conditions (A B : ℝ × ℝ) : Prop :=
  dist A foci2 = 2 * dist foci2 B ∧ dist A B = dist B foci1

theorem equation_of_ellipse (A B : ℝ × ℝ) (a b : ℝ)
  (h1 : foci1 = (-1, 0))
  (h2 : foci2 = (1, 0))
  (h3 : intersection_points_conditions A B = true)
  (h4 : (A, B) = ... ) : -- you can fill in specific values if necessary
  (frac (x^2) 3) + (frac (y^2) 2) = 1 :=
sorry

end equation_of_ellipse_l522_522902


namespace solve_logarithmic_equation_l522_522862

theorem solve_logarithmic_equation (x : ℝ) (h : log 64 (3 * x + 2) = -1/3) : x = -7/12 :=
sorry

end solve_logarithmic_equation_l522_522862


namespace find_x_plus_z_l522_522869

theorem find_x_plus_z :
  ∃ (x y z : ℝ), 
  (x + y + z = 0) ∧
  (2016 * x + 2017 * y + 2018 * z = 0) ∧
  (2016^2 * x + 2017^2 * y + 2018^2 * z = 2018) ∧
  (x + z = 4036) :=
sorry

end find_x_plus_z_l522_522869


namespace xy_sum_zero_l522_522548

theorem xy_sum_zero (x y : ℝ) (h : (x + sqrt (x^2 + 1)) * (y + sqrt (y^2 + 1)) = 1) : x + y = 0 :=
sorry

end xy_sum_zero_l522_522548


namespace boys_neither_sport_l522_522977

theorem boys_neither_sport (Total Boys B F BF N : ℕ) (H_total : Total = 22) (H_B : B = 13) (H_F : F = 15) (H_BF : BF = 18) :
    N = Total - (B + F - BF) :=
sorry

end boys_neither_sport_l522_522977


namespace new_average_of_modified_integers_is_25_5_l522_522022

theorem new_average_of_modified_integers_is_25_5 : 
  ∀ (seq : ℕ → ℤ), (∀ n, seq (n + 1) = seq n + 1) → 
  (∑ i in finset.range 20, seq i) / 20 = 35 →
  let new_seq : ℕ → ℤ := λ i, if i < 19 then seq i - (19 - i) else seq i in
  (∑ i in finset.range 20, new_seq i) / 20 = 25.5 := 
by
  intros seq seq_step avg_cond new_seq_def
  sorry

end new_average_of_modified_integers_is_25_5_l522_522022


namespace aqua_park_earnings_l522_522464

/-- Define the costs and groups of visitors. --/
def admission_fee : ℕ := 12
def tour_fee : ℕ := 6
def group1_size : ℕ := 10
def group2_size : ℕ := 5

/-- Define the total earnings of the aqua park. --/
def total_earnings : ℕ := (admission_fee + tour_fee) * group1_size + admission_fee * group2_size

/-- Prove that the total earnings are $240. --/
theorem aqua_park_earnings : total_earnings = 240 :=
by
  -- proof steps would go here
  sorry

end aqua_park_earnings_l522_522464


namespace primes_div_conditions_unique_l522_522512

theorem primes_div_conditions_unique (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (p ∣ q + 6) ∧ (q ∣ p + 7) → (p = 19 ∧ q = 13) :=
sorry

end primes_div_conditions_unique_l522_522512


namespace discount_percentage_l522_522478

variable (P : ℝ) -- Original price of the dress
variable (D : ℝ) -- Discount percentage

theorem discount_percentage
  (h1 : P * (1 - D / 100) = 68)
  (h2 : 68 * 1.25 = 85)
  (h3 : 85 - P = 5) :
  D = 15 :=
by
  sorry

end discount_percentage_l522_522478


namespace exists_n_divisible_by_2006_l522_522272

def a_n (n : ℤ) := n^3 - (2 * n + 1)^2

theorem exists_n_divisible_by_2006 :
  ∃ (n : ℤ), a_n n % 2006 = 0 :=
begin
  sorry
end

end exists_n_divisible_by_2006_l522_522272


namespace find_m_l522_522887

def vec (x y : ℝ) : ℝ × ℝ := (x, y)

variables (m : ℝ)

def a := vec 1 (-real.sqrt 3)
def b := vec 2 m

def add_vec (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def sub_vec (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

def vec_parallel (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

theorem find_m : vec_parallel (add_vec a b) (sub_vec a b) → m = -2 * real.sqrt 3 :=
by intros h; sorry

end find_m_l522_522887


namespace hyperbola_equation_l522_522568

variable (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
variable (C : ℝ → ℝ → Prop) (C_eq : C = λ x y, x^2 / a^2 - y^2 / b^2 = 1)
variable (A : ℝ × ℝ) (A_eq : A = (3/2, sqrt 3 / 2))
variable (asymptote : ℝ → ℝ → Prop) (asymptote_eq : asymptote = λ x y, y = b / a * x ∨ y = -b / a * x)
variable (circle_intersect : ℝ → ℝ → Prop)
variable (circle_with_OF : circle_intersect O)
variable (origin O : ℝ × ℝ) (origin_eq : O = (0, 0))

theorem hyperbola_equation : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a = sqrt 3 * b ∧ b = sqrt (3/2) ∧ 
  (C = λ x y, x^2 / 3 - y^2 = 1) :=
by
  sorry

end hyperbola_equation_l522_522568


namespace ratio_XR_RS_l522_522487

-- Definitions based on the conditions
def total_hexagon_area : ℝ := 13
def area_per_side : ℝ := total_hexagon_area / 2
def area_unit_squares : ℝ := 2
def base_triangle : ℝ := 4
def area_triangle : ℝ := area_per_side - area_unit_squares
def height_triangle : ℝ := (2 * area_triangle) / base_triangle

-- Theorem statement
theorem ratio_XR_RS (XR RS : ℝ) (h1 : XR + RS = base_triangle)
                    (h2 : area_triangle = 4.5)
                    (h3 : total_hexagon_area = 13)
                    (h4 : area_per_side = 6.5)
                    (h5 : area_unit_squares = 2)
                    (h6 : height_triangle = 2.25) :
  XR / RS = 1 :=
by {
  sorry
}

end ratio_XR_RS_l522_522487


namespace number_of_incorrect_propositions_is_2_l522_522774

-- Definitions based on the conditions given in the original problem
def prop1 (l1 l2 l : Type) [line l1] [line l2] [line l] := 
  perpendicular l1 l ∧ perpendicular l2 l → parallel l1 l2

def prop2 (l1 l2 : Type) [line l1] [line l2] [plane p] := 
  perpendicular l1 p ∧ perpendicular l2 p → parallel l1 l2

def prop3 (p1 p2 : Type) [plane p1] [plane p2] [line l] := 
  perpendicular p1 l ∧ perpendicular p2 l → parallel p1 p2

def prop4 (p1 p2 : Type) [plane p1] [plane p2] [plane p] := 
  perpendicular p1 p ∧ perpendicular p2 p → perpendicular p1 p2

-- The main theorem we are supposed to prove
theorem number_of_incorrect_propositions_is_2 :
  ¬prop1 ∧ prop2 ∧ prop3 ∧ ¬prop4 → incorrect_propositions_count = 2 :=
sorry

end number_of_incorrect_propositions_is_2_l522_522774


namespace simplify_and_evaluate_l522_522354

def a : Int := 1
def b : Int := -2

theorem simplify_and_evaluate :
  ((a * b - 3 * a^2) - 2 * b^2 - 5 * a * b - (a^2 - 2 * a * b)) = -8 := by
  sorry

end simplify_and_evaluate_l522_522354


namespace family_functions_count_l522_522596

-- Definition of a "family function" condition.
def is_family_function (f : ℝ → ℝ) (domain : set ℝ) (range : set ℝ) : Prop :=
  ∀ x ∈ domain, f x^2 ∈ range

-- Analytical expression for y = x^2.
def analytical_expression : ℝ → ℝ := λ x, x^2

-- The specific range considered.
def range_set : set ℝ := {1, 9}

-- Specific domains for the given problem.
def domain_1 : set ℝ := {1, 3}
def domain_2 : set ℝ := {1, -3}
def domain_3 : set ℝ := {-1, 3}
def domain_4 : set ℝ := {-1, -3}
def domain_5 : set ℝ := {1, -1, 3}
def domain_6 : set ℝ := {1, -1, -3}
def domain_7 : set ℝ := {1, -3, 3}
def domain_8 : set ℝ := {-1, -3, 3}
def domain_9 : set ℝ := {-1, 1, 3, -3}

-- The proof problem itself.
theorem family_functions_count : 
  ∃ (domains : list (set ℝ)), 
  (∀ d ∈ domains, is_family_function analytical_expression d range_set) 
  ∧ list.length domains = 9 :=
by {
  sorry
}

end family_functions_count_l522_522596


namespace ratio_of_breadth_to_length_is_six_l522_522365

-- Define constants and conditions
def breadth : ℝ := 420
def playground_area : ℝ := 4200
def playground_fraction : ℝ := 1 / 7

-- Calculate total area of the landscape based on the area of the playground
def total_landscape_area : ℝ := playground_area / playground_fraction

-- Define length based on total area and breadth
def length : ℝ := total_landscape_area / breadth

-- Theorem statement to prove the ratio of breadth to length
theorem ratio_of_breadth_to_length_is_six : breadth / length = 6 := by
  sorry

end ratio_of_breadth_to_length_is_six_l522_522365


namespace ScientificNotation_of_45400_l522_522690

theorem ScientificNotation_of_45400 :
  45400 = 4.54 * 10^4 := sorry

end ScientificNotation_of_45400_l522_522690


namespace four_distinct_real_solutions_l522_522198

theorem four_distinct_real_solutions (k : ℝ) :
  (∃ (x1 x2 x3 x4 : ℝ), x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ 
                         x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧ 
                         k * x1^2 + abs(x1) / (x1 + 6) = 0 ∧ 
                         k * x2^2 + abs(x2) / (x2 + 6) = 0 ∧ 
                         k * x3^2 + abs(x3) / (x3 + 6) = 0 ∧ 
                         k * x4^2 + abs(x4) / (x4 + 6) = 0)
    → k < -1 / 9 :=
sorry

end four_distinct_real_solutions_l522_522198


namespace abs_diff_eq_seven_l522_522240

theorem abs_diff_eq_seven (m n : ℤ) (h1 : |m| = 5) (h2 : |n| = 2) (h3 : m * n < 0) : |m - n| = 7 := 
sorry

end abs_diff_eq_seven_l522_522240


namespace notebooks_to_sell_to_earn_profit_l522_522033

-- Define the given conditions
def notebooks_purchased : ℕ := 2000
def cost_per_notebook : ℚ := 0.15
def selling_price_per_notebook : ℚ := 0.30
def desired_profit : ℚ := 120

-- Define the total cost
def total_cost := notebooks_purchased * cost_per_notebook

-- Define the total revenue needed
def total_revenue_needed := total_cost + desired_profit

-- Define the number of notebooks to be sold to achieve the total revenue
def notebooks_to_sell := total_revenue_needed / selling_price_per_notebook

-- Prove that the number of notebooks to be sold is 1400 to make a profit of $120
theorem notebooks_to_sell_to_earn_profit : notebooks_to_sell = 1400 := 
by {
  sorry
}

end notebooks_to_sell_to_earn_profit_l522_522033


namespace cyclic_quadrilateral_right_triangle_equality_l522_522793

theorem cyclic_quadrilateral_right_triangle_equality 
    (A B C D O : Type) 
    [is_triangle A B C] 
    [is_right_triangle A B C]
    (angle_BAC : angle (B, A, C) = 90)
    (radius_O : radius_circle_in_triangle A B C O = 1)
    (BC_eq_5 : length B C = 5)
    (D_on_circle : point D = extension A O)
    : length BD = length CD ∧ length CD = length OD ∧ length BD = length OD :=
by 
  sorry

end cyclic_quadrilateral_right_triangle_equality_l522_522793


namespace false_statement_of_quadratic_l522_522531

-- Define the function f and the conditions
def f (a b c x : ℝ) := a * x^2 + b * x + c

theorem false_statement_of_quadratic (a b c x0 : ℝ) (h₀ : a > 0) (h₁ : 2 * a * x0 + b = 0) :
  ¬ ∀ x : ℝ, f a b c x ≤ f a b c x0 := by
  sorry

end false_statement_of_quadratic_l522_522531


namespace Luke_points_per_round_l522_522317

theorem Luke_points_per_round (total_points : ℕ) (number_of_rounds : ℕ) (h1 : total_points = 300) (h2 : number_of_rounds = 5) : 
  total_points / number_of_rounds = 60 := 
by 
  -- We need the lemma to divide by non-zero natural number
  have h3 : number_of_rounds ≠ 0 := by simp [h2]
  calc
    total_points / number_of_rounds = 300 / 5 : by rw [h1, h2]
    ... = 60 : by norm_num

end Luke_points_per_round_l522_522317


namespace ABT_equilateral_area_ratio_triangle_to_quadrilateral_l522_522369

section EquilateralProof

-- Definitions of the key points and points' relationships in the problem
variables {A B C D O T : Type}
variables {point : Type} [HasPoint point]

-- Given conditions
axiom diagonals_intersect (ABCD : Type) (O : point) : 
  Intersection (diagonal (ABCD)) O

axiom BOC_equilateral (B O C : point) :
  EquilateralTriangle B O C

axiom AOD_equilateral (A O D : point) :
  EquilateralTriangle A O D

axiom T_symmetric (O T C D M : point) :
  MidPoint M C D ∧ Reflect O M T

-- Data given for part (b)
axiom BC_length (B C : point) :
  Distance B C = 3

axiom AD_length (A D : point) :
  Distance A D = 4

-- Part a: Prove ABC is an equilateral triangle
theorem ABT_equilateral {A B T : point} :
  EquilateralTriangle A B T :=
by sorry

-- Part b: Ratio of areas
theorem area_ratio_triangle_to_quadrilateral {A B C D O T : point}
  (BC : Distance B C = 3) (AD : Distance A D = 4) :
  AreaRatio (Triangle A B T) (Quadrilateral A B C D) = 37 / 49 :=
by sorry

end EquilateralProof

end ABT_equilateral_area_ratio_triangle_to_quadrilateral_l522_522369


namespace log_base_a_conditions_l522_522529

theorem log_base_a_conditions (a m n : ℝ) (ha : 0 < a ∧ a < 1) (hmn : 0 < log a m ∧ log a m < log a n) : n < m ∧ m < 1 :=
by
  sorry

end log_base_a_conditions_l522_522529


namespace same_fee_probability_is_correct_xi_4_probability_is_correct_xi_6_probability_is_correct_l522_522761

noncomputable theory

def prob_A_2_hours : rat := 1 / 4
def prob_B_2_hours : rat := 1 / 2
def prob_A_3_hours : rat := 1 / 2
def prob_B_3_hours : rat := 1 / 4
def prob_A_4_hours : rat := 1 / 4
def prob_B_4_hours : rat := 1 / 4

def prob_same_fee : rat :=
  prob_A_2_hours * prob_B_2_hours + prob_A_3_hours * prob_B_3_hours + prob_A_4_hours * prob_B_4_hours

theorem same_fee_probability_is_correct :
  prob_same_fee = 5 / 16 := by
  sorry

def P_xi_4 : rat :=
  prob_A_2_hours * prob_B_4_hours + prob_B_2_hours * prob_A_4_hours + prob_A_3_hours * prob_B_3_hours

def P_xi_6 : rat :=
  prob_A_4_hours * prob_B_2_hours + prob_B_4_hours * prob_A_2_hours

theorem xi_4_probability_is_correct :
  P_xi_4 = 5 / 16 := by
  sorry

theorem xi_6_probability_is_correct :
  P_xi_6 = 3 / 16 := by
  sorry

end same_fee_probability_is_correct_xi_4_probability_is_correct_xi_6_probability_is_correct_l522_522761


namespace average_marks_l522_522061

/--
Given:
1. The average marks in physics (P) and mathematics (M) is 90.
2. The average marks in physics (P) and chemistry (C) is 70.
3. The student scored 110 marks in physics (P).

Prove that the average marks the student scored in the 3 subjects (P, C, M) is 70.
-/
theorem average_marks (P C M : ℝ) 
  (h1 : (P + M) / 2 = 90)
  (h2 : (P + C) / 2 = 70)
  (h3 : P = 110) : 
  (P + C + M) / 3 = 70 :=
sorry

end average_marks_l522_522061


namespace num_ordered_pairs_l522_522681

theorem num_ordered_pairs (n : ℕ) (h : n = 6) :
  let pairs := { (f, m) | f m : ℕ, (f ≥ 0) ∧ (m ≥ 0) ∧ -- definitions of f and m based on their conditions
                       -- Conditions to define valid (f, m) pairs based on people's positions around the table
                      (some_condition_here)
              }
  in pairs.size = 10 :=
    by sorry

end num_ordered_pairs_l522_522681


namespace main_proof_l522_522537

variables {n : ℕ}
variables {a_n : ℕ → ℕ} {S_n T_n : ℕ → ℕ} 
variables {b_n C_n d_n : ℕ → ℕ}

-- Define arithmetic sequence \a_n\.
def is_arith_seq : Prop := 
  a_n 2 = 4 ∧ S_5 = 30 ∧ ∀ n, S_n (n + 1) = S_n n + a_n (n + 1)

-- Define the recurrence relations for \b_n\ and C_n.
def is_b_seq : Prop :=
  b_n 1 = 0 ∧ ∀ n, n ≥ 2 → b_n n = 2 * b_n (n - 1) + 1

def is_C_seq : Prop :=
  ∀ n, C_n n = b_n n + 1

-- Define the sequence \d_n\ and the sum T_n.
def is_d_seq (a_n b_n : ℕ → ℕ) : Prop :=
  ∀ n, d_n n = 4 / (a_n n * a_n (n + 1)) + b_n n 

def arith_seq_properties : Prop :=
  is_arith_seq ∧ ∀ n, a_n n = 2 * n 

def geom_seq_properties : Prop :=
  is_b_seq ∧ is_C_seq ∧ ∀ n, C_n (n + 1) / C_n n = 2 ∧ b_n n = 2^(n-1) - 1

def summation_properties : Prop :=
  ∀ T_n d_n, is_d_seq a_n b_n → S_n n = T_n n ∧ T_n n = 2^n - n - 1 / (n + 1)

-- Main theorem collecting all properties
theorem main_proof : arith_seq_properties ∧ geom_seq_properties ∧ summation_properties :=
  sorry

end main_proof_l522_522537


namespace probability_of_sequence_l522_522394

theorem probability_of_sequence :
  let total_cards := 52
  let face_cards := 12
  let hearts := 13
  let first_card_face_prob := (face_cards : ℝ) / total_cards
  let second_card_heart_prob := (10 : ℝ) / (total_cards - 1)
  let third_card_face_prob := (11 : ℝ) / (total_cards - 2)
  let total_prob := first_card_face_prob * second_card_heart_prob * third_card_face_prob
  total_prob = 1 / 100.455 :=
by
  sorry

end probability_of_sequence_l522_522394


namespace fourth_row_number_l522_522509

def valid_grid (grid : Matrix (Fin 6) (Fin 6) ℕ) : Prop :=
  ∀ i j, grid i j ∈ {1, 2, 3, 4, 5, 6} ∧
         (∀ k, grid i k ≠ grid i j → k ≠ j) ∧  -- Unique in row
         (∀ k, grid k j ≠ grid i j → k ≠ i) ∧  -- Unique in column
         (∀ u v, (u / 3 = i / 3) → (v / 3 = j / 3) → (u, v) ≠ (i, j) → grid u v ≠ grid i j) -- Unique in sub-square

def diagonal_sums (grid : Matrix (Fin 6) (Fin 6) ℕ) (diag1 diag2 : Fin 6 → ℕ) : Prop :=
  (∀ i, grid i i = diag1 i) ∧ (∀ i, grid i (5 - i) = diag2 i)

theorem fourth_row_number (grid : Matrix (Fin 6) (Fin 6) ℕ) 
  (diag1 diag2 : Fin 6 → ℕ) :
  valid_grid grid ∧ diagonal_sums grid diag1 diag2 →
  let row := grid 3 
  in (row 0 * 10000 + row 1 * 1000 + row 2 * 100 + row 3 * 10 + row 4 = 35126) :=
sorry

end fourth_row_number_l522_522509


namespace pyramid_surface_area_l522_522784

theorem pyramid_surface_area :
  let s := 8
  let A := (0, 0, 0)
  let B := (8, 0, 0)
  let C := (8, 8, 0)
  let A1 := (0, 0, 8)
  let B1 := (8, 0, 8)
  let D := (0, 8, 0)
  let D1 := (0, 8, 8)
  let C1 := (8, 8, 8)
  let M := (8, 4, 0)
  let N := (8, 0, 4)
  let AM := (8, 4, 0)
  let AN := (8, 0, 4)
  let cross_product_AM_AN := (16, -32, -32)
  let cross_product_magnitude := 48
  let area_triangle_AMN := 24
  in area_triangle_AMN = 24 :=
sorry

end pyramid_surface_area_l522_522784


namespace min_value_f_eq_neg_one_tangent_parallel_2e_cubed_function_ordering_l522_522921

noncomputable def f (x : ℝ) : ℝ :=
  x^3 * (3 * Real.log x - 1)

noncomputable def g (x a : ℝ) : ℝ :=
  f(x) - a

theorem min_value_f_eq_neg_one : f 1 = -1 := sorry

theorem tangent_parallel_2e_cubed (m : ℝ) (h : m = Real.exp 1) : f m = 2 * Real.exp 3 := sorry

theorem function_ordering :
  f (Real.log ((3 * Real.exp 1) / 2)) < f (3 / 2) ∧ f (3 / 2) < f (Real.log 2 3) := sorry

end min_value_f_eq_neg_one_tangent_parallel_2e_cubed_function_ordering_l522_522921


namespace stickers_distribution_l522_522583

theorem stickers_distribution : 
  ∃ (f : Fin 3 → ℕ), (∑ i, f i = 10) ∧ (∀ i, f i ≥ 1) ∧ (fintype.card {f // (∑ i, f i = 10) ∧ (∀ i, f i ≥ 1)} = 36) :=
sorry

end stickers_distribution_l522_522583


namespace interior_angle_of_regular_heptagon_l522_522746

theorem interior_angle_of_regular_heptagon : 
  let n := 7 in 
  let sum_of_interior_angles := (n - 2) * 180 in
  let interior_angle := sum_of_interior_angles / n in
  interior_angle = 128.57 :=
by
  sorry

end interior_angle_of_regular_heptagon_l522_522746


namespace city_rentals_cost_per_mile_l522_522674

-- The parameters provided in the problem
def safety_base_rate : ℝ := 21.95
def safety_per_mile_rate : ℝ := 0.19
def city_base_rate : ℝ := 18.95
def miles_driven : ℝ := 150.0

-- The cost expressions based on the conditions
def safety_total_cost (miles: ℝ) : ℝ := safety_base_rate + safety_per_mile_rate * miles
def city_total_cost (miles: ℝ) (city_per_mile_rate: ℝ) : ℝ := city_base_rate + city_per_mile_rate * miles

-- The cost equality condition for 150 miles
def cost_condition : Prop :=
  safety_total_cost miles_driven = city_total_cost miles_driven 0.21

-- Prove that the cost per mile for City Rentals is 0.21 dollars
theorem city_rentals_cost_per_mile : cost_condition :=
by
  -- Start the proof
  sorry

end city_rentals_cost_per_mile_l522_522674


namespace minimum_value_of_function_l522_522238

theorem minimum_value_of_function (x : ℝ) (hx : x > 4) : 
    (∃ y : ℝ, y = x + 9 / (x - 4) ∧ (∀ z : ℝ, (∃ w : ℝ, w > 4 ∧ z = w + 9 / (w - 4)) → z ≥ 10) ∧ y = 10) :=
sorry

end minimum_value_of_function_l522_522238


namespace mushroom_pickers_at_least_50_l522_522351

-- Given conditions
variables (a : Fin 7 → ℕ) -- Each picker collects a different number of mushrooms.
variables (distinct : ∀ i j, i ≠ j → a i ≠ a j)
variable (total_mushrooms : (Finset.univ.sum a) = 100)

-- The proof that at least three of the pickers collected at least 50 mushrooms together
theorem mushroom_pickers_at_least_50 (a : Fin 7 → ℕ) (distinct : ∀ i j, i ≠ j → a i ≠ a j)
    (total_mushrooms : (Finset.univ.sum a) = 100) :
    ∃ i j k : Fin 7, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ (a i + a j + a k) ≥ 50 :=
sorry

end mushroom_pickers_at_least_50_l522_522351


namespace conditional_probability_l522_522779

theorem conditional_probability :
  let A := λ (r : ℝ), r = 0.5
  let AB := λ (r : ℝ), r = 0.2
  (P (A) = 1/2) →
  (P (A ∩ B) = 1/5) →
  P (B | A) = 2/5 :=
by
  intro P_A P_AB
  sorry

end conditional_probability_l522_522779


namespace coral_must_read_pages_to_finish_book_l522_522102

theorem coral_must_read_pages_to_finish_book
  (total_pages first_week_read second_week_percentage pages_remaining first_week_left second_week_read : ℕ)
  (initial_pages_read : ℕ := total_pages / 2)
  (remaining_after_first_week : ℕ := total_pages - initial_pages_read)
  (read_second_week : ℕ := remaining_after_first_week * second_week_percentage / 100)
  (remaining_after_second_week : ℕ := remaining_after_first_week - read_second_week)
  (final_pages_to_read : ℕ := remaining_after_second_week):
  total_pages = 600 → first_week_read = 300 → second_week_percentage = 30 →
  pages_remaining = 300 → first_week_left = 300 → second_week_read = 90 →
  remaining_after_first_week = 300 - 300 →
  remaining_after_second_week = remaining_after_first_week - second_week_read →
  third_week_read = remaining_after_second_week →
  third_week_read = 210 := by
  sorry

end coral_must_read_pages_to_finish_book_l522_522102


namespace total_number_of_workers_is_49_l522_522770

-- Definitions based on the conditions
def avg_salary_all_workers := 8000
def num_technicians := 7
def avg_salary_technicians := 20000
def avg_salary_non_technicians := 6000

-- Prove that the total number of workers in the workshop is 49
theorem total_number_of_workers_is_49 :
  ∃ W, (avg_salary_all_workers * W = avg_salary_technicians * num_technicians + avg_salary_non_technicians * (W - num_technicians)) ∧ W = 49 := 
sorry

end total_number_of_workers_is_49_l522_522770


namespace possible_values_expression_l522_522171

-- Defining the main expression 
def main_expression (a b c d : ℝ) : ℝ :=
  (a / |a|) + (b / |b|) + (c / |c|) + (d / |d|) + (abcd / |abcd|)

-- The theorem that we need to prove
theorem possible_values_expression (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) : 
  main_expression a b c d ∈ {5, 1, -3} :=
sorry

end possible_values_expression_l522_522171


namespace problem1_minimum_value_problem2_triangle_area_l522_522564

def f (x : ℝ) := (Real.cos x) * ((Real.cos x) + (Real.sqrt 3) * (Real.sin x))

theorem problem1_minimum_value :
  ∃ x : ℝ, f x = -1 / 2 :=
sorry

theorem problem2_triangle_area 
  (C : ℝ)
  (a b c : ℝ)
  (C_pos : 0 < C)
  (C_lt_pi : C < Real.pi)
  (hyp_fC : f C = 1)
  (hyp_c : c = Real.sqrt 7)
  (hyp_a_b_sum : a + b = 4) :
  let area := (1 / 2) * a * b * Real.sin C in
  area = (3 * Real.sqrt 3) / 4 :=
sorry

end problem1_minimum_value_problem2_triangle_area_l522_522564


namespace girlsCombinedAvgIs83_l522_522077

section AverageScoreGirls
  -- Define variables
  variables (LB LG MB MG : ℕ)   -- Number of boys/girls in Lincoln and Monroe.
  -- Define the given conditions
  def ratioLincoln := (LB : ℚ) / (LG : ℚ) = 3 / 4
  def ratioMonroe := (MB : ℚ) / (MG : ℚ) = 5 / 2
  def avgBoysCombined := (68 * LB + 85 * MB) / (LB + MB) = 78
  def avgGirlsLincoln := 80
  def avgGirlsMonroe := 95

  -- Define the function to calculate the overall girls' average
  noncomputable def avgGirlsCombined (LB LG MB MG : ℕ) : ℚ :=
      let totalGirls := (LG + MG : ℚ)
      let weightedAvg := ((LG * 80) + (MG * 95)) / totalGirls
      in weightedAvg

  -- The theorem we want to prove
  theorem girlsCombinedAvgIs83
    (hL : ratioLincoln)
    (hM : ratioMonroe)
    (hBoys : avgBoysCombined)
    (hG : avgGirlsCombined LB LG MB MG = 83) :
    avgGirlsCombined LB LG MB MG = 83 := sorry
end AverageScoreGirls

end girlsCombinedAvgIs83_l522_522077


namespace triangle_max_min_diff_eq_fourteen_l522_522274

theorem triangle_max_min_diff_eq_fourteen :
  ∀ (A B C H : Point) (BC : ℝ) (AH : ℝ), BC = 10 → AH = 6 →
  let AB := dist A B
  let AC := dist A C
  let BH := dist B H
  let HC := 10 - BH
  let h := AH
  let AB_squared := BH^2 + h^2
  let AC_squared := (10 - BH)^2 + h^2
  let expr := AB_squared + AC_squared in
  let expr_fn := λ x, 2*x^2 - 20*x + 136 in
  (expr_fn 5 = 186) →
  (expr_fn 0 = 172) →
  (expr_fn 10 = 172) →
  let N := expr_fn 5
  let n := expr_fn 0 in
  N - n = 14 :=
sorry

end triangle_max_min_diff_eq_fourteen_l522_522274


namespace min_value_f_l522_522705

noncomputable def f (x : ℝ) : ℝ := Math.sin(2 * x - π / 3)

theorem min_value_f :
  ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f x = - (Real.sqrt 3) / 2 :=
by
  sorry

end min_value_f_l522_522705


namespace parabola_focus_distance_l522_522215

noncomputable def distance_to_focus (p : ℝ) (M : ℝ × ℝ) : ℝ :=
  let focus := (p, 0)
  let (x1, y1) := M
  let (x2, y2) := focus
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) + p

theorem parabola_focus_distance
  (M : ℝ × ℝ) (p : ℝ)
  (hM : M = (2, 2))
  (hp : p = 1) :
  distance_to_focus p M = Real.sqrt 5 + 1 :=
by
  sorry

end parabola_focus_distance_l522_522215


namespace possible_values_of_expression_l522_522165

theorem possible_values_of_expression (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (v : ℤ), v ∈ ({5, 1, -3, -5} : Set ℤ) ∧ v = (Int.sign a + Int.sign b + Int.sign c + Int.sign d + Int.sign (a * b * c * d)) :=
by
  sorry

end possible_values_of_expression_l522_522165


namespace simplify_expr_l522_522084

variable (x y : ℝ)
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)

theorem simplify_expr : 
  (x ^ (-2) - y ^ (-2)) / (x ^ (-1) - y ^ (-1)) = (y + x) / (x * y) :=
sorry

end simplify_expr_l522_522084


namespace fruit_sequences_l522_522494

-- Define the number of each type of fruit
def apples : ℕ := 4
def oranges : ℕ := 2
def banana : ℕ := 1
def pears : ℕ := 2
def total_days : ℕ := 8

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem stating the number of sequences in which Cory can eat the fruits
theorem fruit_sequences : 
  fact total_days / (fact apples * fact oranges * fact banana * fact pears) = 420 :=
by
  sorry

end fruit_sequences_l522_522494


namespace number_of_distinct_arrangements_l522_522819

-- Given conditions: There are 7 items and we need to choose 4 out of these 7.
def binomial_coefficient (n k : ℕ) : ℕ :=
  (n.choose k)

-- Given condition: Calculate the number of sequences of arranging 4 selected items.
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- The statement in Lean 4 to prove that the number of distinct arrangements is 840.
theorem number_of_distinct_arrangements : binomial_coefficient 7 4 * factorial 4 = 840 :=
by
  sorry

end number_of_distinct_arrangements_l522_522819


namespace greg_spent_on_shirt_l522_522012

-- Define the conditions in Lean
variables (S H : ℤ)
axiom condition1 : H = 2 * S + 9
axiom condition2 : S + H = 300

-- State the theorem to prove
theorem greg_spent_on_shirt : S = 97 :=
by
  sorry

end greg_spent_on_shirt_l522_522012


namespace quadrilateral_area_2023_l522_522775

-- Define the vertices A, B, C, and D
def A := (1, 3)
def B := (1, 1)
def C := (3, 1)
def D := (2023, 2024)

-- Define a function to calculate the area of a convex quadrilateral given its vertices
noncomputable def quadrilateral_area (A B C D : (ℕ × ℕ)) : ℕ :=
  let triangle_area (P Q R : (ℕ × ℕ)) : ℕ :=
    (Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2)
  in
  ((triangle_area A B C).natAbs + (triangle_area A C D).natAbs) / 2

-- Define the theorem stating the area of the given quadrilateral
theorem quadrilateral_area_2023 : quadrilateral_area A B C D = 2042113 := by
  sorry

end quadrilateral_area_2023_l522_522775


namespace aqua_park_earnings_l522_522466

def admission_cost : ℕ := 12
def tour_cost : ℕ := 6
def group1_size : ℕ := 10
def group2_size : ℕ := 5

theorem aqua_park_earnings :
  (group1_size * admission_cost + group1_size * tour_cost) + (group2_size * admission_cost) = 240 :=
by
  sorry

end aqua_park_earnings_l522_522466


namespace parametric_to_standard_l522_522377

theorem parametric_to_standard (t : ℝ) : 
  (x = (2 + 3 * t) / (1 + t)) ∧ (y = (1 - 2 * t) / (1 + t)) → (3 * x + y - 7 = 0) ∧ (x ≠ 3) := 
by 
  sorry

end parametric_to_standard_l522_522377


namespace inequality_f_2_pow_n_l522_522524

-- Define the function that counts the distinct representations
noncomputable def f (n : ℕ) : ℕ :=
  -- Function definition for illustration. Replace with actual definition.
  sorry

-- Define the theorem stating the required inequality
theorem inequality_f_2_pow_n (n : ℕ) (h : n ≥ 3) : 
  2^((n^2) / 4) < f(2^n) ∧ f(2^n) < 2^((n^2) / 2) := 
by {
  -- Proof omitted
  sorry
}

end inequality_f_2_pow_n_l522_522524


namespace existence_of_good_point_l522_522777

def is_good_point (circle : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ i : ℕ, (0 ≤ i < n) → 
    ((∀ j : ℕ, 0 ≤ j ≤ n-1 → ∑ k in finset.range (j+1), circle ((i + k) % n) > 0) ∧
     (∀ j : ℕ, 0 ≤ j ≤ n-1 → ∑ k in finset.range (j+1), circle ((i + n - k) % n) > 0))

theorem existence_of_good_point :
  ∀ (circle : ℕ → ℤ), (cardinal.mk (finset.filter (λ x : ℕ, circle x = -1) (finset.range 2000)) < 667) →
  ∃ i : ℕ, 0 ≤ i < 2000 ∧ is_good_point circle 2000 :=
begin
  intros circle h,
  sorry
end

end existence_of_good_point_l522_522777


namespace g_ten_l522_522490

-- Define the function g and its properties
def g : ℝ → ℝ := sorry

axiom g_property1 : ∀ x y : ℝ, g (x * y) = 2 * g x * g y
axiom g_property2 : g 0 = 2

-- Prove that g 10 = 1 / 2
theorem g_ten : g 10 = 1 / 2 :=
by
  sorry

end g_ten_l522_522490


namespace union_complement_U_A_B_l522_522220

def U : Set Int := {-1, 0, 1, 2, 3}

def A : Set Int := {-1, 0, 1}

def B : Set Int := {0, 1, 2}

def complement_U_A : Set Int := {u | u ∈ U ∧ u ∉ A}

theorem union_complement_U_A_B : (complement_U_A ∪ B) = {0, 1, 2, 3} :=
by
  sorry

end union_complement_U_A_B_l522_522220


namespace range_of_x_max_value_of_a_l522_522782

-- Condition definitions for the problem
def profit_A (x : ℝ) : ℝ := 12 * (500 - x) * (1 + 0.5 * x / 100)
def profit_B (a x : ℝ) : ℝ := 12 * (a - 13 / 1000 * x) * x

-- Question (1)
theorem range_of_x (x : ℝ) : profit_A x ≥ 12 * 500 → 0 < x ∧ x ≤ 300 :=
by
  sorry

-- Question (2)
theorem max_value_of_a (a x : ℝ) : (∀ x, 0 < x ∧ x ≤ 300 → profit_B a x ≤ profit_A x) → 0 < a ∧ a ≤ 5.5 :=
by
  sorry

end range_of_x_max_value_of_a_l522_522782


namespace solution_satisfies_conditions_l522_522072

-- Define the conditions
def satisfies_functional_eq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x + y) = f(x) * f(y)

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f(x) < f(y)

-- Prove the solution
theorem solution_satisfies_conditions : satisfies_functional_eq (λ x : ℝ, (3 : ℝ)^x) ∧ is_monotonically_increasing (λ x : ℝ, (3 : ℝ)^x) :=
by
  -- Solution is not required, so we add sorry to complete the theorem
  sorry

end solution_satisfies_conditions_l522_522072


namespace independent_events_probability_l522_522899

variable (a b : Type) [ProbabilitySpace a] [ProbabilitySpace b]

theorem independent_events_probability (p : a → ℝ) (pa pb : Set a) :
  p(pa ∩ pb) = 0.16 →
  p(pb) = 2/5 →
  p(pa ∩ pb) = p(pa) * p(pb) →
  p(pa) = 0.4 :=
by
  intros h1 h2 h3
  sorry

end independent_events_probability_l522_522899


namespace total_baseball_fans_l522_522975

variable (Y M R : ℕ)

open Nat

theorem total_baseball_fans (h1 : 3 * M = 2 * Y) 
    (h2 : 4 * R = 5 * M) 
    (h3 : M = 96) : Y + M + R = 360 := by
  sorry

end total_baseball_fans_l522_522975


namespace hyperbola_equation_l522_522206

theorem hyperbola_equation (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
    (h₃ : ∀ (F₁ F₂ : ℝ × ℝ), (F₁.1 ^ 2 + F₂.2 ^ 2 ) = 5 ∧ |F₁ - F₂| = 2 * sqrt 5) 
    (h₄ : ∀ (P : ℝ × ℝ), P = (1, 2) → P ∈ (λ x y, y = (b / a) * x)) 
    (h₅ : (1:ℝ)^2 + (2:ℝ)^2 = 5)
    : (∀ x y, (x^2 - y^2 / 4 = 1)) :=
by sorry

end hyperbola_equation_l522_522206


namespace semi_circle_geometry_AF_squared_plus_BE_squared_eq_AB_squared_l522_522335

theorem semi_circle_geometry_AF_squared_plus_BE_squared_eq_AB_squared
  (A B M C D E F : Point)
  (hAB_diameter : Diameter A B M)
  (hrect_construction : Rectangle A B C D)
  (hC_side_square_inscribed : side AC = side (Square (InscribedCircle AB)))
  (h_CM_DM_intersect : ∃ (M : Point), CM E M ∧ DM F M ∧ CM E AB ∧ DM F AB) :
  AF^2 + BE^2 = AB^2 :=
begin
  sorry
end

end semi_circle_geometry_AF_squared_plus_BE_squared_eq_AB_squared_l522_522335


namespace number_of_traced_squares_in_6x6_grid_l522_522119

theorem number_of_traced_squares_in_6x6_grid :
  let m := 6 
  in (∑ k in finset.range m, (m - k) ^ 2) = 54 := 
by
  -- m is the dimension of the grid (6x6)
  let m := 6 
  -- calculate the number of squares that can be formed
  have : (∑ k in finset.range m, (m - k) ^ 2) = 
    (∑ k in finset.range 6, (6 - k) ^ 2) :=
        rfl
  -- simplify the sum 5^2 + 4^2 + 3^2 + 2^2 + 1^2
  have : (∑ k in finset.range 6, (6 - k) ^ 2) = 25 + 16 + 9 + 4 = 54 := 
        sorry
  sorry

end number_of_traced_squares_in_6x6_grid_l522_522119


namespace number_of_books_l522_522372

theorem number_of_books (pages_per_book total_pages : ℕ) (h1 : pages_per_book = 478) (h2 : total_pages = 3824) : total_pages / pages_per_book = 8 :=
by
  rw [h1, h2]
  norm_num

end number_of_books_l522_522372


namespace min_value_of_f_at_1_l522_522185

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x + (1 / (2 * x))

theorem min_value_of_f_at_1 :
  ∀ (f : ℝ → ℝ),
    (∀ x : ℝ, 0 < x → f x = (1 / 2) * x + 1 / (2 * x)) →
    f 1 = 1 →
    ∀ x : ℝ, 0 < x → ((x = 1) ↔ (f(x) = 1)) :=
by
  sorry

end min_value_of_f_at_1_l522_522185


namespace decompose_one_into_five_unit_fractions_l522_522762

theorem decompose_one_into_five_unit_fractions :
  1 = (1/2) + (1/3) + (1/7) + (1/43) + (1/1806) :=
by
  sorry

end decompose_one_into_five_unit_fractions_l522_522762


namespace congruence_from_overlap_l522_522011

-- Definitions used in the conditions
def figure := Type
def equal_area (f1 f2 : figure) : Prop := sorry
def equal_perimeter (f1 f2 : figure) : Prop := sorry
def equilateral_triangle (f : figure) : Prop := sorry
def can_completely_overlap (f1 f2 : figure) : Prop := sorry

-- Theorem that should be proven
theorem congruence_from_overlap (f1 f2 : figure) (h: can_completely_overlap f1 f2) : f1 = f2 := sorry

end congruence_from_overlap_l522_522011


namespace cannot_form_triangle_l522_522413

theorem cannot_form_triangle (a b c : ℕ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  ¬ ∃ a b c : ℕ, (a, b, c) = (1, 2, 3) := 
  sorry

end cannot_form_triangle_l522_522413


namespace area_of_triangle_formed_by_tangents_l522_522120

theorem area_of_triangle_formed_by_tangents (r : ℝ) :
  let a := 2 * r * (sqrt 3 + 1)
  let S := r^2 * (4 + 2 * sqrt 3) * sqrt 3 in
  S = 2 * r^2 * (2 * sqrt 3 + 3) :=
by {
  let a := 2 * r * (sqrt 3 + 1),
  let S := (a^2 * sqrt 3) / 4,
  calc 
    S = r^2 * (4 + 2 * sqrt 3) * sqrt 3 : sorry
      ... = 2 * r^2 * (2 * sqrt 3 + 3) : sorry
}

end area_of_triangle_formed_by_tangents_l522_522120


namespace lattice_points_circle_ellipse_l522_522903

theorem lattice_points_circle_ellipse (a : ℝ) :
  (∃ n m : ℕ, n = 25 ∧ m = 5 ∧ 
    m = (5 + 2 * int.floor a + 4 * ∑ x in finset.range (int.floor a + 1), int.floor (2 * real.sqrt (1 - (x ^ 2 / a ^ 2))))) →
  22 ≤ a ∧ a < 23 :=
sorry

end lattice_points_circle_ellipse_l522_522903


namespace maximum_sum_product_l522_522710

theorem maximum_sum_product (n : ℕ) (x : Fin n → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ x i)
  (h_sum : ∑ i, x i = 1) :
  (∑ i in Finset.range (n - 1), x i * x (i + 1)) ≤ 1/4 :=
sorry

end maximum_sum_product_l522_522710


namespace area_of_shaded_region_l522_522697

-- Given conditions
def side_length := 8
def area_of_square := side_length * side_length
def area_of_triangle := area_of_square / 4

-- Lean 4 statement for the equivalence
theorem area_of_shaded_region : area_of_triangle = 16 :=
by
  sorry

end area_of_shaded_region_l522_522697


namespace number_of_umbrella_numbers_is_40_l522_522960

def is_umbrella_number (n : ℕ) : Prop :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  d1 > d0 ∧ d1 > d2

def form_three_digit_numbers (d : Set ℕ) : ℕ := 
  d.toList.permutations.filter (λ l => l.length = 3).map (λ l => l.head! * 100 + l.tail!.head! * 10 + l.tail!.tail!.head!).filter is_umbrella_number

theorem number_of_umbrella_numbers_is_40 : form_three_digit_numbers {1, 2, 3, 4, 5, 6} = 40 :=
by sorry

end number_of_umbrella_numbers_is_40_l522_522960


namespace sara_pumpkins_l522_522675

theorem sara_pumpkins : 
  (initial_pumpkins rabbits_ate : ℕ)-> initial_pumpkins = 43 -> rabbits_ate = 23 -> initial_pumpkins - rabbits_ate = 20 :=
by
  intros initial_pumpkins rabbits_ate h1 h2
  rw [h1, h2]
  simp
  done

end sara_pumpkins_l522_522675


namespace line_AB_l522_522572

-- Statements for circles and intersection
def circle_C1 (x y: ℝ) : Prop := x^2 + y^2 = 1
def circle_C2 (x y: ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 1

-- Points A and B are defined as the intersection points of circles C1 and C2
axiom A (x y: ℝ) : circle_C1 x y ∧ circle_C2 x y
axiom B (x y: ℝ) : circle_C1 x y ∧ circle_C2 x y

-- The goal is to prove that the line passing through points A and B has the equation x - y = 0
theorem line_AB (x y: ℝ) : circle_C1 x y → circle_C2 x y → (x - y = 0) :=
by
  sorry

end line_AB_l522_522572


namespace arcsin_sqrt_three_over_two_l522_522481

theorem arcsin_sqrt_three_over_two : Real.arcsin (Real.sqrt 3 / 2) = Real.pi / 3 :=
by
  -- The proof is omitted
  sorry

end arcsin_sqrt_three_over_two_l522_522481


namespace equation_relating_price_and_tax_and_discount_l522_522602

variable (c t d : ℚ)

theorem equation_relating_price_and_tax_and_discount
  (h1 : 1.30 * c * ((100 + t) / 100) * ((100 - d) / 100) = 351) :
    1.30 * c * (100 + t) * (100 - d) = 3510000 := by
  sorry

end equation_relating_price_and_tax_and_discount_l522_522602


namespace area_of_isosceles_triangle_l522_522698

theorem area_of_isosceles_triangle (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : 
  let r := 1 in
  let base := 2 * r in
  let height := r / sin θ in
  let area := 1 / (sin θ * cos θ) in
  area = (1 / (sin θ * cos θ)) :=
by
  sorry

end area_of_isosceles_triangle_l522_522698


namespace tori_original_height_l522_522395

-- Definitions for given conditions
def current_height : ℝ := 7.26
def height_gained : ℝ := 2.86

-- Theorem statement
theorem tori_original_height : current_height - height_gained = 4.40 :=
by sorry

end tori_original_height_l522_522395


namespace minimum_distance_midpoint_to_line_l522_522986

noncomputable def P_coordinates : ℝ × ℝ :=
  (3 * Real.cos (Real.pi / 4), 3 * Real.sin (Real.pi / 4))

def curve_C (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.cos (θ - Real.pi / 4)

def line_l (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.cos θ + 4 * ρ * Real.sin θ = Real.sqrt 2

def midpoint_M (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.sqrt 2 / 2 + Real.cos θ, Real.sqrt 2 / 2 + Real.sin θ)

def distance_to_line (M : ℝ × ℝ) : ℝ :=
  Real.abs (2 * M.1 + 4 * M.2 - Real.sqrt 2) / Real.sqrt (2^2 + 4^2)

theorem minimum_distance_midpoint_to_line :
  ∀ θ : ℝ, let P := P_coordinates in
  let Q := point_Q θ in
  let M := midpoint_M P Q in
  distance_to_line M = (Real.sqrt 10 - 1) / 2 :=
begin
  sorry
end

end minimum_distance_midpoint_to_line_l522_522986


namespace ball_falls_in_hole_l522_522791

theorem ball_falls_in_hole (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (m n : ℤ), n ≠ 0 ∧ a / b = m / n :=
begin
  -- mathematical proof goes here
  sorry
end

end ball_falls_in_hole_l522_522791


namespace no_positive_x_satisfies_equation_l522_522502

theorem no_positive_x_satisfies_equation : 
  ¬ ∃ (x : ℝ), (0 < x) ∧ (log 4 x * log x 9 = 2 * log 4 9) := 
sorry

end no_positive_x_satisfies_equation_l522_522502


namespace irrational_numbers_not_all_square_roots_l522_522415

theorem irrational_numbers_not_all_square_roots :
  (∀ x y : ℝ, x ≠ y → ∠ (x, y) = ∠ (y, x)) → 
  (∀ z : ℝ, ∃! w : ℝ, z = w) → 
  (∀ l₁ l₂ : Line, (∀ t : Transversal, alternateInteriorAnglesEqual t l₁ l₂ → parallel l₁ l₂)) →
  ¬ (∀ a : ℝ, irrational a → ¬ ∃ b : ℝ, a = sqrt b) :=
by
  intros h_angle h_number_line h_parallel
  sorry

end irrational_numbers_not_all_square_roots_l522_522415


namespace sum_first_n_terms_l522_522976

-- Define the sequence a_n
def geom_seq (a : ℕ → ℕ) (r : ℕ) : Prop :=
  ∀ n, a (n + 1) = r * a n

-- Define the main conditions from the problem
axiom a7_cond (a : ℕ → ℕ) : a 7 = 8 * a 4
axiom arithmetic_seq_cond (a : ℕ → ℕ) : (1 / 2 : ℝ) * a 2 < (a 3 - 4) ∧ (a 3 - 4) < (a 4 - 12)

-- Define the sequences a_n and b_n using the conditions
def a_n (n : ℕ) : ℕ := 2^(n + 1)
def b_n (n : ℕ) : ℤ := (-1)^n * (Int.ofNat (n + 1))

-- Define the sum of the first n terms of b_n
noncomputable def T_n (n : ℕ) : ℤ :=
  (Finset.range n).sum b_n

-- Main theorem statement
theorem sum_first_n_terms (k : ℕ) : |T_n k| = 20 → k = 40 ∨ k = 37 :=
sorry

end sum_first_n_terms_l522_522976


namespace solve_equation_and_compute_l522_522682

theorem solve_equation_and_compute :
  ∃ y : ℤ, 3 * (y^2 + 6 * y + 12) = 777 ∧ 4 * y - 8 = 2 * y + 18 :=
begin
  sorry
end

end solve_equation_and_compute_l522_522682


namespace problem_I_problem_II_l522_522535

-- Definitions
def p (x : ℝ) : Prop := (x + 2) * (x - 3) ≤ 0
def q (m : ℝ) (x : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

-- Problem (I)
theorem problem_I (m : ℝ) : m > 0 → (∀ x : ℝ, q m x → p x) → 0 < m ∧ m ≤ 2 := by
  sorry

-- Problem (II)
theorem problem_II (x : ℝ) : 7 > 0 → 
  (p x ∨ q 7 x) ∧ ¬(p x ∧ q 7 x) → 
  (-6 ≤ x ∧ x < -2) ∨ (3 < x ∧ x ≤ 8) := by
  sorry

end problem_I_problem_II_l522_522535


namespace check_basis_l522_522802

structure Vector2D :=
  (x : ℤ)
  (y : ℤ)

def are_collinear (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.y - v2.x * v1.y = 0

def can_be_basis (v1 v2 : Vector2D) : Prop :=
  ¬ are_collinear v1 v2

theorem check_basis :
  can_be_basis ⟨-1, 2⟩ ⟨5, 7⟩ ∧
  ¬ can_be_basis ⟨0, 0⟩ ⟨1, -2⟩ ∧
  ¬ can_be_basis ⟨3, 5⟩ ⟨6, 10⟩ ∧
  ¬ can_be_basis ⟨2, -3⟩ ⟨(1 : ℤ)/2, -(3 : ℤ)/4⟩ :=
by
  sorry

end check_basis_l522_522802


namespace KW_is_100_percent_more_than_B_l522_522818

-- Define the assets of companies A and B and the price of Company KW
variables (A B KW : Real)
-- Define the condition that KW is 20% more than A's assets
def price_of_KW_is_20_percent_more_than_A := KW = 1.20 * A

-- Define the condition that KW is 75% of the combined assets of A and B
def KW_is_75_percent_of_A_and_B_combined := KW = 0.75 * (A + B)

-- Define the percentage by which the price of KW is more than B's assets
def percentage_more_than_B := 1.0  -- Representing 100%

-- The goal to prove
theorem KW_is_100_percent_more_than_B
    (h1 : KW = 1.20 * A)
    (h2 : KW = 0.75 * (A + B)) :
    KW = B + percentage_more_than_B * B :=
by 
    calc KW = 1.20 * A : h1
       ... = 0.75 * (A + B) : h2
       ... = B + 1.0 * B : sorry

end KW_is_100_percent_more_than_B_l522_522818


namespace concurrency_or_parallel_l522_522900

/-- Given that quadrilateral ABCD is a convex quadrilateral, and P, Q, R, S are points on sides 
    AB, BC, CD, and DA respectively, with PR and QS intersecting at point O. If quadrilaterals 
    APOS, BQOP, CROQ, and DSOR each have an incircle, then the lines AC, PQ, and RS are either 
    concurrent or parallel. -/
theorem concurrency_or_parallel {A B C D P Q R S O : Point}
  (AB_convex : convex_quad A B C D)
  (P_on_AB : lies_on P (segment A B))
  (Q_on_BC : lies_on Q (segment B C))
  (R_on_CD : lies_on R (segment C D))
  (S_on_DA : lies_on S (segment D A))
  (O_on_PR_QS : intersection O (line PR) (line QS))
  (APOS_incircle : exists incircle APOS)
  (BQOP_incircle : exists incircle BQOP)
  (CROQ_incircle : exists incircle CROQ)
  (DSOR_incircle : exists incircle DSOR)
  : concurrent_or_parallel (line AC) (line PQ) (line RS) :=
sorry

end concurrency_or_parallel_l522_522900


namespace length_AC_is_12_sqrt_3_l522_522616

noncomputable def length_of_diagonal_AC
    (A B C D : Type)
    (AB CD BC DA : ℝ)
    (angle_ADC : ℝ) 
    (h_AB : AB = 12) 
    (h_CD : CD = 12)
    (h_BC : BC = 15)
    (h_DA : DA = 15)
    (h_angle_ADC : angle_ADC = 120)
    : ℝ :=
  let AD := 12
  let CD := 12
  let cos_120 := -1 / 2
  real.sqrt (AD * AD + CD * CD - 2 * AD * CD * cos_120)

-- Now we state the theorem to be proved
theorem length_AC_is_12_sqrt_3
    (A B C D : Type)
    (AB CD BC DA : ℝ)
    (angle_ADC : ℝ)
    (h_AB : AB = 12)
    (h_CD : CD = 12)
    (h_BC : BC = 15)
    (h_DA : DA = 15)
    (h_angle_ADC : angle_ADC = 120) :
  length_of_diagonal_AC A B C D AB CD BC DA angle_ADC h_AB h_CD h_BC h_DA h_angle_ADC = 12 * real.sqrt 3 :=
sorry

end length_AC_is_12_sqrt_3_l522_522616


namespace rice_weight_employees_l522_522460

noncomputable def rice_less_than_9_9 (n : ℕ) (ξ : ℝ → ℝ) (σ : ℝ) : ℝ :=
let p_9_9 : ℝ := (1 - 0.96) / 2 in
n * p_9_9

theorem rice_weight_employees : rice_less_than_9_9 2000 (λ x, pdf (NormalDist.mk 10 σ ^ 2) x) = 40 :=
by
  sorry

end rice_weight_employees_l522_522460


namespace decimal_to_base4_100_has_4_digits_l522_522271

theorem decimal_to_base4_100_has_4_digits : ∀ (x : ℕ), x = 100 → (Nat.digits 4 x).length = 4 :=
by
  intros x hx
  rw hx
  sorry

end decimal_to_base4_100_has_4_digits_l522_522271


namespace donuts_selection_count_l522_522473

noncomputable def donuts_selection_combinations : ℕ :=
  let combinations := (finset.card (finset.Icc (1, 1, 1, 1, 1) (9, 9, 9, 9, 9)))
  combinations

theorem donuts_selection_count : donuts_selection_combinations = 70 :=
sorry

end donuts_selection_count_l522_522473


namespace inequality_holds_h_increasing_on_1_to_infty_l522_522345

theorem inequality_holds (x : ℝ) (hx : 1 ≤ x) : 
    x / 2 ≥ (x - 1) / (x + 1) :=
sorry

theorem h_increasing_on_1_to_infty (x : ℝ) (hx : 1 ≤ x) : 
    monotone_on (λ x, real.log x - (x - 1)) (set.Ici 1) :=
sorry

end inequality_holds_h_increasing_on_1_to_infty_l522_522345


namespace total_earnings_l522_522580

theorem total_earnings : 
  let wage : ℕ := 10
  let hours_monday : ℕ := 7
  let tips_monday : ℕ := 18
  let hours_tuesday : ℕ := 5
  let tips_tuesday : ℕ := 12
  let hours_wednesday : ℕ := 7
  let tips_wednesday : ℕ := 20
  let total_hours : ℕ := hours_monday + hours_tuesday + hours_wednesday
  let earnings_from_wage : ℕ := total_hours * wage
  let total_tips : ℕ := tips_monday + tips_tuesday + tips_wednesday
  let total_earnings : ℕ := earnings_from_wage + total_tips
  total_earnings = 240 :=
by
  sorry

end total_earnings_l522_522580


namespace money_per_postcard_l522_522737

def postcards_per_day : ℕ := 30
def days : ℕ := 6
def total_earning : ℕ := 900
def total_postcards := postcards_per_day * days
def price_per_postcard := total_earning / total_postcards

theorem money_per_postcard :
  price_per_postcard = 5 := 
sorry

end money_per_postcard_l522_522737


namespace range_of_x_for_f_positive_l522_522560

variable (a : ℝ) (x : ℝ)

theorem range_of_x_for_f_positive (h1 : 0 < a ∧ a < 1) : 
  (log 2 (a^(2 * x) - 4 * a^x + 1)) > 0 ↔ x > 2 * (log 2 4) / (log 2 a) :=
sorry

end range_of_x_for_f_positive_l522_522560


namespace trapezoid_perimeter_l522_522622

variables (A B C D : Type)
variables (AB CD BC : ℝ)
variables (h : ℝ)

-- Given conditions
def isosceles_trapezoid (AB CD BC : ℝ) (h : ℝ) : Prop :=
  AB = 12 ∧ CD = 12 ∧ h = AB / 2 ∧ BC = 5

-- Prove that the perimeter = 34
theorem trapezoid_perimeter (AB CD BC : ℝ) (h : ℝ) :
  isosceles_trapezoid AB CD BC h →
  AB + BC + CD + (sqrt (BC^2 - h^2)) = 34 :=
sorry

end trapezoid_perimeter_l522_522622


namespace bernoulli_poly_deriv_l522_522668

theorem bernoulli_poly_deriv (B : ℕ → ℤ) (z : ℂ) :
  (∀ n, B n z = ∑ k in finset.range (n + 1), nat.choose n k * B k * z^(n - k)) →
  (∀ n, deriv (B n z) = n * B (n-1) z) :=
by
  sorry

end bernoulli_poly_deriv_l522_522668


namespace sixty_first_permutation_l522_522731

noncomputable def find_61st_permutation : Nat :=
  let digits := [1, 4, 5, 7, 8]
  let permutations := List.permutations digits
  let sorted_permutations := List.sorted permutations (≤)
  sorted_permutations.get! 60  -- 61st element (index is 0-based)

theorem sixty_first_permutation :
  find_61st_permutation = [5, 7, 1, 4, 8] :=
by
  sorry

end sixty_first_permutation_l522_522731


namespace arcsin_sqrt_three_over_two_l522_522482

theorem arcsin_sqrt_three_over_two : Real.arcsin (Real.sqrt 3 / 2) = Real.pi / 3 :=
by
  -- The proof is omitted
  sorry

end arcsin_sqrt_three_over_two_l522_522482


namespace measure_angle_A_l522_522994

open Real.Angle

theorem measure_angle_A (B C A : Real.Angle) (hB : B = 18) (hC : C = 3 * B) (hSum : A + B + C = 180) :
  A = 108 :=
by sorry

end measure_angle_A_l522_522994


namespace larger_square_side_length_l522_522380

variable (P : ℕ) -- Perimeter of the smaller square paper.
variable (H : ℕ) -- Height of the computer screen.
variable (S : ℕ) -- Side length of the smaller square paper.
variable (L : ℕ) -- Side length of the larger square paper.

-- Conditions
def condition1 := P = H - 20
def condition2 := S = 20
def condition3 := 4 * S = P
def condition4 := 4 * L = H

-- Proof statement
theorem larger_square_side_length : 
  condition1 → condition2 → condition3 → condition4 → L = 25 := 
by 
  intro h1 h2 h3 h4
  sorry

end larger_square_side_length_l522_522380


namespace trapezoid_diagonal_l522_522489

noncomputable def length_of_diagonal (long_base short_base leg : ℕ) : ℕ :=
  nat.sqrt (long_base^2 + (leg^2 - ((long_base - short_base)/2)^2))

theorem trapezoid_diagonal
  (long_base short_base leg diagonal : ℕ)
  (h_long_base : long_base = 24)
  (h_short_base : short_base = 10)
  (h_leg : leg = 11)
  (h_diagonal : diagonal = 19) :
  length_of_diagonal long_base short_base leg = diagonal :=
by
  rw [h_long_base, h_short_base, h_leg, h_diagonal]
  exact sorry

end trapezoid_diagonal_l522_522489


namespace point_D_not_on_graph_l522_522416

-- Define the function f
def f : ℝ → ℝ := λ x, (x - 1) / (x + 2)

-- Define the point
def point_not_on_graph (x y : ℝ) := y ≠ f x

-- The theorem to prove
theorem point_D_not_on_graph : point_not_on_graph (-2) 1 :=
by sorry

end point_D_not_on_graph_l522_522416


namespace number_of_integers_divisible_by_10_l522_522231

theorem number_of_integers_divisible_by_10 (a b : ℕ) (h1 : a = 100) (h2 : b = 500) :
  ∃ n : ℕ, n = 41 ∧ ∀ k : ℕ, (100 ≤ k ∧ k ≤ 500 ∧ k % 10 = 0) ↔  ∃ i : ℕ, (1 ≤ i ∧ i ≤ n ∧ k = 100 + (i - 1) * 10) :=
by
  let n := 41
  existsi n
  split
  { -- n = 41
    refl },
  { -- Main equivalence part
    assume k,
    split
    { -- Left to right ↔
      intro hk
      rcases hk with ⟨h100k, hk500, hkmod⟩
      use ((k - 100) / 10 + 1) 
      split
      { -- 1 ≤ i
        linarith [(k - 100) / 10], },
      { -- i ≤ n
        linarith [div_le_iff_lt h100k], }, 
      { -- k = 100 + (i - 1) * 10
        calc 
        k = 100 + 10 * ((k - 100) / 10) : by linarith [(k - 100) / 10 * 10]
        ... = 100 + 10 * ((k - 100) / 10) : by ring, } }, 
    { -- Right to left ↔
      rintro ⟨i, rangei_ltor,_⟩
      obtain ⟨rangei,_⟩ := rangei_ltor
      rcases rangei with ⟨range1,_⟩
      split
      { -- 100 ≤ k
        linarith }
      { -- k ≤ 500
        linarith },
      { -- k % 10 = 0
        calc 
        k = 100 + 10 * (i - 1) : _,
      rw add_mul,
      norm_cast,
      ring } } sorry

end number_of_integers_divisible_by_10_l522_522231


namespace find_number_l522_522244

theorem find_number (N p q : ℝ) (h₁ : N / p = 8) (h₂ : N / q = 18) (h₃ : p - q = 0.2777777777777778) : N = 4 :=
sorry

end find_number_l522_522244


namespace supporters_second_team_percentage_l522_522839

theorem supporters_second_team_percentage (n : ℕ) (p1 : ℕ) (p_not_either : ℕ) 
  (h_n : n = 50) (h_p1_percentage : p1 = Nat.mul 50 40 / 100) (h_p_not_either : p_not_either = 3) : 
  (50 - p1 - p_not_either) * 100 / 50 = 54 :=
by 
  rw [h_n, h_p1_percentage, h_p_not_either]
  norm_num
  -- Additional details can be filled in here to complete the proof
  sorry

end supporters_second_team_percentage_l522_522839


namespace ratio_of_c_and_d_l522_522249

theorem ratio_of_c_and_d
  (x y c d : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hd : d ≠ 0) 
  (h1 : 8 * x - 6 * y = c)
  (h2 : 9 * y - 12 * x = d) :
  c / d = -2 / 3 := 
  sorry

end ratio_of_c_and_d_l522_522249


namespace sufficient_condition_for_root_l522_522667

theorem sufficient_condition_for_root (m : ℝ) (f : ℝ → ℝ) (h1 : m > 7) (h2 : f = λ x, x^2 + m * x + 9) : 
  (∃ x, f x = 0) ↔ m > 7 :=
sorry

end sufficient_condition_for_root_l522_522667


namespace ratio_of_radii_l522_522610

theorem ratio_of_radii (a b : ℝ) (h1 : π * b^2 - π * a^2 = 4 * π * a^2) : a / b = Real.sqrt(5) / 5 := by
  sorry

end ratio_of_radii_l522_522610


namespace general_formula_sequence_sum_first_n_terms_inequality_sequence_l522_522154

-- Proof Problem 1
theorem general_formula_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, 0 < n → 2 * n * S (n + 1) - 2 * (n + 1) * S n = n^2 + n) :
  a = (λ n, n) := sorry

-- Proof Problem 2
theorem sum_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℚ) (T : ℕ → ℚ) (h1 : ∀ n : ℕ, b n = n / (2 * (n + 3) * S n))
  (h2 : ∀ n : ℕ, T n = ∑ i in finset.range n, b i)
  (h3 : ∀ n : ℕ, S n = n * (n + 1) / 2) :
  ∀ n : ℕ, T n =  5/12 - (2 * n + 5) / (2 * (n + 2) * (n + 3)) := sorry

-- Proof Problem 3
theorem inequality_sequence (a : ℕ → ℕ) (h1 : ∀ n : ℕ, a n = n) :
  ∀ n : ℕ, 2 ≤ n → ∑ i in finset.range (n - 1), 1 / (a (i + 2)^3 : ℚ) < 1 / 4 := sorry

end general_formula_sequence_sum_first_n_terms_inequality_sequence_l522_522154


namespace modulus_of_squared_complex_l522_522195

theorem modulus_of_squared_complex (z : ℂ) (h : z = (2 - (1 : ℂ).im) ^ 2) : complex.abs z = 5 :=
sorry

end modulus_of_squared_complex_l522_522195


namespace youngest_brother_age_difference_l522_522673

def Rick_age : ℕ := 15
def Oldest_brother_age : ℕ := 2 * Rick_age
def Middle_brother_age : ℕ := Oldest_brother_age / 3
def Smallest_brother_age : ℕ := Middle_brother_age / 2
def Youngest_brother_age : ℕ := 3

theorem youngest_brother_age_difference :
  Smallest_brother_age - Youngest_brother_age = 2 :=
by
  -- sorry to skip the proof
  sorry

end youngest_brother_age_difference_l522_522673


namespace rational_squares_sum_l522_522669

theorem rational_squares_sum (n : ℕ) (h : 0 < n) :
  ∃ S : fin n → ℚ, (∀ i j, i ≠ j → S i ≠ S j) ∧ (∀ i, 0 < S i) ∧ (∑ i in finset.univ.fin n, (S i)^2) = n :=
sorry

end rational_squares_sum_l522_522669


namespace rupert_jumps_more_l522_522348

theorem rupert_jumps_more (Ronald_jumps Rupert_jumps total_jumps : ℕ)
  (h1 : Ronald_jumps = 157)
  (h2 : total_jumps = 243)
  (h3 : Rupert_jumps + Ronald_jumps = total_jumps) :
  Rupert_jumps - Ronald_jumps = 86 :=
by
  sorry

end rupert_jumps_more_l522_522348


namespace find_m_l522_522242

-- Define the lines l1 and l2
def line1 (x y : ℝ) (m : ℝ) : Prop := x + m^2 * y + 6 = 0
def line2 (x y : ℝ) (m : ℝ) : Prop := (m - 2) * x + 3 * m * y + 2 * m = 0

-- The statement that two lines are parallel
def lines_parallel (m : ℝ) : Prop :=
  ∀ (x y : ℝ), line1 x y m → line2 x y m

-- The mathematically equivalent proof problem
theorem find_m (m : ℝ) (H_parallel : lines_parallel m) : m = 0 ∨ m = -1 :=
sorry

end find_m_l522_522242


namespace period_of_sine_3x_pipluspi_l522_522115

noncomputable def period_of_sine_function_coefficient (a : ℝ) : ℝ :=
  let standard_period := 2 * Real.pi
  in standard_period / a

theorem period_of_sine_3x_pipluspi : period_of_sine_function_coefficient 3 = 2 * Real.pi / 3 := by
  sorry

end period_of_sine_3x_pipluspi_l522_522115


namespace extremum_point_iff_nonnegative_condition_l522_522203

noncomputable def f (a x : ℝ) : ℝ := Real.log (1 + x) - (a * x) / (x + 1)

theorem extremum_point_iff (a : ℝ) (h : 0 < a) :
  (∃ (x : ℝ), x = 1 ∧ ∀ (f' : ℝ), f' = (1 + x - a) / (x + 1)^2 ∧ f' = 0) ↔ a = 2 :=
by
  sorry

theorem nonnegative_condition (a : ℝ) (h0 : 0 < a) :
  (∀ (x : ℝ), x ∈ Set.Ici 0 → f a x ≥ 0) ↔ 0 < a ∧ a ≤ 1 :=
by
  sorry

end extremum_point_iff_nonnegative_condition_l522_522203


namespace least_addition_divisible_by_9_l522_522768

theorem least_addition_divisible_by_9 :
  let current_savings := 642986
  let additions_needed := 1
  divisible_by_9 := (current_savings + additions_needed) % 9 = 0
  ∃ addition, ∀ x, x < addition -> (current_savings + x) % 9 ≠ 0 := 
by
  let current_savings := 642986 in
  let additions_needed := 1 in
  let divisible_by_9 := (current_savings + additions_needed) % 9 = 0 in
  trivial

end least_addition_divisible_by_9_l522_522768


namespace smallest_sum_of_three_diff_numbers_l522_522097

theorem smallest_sum_of_three_diff_numbers :
  ∃ a b c ∈ ({0, 5, -2, 18, -4, 3} : set ℤ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a + b + c = -6) := 
by { sorry }

end smallest_sum_of_three_diff_numbers_l522_522097


namespace black_greater_than_gray_by_103_l522_522989

def a := 12
def b := 9
def c := 7
def d := 3

def area (side: ℕ) := side * side

def black_area_sum : ℕ := area a + area c
def gray_area_sum : ℕ := area b + area d

theorem black_greater_than_gray_by_103 :
  black_area_sum - gray_area_sum = 103 := by
  sorry

end black_greater_than_gray_by_103_l522_522989


namespace problem_1_problem_2_ln_ineq_problem_3_l522_522488

noncomputable def a : ℕ → ℝ
| 1       := 1
| (n + 1) := (1 + 1 / (n^2 + n)) * a n + 1 / (2^n)

noncomputable def b : ℕ → ℝ
| n := if h : n > 0 then (a (n + 1) - a n) / a n else 0

theorem problem_1 : ∀ (n : ℕ), n ≥ 2 → a n ≥ 2 := by sorry

theorem problem_2 : ∀ (n : ℕ), ∑ i in finset.range n, b (i + 1) < 7 / 4 := by sorry

theorem ln_ineq (x : ℝ) (hx : x > 0) : real.log (1 + x) < x :=
begin
  -- Here you'd include the mathematical proof for the given inequality,
  -- but it is asserted as a true theorem in the problem setup.
  sorry
end

theorem problem_3 : ∀ (n : ℕ), a n < 2 * real.exp (3 / 4) := by sorry

end problem_1_problem_2_ln_ineq_problem_3_l522_522488


namespace bisects_segment_proof_l522_522573

structure IntersectionData where
  (a b : Line)
  (P : Point)
  (P_not_on_a : P ∉ a.points)
  (P_not_on_b : P ∉ b.points)
  (P_not_on_f : P ∉ angle_bisector(a, b).points)
  (f : Line := angle_bisector(a, b))
  (A : Point := a.intersection(P.perpendicular))
  (F : Point := f.intersection(a.perpendicular(A)))
  (G H : Point)
  (g : Line := P.perpendicular_to(F))

def bisects_segment (p q r : Point) : Prop :=
  ∃ (m : Point), dist p m = dist m q ∧ Line.contains r m

theorem bisects_segment_proof
  (data : IntersectionData)
  (GH_exists : data.G ∈ data.g.points ∧ data.H ∈ data.g.points)
  (G_on_a : data.G ∈ data.a.points)
  (H_on_b : data.H ∈ data.b.points)
  : bisects_segment data.G data.H data.P := by
  sorry

end bisects_segment_proof_l522_522573


namespace f_sub_f_succ_l522_522147

def f (n : ℕ) : ℝ :=
  (Finset.range (3 * n - 1)).sum (λ i, 1 / (i + n + 1))

theorem f_sub_f_succ (k : ℕ) (hk : 0 < k) :
  f (k + 1) - f k = (1 / (3 * k) + 1 / (3 * k + 1) + 1 / (3 * k + 2) - 1 / (k + 1)) :=
by
  sorry

end f_sub_f_succ_l522_522147


namespace orthocenter_intersection_l522_522428

open EuclideanGeometry
open Triangle

variables (A B C D K M : Point)

theorem orthocenter_intersection
  (h1 : ∠BDA = 90°)
  (h2 : ∠CDA = 90°)
  (h3 : line_intersect_circle_again (A, C) B K)
  (h4 : line_intersect_circle_again (A, B) C M) :
  concurrent (line B M) (line C K) (line A D) :=
sorry

end orthocenter_intersection_l522_522428


namespace PQRS_is_rhombus_l522_522641

theorem PQRS_is_rhombus 
  (A B C D E P Q R S : Point) 
  (h_parallelogram : parallelogram A B C D)
  (h_diagonals_intersect : intersect_at_diagonals E A C B D)
  (h_P : circumcenter A B E P)
  (h_Q : circumcenter B C E Q)
  (h_R : circumcenter C D E R)
  (h_S : circumcenter A D E S) :
  rhombus P Q R S := 
sorry

end PQRS_is_rhombus_l522_522641


namespace main_theorem_l522_522108

noncomputable def problem_statement : Prop :=
  let a := cos^2 (π / 9)
  let b := cos^2 (2 * π / 9)
  let c := cos^2 (4 * π / 9)
  cubic_polynomial : (x - a) * (x - b) * (x - c) = x^3 - (9 / 2) * x^2 + (27 / 16) * x - (1 / 16)
  ∧ sqrt((3 - a) * (3 - b) * (3 - c)) = (5 * sqrt(5)) / 4

theorem main_theorem : problem_statement :=
  by
    sorry

end main_theorem_l522_522108


namespace pattys_sandwich_shop_cost_l522_522807

theorem pattys_sandwich_shop_cost :
  let sandwich_cost := 4
  let soda_cost := 3
  let num_sandwiches := 7
  let num_sodas := 5
  let total_cost := num_sandwiches * sandwich_cost + num_sodas * soda_cost
  total_cost < 50 →
  total_cost = 43 := by
  let sandwich_cost := 4
  let soda_cost := 3
  let num_sandwiches := 7
  let num_sodas := 5
  let total_cost := num_sandwiches * sandwich_cost + num_sodas * soda_cost
  intros h
  change total_cost = 43 -- restates the goal in terms of total_cost
  simp [sandwich_cost, soda_cost, num_sandwiches, num_sodas, total_cost]
  sorry

end pattys_sandwich_shop_cost_l522_522807


namespace power_function_sqrt_l522_522217

theorem power_function_sqrt (α : ℝ) (f : ℝ → ℝ) (h_def : ∀ x : ℝ, f x = x^α) (h_point : f 2 = Real.sqrt 2) : ∀ x, f x = Real.sqrt x :=
by
  have h_power : 2^α = Real.sqrt 2 := by rw [h_def 2, h_point]
  sorry

end power_function_sqrt_l522_522217


namespace sum_of_coeffs_l522_522875

theorem sum_of_coeffs (a0 a1 a2 a3 a4 a5 : ℤ)
  (h1 : (1 - 2 * (0 : ℤ))^5 = a0)
  (h2 : (1 - 2 * (1 : ℤ))^5 = a0 + a1 + a2 + a3 + a4 + a5) :
  a1 + a2 + a3 + a4 + a5 = -2 := by
  sorry

end sum_of_coeffs_l522_522875


namespace sum_of_three_squares_l522_522462

theorem sum_of_three_squares (a b c : ℤ) (h1 : 2 * a + 2 * b + c = 27) (h2 : a + 3 * b + c = 25) : 3 * c = 33 :=
  sorry

end sum_of_three_squares_l522_522462


namespace f_geq_1_for_all_x_l522_522151

noncomputable def y (a : ℝ) (x : ℝ) : ℝ := (1/2) * (a^x + a^(-x))

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  let y_val := y a x
  in y_val + real.sqrt(y_val^2 - 1)

theorem f_geq_1_for_all_x (a : ℝ) (x : ℝ) (h : a > 1) : f a x ≥ 1 :=
sorry

end f_geq_1_for_all_x_l522_522151


namespace inverse_proportion_quadrants_l522_522966

theorem inverse_proportion_quadrants (k : ℝ) :
  (∀ (x : ℝ), x ≠ 0 → ((x > 0 → (k - 3) / x > 0) ∧ (x < 0 → (k - 3) / x < 0))) → k > 3 :=
by
  intros h
  sorry

end inverse_proportion_quadrants_l522_522966


namespace odds_against_y_winning_l522_522981

/- 
   Define the conditions: 
   odds_w: odds against W winning is 4:1
   odds_x: odds against X winning is 5:3
-/
def odds_w : ℚ := 4 / 1
def odds_x : ℚ := 5 / 3

/- 
   Calculate the odds against Y winning 
-/
theorem odds_against_y_winning : 
  (4 / (4 + 1)) + (5 / (5 + 3)) < 1 ∧
  (1 - ((4 / (4 + 1)) + (5 / (5 + 3)))) = 17 / 40 ∧
  ((1 - (17 / 40)) / (17 / 40)) = 23 / 17 := by
  sorry

end odds_against_y_winning_l522_522981


namespace polynomial_degree_and_type_l522_522383

def is_degree (p : Polynomial ℕ) (d : ℕ) : Prop :=
  p.degree = d

def is_trinomial (p : Polynomial ℕ) : Prop :=
  p.support.card = 3

theorem polynomial_degree_and_type (x y : ℕ) :
  is_degree (monomial 3 1 + monomial 2 1 + monomial 0 (-2)) 3 ∧ is_trinomial (monomial 3 1 + monomial 2 1 + monomial 0 (-2)) :=
by sorry

end polynomial_degree_and_type_l522_522383


namespace sum_of_cubes_l522_522576

-- Conditions as definitions in Lean 4
def sum_of_n (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_of_products_n_nplus1 (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 3

def sum_of_products_n_nplus1_nplus2 (n : ℕ) : ℕ := n * (n + 1) * (n + 2) * (n + 3) / 4

-- The main proof problem statement
theorem sum_of_cubes (n : ℕ) :
  (∑ k in Finset.range (n + 1), k^3) = n^2 * (n + 1)^2 / 4 :=
by
  sorry

end sum_of_cubes_l522_522576


namespace Q1_subsets_of_B_Q2_range_of_a_l522_522219

open Set

def A := { x : ℝ | 2^(x-6) ≤ 2^(-2*x) ∧ 2^(-2*x) ≤ 1 }
def B := { x : ℕ | x ∈ A }
def C (a : ℝ) := { x : ℝ | a ≤ x ∧ x ≤ a + 1 }

theorem Q1_subsets_of_B : B = {0, 1, 2} ∧ 
  (∀ s, s ⊆ B ↔ s ∈ ({∅, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}} : set (set ℕ))) :=
by
  sorry

theorem Q2_range_of_a (a : ℝ) (h : A ∩ C a = C a) : 0 ≤ a ∧ a ≤ 1 :=
by
  sorry

end Q1_subsets_of_B_Q2_range_of_a_l522_522219


namespace student_failed_by_l522_522060

-- Define the conditions explicitly as Lean definitions
def passing_criterion_percentage : ℝ := 0.6
def student_marks : ℕ := 80
def max_marks : ℕ := 200
def passing_marks : ℕ := (passing_criterion_percentage * max_marks).to_nat

-- Theorem stating by how many marks the student failed the test
theorem student_failed_by : (passing_marks - student_marks) = 40 := by
  -- No implementation of the proof
  sorry

end student_failed_by_l522_522060


namespace square_pyramid_planes_l522_522234

theorem square_pyramid_planes : 
  let edges := [(1,2), (2,3), (3,4), (4,1), (5,1), (5,2), (5,3), (5,4)] in
  let pairs := (⊠) edges edges in
  (count_valid_plane_determining_pairs pairs = 22) :=  sorry

def pair_edges (e₁ e₂ : ℕ × ℕ) : Prop :=
  -- define the conditions under which two edges determine a plane
  sorry

def count_valid_plane_determining_pairs (pairs : List ((ℕ × ℕ) × (ℕ × ℕ))) : ℕ :=
  (pairs.filter (λ (p : ((ℕ × ℕ) × (ℕ × ℕ))), pair_edges p.fst p.snd)).length

#print axioms square_pyramid_planes

end square_pyramid_planes_l522_522234


namespace domain_of_g_l522_522499

noncomputable def g (t : ℝ) : ℝ := 1 / ((t - 2)^3 + (t + 2)^3)

theorem domain_of_g : ∀ t : ℝ, t ≠ 0 ↔ g t ∈ ℝ :=
by
  sorry

end domain_of_g_l522_522499


namespace rectangle_area_l522_522132

theorem rectangle_area 
  (P : ℝ) (r : ℝ) (hP : P = 40) (hr : r = 3 / 2) : 
  ∃ (length width : ℝ), 2 * (length + width) = P ∧ length = 3 * (width / 2) ∧ (length * width) = 96 :=
by
  sorry

end rectangle_area_l522_522132


namespace min_parabola_distance_l522_522307

-- Define the points and the parabola
def Point (x y : ℝ) := (x, y)

def A : Point := (2, 0)
def B : Point := (7, 6)

def onParabola (P : Point) : Prop := (P.2)^2 = 8 * P.1

-- Define the distance function
def distance (P Q : Point) : ℝ := real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

-- Define a function that checks the minimum value of AP + BP
noncomputable def minSumDistance (P : Point) : ℝ := distance A P + distance B P

-- The proof statement
theorem min_parabola_distance :
  ∃ (P : Point), onParabola P ∧ minSumDistance P = 9 :=
sorry

end min_parabola_distance_l522_522307


namespace unique_coloring_l522_522093

-- Define the setup as a type for hexagons
structure Hexagon :=
(color : Bool)   -- True represents red, False represents yellow

-- Constraints and setup
def adjacent (a b : Hexagon) : Prop :=
-- Placeholder for the adjacency relationship
sorry

def valid_coloring (hexes : List Hexagon) : Prop :=
∀ (h₁ h₂ : Hexagon), adjacent h₁ h₂ → (h₁.color ≠ h₂.color)

-- Initial condition: The hexagon labeled R is red
def hex_R : Hexagon := { color := true }

-- Define the full set of hexagons
def hexagons := sorry  -- Placeholder for the actual list of hexagons

theorem unique_coloring :
  ∃! hexes : List Hexagon,
    hex_R ∈ hexes ∧ valid_coloring hexes :=
sorry

end unique_coloring_l522_522093


namespace number_of_integer_values_for_a_l522_522182

theorem number_of_integer_values_for_a :
  (∃ (a : Int), ∃ (p q : Int), p * q = -12 ∧ p + q = a ∧ p ≠ q) →
  (∃ (n : Nat), n = 6) := by
  sorry

end number_of_integer_values_for_a_l522_522182


namespace even_function_max_value_l522_522964

variable {f : ℝ → ℝ}

-- Conditions: f is an even function and has a maximum value of 6 on the interval [-3, -1]
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def has_max_on_interval (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop := ∀ x ∈ set.Icc a b, f x ≤ m ∧ ∃ c ∈ set.Icc a b, f c = m

theorem even_function_max_value :
  is_even_function f →
  has_max_on_interval f (-3) (-1) 6 →
  has_max_on_interval f 1 3 6 :=
by
  intros h_even h_max
  sorry

end even_function_max_value_l522_522964


namespace correct_calculation_l522_522757

theorem correct_calculation :
  (∀ a b : ℝ, (4 * a * b)^2 = 16 * a^2 * b^2) ∧
  (∀ a : ℝ, a^2 * a^3 = a^5) ∧
  (∀ a : ℝ, a^2 + a^2 = 2 * a^2) ∧
  (∀ a b : ℝ, (-3 * a^3 * b)^2 = 9 * a^6 * b^2)
: option_D_correct :=
  sorry

end correct_calculation_l522_522757


namespace circles_intersect_l522_522194

noncomputable def circle_center_radius (h : real) := 
  (0, h, h)

def dist_center_to_line (h : real) := 
  h / real.sqrt 2

def segment_length (h : real) := 
  2 * real.sqrt (h^2 - (h^2 / 2)) 

def centers_distance (a b : real) := 
  real.sqrt (1 + (a - b)^2)
  
theorem circles_intersect (a : real) (a_pos : a > 0)  
  (segment_len : segment_length a = 2) :
  let M := circle_center_radius a,
      N := (1, 1, 1) in
  (M.2 + N.2 = real.sqrt 2 + 1) ∧ 
  (M.2 - N.2 = real.sqrt 2 - 1) ∧ 
  (M.0 - 1, M.1 - 1 < M.2 + N.2) := sorry

end circles_intersect_l522_522194


namespace tangent_line_at_point_monotonic_intervals_l522_522149

noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2) * x^2 - 2*x + 5

theorem tangent_line_at_point :
  let df := (fun x => (3 * x^2 - x - 2)) in
  df 0 = -2 ∧ (λ x y, 2 * x + y - 5 = 0) (0 : ℝ) (5 : ℝ) := 
by
  let df := (fun x => (3 * x^2 - x - 2))
  split
  case left => simp [df]
  case right => simp

theorem monotonic_intervals :
  let df := (fun x => (3 * x^2 - x - 2)) in
  ((∀ x, x < -(2/3) → df x > 0) ∧ (∀ x, x > 1 → df x > 0)) ∧ 
  (∀ x, - (2/3) < x ∧ x < 1 → df x < 0) :=
by
  let df := (fun x => (3 * x^2 - x - 2))
  split
  case left => split
    case left => sorry -- proof of df(x) > 0 for x < -2/3
    case right => sorry -- proof of df(x) > 0 for x > 1
  case right => sorry -- proof of df(x) < 0 for -2/3 < x < 1

end tangent_line_at_point_monotonic_intervals_l522_522149


namespace factorize_a3_minus_ab2_l522_522128

theorem factorize_a3_minus_ab2 (a b: ℝ) : 
  a^3 - a * b^2 = a * (a + b) * (a - b) :=
by
  sorry

end factorize_a3_minus_ab2_l522_522128


namespace find_roots_l522_522861

theorem find_roots (x : ℝ) : (x^2 + x = 0) ↔ (x = 0 ∨ x = -1) := 
by sorry

end find_roots_l522_522861


namespace correct_exponentiation_l522_522410

theorem correct_exponentiation : ∀ (x : ℝ), (x^(4/5))^(5/4) = x :=
by
  intro x
  sorry

end correct_exponentiation_l522_522410


namespace find_circle_center_l522_522781

-- Definitions of conditions
def is_tangent_to_parabola (center : ℝ × ℝ) : Prop :=
  let (a, b) := center in
  let m := 6 in
  (b - 9) / (a - 3) = -1 / m

def on_perpendicular_bisector (center : ℝ × ℝ) : Prop :=
  let (a, b) := center in
  let midpoint := (3 / 2, 11 / 2) in
  (b - midpoint.2) / (a - midpoint.1) = -3 / 7

-- The main statement
theorem find_circle_center (center : ℝ × ℝ) (h1 : on_perpendicular_bisector center) (h2 : is_tangent_to_parabola center) : 
  center = (-27 / 13, 118 / 13) :=
by
  sorry

end find_circle_center_l522_522781


namespace percent_of_12356_l522_522025

theorem percent_of_12356 : 12356 * 0.001 = 12.356 :=
by exact Mod.generalize {123.56 / 10, apply_mod, step}.

end percent_of_12356_l522_522025


namespace min_value_x_add_y_div_2_l522_522962

theorem min_value_x_add_y_div_2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y - 2 * x - y = 0) :
  ∃ x y, 0 < x ∧ 0 < y ∧ (x * y - 2 * x - y = 0 ∧ x + y / 2 = 4) :=
sorry

end min_value_x_add_y_div_2_l522_522962


namespace solve_for_x_l522_522618

-- Definitions for the problem conditions
def perimeter_triangle := 14 + 12 + 12
def perimeter_rectangle (x : ℝ) := 2 * x + 16

-- Lean 4 statement for the proof problem 
theorem solve_for_x (x : ℝ) : 
  perimeter_triangle = perimeter_rectangle x → 
  x = 11 := 
by 
  -- standard placeholders
  sorry

end solve_for_x_l522_522618


namespace parabola_min_y1_y2_squared_l522_522216

theorem parabola_min_y1_y2_squared (x1 x2 y1 y2 : ℝ) :
  (y1^2 = 4 * x1) ∧
  (y2^2 = 4 * x2) ∧
  (x1 * x2 = 16) →
  (y1^2 + y2^2 ≥ 32) :=
by
  intro h
  sorry

end parabola_min_y1_y2_squared_l522_522216


namespace max_factors_b_pow_n_l522_522547

theorem max_factors_b_pow_n (b n : ℕ) (h_b : b ≤ 20) (h_n : n ≤ 20) (h_pos_b : 0 < b) (h_pos_n : 0 < n) :
  ∃ b, b ≤ 20 ∧ ∃ n, n ≤ 20 ∧ 0 < b ∧ 0 < n ∧ (∀ b' n', (b' ≤ 20 ∧ n' ≤ 20 (0 < b') ∧ 0 < n') → 
    (num_factors (b'^n') ≤ 861)) := sorry

end max_factors_b_pow_n_l522_522547


namespace max_value_min_4x_y_4y_x2_5y2_l522_522306

theorem max_value_min_4x_y_4y_x2_5y2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ t, t = min (4 * x + y) (4 * y / (x^2 + 5 * y^2)) ∧ t ≤ 2 :=
by
  sorry

end max_value_min_4x_y_4y_x2_5y2_l522_522306


namespace first_year_with_sum_of_digits_15_l522_522747

def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem first_year_with_sum_of_digits_15 : 
  ∃ y, y > 2021 ∧ sum_of_digits y = 15 ∧ ∀ z, z > 2021 ∧ sum_of_digits z = 15 → z ≥ y :=
begin
  use 2049,
  split,
  { linarith },
  split,
  { -- Sum of digits of 2049 is 15
    sorry },
  { -- For all z, if z > 2021 and sum of digits of z is 15, then z >= 2049
    sorry }
end

end first_year_with_sum_of_digits_15_l522_522747


namespace highest_score_l522_522693

variable (avg runs_excluding: ℕ)
variable (innings remaining_innings total_runs total_runs_excluding H L: ℕ)

axiom batting_average (h_avg: avg = 60) (h_innings: innings = 46) : total_runs = avg * innings
axiom diff_highest_lowest_score (h_diff: H - L = 190) : true
axiom avg_excluding_high_low (h_avg_excluding: runs_excluding = 58) (h_remaining_innings: remaining_innings = 44) : total_runs_excluding = runs_excluding * remaining_innings
axiom sum_high_low : total_runs - total_runs_excluding = 208

theorem highest_score (h_avg: avg = 60) (h_innings: innings = 46) (h_diff: H - L = 190) (h_avg_excluding: runs_excluding = 58) (h_remaining_innings: remaining_innings = 44)
    (calc_total_runs: total_runs = avg * innings) 
    (calc_total_runs_excluding: total_runs_excluding = runs_excluding * remaining_innings)
    (calc_sum_high_low: total_runs - total_runs_excluding = 208) : H = 199 :=
by
  sorry

end highest_score_l522_522693


namespace conference_room_probability_l522_522437

theorem conference_room_probability :
  let m := 16
  let n := 2925
  ∑ k in {m, n}, k = 2941 := by
sorry

end conference_room_probability_l522_522437


namespace possible_values_expression_l522_522166

theorem possible_values_expression (a b c d : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) :
  ∃ (x : ℝ), x ∈ {5, 1, -3} ∧ x = (a / |a| + b / |b| + c / |c| + d / |d| + (abcd / |abcd|)) :=
by
  sorry

end possible_values_expression_l522_522166


namespace num_perfect_square_factors_108000_l522_522934

def is_even (n : ℕ) : Prop := n % 2 = 0

def perfect_square_factors_count (n : ℕ) : ℕ := 
  if n = 108000 then 
    let valid_a := {0, 2, 4}
    let valid_b := {0, 2}
    let valid_c := {0, 2}
    3 * 2 * 2
  else 
    0

theorem num_perfect_square_factors_108000 : perfect_square_factors_count 108000 = 12 := by
  sorry

end num_perfect_square_factors_108000_l522_522934


namespace midpoint_x_coordinate_l522_522156

theorem midpoint_x_coordinate (M N : ℝ × ℝ)
  (hM : M.1 ^ 2 = 4 * M.2)
  (hN : N.1 ^ 2 = 4 * N.2)
  (h_dist : (Real.sqrt ((M.1 - 1)^2 + M.2^2)) + (Real.sqrt ((N.1 - 1)^2 + N.2^2)) = 6) :
  (M.1 + N.1) / 2 = 2 := 
sorry

end midpoint_x_coordinate_l522_522156


namespace loan_payment_difference_is_382_l522_522067

def loan_amount : ℝ := 10000
def interest_quarterly := 0.1 / 4
def period_5_years := 5
def periods_in_5_years := period_5_years * 4
def final_period := periods_in_5_years * 2
def interest_simple := 0.12

noncomputable def amount_after_5_years : ℝ := loan_amount * (1 + interest_quarterly)^periods_in_5_years
noncomputable def half_payment : ℝ := amount_after_5_years / 2
noncomputable def remaining_after_5_years : ℝ := half_payment * (1 + interest_quarterly)^periods_in_5_years
noncomputable def total_compounded : ℝ := half_payment + remaining_after_5_years
noncomputable def total_simple : ℝ := loan_amount + (loan_amount * interest_simple * 10)
noncomputable def positive_difference : ℝ := abs (total_simple - total_compounded)

theorem loan_payment_difference_is_382 : round positive_difference = 382 := 
by 
  sorry

end loan_payment_difference_is_382_l522_522067


namespace sqrt_product_cos_squared_l522_522106

theorem sqrt_product_cos_squared :
  let x1 := cos (π / 9) ^ 2
  let x2 := cos (2 * π / 9) ^ 2
  let x3 := cos (4 * π / 9) ^ 2
  512 * x1 ^ 9 - 2304 * x1 ^ 7 + 3360 * x1 ^ 5 - 1680 * x1 ^ 3 + 315 * x1 - 9 = 0
  ∧ 512 * x2 ^ 9 - 2304 * x2 ^ 7 + 3360 * x2 ^ 5 - 1680 * x2 ^ 3 + 315 * x2 - 9 = 0
  ∧ 512 * x3 ^ 9 - 2304 * x3 ^ 7 + 3360 * x3 ^ 5 - 1680 * x3 ^ 3 + 315 * x3 - 9 = 0
  → sqrt ((3 - x1) * (3 - x2) * (3 - x3)) = 9 * sqrt 3 / 8 :=
by
  sorry

end sqrt_product_cos_squared_l522_522106


namespace collinear_S_T_K_l522_522311

open_locale real

theorem collinear_S_T_K 
  (O : Point) (A B : Point) (C : Point) (K H M T S: Point) 
  (circle_O : Circle O)
  (AB_diameter : is_diameter A B circle_O)
  (C_on_circle : on_circle C circle_O)
  (m_tangent : tangent_at m circle_O A)
  (n_tangent : tangent_at n circle_O B)
  (BC_inter_m : intersects B C m K)
  (bisect_CAK : bisects_angle C A K H)
  (M_midpoint_arc : midpoint_of_arc M A B circle_O)
  (HM_inter_circle : intersects H M circle_O S)
  (M_tangent_inter_n : tangent_intersects M n T) :
  collinear S T K :=
by sorry

end collinear_S_T_K_l522_522311


namespace Niko_total_profit_l522_522330

-- Definitions based on conditions
def cost_per_pair : ℕ := 2
def total_pairs : ℕ := 9
def profit_margin_4_pairs : ℚ := 0.25
def profit_per_other_pair : ℚ := 0.2
def pairs_with_margin : ℕ := 4
def pairs_with_fixed_profit : ℕ := 5

-- Calculations based on definitions
def total_cost : ℚ := total_pairs * cost_per_pair
def profit_on_margin_pairs : ℚ := pairs_with_margin * (profit_margin_4_pairs * cost_per_pair)
def profit_on_fixed_profit_pairs : ℚ := pairs_with_fixed_profit * profit_per_other_pair
def total_profit : ℚ := profit_on_margin_pairs + profit_on_fixed_profit_pairs

-- Statement to prove
theorem Niko_total_profit : total_profit = 3 := by
  sorry

end Niko_total_profit_l522_522330


namespace find_point_X_on_BC_l522_522773

theorem find_point_X_on_BC 
  (A B C X : Type) 
  (a b c : ℝ) 
  (h_triangle : triangle A B C) 
  (BX XC : ℝ) 
  (hx : BX = x) 
  (hxc : XC = a - x) : 
  ∃ x : ℝ, 
  4 * a * x^2 + 4 * (b^2 - c^2 - a^2) * x + (-a * b^2 + 3 * a * c^2 + a^3 - 2 * a * b * c) = 0 := 
sorry

end find_point_X_on_BC_l522_522773


namespace sara_wrapping_paper_l522_522349

theorem sara_wrapping_paper (s : ℚ) (l : ℚ) (total : ℚ) : 
  total = 3 / 8 → 
  l = 2 * s →
  4 * s + 2 * l = total → 
  s = 3 / 64 :=
by
  intros h1 h2 h3
  sorry

end sara_wrapping_paper_l522_522349


namespace odd_numbered_side_even_numbered_side_l522_522662

theorem odd_numbered_side (y : ℕ) (h : y = 239) : ∃ x : ℕ, 2 * x^2 - 1 = y^2 ∧ x = 169 :=
by {
  use 169,
  split,
  {
    rw h,
    norm_num,
  },
  refl,
}

theorem even_numbered_side (y : ℕ) (h : y = 408) : ∃ x : ℕ, 2 * (x^2 + x) = y^2 ∧ x = 288 :=
by {
  use 288,
  split,
  {
    rw h,
    norm_num,
  },
  refl,
}

#print odd_numbered_side
#print even_numbered_side

end odd_numbered_side_even_numbered_side_l522_522662


namespace compute_value_l522_522302

def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := x / 4
def f_inv (x : ℝ) : ℝ := x - 3
def g_inv (x : ℝ) : ℝ := 4 * x

theorem compute_value : f (g_inv (f_inv (f_inv (f_inv (g (f 23)))))) = -7 :=
by
  sorry

end compute_value_l522_522302


namespace Fred_says_1_l522_522863

-- Define the players in the order given
inductive Player
| Fred | Gail | Henry | Iggy | Joan

open Player

-- Define a function to determine the player counting down from a given number
def who_says_1 : Player :=
  let start := 34 in
  let players : List Player := [Henry, Iggy, Joan, Fred, Gail] in
  let idx := (start - 1) % players.length in
  players[idx]

-- Formal statement to prove
theorem Fred_says_1 : who_says_1 = Fred :=
by
  -- Proof is omitted using sorry
  sorry

end Fred_says_1_l522_522863


namespace width_of_each_road_is_10_l522_522053

variables (w : ℝ)
def area_of_lengths_covered_by_roads (w : ℝ) := 80 * w + 60 * w - w * w
def total_cost_of_roads (w : ℝ) := 5 * area_of_lengths_covered_by_roads w
def total_roads_cost_condition := total_cost_of_roads w = 6500

theorem width_of_each_road_is_10 (h : total_roads_cost_condition) : w = 10 :=
by
  -- The proof will be developed here based on the actual derivations.
  sorry

end width_of_each_road_is_10_l522_522053


namespace estimated_households_exceeding_320_l522_522607

noncomputable def community_electricity_consumption (n : ℕ) (mean stddev : ℝ) : ℕ :=
  let households := n
  let mu := mean
  let sigma := stddev
  let upper_bound := 320
  let probability_exceeding := (1 - 0.954) / 2
  let expected_households := households * probability_exceeding
  (expected_households).toNat

theorem estimated_households_exceeding_320 :
  community_electricity_consumption 1000 300 10 = 23 :=
sorry

end estimated_households_exceeding_320_l522_522607


namespace willam_percentage_l522_522766

theorem willam_percentage (total_tax : ℕ) (willam_tax : ℕ) (fraction : ℕ) 
  (h1 : total_tax = 3840)
  (h2 : willam_tax = 480)
  (h3 : fraction = 50) :
  willam_tax * 100 / total_tax = 12.5 :=
by
  sorry

end willam_percentage_l522_522766


namespace equidistant_point_l522_522424

def Point := (ℝ × ℝ × ℝ)

def A (y : ℝ) : Point := (0, y, 0)
def B : Point := (0, 5, -9)
def C : Point := (-1, 0, 5)

def dist (p1 p2 : Point) : ℝ :=
  match p1, p2 with
  | (x1, y1, z1), (x2, y2, z2) => Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)

theorem equidistant_point (y : ℝ) (h : dist (A y) B = dist (A y) C) : y = 8 :=
by
  sorry

end equidistant_point_l522_522424


namespace average_cookies_l522_522474

theorem average_cookies (cookie_counts : List ℕ) (h : cookie_counts = [8, 10, 12, 15, 16, 17, 20]) :
  (cookie_counts.sum : ℚ) / cookie_counts.length = 14 := by
    -- Proof goes here
  sorry

end average_cookies_l522_522474


namespace number_of_strawberries_l522_522658

def total_berries : ℕ := 120
def fraction_raspberries : ℚ := 1 / 4
def fraction_blackberries : ℚ := 3 / 8
def fraction_blueberries : ℚ := 1 / 6

theorem number_of_strawberries : 
  let s := total_berries - (fraction_raspberries * total_berries + 
                            fraction_blackberries * total_berries + 
                            fraction_blueberries * total_berries) in
  s = 25 :=
by
  sorry

end number_of_strawberries_l522_522658


namespace radius_of_semicircular_cubicle_l522_522450

noncomputable def radius_of_semicircle (P : ℝ) : ℝ := P / (Real.pi + 2)

theorem radius_of_semicircular_cubicle :
  radius_of_semicircle 71.9822971502571 = 14 := 
sorry

end radius_of_semicircular_cubicle_l522_522450


namespace area_enclosed_by_sin_half_l522_522363

noncomputable def enclosed_area_sine (a b : ℝ) (f g : ℝ → ℝ) : ℝ :=
  ∫ x in a..b, f x - g x

theorem area_enclosed_by_sin_half : 
    enclosed_area_sine (π / 6) (5 * π / 6) (fun x => sin x) (fun x => 1 / 2) 
    = (sqrt 3 - π / 3) := 
  sorry

end area_enclosed_by_sin_half_l522_522363


namespace arc_length_of_sector_l522_522901

-- Define the problem conditions
variables (S : ℝ := 4) (α : ℝ := 2)
-- Define the radius, area, and length of the arc
noncomputable def r := real.sqrt ((2 * S) / α)
noncomputable def l := r * α

-- Theorem statement
theorem arc_length_of_sector : l = 4 := 
by
  -- Leave the proof out, since it's not required
  sorry

end arc_length_of_sector_l522_522901


namespace problem1_problem2_problem3_l522_522030

-- Definition for the first problem
def A_10_m_eq_10_9_5 (m : ℕ) : Prop :=
  A 10 m = 10 * 9 * 8 * 7 * 6 * 5

-- Theorem statement for the first problem
theorem problem1 : ∃ m : ℕ, A_10_m_eq_10_9_5 m ∧ m = 6 := 
sorry

-- Theorem statement for the second problem
theorem problem2 : ∃ n : ℕ, n = 3! ∧ n = 6 := 
sorry

-- Definition and theorem statement for the third problem
def arrangements (total : ℕ) : Prop :=
  total = 2 * 4 * 4!

theorem problem3 : ∃ n : ℕ, arrangements n ∧ n = 192 :=
sorry

end problem1_problem2_problem3_l522_522030


namespace cos_angle_focus_l522_522158

theorem cos_angle_focus (P F1 F2 : ℝ × ℝ): 
  (∀ x y : ℝ, (x, y) ∈ P → x^2 / 4 + y^2 = 1) ∧ 
    (∀ x y : ℝ, (x, y) ∈ (F1, F2) → (∃ m : ℝ, |PF1| = 3 * |PF2| = 3 * m) ∧ (p + m = 4 )) → 
  cos_angle F1 P F2 = -1 / 3 :=
  sorry

end cos_angle_focus_l522_522158


namespace problem_statement_l522_522870

def f : ℝ → ℝ
| x => if x > 0 then -Real.cos (π * x) else f (x + 1) + 1

theorem problem_statement : f (4/3) + f (-4/3) = 3 := 
sorry

end problem_statement_l522_522870


namespace rational_function_invariance_l522_522671

/-- Definition of the rational function R(x) as given by the solution -/
def R (x : ℚ) : ℚ :=
  x^2 + 1/(x^2) + (1 - x)^2 + 1/((1 - x)^2) + (x^2)/((1 - x)^2) + ((x - 1)^2)/(x^2)

/-- Proof problem stating that the rational function R(x) is non-constant
    and satisfies the invariance properties R(x) = R(1/x) and R(x) = R(1-x). -/
theorem rational_function_invariance (x : ℚ) : 
  R(x) ≠  R(1 : ℚ) ∧ R(x) = R(1/x) ∧ R(x) = R(1 - x) :=
sorry

end rational_function_invariance_l522_522671


namespace locus_of_M_is_circle_l522_522539

noncomputable def ellipse_foci (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

def moving_point_on_ellipse (a b : ℝ) : Set (ℝ × ℝ) := ellipse_foci a b

def perpendicular_from_f_to_external_bisector (F₁ F₂ P M : ℝ × ℝ) : Prop :=
  ∃ (L : ℝ) (B : Set (ℝ × ℝ)), B = {p | p ≠ P ∧ angle ∠ F₁ P F₂ = angle ∠ P M F₂} ∧
  L = foot F₂ B ∧ M = L

theorem locus_of_M_is_circle (a b : ℝ) (F₁ F₂ : ℝ × ℝ) :
  ∃ (O : ℝ × ℝ) (r : ℝ), locus_of_point_m (a b) F₁ F₂ r where
  O = origin,
  r = a :=
sorry

end locus_of_M_is_circle_l522_522539


namespace find_multiple_l522_522046

theorem find_multiple :
  let number := 220030
  let sum := 555 + 445
  let difference := 555 - 445
  let remainder := 30
  ∃ (multiple : ℕ), (number % sum = remainder) ∧ ((number / sum) = multiple * difference) :=
by
  let number := 220030
  let sum := 555 + 445
  let difference := 555 - 445
  let remainder := 30
  have h_sum : sum = 555 + 445 := rfl
  have h_diff : difference = 555 - 445 := rfl
  have h_remainder : number % sum = remainder := sorry
  have h_multiple : ∃ (multiple : ℕ), (number / sum) = multiple * difference := sorry
  use 2
  split
  · exact h_remainder
  · exact h_multiple

end find_multiple_l522_522046


namespace find_x_modulo_l522_522859

theorem find_x_modulo (k : ℤ) : ∃ x : ℤ, x = 18 + 31 * k ∧ ((37 * x) % 31 = 15) := by
  sorry

end find_x_modulo_l522_522859


namespace probability_point_above_parabola_l522_522358

theorem probability_point_above_parabola :
  let S := {p : ℕ × ℕ | p.1 ∈ Finset.range 10 \{0} ∧ p.2 ∈ Finset.range 10 \{0}},
      valid_points := {p ∈ S | ∀ x : ℝ, p.2 > p.2 * x^2 - p.1 * x},
      A := Finset.card valid_points,
      T := 81 in
  A = 16 → (A : ℝ) / T = 16 / 81 :=
by {
  sorry
}

end probability_point_above_parabola_l522_522358


namespace log_power_eq_mul_log_number_of_digits_in_219_power_220_closer_approximation_l522_522384

-- Definitions from the conditions
variable (a M : ℝ) (n : ℝ)
variable (H_a_pos : 0 < a) (H_a_ne : a ≠ 1) (H_M_pos : 0 < M)
variable (lg219 : ℝ := 2.34)
variable (lg3 : ℝ := 0.4771)


-- Problem 1: Prove the property of logarithms
theorem log_power_eq_mul_log : log a (M ^ n) = n * log a M :=
by
  sorry

-- Problem 2: Calculate the number of digits in 219^220
theorem number_of_digits_in_219_power_220 : 
  let t := 219 ^ 220 in
  ⌊log 10 t⌋ + 1 = 515 :=
by
  sorry

-- Problem 3: Determine which student's approximation is closer
variable (M := 3 ^ 361) (N := 10 ^ 80) 

theorem closer_approximation : 
  let A := 10 ^ 73 
  let B := 10 ^ 93 in
  abs (log 10 (M / N) - log 10 A) < abs (log 10 (M / N) - log 10 B) :=
by
  sorry

end log_power_eq_mul_log_number_of_digits_in_219_power_220_closer_approximation_l522_522384


namespace convert_radian_to_degree_part1_convert_radian_to_degree_part2_convert_radian_to_degree_part3_convert_degree_to_radian_part1_convert_degree_to_radian_part2_l522_522493

noncomputable def pi_deg : ℝ := 180 -- Define pi in degrees
notation "°" => pi_deg -- Define a notation for degrees

theorem convert_radian_to_degree_part1 : (π / 12) * (180 / π) = 15 := 
by
  sorry

theorem convert_radian_to_degree_part2 : (13 * π / 6) * (180 / π) = 390 := 
by
  sorry

theorem convert_radian_to_degree_part3 : -(5 / 12) * π * (180 / π) = -75 := 
by
  sorry

theorem convert_degree_to_radian_part1 : 36 * (π / 180) = (π / 5) := 
by
  sorry

theorem convert_degree_to_radian_part2 : -105 * (π / 180) = -(7 * π / 12) := 
by
  sorry

end convert_radian_to_degree_part1_convert_radian_to_degree_part2_convert_radian_to_degree_part3_convert_degree_to_radian_part1_convert_degree_to_radian_part2_l522_522493


namespace smallest_whole_number_larger_than_any_triangle_perimeter_l522_522407

theorem smallest_whole_number_larger_than_any_triangle_perimeter (a b : ℕ) (h₁ : a = 7) (h₂ : b = 21)  :
  ∃ n : ℕ, n = 57 ∧ ∀ s : ℕ, (14 < s ∧ s < 28) → 7 + 21 + s < n :=
by
  have h₃ : ∀ s : ℕ, (14 < s ∧ s < 28) → 7 + 21 + s < 57,
  {
    intros s hs,
    cases hs with hs1 hs2,
    linarith,
  }
  use 57,
  split,
  refl,
  exact h₃,

end smallest_whole_number_larger_than_any_triangle_perimeter_l522_522407


namespace students_spend_185_minutes_in_timeout_l522_522319

variable (tR tF tS t_total : ℕ)

-- Conditions
def running_timeouts : ℕ := 5
def food_timeouts : ℕ := 5 * running_timeouts - 1
def swearing_timeouts : ℕ := food_timeouts / 3
def total_timeouts : ℕ := running_timeouts + food_timeouts + swearing_timeouts
def timeout_duration : ℕ := 5

-- Total time spent in time-out
def total_timeout_minutes : ℕ := total_timeouts * timeout_duration

theorem students_spend_185_minutes_in_timeout :
  total_timeout_minutes = 185 :=
by
  -- The answer is directly given by the conditions and the correct answer identified.
  sorry

end students_spend_185_minutes_in_timeout_l522_522319


namespace possible_values_of_expression_l522_522162

theorem possible_values_of_expression (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (v : ℤ), v ∈ ({5, 1, -3, -5} : Set ℤ) ∧ v = (Int.sign a + Int.sign b + Int.sign c + Int.sign d + Int.sign (a * b * c * d)) :=
by
  sorry

end possible_values_of_expression_l522_522162


namespace solve_sequence_problem_l522_522905

noncomputable def sequence_sum (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, S n = 2 * a n + n - 4

noncomputable def sequence_general_formula (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = 2 ^ n + 1

noncomputable def max_ratio_value (max_val : ℚ) (a : ℕ → ℤ) : Prop :=
  max_val = 2 / 5 ∧ ∀ n, (rat.of_int n / a n : ℚ) ≤ 2 / 5

theorem solve_sequence_problem (S : ℕ → ℤ) (a : ℕ → ℤ) :
  sequence_sum S a →
  sequence_general_formula a ∧ max_ratio_value (2 / 5) a :=
by
  intros h_seq_sum
  sorry

end solve_sequence_problem_l522_522905


namespace odd_function_properties_l522_522181

-- Define the function f(x) for x in [-1, 0]
def f_neg (x : ℝ) : ℝ := (1 / 4 ^ x) - (b / 2 ^ x)

-- Define the function f(x) for x in [0, 1]
def f_pos (x : ℝ) : ℝ := 2 ^ x - 4 ^ x

theorem odd_function_properties
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_def_neg : ∀ x ∈ Icc (-1 : ℝ) (0 : ℝ), f x = f_neg x)
  (h_def_pos : ∀ x ∈ Icc (0 : ℝ) (1 : ℝ), f x = f_pos x) :
  b = 1 ∧ f = λ x, if x ∈ Icc (0 : ℝ) (1 : ℝ) then f_pos x else f_neg x ∧
  (∀ x ∈ Icc (0 : ℝ) (1 : ℝ), f x ≤ 0) ∧ (∀ x, f 0 = 0 ∧ f 1 = 0 ∧ f 2 = -2) :=
by
  sorry

end odd_function_properties_l522_522181


namespace lights_on_fourth_tier_l522_522258

def number_lights_topmost_tier (total_lights : ℕ) : ℕ :=
  total_lights / 127

def number_lights_tier (tier : ℕ) (lights_topmost : ℕ) : ℕ :=
  2^(tier - 1) * lights_topmost

theorem lights_on_fourth_tier (total_lights : ℕ) (H : total_lights = 381) : number_lights_tier 4 (number_lights_topmost_tier total_lights) = 24 :=
by
  rw [H]
  sorry

end lights_on_fourth_tier_l522_522258


namespace hexagon_coloring_l522_522121

-- Define the vertices of the hexagon
inductive Vertex
| A | B | C | D | E | F
deriving DecidableEq

-- Define the condition that no two ends of each diagonal have the same color
def distinct_colors (color : Vertex → ℕ) : Prop :=
  color Vertex.A ≠ color Vertex.D ∧
  color Vertex.B ≠ color Vertex.E ∧
  color Vertex.C ≠ color Vertex.F

-- The main statement of the problem
theorem hexagon_coloring (colors : ℕ)
  (seven_colors : colors = 7) :
  ∃ (X : ℕ), (∑ c: Vertex → ℕ in finset.range (7^6), 
  if distinct_colors c then 1 else 0) = X :=
sorry

end hexagon_coloring_l522_522121


namespace ratio_division_l522_522971

variables {A B C F H E : Type*}
variables [add_monoid A] [add_monoid B] [add_monoid C]
variables [linear_ordered_field F] [linear_ordered_field H]
variables [linear_ordered_field E]

-- Define the points and ratios
noncomputable def point_F : F := sorry
noncomputable def point_H : H := sorry
noncomputable def point_E : E := sorry

-- Define conditions
def divides_ratio (F : Type*) (AC : Type*) : Prop := sorry -- F divides AC in ratio 2:3
def midpoint (H : Type*) (CF : Type*) : Prop := sorry -- H is midpoint of CF
def intersection (E : Type*) (AH BC : Type*) : Prop := sorry -- E is intersection of AH with BC

-- Main statement
theorem ratio_division (A B C : Type*) : divides_ratio F AC → midpoint H CF → intersection E AH BC → ratio E BC = 2 / 7 := 
sorry

end ratio_division_l522_522971


namespace total_toys_per_week_l522_522786

def toys_per_day := 1100
def working_days_per_week := 5

theorem total_toys_per_week : toys_per_day * working_days_per_week = 5500 :=
by
  sorry

end total_toys_per_week_l522_522786


namespace solutions_of_equation_l522_522722

theorem solutions_of_equation : 
  (∀ x : ℝ, x * (5 * x + 2) = 6 * (5 * x + 2) → x = 6 ∨ x = -2 / 5) :=
by
  intro x
  intro h
  have h_pos : (x - 6) * (5 * x + 2) = 0 := by sorry
  cases h_pos with
  | inl h₁ => convert h₁; field_simp
  | inr h₂ => convert h₂; ring_nf; field_simp
  sorry

end solutions_of_equation_l522_522722


namespace sum_divisible_by_10_l522_522144

-- Define the problem statement
theorem sum_divisible_by_10 {n : ℕ} : (n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2) % 10 = 0 ↔ ∃ t : ℕ, n = 5 * t + 1 :=
by sorry

end sum_divisible_by_10_l522_522144


namespace angle_bisectors_lie_on_altitudes_l522_522099

theorem angle_bisectors_lie_on_altitudes
  (A1 B1 C1 : Point) 
  (circumcircle : is_circumcircle A1 B1 C1) 
  (intersect_circumcircle : ∀ P Q R : Point, 
      ((altitude P Q) ∩ (circumcircle)) ∧ 
      ((altitude Q R) ∩ (circumcircle)) ∧ 
      ((altitude R P) ∩ (circumcircle))) :
  ∃ (A B C : Point), 
  (constructed_triangle A B C (angle_bisectors A1 B1 C1)) :=
    sorry

end angle_bisectors_lie_on_altitudes_l522_522099


namespace train_speed_l522_522062

theorem train_speed (length_of_train : ℝ) (time_to_pass_tree : ℝ) (speed_conversion_factor : ℝ) 
  (h_length : length_of_train = 275) (h_time : time_to_pass_tree = 11) (h_conversion : speed_conversion_factor = 3.6) : 
  length_of_train / time_to_pass_tree * speed_conversion_factor = 90 :=
by
  rw [h_length, h_time, h_conversion]
  norm_num
  sorry

end train_speed_l522_522062


namespace period_of_my_function_l522_522113

-- Define the sine function
def my_sine (x : ℝ) := Real.sin x

-- Define the function y = sin(3x + π)
def my_function (x : ℝ) := my_sine (3 * x + Real.pi)

-- Define the period
def period (f : ℝ → ℝ) (T : ℝ) := ∀ x, f (x + T) = f x

-- Statement to prove
theorem period_of_my_function : period my_function (2 * Real.pi / 3) :=
sorry

end period_of_my_function_l522_522113


namespace Niko_total_profit_l522_522329

-- Definitions based on conditions
def cost_per_pair : ℕ := 2
def total_pairs : ℕ := 9
def profit_margin_4_pairs : ℚ := 0.25
def profit_per_other_pair : ℚ := 0.2
def pairs_with_margin : ℕ := 4
def pairs_with_fixed_profit : ℕ := 5

-- Calculations based on definitions
def total_cost : ℚ := total_pairs * cost_per_pair
def profit_on_margin_pairs : ℚ := pairs_with_margin * (profit_margin_4_pairs * cost_per_pair)
def profit_on_fixed_profit_pairs : ℚ := pairs_with_fixed_profit * profit_per_other_pair
def total_profit : ℚ := profit_on_margin_pairs + profit_on_fixed_profit_pairs

-- Statement to prove
theorem Niko_total_profit : total_profit = 3 := by
  sorry

end Niko_total_profit_l522_522329


namespace find_abc_sum_l522_522001

-- Conditions
def isosceles_right_triangle (A B C : Point) : Prop :=
  dist A B = dist A C ∧ ∠ B A C = 90

def midpoint (M B C : Point) : Prop :=
  dist M B = dist M C

def right_angle_triangle_at (A I E : Point) : Prop :=
  ∠ A I E = 90

def area_triangle (E M I : Point) (area : ℝ) : Prop :=
  triangle_area E M I = area

-- The theorem
theorem find_abc_sum 
  (A B C M I E : Point)
  (h1 : isosceles_right_triangle A B C)
  (h2 : midpoint M B C)
  (h3 : I ∈ line_segment A C)
  (h4 : E ∈ line_segment A B)
  (h5 : right_angle_triangle_at A I E)
  (h6 : AI > AE)
  (h7 : area_triangle E M I 5)
  : ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ b ∉ { p * p | p : ℕ } ∧ CI = a - sqrt b / c ∧ a + b + c = 28 :=
sorry

end find_abc_sum_l522_522001


namespace number_of_even_digits_in_base5_of_312_l522_522137

theorem number_of_even_digits_in_base5_of_312 : 
  ∃ n : ℕ, n = 312 ∧ (number_of_even_digits (base5_repr 312) = 4) := by
  sorry

-- Helper functions which might be necessary to define if they don't exist in the imported library

-- convert a natural number to its base-5 representation
def base5_repr (n : ℕ) : list ℕ := sorry

-- count the number of even digits in a list of natural numbers
def number_of_even_digits : list ℕ → ℕ :=
  list.countp (λ d, d % 2 = 0)

end number_of_even_digits_in_base5_of_312_l522_522137


namespace percentage_of_first_over_second_l522_522399

theorem percentage_of_first_over_second (X : ℝ) :
  let A := 1.71 * X in
  let B := 1.80 * X in
  (A / B) * 100 = 95 :=
by
  sorry

end percentage_of_first_over_second_l522_522399


namespace number_of_arrangements_l522_522739

noncomputable def NumberArrangements : ℕ :=
  let total_students := 5
  let total_boys := 2
  let total_girls := 3
  let middle_positions := 3
  let adjacent_girls_possibilities := 3!
  let permutation_adjacent_girls := 2!
  let remaining_positions := 2
  let exclude_overcounting := 2
  (middle_positions * adjacent_girls_possibilities * permutation_adjacent_girls * remaining_positions) / exclude_overcounting

theorem number_of_arrangements (boyA_not_at_ends : true) (two_girls_adjacent : true) :
  NumberArrangements = 36 := by
  sorry

end number_of_arrangements_l522_522739


namespace distance_AB_l522_522558

open Real

def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
  {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

theorem distance_AB (a b : ℝ) (h : a > b ∧ b > 0) :
  ∀ (A B : ℝ × ℝ), A ∈ ellipse a b → B ∈ ellipse a b → 
                   ((mutually_perpendicular A B) ∧ (centered_at_O A B)) →
                   (ab / sqrt (a^2 + b^2) ≤ dist A B ∧ dist A B ≤ sqrt (a^2 + b^2)) := 
by
  sorry

end distance_AB_l522_522558


namespace value_range_a7_l522_522295

section arithmetic_sequence

variables {a : ℕ → ℚ} {n : ℕ}

-- Define the sum of first n terms as the nth accumulation of a sequence
def S (n : ℕ) : ℚ := (Finset.range n).sum a

-- Conditions given in the problem
variables (h1 : S 4 ≥ 10) (h2 : S 5 ≤ 15) (h3 : S 7 ≥ 21)

-- Goal to prove
theorem value_range_a7 (a : ℕ → ℚ) (h1 : S 4 ≥ 10) (h2 : S 5 ≤ 15) (h3 : S 7 ≥ 21) :
  ∃ x, a 7 = x ∧ 3 ≤ x ∧ x ≤ 7 :=
begin
  sorry
end

end arithmetic_sequence

end value_range_a7_l522_522295


namespace first_term_geometric_sequence_l522_522829

theorem first_term_geometric_sequence :
  ∃ (x : ℚ), x * 2^4 = 12 :=
begin
  use 3/4,
  sorry
end

end first_term_geometric_sequence_l522_522829


namespace consecutive_sum_impossible_l522_522421

theorem consecutive_sum_impossible (n : ℕ) :
  (¬ (∃ (a b : ℕ), a < b ∧ n = (b - a + 1) * (a + b) / 2)) ↔ ∃ s : ℕ, n = 2 ^ s :=
sorry

end consecutive_sum_impossible_l522_522421


namespace total_books_to_read_l522_522347

theorem total_books_to_read (books_per_week : ℕ) (weeks : ℕ) (total_books : ℕ) 
  (h1 : books_per_week = 6) 
  (h2 : weeks = 5) 
  (h3 : total_books = books_per_week * weeks) : 
  total_books = 30 :=
by
  rw [h1, h2] at h3
  exact h3

end total_books_to_read_l522_522347


namespace point_in_third_quadrant_l522_522961

theorem point_in_third_quadrant (α : ℝ) (h1 : π / 2 < α ∧ α < π) :
  ∃ (quadrant : ℕ), quadrant = 3 ∧ (quadrant_of_point (sin α, cos α) = quadrant) :=
by
  sorry

end point_in_third_quadrant_l522_522961


namespace triangle_incenter_d1_d2_d3_l522_522670

variable (a b c p d1 d2 d3 : ℝ)
variable (p_eq : p = (a + b + c) / 2) -- semi-perimeter definition

theorem triangle_incenter_d1_d2_d3 :
  (d1 ^ 2 / ((p - a) / a)) : (d2 ^ 2 / ((p - b) / b)) : (d3 ^ 2 / ((p - c) / c)) = 1 := 
by
  sorry -- Proof is omitted

end triangle_incenter_d1_d2_d3_l522_522670


namespace coral_third_week_pages_l522_522100

theorem coral_third_week_pages :
  let total_pages := 600
  let week1_read := total_pages / 2
  let remaining_after_week1 := total_pages - week1_read
  let week2_read := remaining_after_week1 * 0.30
  let remaining_after_week2 := remaining_after_week1 - week2_read
  remaining_after_week2 = 210 :=
by
  sorry

end coral_third_week_pages_l522_522100


namespace volume_of_pyramid_ACE_l522_522924

variables {P A B C D E M : Point}
variables (PA PB PAC ABCD PAB ACE : Plane)
variables (AC CD EF: ℝ) (AD_angle A_angle : ℝ)

-- Conditions provided
axiom On_plane_P_ABC : P ∈ ABC
axiom On_surface_CD_ABCD : CD ∈ ABCD
axiom On_surface_PA_PB : PA ∈ PB
axiom On_plane_ME_PAB : ME ∈ PAB
axiom AC_value : AC = _ -- Add appropriate value if available
axiom AD_angle_value : AD_angle = 60
axiom CD_value : CD = 4 * (sqrt 3)
axiom EF_value : EF = 2 * (sqrt 3)
axiom CD_perp_PAC : CD ⊥ PAC
axiom C_perp_PC : C ⊥ PC
axiom A_angle_value : A_angle = 9
axiom EF_half_CD : EF = 1/2 * CD

noncomputable def volume_pyramid (a c e : Point) : ℝ :=
sorry

theorem volume_of_pyramid_ACE :
  volume_pyramid A C E = (16 * (sqrt 3)) / 3 :=
sorry

end volume_of_pyramid_ACE_l522_522924


namespace pirate_ship_overtake_at_230_pm_l522_522448

def initial_distance := 15 -- miles
def pirate_ship_initial_speed := 14 -- mph
def trading_vessel_speed := 10 -- mph
def reduced_pirate_ship_speed := 12 -- mph
def start_time := 10 -- hours (10:00 a.m. corresponds to 10 in 24-hour format)

def pirate_ship_overtake_time :=
  let initial_relative_speed := pirate_ship_initial_speed - trading_vessel_speed
  let distance_covered_by_pirate_ship_in_3hrs := pirate_ship_initial_speed * 3
  let distance_covered_by_trading_vessel_in_3hrs := trading_vessel_speed * 3
  let remaining_distance := initial_distance + distance_covered_by_trading_vessel_in_3hrs - distance_covered_by_pirate_ship_in_3hrs
  let new_relative_speed := reduced_pirate_ship_speed - trading_vessel_speed
  let time_to_overtake_remaining_distance := remaining_distance / new_relative_speed
  let total_time_hours := 3 + time_to_overtake_remaining_distance -- 3 hours chase + additional time
  let total_overtake_time := start_time + total_time_hours
  if total_overtake_time >= 12 then -- Convert to 12-hour format if necessary
    (total_overtake_time - 12, "p.m.")
  else
    (total_overtake_time, "a.m.")

theorem pirate_ship_overtake_at_230_pm : pirate_ship_overtake_time = (2.5, "p.m.") := by
  sorry

end pirate_ship_overtake_at_230_pm_l522_522448


namespace trapezoid_area_l522_522468

theorem trapezoid_area (a b d1 d2: ℝ) (h1: a = 36) (h2: b = 60) (h3: d1 = 48) (h4: d2 = 48) :
  let height := (36 * 48) / 60 in
  let shorter_base := 60 - 2 * (Real.sqrt (36^2 - height^2)) in
  (a + shorter_base) * height / 2 = 1105.92 :=
by
  sorry

end trapezoid_area_l522_522468


namespace valid_votes_l522_522021

theorem valid_votes (V : ℝ) 
  (h1 : 0.70 * V - 0.30 * V = 176): V = 440 :=
  sorry

end valid_votes_l522_522021


namespace total_time_is_correct_l522_522321

-- Definitions based on conditions
def timeouts_for_running : ℕ := 5
def timeouts_for_throwing_food : ℕ := 5 * timeouts_for_running - 1
def timeouts_for_swearing : ℕ := timeouts_for_throwing_food / 3

-- Definition for total time-outs
def total_timeouts : ℕ := timeouts_for_running + timeouts_for_throwing_food + timeouts_for_swearing
-- Each time-out is 5 minutes
def timeout_duration : ℕ := 5

-- Total time in minutes
def total_time_in_minutes : ℕ := total_timeouts * timeout_duration

-- The proof statement
theorem total_time_is_correct : total_time_in_minutes = 185 := by
  sorry

end total_time_is_correct_l522_522321


namespace range_of_k_l522_522885

def p (k : ℝ) : Prop := k^2 + 3 * k - 4 ≤ 0
def q (k : ℝ) : Prop := ∀ x > 0, x + k + 1 / x ≥ 0

theorem range_of_k :
  ¬ (∃ k : ℝ, p k ∧ q k) ∧ (∃ k : ℝ, p k ∨ q k) → ∀ k : ℝ, k ∈ set.Icc (-4 : ℝ) (-2) ∪ set.Ioc (1 : ℝ) ⊤ :=
sorry

end range_of_k_l522_522885


namespace translation_left_by_pi_six_gives_new_function_l522_522738

-- Define the original function
def original_function (x : ℝ) : ℝ := sin x

-- Define the translation amount
def translation_amount : ℝ := π / 6

-- Define the new function expected after translation
def new_function (x : ℝ) : ℝ := sin (x + translation_amount)

-- Formulate the proof problem
theorem translation_left_by_pi_six_gives_new_function :
  ∀ x : ℝ, new_function x = sin (x + π / 6) :=
by
  intros
  sorry

end translation_left_by_pi_six_gives_new_function_l522_522738


namespace increase_of_100_by_50_percent_is_150_l522_522776

theorem increase_of_100_by_50_percent_is_150 :
  let initial_number := 100
  let increase_percentage := 0.50
  let increased_amount := initial_number * increase_percentage
  let final_number := initial_number + increased_amount
  final_number = 150 :=
by
  let initial_number := 100
  let increase_percentage := 0.50
  let increased_amount := initial_number * increase_percentage
  let final_number := initial_number + increased_amount
  sorry

end increase_of_100_by_50_percent_is_150_l522_522776


namespace parallel_or_contained_l522_522540

variables {Point Line Plane : Type} [HasPerp Line Plane] [HasPerp Line Line] [HasSubset Line Plane] [HasParallel Line Line] [HasParallel Line Plane]

-- Given assumptions:
axiom distinct_lines (l m : Line) (h : l ≠ m) : Prop
axiom distinct_planes (α β γ : Plane) (h : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) : Prop
axiom perpendicular_line_plane (l : Line) (γ : Plane) : l ⊥ γ
axiom perpendicular_plane_plane (α : Plane) (γ : Plane) : α ⊥ γ
axiom parallel_line_plane (l : Line) (α : Plane) : l ∥ α

-- Mathematical equivalent proof problem:
theorem parallel_or_contained (l : Line) (α γ : Plane) (hlγ : l ⊥ γ) (hαγ : α ⊥ γ) : (l ∥ α) ∨ (l ⊆ α) :=
sorry

end parallel_or_contained_l522_522540


namespace sign_pyramid_combinations_l522_522823

-- Define a ±1 data type for easy reference
inductive Sign : Type
| pos : Sign
| neg : Sign

open Sign

/-- A function to determine the sign of a top cell in the pyramid structure given bottom row -/
noncomputable def pyramidTopSign (a b c d e : Sign) : Sign :=
  let ab := if a = b then pos else neg
  let bc := if b = c then pos else neg
  let cd := if c = d then pos else neg
  let de := if d = e then pos else neg
  let ab_bc := if ab = bc then pos else neg
  let bc_cd := if bc = cd then pos else neg
  let cd_de := if cd = de then pos else neg
  let ab_bc_bc_cd := if ab_bc = bc_cd then pos else neg
  let bc_cd_cd_de := if bc_cd = cd_de then pos else neg
  if ab_bc_bc_cd = bc_cd_cd_de then pos else neg

/-- Prove that there are exactly 16 ways to fill the bottom cells so all intermediate top cells are "+" -/
theorem sign_pyramid_combinations : 
  {l : List Sign // l.length = 5 ∧ pyramidTopSign (l.nth 0).get_or_else pos
                                         (l.nth 1).get_or_else pos 
                                         (l.nth 2).get_or_else pos 
                                         (l.nth 3).get_or_else pos 
                                         (l.nth 4).get_or_else pos = pos} 
  = 16 := sorry

end sign_pyramid_combinations_l522_522823


namespace range_of_a_plus_b_l522_522594

theorem range_of_a_plus_b (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : |Real.log a| = |Real.log b|) (h₄ : a ≠ b) :
  2 < a + b :=
by
  sorry

end range_of_a_plus_b_l522_522594


namespace total_cost_for_3000_pencils_correct_l522_522434

-- Define the given conditions
def cost_per_box : ℕ := 50
def pencils_per_box : ℕ := 200
def discount : ℝ := 0.10
def total_pencils : ℕ := 3000

-- Define the target cost
def expected_total_cost : ℝ := 700.0

-- Define helper calculations
def cost_per_pencil : ℝ := cost_per_box / pencils_per_box
def cost_first_1000 : ℝ := 1000 * cost_per_pencil
def cost_next_1000 : ℝ := (1000 * cost_per_pencil) * (1 - discount) 
def total_disounted_cost : ℝ := cost_first_1000 + 2 * cost_next_1000

-- The final theorem to be proven
theorem total_cost_for_3000_pencils_correct : 
  total_disounted_cost = expected_total_cost := 
by
  sorry

end total_cost_for_3000_pencils_correct_l522_522434


namespace bus_stop_time_l522_522126

theorem bus_stop_time (speed_excluding_stoppages speed_including_stoppages : ℝ) 
  (h1 : speed_excluding_stoppages = 60)
  (h2 : speed_including_stoppages = 50) :
  let stopping_time := ((speed_excluding_stoppages - speed_including_stoppages) / speed_excluding_stoppages) * 60 in
  stopping_time = 10 :=
by
  -- proof omitted
  sorry

end bus_stop_time_l522_522126


namespace added_number_is_6_l522_522445

theorem added_number_is_6 : ∃ x : ℤ, (∃ y : ℤ, y = 9 ∧ (2 * y + x) * 3 = 72) → x = 6 := 
by
  sorry

end added_number_is_6_l522_522445


namespace simplify_expression_l522_522679

theorem simplify_expression : (sqrt (3 * 5) * sqrt (5^2 * 3^3) = 45 * sqrt 5) :=
by
  sorry

end simplify_expression_l522_522679


namespace range_of_a_l522_522569

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 1 < x → a * log x > 1 - 1 / x) ↔ a ≥ 1 := 
sorry

end range_of_a_l522_522569


namespace isosceles_right_triangle_area_l522_522074

theorem isosceles_right_triangle_area
  (a b c : ℝ) 
  (h1 : a = b) 
  (h2 : c = a * Real.sqrt 2) 
  (area : ℝ) 
  (h_area : area = 50)
  (h3 : (1/2) * a * b = area) :
  (a + b + c) / area = 0.4 + 0.2 * Real.sqrt 2 :=
by
  sorry

end isosceles_right_triangle_area_l522_522074


namespace closest_point_on_parabola_to_line_l522_522695

theorem closest_point_on_parabola_to_line :
  ∃ (x y : ℝ), y = x^2 ∧ (∀ x1 y1, y1 = x1^2 → d (x, y) ≤ d (x1, y1)) ∧ x = 1 ∧ y = 1
  where 
    d (P Q : ℝ × ℝ) : ℝ := |2 * P.1 - P.2 - 4| / real.sqrt 5 :=
proof
  sorry

end closest_point_on_parabola_to_line_l522_522695


namespace slope_of_symmetric_line_l522_522243

open Real

theorem slope_of_symmetric_line :
  let A := (4, 0)
  let B := (0, 2)
  (∃ l : ℝ, is_symmetric_with_respect_to_line A B l) → slope l = 2 :=
sorry

-- definition for is_symmetric_with_respect_to_line would be included if needed.

end slope_of_symmetric_line_l522_522243


namespace mean_score_l522_522419

theorem mean_score (mu sigma : ℝ) 
  (h1 : 86 = mu - 7 * sigma) 
  (h2 : 90 = mu + 3 * sigma) :
  mu = 88.8 :=
by
  -- skipping the proof
  sorry

end mean_score_l522_522419


namespace mode_expected_median_l522_522052

noncomputable def pdf (x : ℝ) : ℝ :=
  if 2 < x ∧ x < 4 then -3 / 4 * x^2 + 9 / 2 * x - 6 else 0

theorem mode_expected_median (X : ℝ) (hX1 : ∀ x, pdf x ≥ 0)
  (hX2 : ∫ x in Ioc 2 4, pdf x = 1) :
  (∃ M0, (∀ y, pdf y ≤ pdf M0) ∧ M0 = 3) ∧
  (∃ μ, μ = ∫ x in Ioc 2 4, x * pdf x ∧ μ = 3) ∧
  (∃ Me, (∫ x in Ioc 2 Me, pdf x = ∫ x in Me 4, pdf x) ∧ Me = 3) :=
sorry

end mode_expected_median_l522_522052


namespace budget_spent_on_salaries_l522_522436

theorem budget_spent_on_salaries :
  ∀ (B R U E S T : ℕ),
  R = 9 ∧
  U = 5 ∧
  E = 4 ∧
  S = 2 ∧
  T = (72 * 100) / 360 → 
  B = 100 →
  (B - (R + U + E + S + T)) = 60 :=
by sorry

end budget_spent_on_salaries_l522_522436


namespace point_coordinates_l522_522985

theorem point_coordinates (m : ℝ) 
  (h1 : dist (0 : ℝ) (Real.sqrt m) = 4) : 
  (-m, Real.sqrt m) = (-16, 4) := 
by
  -- The proof will use the conditions and solve for m to find the coordinates
  sorry

end point_coordinates_l522_522985


namespace remaining_painting_time_l522_522788

-- Define the given conditions as Lean definitions
def total_rooms : ℕ := 9
def hours_per_room : ℕ := 8
def rooms_painted : ℕ := 5

-- Formulate the main theorem to prove the remaining time is 32 hours
theorem remaining_painting_time : 
  (total_rooms - rooms_painted) * hours_per_room = 32 := 
by 
  sorry

end remaining_painting_time_l522_522788


namespace tangent_bd_circumcircle_tsh_l522_522987

variables {A B C D H S T : Type*}
variables [convex_quadrilateral ABCD]
variables (∠ABC : ∠ABC = 90°) (∠CDA : ∠CDA = 90°)
variables (foot: H = foot_of_perpendicular A BD)
variables (on_segment_S : S ∈ segment A B) (on_segment_T : T ∈ segment A D)
variables (H_inside_SCT : H ∈ inside_triangle SCT)
variables (angle_CHS_CSB : ∠CHS - ∠CSB = 90°) (angle_THC_DTC : ∠THC - ∠DTC = 90°)

theorem tangent_bd_circumcircle_tsh :
  tangent_line_circumcircle BD (circumcircle T S H) :=
sorry

end tangent_bd_circumcircle_tsh_l522_522987


namespace sold_out_performance_revenue_l522_522356

-- Define the conditions
def overhead_cost : ℕ := 81000
def production_cost_per_performance : ℕ := 7000
def performances : ℕ := 9

-- Define the proof problem
theorem sold_out_performance_revenue :
  ∃ (x : ℕ), (performances * x = overhead_cost + (performances * production_cost_per_performance)) ∧ x = 16000 :=
begin
  use 16000,
  split,
  { -- Verification of the equation
    calc
      performances * 16000
        = 9 * 16000 : by refl
    ... = 144000 : by norm_num
    ... = overhead_cost + (performances * production_cost_per_performance) : by norm_num },
  { refl }
end

end sold_out_performance_revenue_l522_522356


namespace max_difference_minuend_subtrahend_l522_522831

theorem max_difference_minuend_subtrahend :
  ∃ (a b c d e f g h i : ℕ),
    -- Minuend: abc
    a ∈ {3, 5, 9} ∧ b ∈ {2, 3, 7} ∧ c ∈ {3, 4, 8, 9} ∧
    -- Subtrahend: def
    d ∈ {2, 3, 7} ∧ e ∈ {3, 5, 9} ∧ f ∈ {1, 4, 7} ∧
    -- Result: ghi
    g ∈ {4, 5, 9} ∧ h = 2 ∧ i ∈ {4, 5, 9} ∧
    -- Difference condition
    let abc := 100 * a + 10 * b + c in
    let def := 100 * d + 10 * e + f in
    let ghi := 100 * g + 10 * h + i in
    -- Three-digit number condition
    100 ≤ ghi ∧ ghi < 1000 ∧
    -- Correct maximum difference is obtained
    ghi = abc - def ∧
    -- Prove the maximum difference given
    abc = 923 ∧ def = 394 ∧ ghi = 529 :=
by {
  sorry
}

end max_difference_minuend_subtrahend_l522_522831


namespace jayden_current_age_l522_522606

def current_age_of_Jayden (e : ℕ) (j_in_3_years : ℕ) : ℕ :=
  j_in_3_years - 3

theorem jayden_current_age (e : ℕ) (h1 : e = 11) (h2 : ∃ j : ℕ, j = ((e + 3) / 2) ∧ j_in_3_years = j) : 
  current_age_of_Jayden e j_in_3_years = 4 :=
by
  sorry

end jayden_current_age_l522_522606


namespace ball_box_distribution_l522_522946

theorem ball_box_distribution : (∃ (f : Fin 4 → Fin 2), true) ∧ (∀ (f : Fin 4 → Fin 2), true) → ∃ (f : Fin 4 → Fin 2), true ∧ f = 16 :=
by sorry

end ball_box_distribution_l522_522946


namespace triangle_area_x_value_l522_522864

theorem triangle_area_x_value :
  ∃ x : ℝ, x > 0 ∧ 100 = (1 / 2) * x * (3 * x) ∧ x = 10 * Real.sqrt 6 / 3 :=
sorry

end triangle_area_x_value_l522_522864


namespace possible_values_of_expression_l522_522180

noncomputable def sign (x : ℝ) : ℝ :=
if x > 0 then 1 else -1

theorem possible_values_of_expression
  (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := (sign a) + (sign b) + (sign c) + (sign d) + (sign (a * b * c * d))
  in expr ∈ {5, 1, -1, -5} :=
by
  let expr := (sign a) + (sign b) + (sign c) + (sign d) + (sign (a * b * c * d))
  sorry

end possible_values_of_expression_l522_522180


namespace angle_between_a_and_c_l522_522575

variables {V : Type*} [inner_product_space ℝ V] (a b c : V)
variables (m : ℝ) (h₀ : m ≠ 0)
variables (ha : ∥a∥ = m) (hb : ∥b∥ = m) (hc : ∥c∥ = m)
variable (h : a + b = √3 • c)

theorem angle_between_a_and_c : real.angle a c = real.angle.pi_div_six :=
begin
  sorry
end

end angle_between_a_and_c_l522_522575


namespace find_seventh_number_in_sequence_l522_522717

theorem find_seventh_number_in_sequence :
  let seq := λ n, match n with
                  | 0 => 11
                  | 1 => 23
                  | 2 => 47
                  | 3 => 83
                  | 4 => 131
                  | 5 => 191
                  | n+6 => seq (n + 5) + (72 + n*12)
                  end in
  seq 6 = 263 :=
by
  -- The proof would go here.
  sorry

end find_seventh_number_in_sequence_l522_522717


namespace hamburgers_leftover_at_end_of_lunch_service_l522_522054

theorem hamburgers_leftover_at_end_of_lunch_service : 
  let initial_hamburgers := 25
  let first_hour_served_hamburgers := 12
  let second_hour_served_hamburgers := 6
  (initial_hamburgers - first_hour_served_hamburgers - second_hour_served_hamburgers) = 7 :=
by
  let initial_hamburgers := 25
  let first_hour_served_hamburgers := 12
  let second_hour_served_hamburgers := 6
  have h := initial_hamburgers - first_hour_served_hamburgers - second_hour_served_hamburgers
  exact eq.symm (eq.trans (sub_eq_of_eq_add (rfl : initial_hamburgers - first_hour_served_hamburgers = 13))
    (sub_eq_of_eq_add (rfl : 13 - second_hour_served_hamburgers = 7)))
  sorry

end hamburgers_leftover_at_end_of_lunch_service_l522_522054


namespace solution_set_inequality_l522_522562

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x + x^2

theorem solution_set_inequality : 
  { x : ℝ | f (Real.log x) + f (Real.log (1 / x)) < 2 * f 1 } = 
  Ioo (1 / Real.exp 1) (Real.exp 1) := 
sorry

end solution_set_inequality_l522_522562


namespace symmetric_point_l522_522366

theorem symmetric_point (a b : ℝ) :
  -- Conditions
  let M := (2 : ℝ, 1 : ℝ) in
  let L := { p : ℝ × ℝ | p.1 + p.2 + 1 = 0 } in
  -- Question
  (∃ (a b : ℝ), 
    let P := (a, b) in
    -- Definition of symmetric to the line
    ∃ (symm_cond1 : (b - 1) / (a - 2) * (-1) = -1)
    (symm_cond2 : (2 + a) / 2 + (1 + b) / 2 + 1 = 0),
      -- Correct Answer
      P = (-2, -3)) := sorry

end symmetric_point_l522_522366


namespace largest_three_digit_geometric_sequence_with_8_l522_522404

theorem largest_three_digit_geometric_sequence_with_8 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ n = 842 ∧ (∃ (a b c : ℕ), n = 100*a + 10*b + c ∧ a = 8 ∧ (a * c = b^2) ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) ) :=
by
  sorry

end largest_three_digit_geometric_sequence_with_8_l522_522404


namespace smallest_k_on_circle_with_conditions_l522_522660

def smallest_possible_value_of_k : ℕ := 143

theorem smallest_k_on_circle_with_conditions :
  ∀ (points : list (ℕ × ℕ)) (colors : fin 1000 → fin smallest_possible_value_of_k), 
  (∀ (s1 s2 s3 s4 s5 : fin 1000 → ℕ), 
    endpoints s1 ≠ endpoints s2 ∧
    endpoints s2 ≠ endpoints s3 ∧ 
    endpoints s3 ≠ endpoints s4 ∧ 
    endpoints s4 ≠ endpoints s5 ∧ 
    endpoints s1 ≠ endpoints s3 ∧ 
    endpoints s1 ≠ endpoints s4 ∧ 
    endpoints s1 ≠ endpoints s5 ∧ 
    endpoints s2 ≠ endpoints s4 ∧ 
    endpoints s2 ≠ endpoints s5 ∧ 
    endpoints s3 ≠ endpoints s5 → 
    ∃ (seg1 seg2 seg3 : fin smallest_possible_value_of_k),
    endpoints seg1.index.1 ≠ endpoints seg1.index.2 ∧
    endpoints seg2.index.1 ≠ endpoints seg2.index.2 ∧
    endpoints seg3.index.1 ≠ endpoints seg3.index.2) :=
by { sorry }

end smallest_k_on_circle_with_conditions_l522_522660


namespace rhinoceros_folds_l522_522055

theorem rhinoceros_folds :
  ∃ (v h : ℕ), v + h = 17 ∧ 
    (∀ n, n > 0 → (by sorry : v ≠ h ∨ h ≠ v) ∧ 
     ∀ v' h', (v' = h) ∧ (h' = v) → false) :=
by sorry

end rhinoceros_folds_l522_522055


namespace relative_stacked_product_2014_l522_522822

-- Define the series and its product
def series (n : ℕ) := fin n → ℝ

def product (s : series n) := ∏ i, s i

-- Given conditions
axiom relative_stacked_product_2013 (a : series 2013) 
  (ha_pos : ∀ i, 0 < a i) : 
  real.log (product a) = 2013

-- Proof problem statement
theorem relative_stacked_product_2014 (a : series 2013) 
  (ha_pos : ∀ i, 0 < a i) : real.log (10 * product (λ i, 10 * a i)) = 4027 :=
sorry

end relative_stacked_product_2014_l522_522822


namespace num_sets_consecutive_integers_sum_150_l522_522080

theorem num_sets_consecutive_integers_sum_150 :
  (∃ c : ℕ, c = (finset.filter (λ n : ℕ, ∃ a : ℤ, 2 * a = (300 / n) - n + 1 ∧ a > 0
    ∧ 300 % n = 0 ∧ (300 / n) - n + 1).card) = 4 :=
by {
  -- Proof is omitted
  sorry
}

end num_sets_consecutive_integers_sum_150_l522_522080


namespace meadowbrook_total_not_74_l522_522993

theorem meadowbrook_total_not_74 (h c : ℕ) : 
  21 * h + 6 * c ≠ 74 := sorry

end meadowbrook_total_not_74_l522_522993


namespace gain_percent_correct_l522_522597

-- Define the cost price and selling price
variables (C S : ℝ)

-- Define the condition
def condition (C S : ℝ) : Prop := 80 * C = 58 * S

-- Define the gain percent calculation
def gain_percent (C S : ℝ) : ℝ := ((S - C) / C) * 100

-- Define the proof statement
theorem gain_percent_correct (h : condition C S) : gain_percent C S = 37.93 :=
by {
  -- Substitute the condition 80 C = 58 S to find S in terms of C
  have h₁ : S = (80 / 58) * C,
  from (eq.symm $ eq_div_iff_mul_eq.mpr h).symm,
  -- Substituting S in the gain percent formula
  have h₂ : gain_percent C S = ((S - C) / C) * 100,
  from rfl,
  rw [h₁, h₂],
  -- Simplifying the calculation to get the correct gain percent
  calc ((80 / 58) * C - C) / C * 100
      = (22 / 58) * 100 : 
  by {
    simp [sub_div, mul_sub, mul_div_cancel, ne_of_gt],
    norm_cast,
  }
}

end gain_percent_correct_l522_522597


namespace product_midpoint_l522_522749

theorem product_midpoint (x1 y1 x2 y2 : ℝ) :
  x1 = 4 → y1 = -1 → x2 = -2 → y2 = 7 →
  let mx := (x1 + x2) / 2 in
  let my := (y1 + y2) / 2 in
  mx * my = 3 :=
by {
  intros,
  simp,
  sorry,
}

end product_midpoint_l522_522749


namespace midpoint_trajectory_ratio_of_areas_l522_522213

open Real

noncomputable def parabola : (ℝ × ℝ) → Prop :=
λ p, (p.2)^2 = 4 * p.1

noncomputable def focus : (ℝ × ℝ) := (1, 0)

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def line_passing_through_focus (k : ℝ) : (ℝ × ℝ) → Prop :=
λ p, p.2 = k * (p.1 - 1)

theorem midpoint_trajectory :
  ∀ (A B : ℝ × ℝ) (k : ℝ), (parabola A) ∧ (parabola B) ∧
  (line_passing_through_focus k A) ∧ (line_passing_through_focus k B) →
  parabola (midpoint A B) :=
sorry

noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)

noncomputable def O : ℝ × ℝ := (0, 0)

noncomputable def P (y1 : ℝ) : ℝ × ℝ := (-4, -16 / y1)
noncomputable def Q (y2 : ℝ) : ℝ × ℝ := (-4, -16 / y2)

theorem ratio_of_areas :
  ∀ (A B M : ℝ × ℝ), (parabola A) ∧ (parabola B) ∧ (midpoint A B = M) →
  ∃ (y1 y2 : ℝ), (P y1, Q y2) ∧ 
  32 * (area_triangle B O M) = area_triangle O (P y1) (Q y2) :=
sorry

end midpoint_trajectory_ratio_of_areas_l522_522213


namespace find_k_l522_522225

variables (a b : ℝ × ℝ) (k : ℝ)
#check (⟨2, 1⟩ : ℝ × ℝ) -- statement only to double-check type, if necessary

def a := (2, 1)
def b := (-3, k)

-- Dot product function for two 2D vectors
def dot (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Condition 1: Definitions of vectors
def a_def : ℝ × ℝ := (2, 1)
def b_def : ℝ × ℝ := (-3, k)

-- Condition 2: Given dot product equation
def dot_product_condition : Prop := dot a (2 • a - b) = 0

-- The main theorem to prove that k = 16 under the given conditions
theorem find_k (h : dot_product_condition) : k = 16 :=
begin
  sorry,
end

end find_k_l522_522225


namespace minimize_G_l522_522235

noncomputable def F (p q : ℝ) : ℝ :=
  2 * p * q + 4 * p * (1 - q) + 2 * (1 - p) * q - 5 * (1 - p) * (1 - q)

noncomputable def G (p : ℝ) : ℝ :=
  max (F p 0) (F p 1)

theorem minimize_G :
  ∀ (p : ℝ), 0 ≤ p ∧ p ≤ 0.75 → G p = G 0 → p = 0 :=
by
  intro p hp hG
  -- The proof goes here
  sorry

end minimize_G_l522_522235


namespace points_distance_product_gt_bound_l522_522427

theorem points_distance_product_gt_bound (n : ℕ) (d : ℝ)
  (P : Fin (n+1) → ℝ × ℝ)
  (h_dgt0 : d > 0)
  (h_min_dist : ∀ i j : Fin (n+1), i ≠ j → dist (P i) (P j) ≥ d) :
  (finset.univ.product (λ k, dist (P 0) (P k))) > (d / 3)^n * real.sqrt ((nat.factorial (n + 1)).to_real) :=
sorry

end points_distance_product_gt_bound_l522_522427


namespace probability_red_greater_blue_and_less_than_twice_blue_l522_522050

-- Define the problem conditions and statement
theorem probability_red_greater_blue_and_less_than_twice_blue :
  let x_range := set.Icc (0: ℝ) (2: ℝ),
      area_square := 4,
      valid_region_area := 1
  in (valid_region_area / area_square) = (1 / 4) :=
by sorry

end probability_red_greater_blue_and_less_than_twice_blue_l522_522050


namespace supremum_expression_l522_522135

theorem supremum_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 1) : 
  ∃ x, is_sup ((- (1 / (2 * a)) - (2 / b))) x ∧ x = (- 9 / 2) := sorry

end supremum_expression_l522_522135


namespace transform_equation_l522_522954

theorem transform_equation (x y : ℝ) (h : y = x + x⁻¹) :
  x^4 + x^3 - 5 * x^2 + x + 1 = 0 ↔ x^2 * (y^2 + y - 7) = 0 := 
sorry

end transform_equation_l522_522954


namespace time_to_pass_l522_522742

def length_of_each_train : ℝ := 25  -- meters
def speed_of_faster_train : ℝ := 46  -- km/hr
def speed_of_slower_train : ℝ := 36  -- km/hr

theorem time_to_pass : 
  let relative_speed := (speed_of_faster_train - speed_of_slower_train) * (1000 / 3600) in -- m/s
  let total_distance := 2 * length_of_each_train in -- meters
  total_distance / relative_speed = 18 :=
by
  sorry

end time_to_pass_l522_522742


namespace conjugate_of_z_l522_522557

def z : ℂ := 2 - I

theorem conjugate_of_z : complex.conj z = 2 + I :=
by
  -- Proof goes here
  sorry

end conjugate_of_z_l522_522557


namespace no_solution_for_sum_eq_one_l522_522834

theorem no_solution_for_sum_eq_one (x : ℕ) (hx : x > 1) : 
  (∑ k in finset.range (x^2 + 1), (1 : ℝ) / (x + k)) ≠ 1 :=
by
  sorry

end no_solution_for_sum_eq_one_l522_522834


namespace sum_of_m_for_minimal_area_triangle_l522_522381

theorem sum_of_m_for_minimal_area_triangle :
  let points := [(2, 8), (14, 17), (6, m)] in
  let m_min_area := [10, 12] in
  sum m_min_area = 22 :=
sorry

end sum_of_m_for_minimal_area_triangle_l522_522381


namespace sufficient_condition_not_necessary_condition_l522_522236

variable {a b : ℝ} 

theorem sufficient_condition (h : a < b ∧ b < 0) : a ^ 2 > b ^ 2 :=
sorry

theorem not_necessary_condition : ¬ (∀ {a b : ℝ}, a ^ 2 > b ^ 2 → a < b ∧ b < 0) :=
sorry

end sufficient_condition_not_necessary_condition_l522_522236


namespace possible_values_of_a1_l522_522718

def sequence_satisfies_conditions (a : ℕ → ℕ) : Prop :=
  (∀ n ≥ 1, a n ≤ a (n + 1) ∧ a (n + 1) ≤ a n + 5) ∧
  (∀ n ≥ 1, n ∣ a n)

theorem possible_values_of_a1 (a : ℕ → ℕ) :
  sequence_satisfies_conditions a → ∃ k ≤ 26, a 1 = k :=
by
  sorry

end possible_values_of_a1_l522_522718


namespace find_avg_mpg_first_car_l522_522805

def avg_mpg_first_car (x : ℝ) : Prop :=
  let miles_per_month := 450 / 3
  let gallons_first_car := miles_per_month / x
  let gallons_second_car := miles_per_month / 10
  let gallons_third_car := miles_per_month / 15
  let total_gallons := 56 / 2
  gallons_first_car + gallons_second_car + gallons_third_car = total_gallons

theorem find_avg_mpg_first_car : avg_mpg_first_car 50 :=
  sorry

end find_avg_mpg_first_car_l522_522805


namespace distance_after_1991_jumps_l522_522536

theorem distance_after_1991_jumps 
  (A B C P : Point) 
  (d₁ d₂ d₃ : ℕ) 
  (h1 : |P - C| = 27) 
  (h2 : P₁ = 2 * A - P) 
  (h3 : P₂ = 2 * B - P₁) 
  (h4 : P₂ = 2 * B - (2 * A - P))
  (h5 : P₃ = 2 * C - P₂)
  (h6 : P₃ = 2 * C - (2 * B - (2 * A - P)))
  (h7 : ∀ k, Pinit = if k % 6 = 0 then P₀ else if k % 6 = 1 then P₁ else if k % 6 = 2 then P₂ else if k % 6 = 3 then P₃ else if k % 6 = 4 then P₄ else 2 * C - P):
|P₅ - P| = 54 :=
by sorry

end distance_after_1991_jumps_l522_522536


namespace solution_largest_a_exists_polynomial_l522_522855

def largest_a_exists_polynomial : Prop :=
  ∃ (P : ℝ → ℝ) (a b c d e : ℝ),
    (∀ x, P x = a * x^4 + b * x^3 + c * x^2 + d * x + e) ∧
    (∀ x, -1 ≤ x ∧ x ≤ 1 → 0 ≤ P x ∧ P x ≤ 1) ∧
    a = 4

theorem solution_largest_a_exists_polynomial : largest_a_exists_polynomial :=
  sorry

end solution_largest_a_exists_polynomial_l522_522855


namespace sin_2alpha_minus_cos_squared_alpha_l522_522543

variable (α : Real)
variable (sin_alpha : Real)
variable (cos_alpha : Real)

-- Given conditions
axiom sin_alpha_value : sin_alpha = -4 / 5
axiom cos_alpha_value : α > π ∧ α < (3 * π) / 2 ∧ cos α = -3 / 5

-- The theorem to prove
theorem sin_2alpha_minus_cos_squared_alpha : sin (2 * α) - cos α ^ 2 = 3 / 5 :=
by
  sorry

end sin_2alpha_minus_cos_squared_alpha_l522_522543


namespace necklace_sum_l522_522646

theorem necklace_sum (H J x S : ℕ) (hH : H = 25) (h1 : H = J + 5) (h2 : x = J / 2) (h3 : S = 2 * H) : H + J + x + S = 105 :=
by 
  sorry

end necklace_sum_l522_522646


namespace swimming_speed_l522_522048

theorem swimming_speed (v : ℝ) (water_speed : ℝ) (swim_time : ℝ) (distance : ℝ) :
  water_speed = 8 →
  swim_time = 8 →
  distance = 16 →
  distance = (v - water_speed) * swim_time →
  v = 10 := 
by
  intros h1 h2 h3 h4
  sorry

end swimming_speed_l522_522048


namespace mary_initial_nickels_l522_522324

variable {x : ℕ}

theorem mary_initial_nickels (h : x + 5 = 12) : x = 7 := by
  sorry

end mary_initial_nickels_l522_522324


namespace f_satisfies_equation_l522_522189

noncomputable def f (x : ℝ) : ℝ := (20 / 3) * x * (Real.sqrt (1 - x^2))

theorem f_satisfies_equation (f : ℝ → ℝ) :
  (∀ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), 2 * f (Real.sin x * -1) + 3 * f (Real.sin x) = 4 * Real.sin x * Real.cos x) →
  (∀ x ∈ Set.Icc (-Real.sqrt 2 / 2) (Real.sqrt 2 / 2), f x = (20 / 3) * x * (Real.sqrt (1 - x^2))) :=
by
  intro h
  sorry

end f_satisfies_equation_l522_522189


namespace t_shape_perimeter_l522_522792

theorem t_shape_perimeter (total_area : ℝ) (n_squares : ℕ) (s : ℝ)
  (h1 : n_squares = 6)
  (h2 : total_area = 576)
  (h3 : total_area = n_squares * s^2)
  (h4 : s = real.sqrt 96)
  : 9 * s = 36 * real.sqrt 6 :=
by sorry

end t_shape_perimeter_l522_522792


namespace adoption_days_l522_522049

def initial_puppies : ℕ := 15
def additional_puppies : ℕ := 62
def adoption_rate : ℕ := 7

def total_puppies : ℕ := initial_puppies + additional_puppies

theorem adoption_days :
  total_puppies / adoption_rate = 11 :=
by
  sorry

end adoption_days_l522_522049


namespace value_of_x_l522_522315

def a : ℝ×ℝ := (x, 1)
def b : ℝ×ℝ := (4, x)
def a_opp_b (a b : ℝ×ℝ) : Prop := ∃ λ < 0, a = (λ * b.1, λ * b.2)

theorem value_of_x (x : ℝ) (a : ℝ×ℝ := (x, 1)) (b : ℝ×ℝ := (4, x)) (h : a_opp_b a b) : x = -2 :=
by sorry

end value_of_x_l522_522315


namespace max_sum_x1_x2_x3_l522_522304

theorem max_sum_x1_x2_x3 : 
  ∀ (x1 x2 x3 x4 x5 x6 x7 : ℕ), 
    x1 < x2 → x2 < x3 → x3 < x4 → x4 < x5 → x5 < x6 → x6 < x7 →
    x1 + x2 + x3 + x4 + x5 + x6 + x7 = 159 →
    x1 + x2 + x3 = 61 :=
by
  intros x1 x2 x3 x4 x5 x6 x7 h1 h2 h3 h4 h5 h6 h_sum
  sorry

end max_sum_x1_x2_x3_l522_522304


namespace arc_length_of_circle_given_central_angle_and_radius_l522_522552

theorem arc_length_of_circle_given_central_angle_and_radius :
  ∀ (r θ : ℝ), r = 2 → θ = 2 → (r * θ) = 4 :=
by
  intros r θ hr hθ
  rw [hr, hθ]
  rfl

end arc_length_of_circle_given_central_angle_and_radius_l522_522552


namespace number_of_students_in_first_group_l522_522728

def total_students : ℕ := 24
def second_group : ℕ := 8
def third_group : ℕ := 7
def fourth_group : ℕ := 4
def summed_other_groups : ℕ := second_group + third_group + fourth_group
def students_first_group : ℕ := total_students - summed_other_groups

theorem number_of_students_in_first_group :
  students_first_group = 5 :=
by
  -- proof required here
  sorry

end number_of_students_in_first_group_l522_522728


namespace equation_one_solution_equation_two_solution_l522_522355

variables (x : ℝ)

theorem equation_one_solution (h : 2 * (x + 3) = 5 * x) : x = 2 :=
sorry

theorem equation_two_solution (h : (x - 3) / 0.5 - (x + 4) / 0.2 = 1.6) : x = -9.2 :=
sorry

end equation_one_solution_equation_two_solution_l522_522355


namespace find_residue_l522_522860

theorem find_residue : 
  (207 * 13 - 22 * 8 + 5) % 17 = 3 := 
by {
  have h1 : 207 % 17 = 15 := by norm_num,
  have h2 : 22 % 17 = 5 := by norm_num,
  have h3 : (207 * 13) % 17 = (15 * 13) % 17 := by rw [h1],
  have h4 : (15 * 13) % 17 = 4 := by norm_num,
  have h5 : (22 * 8) % 17 = (5 * 8) % 17 := by rw [h2],
  have h6 : (5 * 8) % 17 = 6 := by norm_num,
  have h7 : (207 * 13 - 22 * 8 + 5) % 17 = (4 - 6 + 5) % 17 := by rw [h4, h6],
  norm_num at h7,
  exact h7
}

end find_residue_l522_522860


namespace count_zero_sequences_l522_522291

def a_sequence (a_1 a_2 a_3 : ℤ) : ℕ → ℤ
| 0     := a_1
| 1     := a_2
| 2     := a_3
| (n+3) := a_sequence (n + 2) * |a_sequence (n + 1) - a_sequence n|

def valid_triple (t : (ℤ × ℤ × ℤ)) : Prop := 
  let (a_1, a_2, a_3) := t 
  1 ≤ a_1 ∧ a_1 ≤ 10 ∧ 1 ≤ a_2 ∧ a_2 ≤ 10 ∧ 1 ≤ a_3 ∧ a_3 ≤ 10

def generate_zero_sequence (t : (ℤ × ℤ × ℤ)) : Prop :=
  ∃ n, a_sequence t.fst t.snd t.snd !n = 0

theorem count_zero_sequences :  ∃ (num_seqs : ℕ), num_seqs = 594 :=
  ∃ num_seqs, num_seqs = 594 := sorry

end count_zero_sequences_l522_522291


namespace smallest_gcd_12a_20b_l522_522951

theorem smallest_gcd_12a_20b (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : Nat.gcd a b = 18) :
  Nat.gcd (12 * a) (20 * b) = 72 := sorry

end smallest_gcd_12a_20b_l522_522951


namespace scientific_notation_example_l522_522625

theorem scientific_notation_example :
  110000 = 1.1 * 10^5 :=
by {
  sorry
}

end scientific_notation_example_l522_522625


namespace metal_rods_in_each_beam_l522_522036

theorem metal_rods_in_each_beam 
  (panels metal_rods_sheets total_rods : ℕ)
  (sheets_per_panel beams_per_panel rods_per_sheet rods_needed : ℕ)
  (h_panels : panels = 10)
  (h_sheets_per_panel : sheets_per_panel = 3)
  (h_beams_per_panel : beams_per_panel = 2)
  (h_rods_per_sheet : rods_per_sheet = 10)
  (h_rods_needed : rods_needed = 380)
  (h_total_rods_sheets : total_rods = 3 * sheets_per_panel * panels * rods_per_sheet)
  (h_total_rods : total_rods = rods_needed - (3 * sheets_per_panel * panels * rods_per_sheet)) :
  let rods_per_beam := (rods_needed - (3 * sheets_per_panel * panels * rods_per_sheet)) / (beams_per_panel * panels) in
  rods_per_beam = 4 :=
sorry

end metal_rods_in_each_beam_l522_522036


namespace log_xy_l522_522948

theorem log_xy : (logb 10 (x * y^5) = 2) ∧ (logb 10 (x^3 * y) = 2) → logb 10 (x * y) = 6 / 7 :=
by
  intro h
  sorry

end log_xy_l522_522948


namespace Hallie_earnings_l522_522579

theorem Hallie_earnings :
  let w := 10
  let hM := 7
  let tM := 18
  let hT := 5
  let tT := 12
  let hW := 7
  let tW := 20
  let mondayEarnings := hM * w + tM
  let tuesdayEarnings := hT * w + tT
  let wednesdayEarnings := hW * w + tW
  let totalEarnings := mondayEarnings + tuesdayEarnings + wednesdayEarnings
  totalEarnings = 240 := by {
    let w := 10
    let hM := 7
    let tM := 18
    let hT := 5
    let tT := 12
    let hW := 7
    let tW := 20
    let mondayEarnings := hM * w + tM
    let tuesdayEarnings := hT * w + tT
    let wednesdayEarnings := hW * w + tW
    let totalEarnings := mondayEarnings + tuesdayEarnings + wednesdayEarnings
    sorry
  }

end Hallie_earnings_l522_522579


namespace combination_20_6_l522_522091

theorem combination_20_6 : Nat.choose 20 6 = 38760 :=
by
  sorry

end combination_20_6_l522_522091


namespace sequence_is_integer_l522_522827

-- Define the sequence (a_n)
def a : ℕ → ℕ
| 0     := 1
| 1     := 1
| 2     := 2
| (n+3) := (a (n+2) * a (n+1) + n.factorial) / a n

-- Prove that every element of the sequence is an integer
theorem sequence_is_integer (n : ℕ) : ∀ k, a k ∈ ℕ := 
sorry

end sequence_is_integer_l522_522827


namespace trigonometric_identity_proof_l522_522183

theorem trigonometric_identity_proof 
  (α : ℝ) 
  (h1 : Real.tan (2 * α) = 3 / 4) 
  (h2 : α ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2))
  (h3 : ∃ x : ℝ, (Real.sin (x + 2) + Real.sin (α - x) - 2 * Real.sin α) = 0) : 
  Real.cos (2 * α) = -4 / 5 ∧ Real.tan (α / 2) = (1 - Real.sqrt 10) / 3 := 
sorry

end trigonometric_identity_proof_l522_522183


namespace solve_y_l522_522830

theorem solve_y (y : ℝ) : (12 - y)^2 = 4 * y^2 ↔ y = 4 ∨ y = -12 := by
  sorry

end solve_y_l522_522830


namespace janet_final_lives_l522_522280

-- Given conditions
def initial_lives : ℕ := 47
def lives_lost_in_game : ℕ := 23
def points_collected : ℕ := 1840
def lives_per_100_points : ℕ := 2
def penalty_per_200_points : ℕ := 1

-- Definitions based on conditions
def remaining_lives_after_game : ℕ := initial_lives - lives_lost_in_game
def lives_earned_from_points : ℕ := (points_collected / 100) * lives_per_100_points
def lives_lost_due_to_penalties : ℕ := points_collected / 200

-- Theorem statement
theorem janet_final_lives : remaining_lives_after_game + lives_earned_from_points - lives_lost_due_to_penalties = 51 :=
by
  sorry

end janet_final_lives_l522_522280


namespace find_values_and_intervals_l522_522561

noncomputable section

open Real

def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := sin (2 * ω * x + φ) - 1

theorem find_values_and_intervals (ω φ k : ℝ) (hω : ω > 0) (hφ : |φ| < π / 2) (h_period : (∃ (T > 0), ∀ x, f x ω φ = f (x + T) ω φ) ∧ T = π / (2 * ω)) (h_point : f 0 ω φ = -(1/2)):
  ω = 2 ∧ φ = π / 6 ∧ 
  (∀ k : ℤ, monotonically_increasing (f x ω φ) (2 * k * π - π / 2) (2 * k * π + π / 2)) ∧ 
  (∀ k', ω' > 0, ∃ x₁ x₂ ∈ [0, π/2], g x' = F x' + k' → g x' = 0 ∧ 0 < k' ∧ k' ≤ 1 - sqrt 3 / 2) :=
sorry

end find_values_and_intervals_l522_522561


namespace solve_for_x_l522_522896

namespace proof_problem

-- Define the operation a * b = 4 * a * b
def star (a b : ℝ) : ℝ := 4 * a * b

-- Given condition rewritten in terms of the operation star
def equation (x : ℝ) : Prop := star x x + star 2 x - star 2 4 = 0

-- The statement we intend to prove
theorem solve_for_x (x : ℝ) : equation x → (x = 2 ∨ x = -4) :=
by
  -- Proof omitted
  sorry

end proof_problem

end solve_for_x_l522_522896


namespace ratio_of_wire_lengths_l522_522813

theorem ratio_of_wire_lengths (b_pieces : ℕ) (b_piece_length : ℕ)
  (c_piece_length : ℕ) (cubes_volume : ℕ) :
  b_pieces = 12 →
  b_piece_length = 8 →
  c_piece_length = 2 →
  cubes_volume = (b_piece_length ^ 3) →
  b_pieces * b_piece_length * cubes_volume
    / (cubes_volume * (12 * c_piece_length)) = (1 / 128) :=
by
  intros h1 h2 h3 h4
  sorry

end ratio_of_wire_lengths_l522_522813


namespace polynomial_root_cubic_sum_l522_522649

theorem polynomial_root_cubic_sum
  (a b c : ℝ)
  (h : ∀ x : ℝ, (Polynomial.eval x (3 * Polynomial.X^3 + 5 * Polynomial.X^2 - 150 * Polynomial.X + 7) = 0)
    → x = a ∨ x = b ∨ x = c) :
  (a + b + 2)^3 + (b + c + 2)^3 + (c + a + 2)^3 = 303 :=
  sorry

end polynomial_root_cubic_sum_l522_522649


namespace smallest_possible_square_area_l522_522031

noncomputable def min_square_area (w1 h1 w2 h2 : ℕ) : ℕ :=
let side_length := max (max w1 h1) (max w2 h2)
in side_length * side_length

theorem smallest_possible_square_area : 
  min_square_area 4 2 3 5 = 25 := 
by sorry

end smallest_possible_square_area_l522_522031


namespace number_of_arrangements_of_4_people_l522_522374

theorem number_of_arrangements_of_4_people : ∃ n : ℕ, n = 4! := 
by 
  existsi 24
  simp [factorial]
  sorry

end number_of_arrangements_of_4_people_l522_522374


namespace length_segment_AD_l522_522637

-- Given the setup of the problem
variables {A B C O D M N : Type}
variables [MetricSpace O] [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace M] [MetricSpace N]
variables (circle : Circle O 333) -- circle centered at O with radius 333
variables (triangle : Triangle ABC) -- triangle ABC inscribed in the circle
variables (midpoint_AB : IsMidpoint M A B) -- M is the midpoint of AB
variables (midpoint_AC : IsMidpoint N A C) -- N is the midpoint of AC
variables (line_AO : Line_through A O) -- line going through points A and O
variables (D_on_line_AO : Lies_on D line_AO) -- D lies on the line through A and O
variables (line_AO_inters_BC : Intersect line_AO BC D) -- Line AO intersects segment BC at D
variables (MN_BO_concur : ∃ P : O, P ∈ circle ∧ Lies_on P (Line_through M N) ∧ Lies_on P (Line_through B O)) -- MN and BO concur on circle

-- BC has a length of 665
variable (length_BC : distance B C = 665)

-- Prove that AD has a length of 444
theorem length_segment_AD : distance A D = 444 := 
sorry -- proof to be added

end length_segment_AD_l522_522637


namespace water_used_on_monday_l522_522361

noncomputable def inverse_proportional_water_use (r : ℕ) (w : ℕ) (k : ℕ) :=
  r * w = k

-- Define the constants and variables
constant sunday_rain : ℕ := 3
constant sunday_water : ℕ := 10
constant monday_rain : ℕ := 5

-- Prove that the water used on Monday is 6 liters
theorem water_used_on_monday : 
  ∃ (monday_water : ℕ), 
    inverse_proportional_water_use sunday_rain sunday_water 30 ∧ 
    monday_rain * monday_water = 30 ∧ 
    monday_water = 6 :=
by
  sorry

end water_used_on_monday_l522_522361


namespace minimum_steiner_tree_distance_unit_square_l522_522139

noncomputable def unit_square_vertices : list (ℝ × ℝ) :=
  [(0,0), (1,0), (0,1), (1,1)]

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def steiner_point (p1 p2 p3 : ℝ × ℝ) : ℝ × ℝ :=
  sorry -- Implementation of Steiner point calculation is complex and omitted here.

def minimum_distance (vertices : list (ℝ × ℝ)) : ℝ :=
  sorry -- Complex function to compute Steiner Tree distance, omitted here but should return 2 * sqrt(2)

theorem minimum_steiner_tree_distance_unit_square :
  minimum_distance unit_square_vertices = 2 * real.sqrt 2 :=
by
  sorry

end minimum_steiner_tree_distance_unit_square_l522_522139


namespace diagonals_in_nine_sided_regular_polygon_l522_522931

noncomputable def diagonals_of_regular_polygon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

noncomputable def longer_diagonals_of_regular_polygon (n : ℕ) : ℕ :=
  (n % 2) * (n / 2 - 2) + (n - 1)

theorem diagonals_in_nine_sided_regular_polygon :
  let n := 9 in
  diagonals_of_regular_polygon n = 27 ∧ longer_diagonals_of_regular_polygon n = 18 :=
by
  let n := 9
  sorry

end diagonals_in_nine_sided_regular_polygon_l522_522931


namespace unique_real_solution_l522_522111

theorem unique_real_solution : ∃! (x : ℝ), (2^(4*x + 2)) * (4^(3*x + 7)) = 8^(5*x + 6) := sorry

end unique_real_solution_l522_522111


namespace trains_clear_time_l522_522024

-- Define the constants based on the problem's conditions
def length_T1 : ℝ := 120  -- meters
def length_T2 : ℝ := 280  -- meters
def speed_T1_kmph : ℝ := 42  -- kmph
def speed_T2_kmph : ℝ := 30  -- kmph

-- Conversion constant from kmph to m/s
def kmph_to_mps : ℝ := 1000 / 3600

-- Convert the speeds to m/s
def speed_T1_mps : ℝ := speed_T1_kmph * kmph_to_mps
def speed_T2_mps : ℝ := speed_T2_kmph * kmph_to_mps

-- Calculate the total length to be covered 
def total_length : ℝ := length_T1 + length_T2

-- Calculate the relative speed
def relative_speed : ℝ := speed_T1_mps + speed_T2_mps

-- Calculate the expected time
def expected_time : ℝ := 20  -- seconds (as per the correct answer)

-- Define the main theorem to be proven
theorem trains_clear_time :
  expected_time = total_length / relative_speed :=
by
  -- Steps to solve the theorem would go here.
  sorry

end trains_clear_time_l522_522024


namespace quotient_when_divided_by_44_l522_522045

theorem quotient_when_divided_by_44 :
  ∃ N Q : ℕ, (N % 44 = 0) ∧ (N % 39 = 15) ∧ (N / 44 = Q) ∧ (Q = 3) :=
by {
  sorry
}

end quotient_when_divided_by_44_l522_522045


namespace distance_one_minute_before_collision_l522_522397

def boat_speed_1 : ℝ := 4 -- speed of the first boat in miles/hr
def boat_speed_2 : ℝ := 20 -- speed of the second boat in miles/hr
def initial_distance : ℝ := 20 -- initial distance in miles
def time_to_collide (distance speed1 speed2 : ℝ) : ℝ := distance / (speed1 + speed2) -- time in hours
def distance_in_one_minute (speed1 speed2 : ℝ) : ℝ := (speed1 + speed2) / 60 -- speed in miles per minute

theorem distance_one_minute_before_collision :
  let combined_speed := boat_speed_1 + boat_speed_2 in
  let time_in_minutes := time_to_collide initial_distance boat_speed_1 boat_speed_2 * 60 - 1 in
  let distance_covered := combined_speed / 60 * time_in_minutes in
  initial_distance - distance_covered = 0.4 :=
by
  sorry

end distance_one_minute_before_collision_l522_522397


namespace color_dominos_l522_522452

-- Declare the size of the grid
def grid_size := 3000

-- Define the type of cell colors
inductive Color
| yellow | blue | red

-- Define the coloring function
def color (i j : ℕ) : Color :=
  match (i - j) % 3 with
  | 0 => Color.yellow
  | 1 => Color.blue
  | _ => Color.red

-- Define the grid and the condition for dominos
structure Grid := (color : ℕ → ℕ → Color)
structure Domino := (cell1 cell2 : ℕ × ℕ)

-- Helper function to check domino positions
def valid_domino (grid : Grid) (domino : Domino) : Prop :=
  let ⟨(i1, j1), (i2, j2)⟩ := domino in
  (i1 ≤ grid_size) ∧ (j1 ≤ grid_size) ∧ (i2 ≤ grid_size) ∧ (j2 ≤ grid_size)
  ∧ ((i1, j1 + 1) = (i2, j2) ∨ (i1 + 1, j1) = (i2, j2)) -- either horizontal or vertical adjacency

-- Define balanced-coloring condition
def balanced_coloring (grid : Grid) : Prop :=
  let counts := {
    yellow := 0,
    blue := 0,
    red := 0,
  } in
  ∑ i in range(grid_size), ∑ j in range(grid_size),
    match grid.color i j with
    | Color.yellow => counts.yellow + 1
    | Color.blue => counts.blue + 1
    | Color.red => counts.red + 1
    | _ => counts
  counts.yellow = counts.blue ∧ counts.blue = counts.red

-- Define no more than two adjacencies condition
def non_touching (grid : Grid) (domino : Domino) : Prop :=
  let ⟨(i1, j1), (i2, j2)⟩ := domino in
  let color1 := grid.color i1 j1 in
  let color2 := grid.color i2 j2 in
  color1 ≠ color2

-- Define the final theorem
theorem color_dominos : ∃ (grid : Grid), (∀ (domino : Domino), valid_domino grid domino) 
  ∧ balanced_coloring grid
  ∧ (∀ (domino : Domino), non_touching grid domino) := 
sorry

end color_dominos_l522_522452


namespace measure_angle_ABC_l522_522868

noncomputable def coneVolume (r h : ℝ) : ℝ :=
  (1 / 3) * π * r^2 * h

theorem measure_angle_ABC :
  ∀ (BC BA : ℝ) (cone_radius cone_height : ℝ),
    BC = 16 →
    cone_radius = 15 →
    coneVolume cone_radius cone_height = 675 * π →
    360 - (cone_radius / BC * 360) = 22.5 :=
by
  intros BC BA cone_radius cone_height hBC hConeRadius hVolume
  have hAngle : 360 - (cone_radius / BC * 360) = 22.5
  { sorry }
  exact hAngle

end measure_angle_ABC_l522_522868


namespace circle_equation_tangent_y_axis_l522_522780

theorem circle_equation_tangent_y_axis (m : ℝ) :
  (∃ p : ℝ × ℝ, p.1 = 3 * m ∧ p.2 = m ∧ (6 - 3 * m) ^ 2 + (1 - m) ^ 2 = 9 * m ^ 2) →
  (9 * m ^ 2 = (x - 3 * m) ^ 2 + (y - m) ^ 2) →
  (m = 1 ∨ m = 37) →
  ((m = 1 → (x - 3)^2 + (y - 1)^2 = 9) ∧
   (m = 37 → (x - 111)^2 + (y - 37)^2 = 9 * 37^2)) :=
begin
  sorry
end

end circle_equation_tangent_y_axis_l522_522780


namespace simplify_polynomial_l522_522678

theorem simplify_polynomial (x : ℝ) : 
  (2 * x^5 - 3 * x^3 + 5 * x^2 - 8 * x + 15) + (3 * x^4 + 2 * x^3 - 4 * x^2 + 3 * x - 7) = 
  2 * x^5 + 3 * x^4 - x^3 + x^2 - 5 * x + 8 :=
by sorry

end simplify_polynomial_l522_522678


namespace possible_values_of_expression_l522_522178

noncomputable def sign (x : ℝ) : ℝ :=
if x > 0 then 1 else -1

theorem possible_values_of_expression
  (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := (sign a) + (sign b) + (sign c) + (sign d) + (sign (a * b * c * d))
  in expr ∈ {5, 1, -1, -5} :=
by
  let expr := (sign a) + (sign b) + (sign c) + (sign d) + (sign (a * b * c * d))
  sorry

end possible_values_of_expression_l522_522178


namespace car_speed_increase_l522_522387

theorem car_speed_increase (x : ℕ) :
  let dists := λ n, 35 + n * x in
  (finset.range 12).sum dists = 552 → x = 2 :=
begin
  sorry
end

end car_speed_increase_l522_522387


namespace delta_is_polynomial_of_degree_k_minus_1_l522_522882

noncomputable def polynomial_decomposition (n k : ℕ) (a : Fin (k+1) → ℝ) : ℝ :=
  ∑ i in Finset.range (k+1), a i * (n ^ (k - i))

theorem delta_is_polynomial_of_degree_k_minus_1
  {k : ℕ}
  (a : Fin (k+1) → ℝ) :
  ∃ b : Fin k → ℝ, 
    polynomial_decomposition (n+1) k a - polynomial_decomposition n k a =
    polynomial_decomposition n (k-1) b :=
sorry

end delta_is_polynomial_of_degree_k_minus_1_l522_522882


namespace closest_to_ground_l522_522708

def height (M N P Q : ℝ) : Prop := 
  N > M ∧ N > P ∧ N > Q

theorem closest_to_ground {M N P Q : ℝ} (h_M : M = -136.1) (h_N : N = -78.9) (h_P : P = -160.4) (h_Q : Q = -80) :
  height M N P Q :=
by
  sorry

end closest_to_ground_l522_522708


namespace find_sum_of_ten_numbers_l522_522528

noncomputable def ten_numbers_sum : Prop :=
  let S1 := {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
  let S2 := {90, 91, 92, 93, 94, 95, 96, 97, 98, 99}
  ∃ (A : Set ℕ) (B : Set ℕ), 
    A ⊆ S1 ∧ B ⊆ S2 ∧ 
    A.card = 5 ∧ B.card = 5 ∧ 
    (∀ x ∈ A, ∀ y ∈ B, (x - y) % 10 ≠ 0) ∧ 
    (∀ x ∈ A, ∀ y ∈ A, x ≠ y → (x - y) % 10 ≠ 0) ∧ 
    (∀ x ∈ B, ∀ y ∈ B, x ≠ y → (x - y) % 10 ≠ 0) ∧
    (A ∪ B).sum = 545

theorem find_sum_of_ten_numbers : ten_numbers_sum :=
  sorry

end find_sum_of_ten_numbers_l522_522528


namespace max_attempts_to_find_password_l522_522446

-- Define the digits in the password
def digits := [2, 9, 6, 6]

-- Define a function to count permutations of a list with possible duplicate elements
def permutation_count (l : List ℕ) : ℕ :=
  let n := l.length
  let freqs := l.foldr (λ x m, m.insertWith (+) x 1) ∅
  let factorial (n : ℕ) : ℕ := n.factorial
  factorial n / freqs.fold (λ _ d k, k * factorial d) 1

-- Prove that the number of permutations of the digits [2, 9, 6, 6] is 12
theorem max_attempts_to_find_password : permutation_count digits = 12 := by
  sorry

end max_attempts_to_find_password_l522_522446


namespace probability_A_given_B_l522_522734

open Set

-- Define the sample space
def Ω : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define event A
def A : Finset ℕ := {2, 3, 5}

-- Define event B
def B : Finset ℕ := {1, 2, 4, 5, 6}

-- Define the probability function
noncomputable def P (s : Finset ℕ) : ℚ :=
  s.card.to_rat / Ω.card.to_rat

-- Prove the conditional probability
theorem probability_A_given_B : P (A ∩ B) / P B = 2 / 5 := by
  sorry

end probability_A_given_B_l522_522734


namespace T_mod_1000_l522_522296

def sum_of_all_four_digit_no_repeats : ℕ :=
  let S := {s : Finset (Fin 10000) | ((∀ x ∈ s.digits 10, x ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9}: Finset ℕ)) ∧
                                    (s.digits 10).nodup ∧ s.card = 4)} in
  S.sum id

def T : ℕ := sum_of_all_four_digit_no_repeats

theorem T_mod_1000 : T % 1000 = 720 := sorry

end T_mod_1000_l522_522296


namespace find_a_minus_b_l522_522685

theorem find_a_minus_b (a b : ℝ) (f g h : ℝ → ℝ) (h_inv : ℝ → ℝ) :
  (∀ x, f x = a * x + b) →
  (∀ x, g x = x^2 + 2 * x + 1) →
  (∀ x, h x = f (g x)) →
  (∀ x, h_inv x = 2 * x + 3) →
  h (h_inv 0.5) = 0 →
  a - b = 7 / 12 :=
begin
  sorry
end

end find_a_minus_b_l522_522685


namespace find_aroon_pin_l522_522471

theorem find_aroon_pin (a b : ℕ) (PIN : ℕ) 
  (h0 : 0 ≤ a ∧ a ≤ 9)
  (h1 : 0 ≤ b ∧ b < 1000)
  (h2 : PIN = 1000 * a + b)
  (h3 : 10 * b + a = 3 * PIN - 6) : 
  PIN = 2856 := 
sorry

end find_aroon_pin_l522_522471


namespace function_increasing_on_interval_l522_522500

theorem function_increasing_on_interval :
  ∀ x : ℝ, (1 / 2 < x) → (x > 0) → (8 * x - 1 / (x^2)) > 0 :=
sorry

end function_increasing_on_interval_l522_522500


namespace eight_pow_n_over_three_eq_512_l522_522842

theorem eight_pow_n_over_three_eq_512 : 8^(9/3) = 512 :=
by
  -- sorry skips the proof
  sorry

end eight_pow_n_over_three_eq_512_l522_522842


namespace solve_x_l522_522898

-- Define the custom multiplication operation *
def custom_mul (a b : ℕ) : ℕ := 4 * a * b

-- Given that x * x + 2 * x - 2 * 4 = 0
def equation (x : ℕ) : Prop := custom_mul x x + 2 * x - 2 * 4 = 0

theorem solve_x (x : ℕ) (h : equation x) : x = 2 ∨ x = -4 := 
by 
  -- proof steps go here
  sorry

end solve_x_l522_522898


namespace betty_oranges_l522_522812

theorem betty_oranges (boxes: ℕ) (oranges_per_box: ℕ) (h1: boxes = 3) (h2: oranges_per_box = 8) : boxes * oranges_per_box = 24 :=
by
  -- proof omitted
  sorry

end betty_oranges_l522_522812


namespace dolls_total_l522_522832

theorem dolls_total (dina_dolls ivy_dolls casey_dolls : ℕ) 
  (h1 : dina_dolls = 2 * ivy_dolls)
  (h2 : (2 / 3 : ℚ) * ivy_dolls = 20)
  (h3 : casey_dolls = 5 * 20) :
  dina_dolls + ivy_dolls + casey_dolls = 190 :=
by sorry

end dolls_total_l522_522832


namespace number_of_correct_propositions_l522_522911

theorem number_of_correct_propositions :
  let prop1 := (∀ A B : α, ∀ (plane : α), A ∈ plane → B ∈ plane → ¬ (A = B → line_of A B ∈ plane))
  let prop2 := (∀ (plane1 plane2 : α), (∃ P, P ∈ plane1 ∧ P ∈ plane2) → ∃ (line : Set α), line ∈ plane1 ∧ line ∈ plane2 ∧ infinite_line line)
  let prop3 := (∀ (l1 l2 l3 : Set α), parallel_lines l1 l2 → parallel_lines l2 l3 → parallel_lines l3 l1 → ∃ (plane : α), coplanar_lines l1 l2 l3 plane)
  let prop4 := (∀ (plane1 plane2 : α), (∃ P Q R : α, collinear P Q R ∧ P ∈ plane1 ∧ Q ∈ plane1 ∧ R ∈ plane1 ∧ P ∈ plane2 ∧ Q ∈ plane2 ∧ R ∈ plane2) → plane1 = plane2)
  (¬ prop1) ∧ prop2 ∧ (¬ prop3) ∧ (¬ prop4) ->
  (let propositions := [prop1, prop2, prop3, prop4] in 
   (propositions.filter id).length = 1) := 
by
  sorry

-- Auxiliary definitions to make the statement complete
def line_of (A B : α) : Set α := sorry
def infinite_line (line : Set α) : Prop := sorry
def parallel_lines (l1 l2 : Set α) : Prop := sorry
def coplanar_lines (l1 l2 l3 : Set α) (plane : α) : Prop := sorry
def collinear (P Q R : α) : Prop := sorry

end number_of_correct_propositions_l522_522911


namespace selling_price_equivalence_l522_522379

noncomputable def cost_price_25_profit : ℝ := 1750 / 1.25
def selling_price_profit := 1520
def selling_price_loss := 1280

theorem selling_price_equivalence
  (cp : ℝ)
  (h1 : cp = cost_price_25_profit)
  (h2 : cp = 1400) :
  (selling_price_profit - cp = cp - selling_price_loss) → (selling_price_loss = 1280) := 
  by
  unfold cost_price_25_profit at h1
  simp [h1] at h2
  sorry

end selling_price_equivalence_l522_522379


namespace area_of_figure_l522_522515

theorem area_of_figure : 
  (∀ x y : ℝ, abs x - 1 ≤ y ∧ y ≤ sqrt (1 - x^2) → (area of the region defined by these inequalities)) = (π / 2 + 1) := 
by
  sorry

end area_of_figure_l522_522515


namespace boy_usual_time_reach_school_l522_522743

theorem boy_usual_time_reach_school (R T : ℝ) (h : (7 / 6) * R * (T - 3) = R * T) : T = 21 := by
  sorry

end boy_usual_time_reach_school_l522_522743


namespace prism_diagonal_length_l522_522429

theorem prism_diagonal_length (x y z : ℝ) (h1 : 4 * x + 4 * y + 4 * z = 24) (h2 : 2 * x * y + 2 * x * z + 2 * y * z = 11) : Real.sqrt (x^2 + y^2 + z^2) = 5 :=
  by
  sorry

end prism_diagonal_length_l522_522429


namespace gcd_360_504_l522_522134

theorem gcd_360_504 : Nat.gcd 360 504 = 72 := by
  sorry

end gcd_360_504_l522_522134


namespace wheel_diameter_l522_522699

theorem wheel_diameter 
  (total_distance : ℝ)
  (num_revolutions : ℕ)
  (h1 : total_distance = 439.824)
  (h2 : num_revolutions = 200)
  :
  let circumference := total_distance / num_revolutions in
  let diameter := circumference / Real.pi in
  diameter ≈ 0.7 
  :=
by
  sorry

end wheel_diameter_l522_522699


namespace solve_inequality_system_l522_522683

theorem solve_inequality_system : 
  (∀ x : ℝ, (1 / 3 * x - 1 ≤ 1 / 2 * x + 1) ∧ ((3 * x - (x - 2) ≥ 6) ∧ (x + 1 > (4 * x - 1) / 3)) → (2 ≤ x ∧ x < 4)) := 
by
  intro x h
  sorry

end solve_inequality_system_l522_522683


namespace solution_l522_522370

noncomputable def problem : Prop :=
  let F := (1 : ℝ, 0 : ℝ)
  let P := (2 : ℝ, 1 : ℝ)
  let O := (0 : ℝ, 0 : ℝ)
  let OF := Real.sqrt (F.1^2 + F.2^2)
  let PF := Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2)
  let area := (1 / 2 : ℝ) * Real.abs (P.1 * F.2 + F.1 * O.2 + O.1 * P.2 - (P.2 * F.1 + F.2 * O.1 + O.2 * P.1))
  (Real.sqrt 2 * OF = PF) → area = (1 / 2 : ℝ)

theorem solution : problem :=
sorry

end solution_l522_522370


namespace common_divisors_count_l522_522944

-- Define the numbers
def num1 := 9240
def num2 := 13860

-- Define the gcd of the numbers
def gcdNum := Nat.gcd num1 num2

-- Prove the number of divisors of the gcd is 48
theorem common_divisors_count : (Nat.divisors gcdNum).card = 48 :=
by
  -- Normally we would provide a detailed proof here
  sorry

end common_divisors_count_l522_522944


namespace coins_in_bag_l522_522032

theorem coins_in_bag (x : ℕ) (h : x + x / 2 + x / 4 = 105) : x = 60 :=
by
  sorry

end coins_in_bag_l522_522032


namespace range_of_a_l522_522214

noncomputable def parabola_locus (x : ℝ) : ℝ := x^2 / 4

def angle_sum_property (a k : ℝ) : Prop :=
  2 * a * k^2 + 2 * k + a = 0

def discriminant_nonnegative (a : ℝ) : Prop :=
  4 - 8 * a^2 ≥ 0

theorem range_of_a (a : ℝ) :
  (- (Real.sqrt 2) / 2) ≤ a ∧ a ≤ (Real.sqrt 2) / 2 :=
  sorry

end range_of_a_l522_522214


namespace proof_problem_l522_522555

variable (a : ℕ → ℝ)

-- Defining the conditions of the problem
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n m k : ℕ, n < m → m < k → 2 * a m = a n + a k

variable (OB OC OA BA AB : ℝ → ℝ)

def vector_condition1 : Prop :=
  BA = (a 3) • OB + (a 2015) • OC

variable (lambda : ℝ)

def vector_condition2 : Prop :=
  AB = lambda • (OC - OB)

theorem proof_problem (h1 : arithmetic_seq a)
  (h2 : vector_condition1 OB OC BA a)
  (h3 : vector_condition2 OB OC AB lambda) :
  a 1 + a 2017 = 0 := 
sorry

end proof_problem_l522_522555


namespace partial_fraction_product_l522_522378

theorem partial_fraction_product :
  ∃ (A B C : ℚ), 
  (∀ x : ℚ, x ≠ 1 ∧ x ≠ -3 ∧ x ≠ 4 → 
    (x^2 - 4) / (x^3 + x^2 - 11 * x - 13) = A / (x - 1) + B / (x + 3) + C / (x - 4)) ∧
  A * B * C = 5 / 196 :=
sorry

end partial_fraction_product_l522_522378


namespace perfect_rectangle_squares_l522_522790

theorem perfect_rectangle_squares (squares : Finset ℕ) 
  (h₁ : 9 ∈ squares) 
  (h₂ : 2 ∈ squares) 
  (h₃ : squares.card = 9) 
  (h₄ : ∀ x ∈ squares, ∃ y ∈ squares, x ≠ y ∧ (gcd x y = 1)) :
  squares = {2, 5, 7, 9, 16, 25, 28, 33, 36} := 
sorry

end perfect_rectangle_squares_l522_522790


namespace log_xy_l522_522947

theorem log_xy : (logb 10 (x * y^5) = 2) ∧ (logb 10 (x^3 * y) = 2) → logb 10 (x * y) = 6 / 7 :=
by
  intro h
  sorry

end log_xy_l522_522947


namespace tangent_parallel_and_point_P_l522_522696

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 3

theorem tangent_parallel_and_point_P (P : ℝ × ℝ) (hP1 : P = (1, f 1)) (hP2 : P = (-1, f (-1))) :
  (f 1 = 3 ∧ f (-1) = 3) ∧ (deriv f 1 = 2 ∧ deriv f (-1) = 2) :=
by
  sorry

end tangent_parallel_and_point_P_l522_522696


namespace original_number_is_10_l522_522253

theorem original_number_is_10 (x : ℝ) (h : 2 * x + 5 = x / 2 + 20) : x = 10 := 
by {
  sorry
}

end original_number_is_10_l522_522253


namespace cost_for_four_bottles_l522_522283

variable (cost_3_bottles : ℝ) (n_bottles : ℝ)

def cost_per_bottle (cost_3_bottles : ℝ) (n_bottles : ℝ) : ℝ :=
  cost_3_bottles / n_bottles

def cost_four_bottles (cost_per_bottle : ℝ) : ℝ :=
  cost_per_bottle * 4

theorem cost_for_four_bottles
  (hc3 : cost_3_bottles = 1.50)
  (hn : n_bottles = 3) :
  cost_four_bottles (cost_per_bottle cost_3_bottles n_bottles) = 2.00 :=
by
  sorry

end cost_for_four_bottles_l522_522283


namespace shorter_leg_of_right_triangle_with_hypotenuse_65_l522_522256

theorem shorter_leg_of_right_triangle_with_hypotenuse_65 (a b : ℕ) (h : a^2 + b^2 = 65^2) : a = 16 ∨ b = 16 :=
by sorry

end shorter_leg_of_right_triangle_with_hypotenuse_65_l522_522256


namespace min_value_a_plus_b_plus_c_l522_522888

theorem min_value_a_plus_b_plus_c (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : 9 * a + 4 * b = a * b * c) : a + b + c ≥ 10 :=
by
  sorry

end min_value_a_plus_b_plus_c_l522_522888


namespace max_line_intersections_l522_522824

-- Definitions for the conditions
def coplanar_circles (n : Nat) : Prop :=
  n = 5

def line_intersects_circles (n : Nat) : Prop :=
  ∀ i : Nat, i < n → (∃ p1 p2 : Point, p1 ≠ p2)

-- Main statement to prove
theorem max_line_intersections (n : Nat) (h1 : coplanar_circles n) (h2 : line_intersects_circles n) :
  ∃ max_points : Nat, max_points = 10 :=
by
  -- We add sorry here to skip the proof
  sorry

end max_line_intersections_l522_522824


namespace odd_function_f_l522_522029

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_f (f_odd : ∀ x : ℝ, f (-x) = - f x)
                       (f_lt_0 : ∀ x : ℝ, x < 0 → f x = x * (x - 1)) :
  ∀ x : ℝ, x > 0 → f x = - x * (x + 1) :=
by
  sorry

end odd_function_f_l522_522029


namespace find_side_b_l522_522995

theorem find_side_b
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : 2 * Real.sin B = Real.sin A + Real.sin C)
  (h2 : Real.cos B = 3 / 5)
  (h3 : (1 / 2) * a * c * Real.sin B = 4) :
  b = 4 * Real.sqrt 6 / 3 := 
sorry

end find_side_b_l522_522995


namespace initial_number_of_players_l522_522390

theorem initial_number_of_players 
  (initial_avg_weight : ℝ) (new_player_weight : ℝ) (new_avg_weight : ℝ)
  (h1 : initial_avg_weight = 180)
  (h2 : new_player_weight = 210)
  (h3 : new_avg_weight = 181.42857142857142)
  : ∃ n : ℕ, (n * initial_avg_weight + new_player_weight = (n + 1) * new_avg_weight) ∧ n = 20 :=
begin
  sorry
end

end initial_number_of_players_l522_522390


namespace find_curve2_l522_522550

-- Define the curves
def curve1 (x y : ℝ) : Prop := x^2 - y^2 = 1
def curve2 (x y : ℝ) (p : ℝ) : Prop := y^2 = 2 * p * x

-- Define the intersection points' property
def intersect_points (x y : ℝ) (p : ℝ) : Prop := curve1 x y ∧ curve2 x y p

-- Point P and circumcircle condition
def passes_through (O M N : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  let x0 := O.1;
  let xt := M.1;
  let yt := N.2;
  M.1 = N.1 ∧ x0 * (P.1 - xt) = yt^2 ∧ yt^2 = 2 * p * x0
  
-- The actual theorem proof statement
theorem find_curve2 (p : ℝ) :
  (∀ x y, intersect_points x y p → passes_through (0, 0) (x, y) (x, -y) (7/2, 0)) →
  p = 3 / 4 :=
sorry

end find_curve2_l522_522550


namespace number_of_special_passwords_l522_522527

/-- Definition of a special password as described in the problem -/
def is_special_password (password : List ℕ) : Prop :=
  password.length = 7 ∧
  ∀ d : ℕ, d > 0 → d ≤ 9 → (count password d = d) → (sublist (repeat d d) password)

/-- The main theorem stating that the number of special passwords is 13 -/
theorem number_of_special_passwords : 
  ∃ passwords : List (List ℕ), (∀ p ∈ passwords, is_special_password p) ∧ passwords.length = 13 :=
sorry

end number_of_special_passwords_l522_522527


namespace problem_part_I_problem_part_II_l522_522920

section part_I

variables {a b : ℝ}
def f (x : ℝ) : ℝ := (a * real.log x) / x + b
def f' (x : ℝ) : ℝ := (a * (1 - real.log x)) / (x ^ 2)
def tangent_at_one := f' 1 = 1 ∧ f 1 = 0

theorem problem_part_I (ha : a = 1) (hb : b = 0) : 
  (∀ x, 0 < x ∧ x < real.exp 1 → 0 < f' x) ∧ 
  (∀ x, x > real.exp 1 → f' x < 0) :=
sorry 

end part_I

section part_II

variables {x1 x2 : ℝ}
def g (x : ℝ) : ℝ := (real.log x) / x
def same_value (x1 x2 : ℝ) : Prop := g x1 = g x2 ∧ x1 ≠ x2

theorem problem_part_II (hx : same_value x1 x2) : x1 + x2 > 2 * real.exp 1 :=
sorry

end part_II

end problem_part_I_problem_part_II_l522_522920


namespace card_distribution_ways_l522_522469

-- Define the card values
def card_value : Type := ℕ
def J : card_value := 1 -- Jack
def Q : card_value := 3 -- Queen
def K : card_value := 5 -- King
def A : card_value := 7 -- Ace

-- Define the total value each player should get
def total_value : ℕ := 16

-- All possible card values
def card_values : list card_value := [A, K, Q, J]

-- Count all possible combinations of cards that sum up to 16 Ft for four cards
def card_combinations : list (list card_value) := 
  [[A, K, Q, J], [A, Q, Q, Q], [K, K, K, J], [K, K, Q, Q], [A, A, J, J]]

-- Condition: Each player receives exactly 4 cards from the combinations listed
def valid_distribution (dist : list (list card_value)) : Prop :=
  dist.length = 4 ∧ all_combinations_are_unique dist ∧ cards_are_distributed_fully dist

-- The main statement to prove
theorem card_distribution_ways : ∃ dist : list (list card_value), valid_distribution(dist) = 24 :=
sorry

end card_distribution_ways_l522_522469


namespace hexagonal_pyramid_cross_section_area_l522_522849

theorem hexagonal_pyramid_cross_section_area :
  ∀ (S A B C D E F : Point) (M N : Point),
  let base_side := 2
  let distance_to_plane := 1
  let BM := median S A B
  let SN := apothem S A F
  plane_P := plane_passing_through C
  plane_is_parallel := plane_P_parallel_to_plane (BM N)
  ∃ section_area,
    section_area = (34 * real.sqrt(3)) / 35 := 
by
  sorry

end hexagonal_pyramid_cross_section_area_l522_522849


namespace sixty_seventh_digit_of_one_seventeenth_is_eight_l522_522745

def decimal_representation_of_one_seventeenth : list ℕ :=
  [0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7].cycle.to_list

def digit_at_position (n : ℕ) (digits : list ℕ) : ℕ :=
  digits.nth_le ((n - 1) % digits.length) (by simp [digits.length, nat.mod_lt])

theorem sixty_seventh_digit_of_one_seventeenth_is_eight :
  digit_at_position 67 decimal_representation_of_one_seventeenth = 8 :=
  sorry

end sixty_seventh_digit_of_one_seventeenth_is_eight_l522_522745


namespace exist_equal_success_rate_l522_522809

noncomputable def S : ℕ → ℝ := sorry -- Definition of S(N), the number of successful free throws

theorem exist_equal_success_rate (N1 N2 : ℕ) 
  (h1 : S N1 < 0.8 * N1) 
  (h2 : S N2 > 0.8 * N2) : 
  ∃ (N : ℕ), N1 ≤ N ∧ N ≤ N2 ∧ S N = 0.8 * N :=
sorry

end exist_equal_success_rate_l522_522809


namespace number_of_lines_through_point_intersect_hyperbola_once_l522_522043

noncomputable def hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 = 1

noncomputable def point_P : ℝ × ℝ :=
  (-4, 1)

noncomputable def line_through (P : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  l P

noncomputable def one_point_intersection (l : ℝ × ℝ → Prop) (H : ℝ → ℝ → Prop) : Prop :=
  ∃! p : ℝ × ℝ, l p ∧ H p.1 p.2

theorem number_of_lines_through_point_intersect_hyperbola_once :
  (∃ (l₁ l₂ : ℝ × ℝ → Prop),
    line_through point_P l₁ ∧
    line_through point_P l₂ ∧
    one_point_intersection l₁ hyperbola ∧
    one_point_intersection l₂ hyperbola ∧
    l₁ ≠ l₂) ∧ ¬ (∃ (l₃ : ℝ × ℝ → Prop),
    line_through point_P l₃ ∧
    one_point_intersection l₃ hyperbola ∧
    ∃! (other_line : ℝ × ℝ → Prop),
    line_through point_P other_line ∧
    one_point_intersection other_line hyperbola ∧
    l₃ ≠ other_line) :=
sorry

end number_of_lines_through_point_intersect_hyperbola_once_l522_522043


namespace bicycle_and_car_speeds_l522_522764

theorem bicycle_and_car_speeds :
  ∃ (bicycle_speed car_speed : ℕ), 
    (bicycle_speed = 15) ∧ 
    (car_speed = 3 * bicycle_speed) ∧ 
    (15 / bicycle_speed - 15 / car_speed = 2 / 3) ∧ 
    (x ≠ 0) ∧
    (car_speed = 45) :=
begin
  sorry
end

end bicycle_and_car_speeds_l522_522764


namespace find_a_l522_522910

theorem find_a (a : ℝ) :
  (binomial 6 3 * 1 + binomial 6 4 * (-2 * a) + binomial 6 5 * a^2 = 56) ↔ (a = -1 ∨ a = 6) := 
by
  sorry

end find_a_l522_522910


namespace cristina_pace_is_4_l522_522328

-- Definitions of the conditions
def head_start : ℝ := 36
def nicky_pace : ℝ := 3
def time : ℝ := 36

-- Definition of the distance Nicky runs
def distance_nicky_runs : ℝ := nicky_pace * time

-- Definition of the total distance Cristina ran to catch up
def distance_cristina_runs : ℝ := distance_nicky_runs + head_start

-- Lean 4 theorem statement to prove Cristina's pace
theorem cristina_pace_is_4 :
  (distance_cristina_runs / time) = 4 := 
by sorry

end cristina_pace_is_4_l522_522328


namespace distance_AB_l522_522261

noncomputable def C1_parametric (θ : ℝ) : ℝ × ℝ := (Real.cos θ, 1 + Real.sin θ)

noncomputable def C1_polar (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ

noncomputable def C2_polar (ρ θ : ℝ) : Prop := ρ^2 * (1 + Real.cos θ^2) = 2

theorem distance_AB :
  let θ := Real.pi / 3
  let ρ1 := 2 * Real.sin θ,
  let ρ2 := 2 * Real.sqrt(10) / 5,
  |ρ1 - ρ2| = Real.sqrt 3 - 2 * Real.sqrt 10 / 5 :=
by
  sorry

end distance_AB_l522_522261


namespace speed_of_river_is_6_l522_522398

-- Define the setup of the problem using the given conditions.
def locations_are_150_km_apart := 150
def boat_A_travel_distance := 90
def boat_B_travel_distance (total_distance : ℕ) (A_distance : ℕ) := total_distance - A_distance
def speed_in_still_water := 30

-- Express the equation where the times taken by both boats are equal.
def boats_meet_at_same_time (river_speed : ℝ) := 
  let time_A := boat_A_travel_distance / (speed_in_still_water + river_speed) in
  let time_B := boat_B_travel_distance locations_are_150_km_apart boat_A_travel_distance / (speed_in_still_water - river_speed) in
  time_A = time_B

-- The proof problem: Given the conditions, prove the speed of the river is 6 km/h.
theorem speed_of_river_is_6 (x : ℝ) (h : boats_meet_at_same_time x) : x = 6 := by
  sorry

end speed_of_river_is_6_l522_522398


namespace number_of_classmates_l522_522930

theorem number_of_classmates (total_apples : ℕ) (apples_per_classmate : ℕ) (people_in_class : ℕ) 
  (h1 : total_apples = 15) (h2 : apples_per_classmate = 5) (h3 : people_in_class = total_apples / apples_per_classmate) : 
  people_in_class = 3 :=
by sorry

end number_of_classmates_l522_522930


namespace alpha_gt_beta_iff_alpha_minus_beta_gt_sin_alpha_minus_sin_beta_l522_522891

-- Define the real numbers α and β
variables (α β : ℝ)

-- Define the function f
def f (x : ℝ) : ℝ := x - sin x

-- State the main theorem that needs to be proved
theorem alpha_gt_beta_iff_alpha_minus_beta_gt_sin_alpha_minus_sin_beta :
  α > β ↔ α - β > sin α - sin β :=
sorry

end alpha_gt_beta_iff_alpha_minus_beta_gt_sin_alpha_minus_sin_beta_l522_522891


namespace combination_20_6_l522_522090

theorem combination_20_6 : Nat.choose 20 6 = 38760 :=
by
  sorry

end combination_20_6_l522_522090


namespace A_B_together_l522_522435

/-- This represents the problem of finding out the number of days A and B together 
can finish a piece of work given the conditions. -/
theorem A_B_together (A_rate B_rate: ℝ) (A_days B_days: ℝ) (work: ℝ) :
  A_rate = 1 / 8 →
  A_days = 4 →
  B_rate = 1 / 12 →
  B_days = 6 →
  work = 1 →
  (A_days * A_rate + B_days * B_rate = work / 2) →
  (24 / (A_rate + B_rate) = 4.8) :=
by
  intros hA_rate hA_days hB_rate hB_days hwork hwork_done
  sorry

end A_B_together_l522_522435


namespace intersection_cardinality_l522_522653

open Set

def A : Set ℕ := {x | 1 ≤ x ∧ x ≤ 99}
def B : Set ℕ := {y | ∃ x ∈ A, y = 2 * x}
def C : Set ℕ := {x | 2 * x ∈ A}

theorem intersection_cardinality :
  (B ∩ C).toFinset.card = 24 := by
  sorry

end intersection_cardinality_l522_522653


namespace line_angle_135_degrees_l522_522601

theorem line_angle_135_degrees :
  let p1 := (0 : ℝ, 0 : ℝ)
  let p2 := (3 : ℝ, -3 : ℝ)
  let k := (p2.2 - p1.2) / (p2.1 - p1.1)  -- slope of the line
  ∃ α : ℝ, tan α = k ∧ α = 135 :=
by
  let p1 := (0 : ℝ, 0 : ℝ)
  let p2 := (3 : ℝ, -3 : ℝ)
  let k := (p2.2 - p1.2) / (p2.1 - p1.1)
  use (135 : ℝ)
  sorry

end line_angle_135_degrees_l522_522601


namespace quadratic_non_residues_are_primitive_roots_l522_522105

def is_quadratic_non_residue_mod (a n : ℕ) : Prop :=
  ∀ b : ℕ, ¬ (a ≡ b^2 [MOD n])

def is_primitive_root_mod (a n : ℕ) : Prop :=
  ∀ b ∈ (Finset.filter (λ x, Nat.gcd x n = 1) (Finset.range n)), ∃ k, a^k ≡ b [MOD n]

def fermat_primes (n : ℕ) : Prop :=
  n = 3 ∨ n = 5 ∨ n = 17 ∨ n = 257 ∨ n = 65537

theorem quadratic_non_residues_are_primitive_roots (n : ℕ) :
  (3 ≤ n ∧ n.toString.length < 20 ∧ ∀ a, is_quadratic_non_residue_mod a n → is_primitive_root_mod a n) ↔ fermat_primes n := by
  sorry

end quadratic_non_residues_are_primitive_roots_l522_522105


namespace circumscribed_spheres_intersection_l522_522221

-- Definitions based on conditions
structure Point := 
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

structure Tetrahedron :=
  (A B C D : Point)

structure Plane :=
  (p₁ p₂ p₃ : Point)

def is_orthocenter (E : Point) (T : Tetrahedron) : Prop :=
  (∃ α β γ : Plane, 
    α.p₁ = T.B ∧ α.p₂ = T.C ∧ α.p₃ = T.D ∧ 
    β.p₁ = T.C ∧ β.p₂ = T.D ∧ β.p₃ = T.A ∧
    γ.p₁ = T.D ∧ γ.p₂ = T.A ∧ γ.p₃ = T.B ∧ 
    ∀ (P : Plane), (P = α ∨ P = β ∨ P = γ) → -- Ensuring planes are correct
      ((E ∈ P ∧ (P.p₁ - P.p₂) ⊥ (T.D - T.A))))

def is_perpendicular (l : ℝ → Point) (v : Point) : Prop := 
  ∀ t : ℝ, (v.x - l t.x) * (v.y - l t.y) * (v.z - l t.z) = 0 -- Line perpendicular property

-- Main proof statement
theorem circumscribed_spheres_intersection
  (A1 A2 A3 : Point)
  (l : ℝ → Point)
  (T1 T2 T3 : Tetrahedron)
  {E : Point}
  (h1 : A1 = l 1)
  (h2 : A2 = l 2)
  (h3 : A3 = l 3)
  (hT1 : is_orthocenter E T1)
  (hT2 : is_orthocenter E T2)
  (hT3 : is_orthocenter E T3) :
  ∃ F : Point, F ∈ ⟨l 0, l 1⟩ ∧ -- F is the foot of the perpendicular from E to line l
  ∀ R : Point,
  (distance E R = distance E F) ∧ (R - E) ⊥ (A3 - A2 - A1 - A2) := sorry

end circumscribed_spheres_intersection_l522_522221


namespace avg_score_B_correct_l522_522453

-- Definitions and conditions
def avg_score_U := 65
def avg_score_C := 77
def ratio_U := 4
def ratio_B := 6
def ratio_C := 5
def combined_avg_score := 75

-- The goal is to prove that avg_score_B = 80
theorem avg_score_B_correct (x : ℕ) :
  let total_students := ratio_U * x + ratio_B * x + ratio_C * x,
      total_score_U := avg_score_U * ratio_U * x,
      total_score_C := avg_score_C * ratio_C * x,
      total_score_combined := combined_avg_score * total_students in
  ∃ avg_score_B : ℕ, 
    total_score_U + avg_score_B * ratio_B * x + total_score_C = total_score_combined ∧ 
    avg_score_B = 80 :=
by
  sorry

end avg_score_B_correct_l522_522453


namespace line_eq_45_deg_y_intercept_2_circle_eq_center_neg2_3_tangent_yaxis_l522_522700

theorem line_eq_45_deg_y_intercept_2 :
  (∃ l : ℝ → ℝ, (l 0 = 2) ∧ (∀ x, l x = x + 2)) := sorry

theorem circle_eq_center_neg2_3_tangent_yaxis :
  (∃ c : ℝ × ℝ → ℝ, (c (-2, 3) = 0) ∧ (∀ x y, c (x, y) = (x + 2)^2 + (y - 3)^2 - 4)) := sorry

end line_eq_45_deg_y_intercept_2_circle_eq_center_neg2_3_tangent_yaxis_l522_522700


namespace range_of_m_l522_522878

noncomputable def f : ℝ → ℝ
| x => x^2 + 2 * x

theorem range_of_m (A B : Set ℝ)
  (hA : ∀ (x : ℝ), x ∈ A → x ∈ set.univ)
  (hB : ∀ (y : ℝ), y ∈ B ↔ ∃ x ∈ A, f x = y) :
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ x1 ∈ A ∧ x2 ∈ A ∧ ∃ m ∈ B, f x1 = m ∧ f x2 = m) → 
  ∀ (m : ℝ), m ∈ B → m > -1 :=
by
  sorry

end range_of_m_l522_522878


namespace n_congruence_mod_9_l522_522081

def n : ℕ := 2 + 333 + 5555 + 77777 + 999999 + 2222222 + 44444444 + 666666666

theorem n_congruence_mod_9 : n % 9 = 4 :=
by
  sorry

end n_congruence_mod_9_l522_522081


namespace find_k_l522_522245

-- Definitions of the lines and the area condition
def line1 : ℝ × ℝ → Prop := λ p, p.1 = 1
def line2 : ℝ × ℝ → Prop := λ p, p.2 = -1
def line3 : ℝ × ℝ → Prop := λ p, p.2 = 3
def line4 (k : ℝ) : ℝ × ℝ → Prop := λ p, p.2 = k * p.1 - 3

-- Condition stating the area of the convex quadrilateral formed by these lines
def area_condition (k : ℝ) : Prop :=
  let x1 := 1
  let y1 := -1
  let y2 := 3
  let x2 := (3 + 3) / k in
  if k < 0 then
    (1 / 2) * ((1 + 2 / k) + (1 + 6 / k)) * (y2 - y1) = 12
  else
    (1 / 2) * ((2 / k - 1) + (6 / k - 1)) * (y2 - y1) = 12

-- Theorem stating that the only solutions for k are -2 or 1
theorem find_k (k : ℝ) (h : area_condition k) : k = -2 ∨ k = 1 :=
sorry

end find_k_l522_522245


namespace number_of_solutions_l522_522375

theorem number_of_solutions (m n : ℕ) (h_pos_m : m > 0) (h_pos_n : n > 0) :
  (4 / m + 2 / n = 1) -> (m > 4 ∧ n > 2) -> ∃! (m n : ℕ), 4 / m + 2 / n = 1 ∧ m > 4 ∧ n > 2) :=
sorry

end number_of_solutions_l522_522375


namespace possible_values_of_expression_l522_522177

noncomputable def sign (x : ℝ) : ℝ :=
if x > 0 then 1 else -1

theorem possible_values_of_expression
  (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := (sign a) + (sign b) + (sign c) + (sign d) + (sign (a * b * c * d))
  in expr ∈ {5, 1, -1, -5} :=
by
  let expr := (sign a) + (sign b) + (sign c) + (sign d) + (sign (a * b * c * d))
  sorry

end possible_values_of_expression_l522_522177


namespace sum_c_k_squared_l522_522495

def c_k (k : ℕ) : ℚ := 
  let rec aux (n : ℕ) : ℚ := if n = 0 then k else aux (n - 1) + 1 / (3 * k + aux (n - 1))
  k + 1 / (3 * k + aux k)

theorem sum_c_k_squared : 
  (∑ k in finset.range 10, (c_k k) ^ 2) = 615 := 
by
  sorry

end sum_c_k_squared_l522_522495


namespace exist_xp_xq_l522_522150

variable {n : ℕ}
variable {x : Fin n → ℝ}
variable {a_coeff : Fin n → ℕ+}
variable {b_coeff : Fin n → ℕ+}

def a (a_coeff : Fin n → ℕ+) (x : Fin n → ℝ) : ℝ :=
  (∑ i, a_coeff i * x i) / (∑ i, a_coeff i)

def b (b_coeff : Fin n → ℕ+) (x : Fin n → ℝ) : ℝ :=
  (∑ i, b_coeff i * x i) / (∑ i, b_coeff i)

theorem exist_xp_xq (a_coeff b_coeff : Fin n → ℕ+) (x : Fin n → ℝ) :
  ∃ (p q : Fin n), |a a_coeff x - b b_coeff x| ≤ |a a_coeff x - x p| ∧ |a a_coeff x - x p| ≤ |x q - x p| :=
by
  sorry

end exist_xp_xq_l522_522150


namespace total_jury_duty_days_l522_522633

-- Conditions
def jury_selection_days : ℕ := 2
def trial_multiplier : ℕ := 4
def evidence_review_hours : ℕ := 2
def lunch_hours : ℕ := 1
def trial_session_hours : ℕ := 6
def hours_per_day : ℕ := evidence_review_hours + lunch_hours + trial_session_hours
def deliberation_hours_per_day : ℕ := 14 - 2

def deliberation_first_defendant_days : ℕ := 6
def deliberation_second_defendant_days : ℕ := 4
def deliberation_third_defendant_days : ℕ := 5

def deliberation_first_defendant_total_hours : ℕ := deliberation_first_defendant_days * deliberation_hours_per_day
def deliberation_second_defendant_total_hours : ℕ := deliberation_second_defendant_days * deliberation_hours_per_day
def deliberation_third_defendant_total_hours : ℕ := deliberation_third_defendant_days * deliberation_hours_per_day

def deliberation_days_conversion (total_hours: ℕ) : ℕ := (total_hours + deliberation_hours_per_day - 1) / deliberation_hours_per_day

-- Total days spent
def total_days_spent : ℕ :=
  let trial_days := jury_selection_days * trial_multiplier
  let deliberation_days := deliberation_days_conversion deliberation_first_defendant_total_hours + deliberation_days_conversion deliberation_second_defendant_total_hours + deliberation_days_conversion deliberation_third_defendant_total_hours
  jury_selection_days + trial_days + deliberation_days

#eval total_days_spent -- Expected: 25

theorem total_jury_duty_days : total_days_spent = 25 := by
  sorry

end total_jury_duty_days_l522_522633


namespace james_weight_loss_l522_522628

noncomputable def weight_loss_time (initial_weight: ℕ) (current_weight: ℕ) (goal_weight: ℕ) (months: ℕ) : ℕ :=
  (initial_weight - current_weight) / months

theorem james_weight_loss :
  ∀ (initial_weight current_weight goal_weight months: ℕ),
    initial_weight = 222 →
    current_weight = 198 →
    goal_weight = 190 →
    months = 12 →
    (current_weight - goal_weight) / (weight_loss_time initial_weight current_weight goal_weight months) = 4 :=
by
  intros
  unfold weight_loss_time
  rw [A₁, A₂, A₃, A₄]
  sorry

end james_weight_loss_l522_522628


namespace problem_solution_l522_522590

theorem problem_solution (x : ℝ) (h : 4^(2 * x) + 8 = 18 * 4^x) : x^2 + 2 = 2.25 ∨ x^2 + 2 = 6 :=
by
  sorry

end problem_solution_l522_522590


namespace total_lychee_yield_25_days_l522_522702

-- Definitions for the conditions
def a : ℕ → ℚ
| 1     := 1
| (n+1) := if n < 10 then 2 * a n + 1
           else if n < 15 then a 10
           else if n < 24 then (a (n + 1) / 2)
           else a (n + 1) / 2

-- Function to compute the sum of yields up to day n
def total_lychee_yield (n : ℕ) : ℚ :=
  (Finset.range n).sum a

-- The main theorem stating the total yield over 25 days, rounded to the nearest integer
theorem total_lychee_yield_25_days : total_lychee_yield 25 = 8173 := sorry

end total_lychee_yield_25_days_l522_522702


namespace square_fold_distance_l522_522059

noncomputable def distance_from_A (area : ℝ) (visible_equal : Bool) : ℝ :=
  if area = 18 ∧ visible_equal then 2 * Real.sqrt 6 else 0

theorem square_fold_distance (area : ℝ) (visible_equal : Bool) :
  area = 18 → visible_equal → distance_from_A area visible_equal = 2 * Real.sqrt 6 :=
by
  sorry

end square_fold_distance_l522_522059


namespace sixth_face_is_A_l522_522432

-- Definitions for the conditions
def cube_faces (faces : Fin 6 → Array (Array Bool)) :=
  ∃ f : Fin 6 → Array (Array Bool), -- A function defining each face of the cube
  ∀ i : Fin 6, 
    0 < i ∧ i < 6 → 
      faces i = f i

theorem sixth_face_is_A (black white : ℕ) (faces : Fin 5 → Array (Array Bool)) 
  (condition1 : black = 15) 
  (condition2 : white = 12) 
  (condition3 : ∃ f_A : Array (Array Bool), ∀ i : Fin 5, faces i = f_A) : 
  ∃ final_face : Array (Array Bool), final_face = (λ _, by exact f_A) :=
sorry

end sixth_face_is_A_l522_522432


namespace not_in_sequence_2002_l522_522497

def largest_proper_factor (n : ℕ) : ℕ :=
  if h : 2 ≤ n then
    Nat.findGreatest (λ m, m < n ∧ n % m = 0) h
  else 0

def sequence_a (A : ℕ) : ℕ → ℕ
| 0        := A
| (n + 1) := let an := sequence_a n in an + largest_proper_factor an

theorem not_in_sequence_2002 (A : ℕ) (hA : A > 1) : ¬ (∃ n, sequence_a A n = 2002) :=
sorry

end not_in_sequence_2002_l522_522497


namespace school_dinner_theater_tickets_l522_522735

theorem school_dinner_theater_tickets (x y : ℕ)
  (h1 : x + y = 225)
  (h2 : 6 * x + 9 * y = 1875) :
  x = 50 :=
by
  sorry

end school_dinner_theater_tickets_l522_522735


namespace min_diagonal_of_rectangle_l522_522967

noncomputable def rectangle_min_diagonal (l w : ℝ) (h : l + w = 10) : ℝ := √(l^2 + w^2)

theorem min_diagonal_of_rectangle {l w : ℝ} (h : l + w = 10) : 
  ∃ (x y : ℝ), (l = x ∧ w = y) ∧ rectangle_min_diagonal l w h = √50 :=
by
  sorry

end min_diagonal_of_rectangle_l522_522967


namespace MrJones_pants_count_l522_522656

theorem MrJones_pants_count (P : ℕ) (h1 : 6 * P + P = 280) : P = 40 := by
  sorry

end MrJones_pants_count_l522_522656


namespace count_factors_multiple_of_150_l522_522237

theorem count_factors_multiple_of_150 (n : ℕ) (h : n = 2^10 * 3^14 * 5^8) : 
  ∃ k, k = 980 ∧ ∀ d : ℕ, d ∣ n → 150 ∣ d → (d.factors.card = k) := sorry

end count_factors_multiple_of_150_l522_522237


namespace integral_definite_l522_522814

open Real

theorem integral_definite :
  ∫ x in (set.Icc (-1 / 2 : ℝ) 0),
  (x / (2 + sqrt (2 * x + 1))) =
  (7 / 6 - 3 * log (3 / 2)) :=
by
  sorry

end integral_definite_l522_522814


namespace half_lake_covered_day_l522_522444

theorem half_lake_covered_day
  (N : ℕ) -- the total number of flowers needed to cover the entire lake
  (flowers_on_day : ℕ → ℕ) -- a function that gives the number of flowers on a specific day
  (h1 : flowers_on_day 20 = N) -- on the 20th day, the number of flowers is N
  (h2 : ∀ d, flowers_on_day (d + 1) = 2 * flowers_on_day d) -- the number of flowers doubles each day
  : flowers_on_day 19 = N / 2 :=
by
  sorry

end half_lake_covered_day_l522_522444


namespace triangle_perimeter_l522_522794

def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

variable (a b c : ℝ)

theorem triangle_perimeter
  (h1 : 90 = (1/2) * 18 * b)
  (h2 : right_triangle 18 b c) :
  18 + b + c = 28 + 2 * Real.sqrt 106 :=
by
  sorry

end triangle_perimeter_l522_522794


namespace range_of_t_l522_522425

-- Definitions for the sequences {a_n} and {b_n}
def a_seq (n : ℕ) := 2n - 2
def b_seq (n : ℕ) := 3 * 2^(2n - 3)

-- Definitions for A_n and B_n
def A_n (n : ℕ) := ∏ i in (finset.range n).filter (λ x, 2 ≤ x + 1), a_seq (i + 1)
def B_n (n : ℕ) := ∑ i in (finset.range n).filter (λ x, 2 ≤ x + 1), b_seq (i + 1)

-- Set M definition
def M := {n : ℕ | A_n n + B_n n < 895}

-- Definition of distance between points P_i and P_j
def u (i j : ℕ) := real.sqrt ((a_seq i - a_seq j)^2 + (b_seq i - b_seq j)^2)

-- Theorem stating the range of real number t
theorem range_of_t (t : ℝ) (i j n : ℕ) (h : i ∈ M ∧ j ∈ M ∧ n ∈ M) :
  (u i j) ^ (2 * t) + 2 * (u i j) ^ i ≤ 8 * abs(A_n n + B_n n - 895) → t ≤ real.log 2 / (real.log (u i j)) := sorry

end range_of_t_l522_522425


namespace diagonal_intersection_of_parallelogram_l522_522339

structure Point where
  x : ℤ
  y : ℤ

def midpoint (p1 p2 : Point) : Point :=
  { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2 }

theorem diagonal_intersection_of_parallelogram :
  let p1 := Point.mk 2 (-3)
  let p2 := Point.mk 10 9
  midpoint p1 p2 = Point.mk 6 3 := by
  let p1 : Point := ⟨2, -3⟩
  let p2 : Point := ⟨10, 9⟩
  show midpoint p1 p2 = ⟨6, 3⟩
  sorry

end diagonal_intersection_of_parallelogram_l522_522339


namespace find_x_solution_l522_522955

theorem find_x_solution
  (x y z : ℤ)
  (h1 : 4 * x + y + z = 80)
  (h2 : 2 * x - y - z = 40)
  (h3 : 3 * x + y - z = 20) :
  x = 20 :=
by
  -- Proof steps go here...
  sorry

end find_x_solution_l522_522955


namespace exactly_one_solves_l522_522337

-- Define the independent probabilities for person A and person B
variables (p₁ p₂ : ℝ)

-- Assume probabilities are between 0 and 1 inclusive
axiom h1 : 0 ≤ p₁ ∧ p₁ ≤ 1
axiom h2 : 0 ≤ p₂ ∧ p₂ ≤ 1

theorem exactly_one_solves : (p₁ * (1 - p₂) + p₂ * (1 - p₁)) = (p₁ * (1 - p₂) + p₂ * (1 - p₁)) := 
by sorry

end exactly_one_solves_l522_522337


namespace smallest_positive_period_pi_triangle_properties_l522_522929

noncomputable def f (x : ℝ) : ℝ := 
  let a : ℝ × ℝ := (Real.sin x, -1)
  let b : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, -1/2)
  (a.1 + b.1, a.2 + b.2) • a - 2

theorem smallest_positive_period_pi : ∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = Real.pi := 
by
  sorry

theorem triangle_properties (A : ℝ) (S : ℝ) : 
  let a := 2 * Real.sqrt 3
  let c := 4
  let b : ℝ := 2
  (S = 2 * Real.sqrt 3) ∧ (A = Real.pi / 3) ∧ (a^2 + c^2 - 2 * a * c * Real.cos A = b^2) := 
by
  sorry

end smallest_positive_period_pi_triangle_properties_l522_522929


namespace area_of_triangle_AEF_l522_522346

theorem area_of_triangle_AEF (ABCD : Parallelogram) (E F : Point) (AE : Length) (CF : Length) :
  parallelogram ABCD ∧
  area ABCD = 48 ∧
  on_side E ABCD.AB ∧
  on_side F ABCD.CD ∧
  AE = (1 / 3) * length ABCD.AB ∧
  CF = (1 / 3) * length ABCD.CD →
  area (triangle.mk ABCD.A E F) = 32 / 9 :=
by
  sorry

end area_of_triangle_AEF_l522_522346


namespace evaluate_expression_l522_522843

theorem evaluate_expression :
  (2^2 + 2^1 + 2^(-2)) / (2^(-3) + 2^(-4) + 2^(-5) + 2^(-6)) = 80 / 3 := by
  sorry

end evaluate_expression_l522_522843


namespace original_number_is_10_l522_522252

theorem original_number_is_10 (x : ℝ) (h : 2 * x + 5 = x / 2 + 20) : x = 10 := 
by {
  sorry
}

end original_number_is_10_l522_522252


namespace compare_fractions_l522_522480

theorem compare_fractions : (-2 / 7) > (-3 / 10) :=
sorry

end compare_fractions_l522_522480


namespace alvin_earns_total_l522_522070

def total_marbles : Nat := 150
def percent_white : ℚ := 0.20
def percent_black : ℚ := 0.25
def percent_blue : ℚ := 0.30
def percent_green : ℚ := 0.15
def percent_red : ℚ := 0.10

def price_white : ℚ := 0.05
def price_black : ℚ := 0.10
def price_blue : ℚ := 0.15
def price_green : ℚ := 0.12
def price_red : ℚ := 0.25

noncomputable def earnings (count : ℚ) (price : ℚ) : ℚ := count * price

def rounded_count (percent : ℚ) (total : ℕ) : ℕ :=
  let count := percent * total
  if count.fract < 0.5 then count.floor.toNat else count.ceil.toNat

theorem alvin_earns_total :
  (earnings (rounded_count percent_white total_marbles) price_white +
   earnings (rounded_count percent_black total_marbles) price_black +
   earnings (rounded_count percent_blue total_marbles) price_blue +
   earnings (rounded_count percent_green total_marbles) price_green +
   earnings (rounded_count percent_red total_marbles) price_red).toReal = 18.56 :=
by
  sorry

end alvin_earns_total_l522_522070


namespace period_of_my_function_l522_522112

-- Define the sine function
def my_sine (x : ℝ) := Real.sin x

-- Define the function y = sin(3x + π)
def my_function (x : ℝ) := my_sine (3 * x + Real.pi)

-- Define the period
def period (f : ℝ → ℝ) (T : ℝ) := ∀ x, f (x + T) = f x

-- Statement to prove
theorem period_of_my_function : period my_function (2 * Real.pi / 3) :=
sorry

end period_of_my_function_l522_522112


namespace eiffel_tower_scale_height_l522_522612

theorem eiffel_tower_scale_height (actual_height : ℕ) (scale_ratio : ℕ) (h_actual_height : actual_height = 324) (h_scale_ratio : scale_ratio = 30) :
  (324 / 30 : ℝ).round = 11 :=
by {
  rw [h_actual_height, h_scale_ratio],
  norm_num, -- simplify the division and rounding
  sorry
}

end eiffel_tower_scale_height_l522_522612


namespace goat_greater_area_proof_l522_522402

-- Define the conditions
def rope_length : ℝ := 10
def shed_length : ℝ := 20
def shed_width : ℝ := 10

-- Define the areas computed
def area_semi_circle (r : ℝ) : ℝ := (1 / 2) * Real.pi * r^2
def area_three_quarter_circle (r : ℝ) : ℝ := (3 / 4) * Real.pi * r^2
def area_quarter_circle (r : ℝ) : ℝ := (1 / 4) * Real.pi * r^2

-- Define the areas for each arrangement
def area_arrangement_I : ℝ := area_semi_circle rope_length
def area_arrangement_II : ℝ := area_three_quarter_circle rope_length + area_quarter_circle (rope_length / 2)

-- Define the difference in areas
def area_difference : ℝ := area_arrangement_II - area_arrangement_I

-- The theorem to prove
theorem goat_greater_area_proof :
  area_arrangement_II > area_arrangement_I ∧ area_difference = 31.25 * Real.pi :=
by
  -- Compute expected values
  have eq1: area_arrangement_I = 50 * Real.pi := by sorry
  have eq2: area_arrangement_II = 81.25 * Real.pi := by sorry
  have diff : area_difference = 31.25 * Real.pi := by sorry

  -- Prove the assertions
  exact And.intro (by linarith [eq1, eq2]) diff

end goat_greater_area_proof_l522_522402


namespace friends_recycled_pounds_l522_522085

theorem friends_recycled_pounds (total_points chloe_points each_points pounds_per_point : ℕ)
  (h1 : each_points = pounds_per_point / 6)
  (h2 : total_points = 5)
  (h3 : chloe_points = pounds_per_point / 6)
  (h4 : pounds_per_point = 28) 
  (h5 : total_points - chloe_points = 1) :
  pounds_per_point = 6 :=
by
  sorry

end friends_recycled_pounds_l522_522085


namespace curve_not_parabola_l522_522140

open Real

noncomputable def curve_represents (θ : ℝ) : Type :=
  if sin θ = 1 then {C : Type // ∃ y : ℝ, x^2 + (y^2 * sin θ) = 4}
  else if sin θ = 0 then {C : Type // ∃ x : ℝ, y^2 * sin θ = 4}
  else if sin θ ∈ Icc (-1 : ℝ) 0 then {H : Type // ∃ x y : ℝ, x^2 + y^2 * sin θ = 4}
  else {E : Type // ∃ x y : ℝ, x^2 + y^2 * sin θ = 4}

theorem curve_not_parabola (θ : ℝ) : ¬(curve_represents θ = {P : Type // ∀ x y : ℝ, (x - h)² = 4a(y - k)}) :=
by
  sorry

end curve_not_parabola_l522_522140


namespace total_earnings_l522_522581

theorem total_earnings : 
  let wage : ℕ := 10
  let hours_monday : ℕ := 7
  let tips_monday : ℕ := 18
  let hours_tuesday : ℕ := 5
  let tips_tuesday : ℕ := 12
  let hours_wednesday : ℕ := 7
  let tips_wednesday : ℕ := 20
  let total_hours : ℕ := hours_monday + hours_tuesday + hours_wednesday
  let earnings_from_wage : ℕ := total_hours * wage
  let total_tips : ℕ := tips_monday + tips_tuesday + tips_wednesday
  let total_earnings : ℕ := earnings_from_wage + total_tips
  total_earnings = 240 :=
by
  sorry

end total_earnings_l522_522581


namespace problem_statement_l522_522305

open Complex

noncomputable def isosceles_right_triangle (z1 z2 : ℂ) : Prop :=
  ∃ (θ : ℂ), θ = exp (π * I / 4) ∧ z2 = θ * z1

theorem problem_statement (a b z1 z2 : ℂ)
    (h1 : z1^2 + a*z1 + b = 0)
    (h2 : z2^2 + a*z2 + b = 0)
    (h3 : isosceles_right_triangle z1 z2) :
    a ≠ 0 → z1 ≠ 0 → a^2 / b = 2 + 2*I :=
by
  sorry

end problem_statement_l522_522305


namespace count_to_top_plus_l522_522094

def bottom_cells : Type := list (bool)
def num_bottom_cells := 5

def sign_rule (c1 c2 : bool) : bool :=
if c1 = c2 then true else false

def pyramid_top (cells : bottom_cells) : bool :=
let l1 := list.map_endo_rel sign_rule cells in
let l2 := list.map_endo_rel sign_rule l1 in
let l3 := list.map_endo_rel sign_rule l2 in
let l4 := list.map_endo_rel sign_rule l3 in
l4.head

noncomputable def num_valid_configurations : ℕ :=
list.length (list.filter (λ cells, pyramid_top cells = true)
(bottom_cells.replicate_with (λ_, [true, false]) num_bottom_cells))

theorem count_to_top_plus : num_valid_configurations = 12 := sorry

end count_to_top_plus_l522_522094


namespace find_angle_CKB_l522_522516

variables (x y t z : ℝ)
-- Given conditions
def angle_CKB_is_obtuse : Prop := sorry  -- Angle ∠CKB is obtuse.
def AM_EQ_KB := x  -- AM = KB = x
def CK_EQ_DL := y  -- CK = DL = y
def MK := t        -- MK = t
def KL := z        -- KL = z

theorem find_angle_CKB (hx : x > 0) (hy : y > 0) (ht : t > 0) (hz : z > 0) 
  (obtuse_CKB : angle_CKB_is_obtuse): 
  ∃ θ : ℝ, (θ = 5 * π / 6 ∨ θ = π - arcsin (1 / 4)) ∧ θ > π / 2 := 
sorry

end find_angle_CKB_l522_522516


namespace negative_number_in_set_l522_522073

-- Definitions for the set of numbers
def numbers : Set ℤ := {-2, 0, 1, 3}

-- Definition stating that a number is negative if it is less than zero
def is_negative (n : ℤ) : Prop := n < 0

-- The theorem to prove the specific problem statement
theorem negative_number_in_set : ∃! n ∈ numbers, is_negative n := by
  sorry

end negative_number_in_set_l522_522073


namespace icosahedron_faces_share_vertex_same_integer_l522_522672

open Set Function

theorem icosahedron_faces_share_vertex_same_integer :
  ∃ (f1 f2 : Fin 20), (f1 ≠ f2) ∧ (∃ (v : Fin 12), shares_vertex v f1 f2) ∧ face_value f1 = face_value f2 := by
  -- Define the conditions, properties, and constraints
  let vertices := 12
  let faces := 20
  let total_sum := 39

  -- Define a regular icosahedron with appropriate properties
  -- Assume shares_vertex and face_value are given or defined elsewhere
  sorry

end icosahedron_faces_share_vertex_same_integer_l522_522672


namespace penny_exceeded_by_32_l522_522268

def bulk_price : ℤ := 5
def min_spend_before_tax : ℤ := 40
def tax_per_pound : ℤ := 1
def penny_payment : ℤ := 240

def total_cost_per_pound : ℤ := bulk_price + tax_per_pound

def min_pounds_for_min_spend : ℤ := min_spend_before_tax / bulk_price

def total_pounds_penny_bought : ℤ := penny_payment / total_cost_per_pound

def pounds_exceeded : ℤ := total_pounds_penny_bought - min_pounds_for_min_spend

theorem penny_exceeded_by_32 : pounds_exceeded = 32 := by
  sorry

end penny_exceeded_by_32_l522_522268


namespace trig_identity_l522_522159

theorem trig_identity 
  (α : ℝ)
  (h1 : sin α = 1 / 2 + cos α)
  (h2 : 0 < α ∧ α < π / 2) 
  : (cos (2 * α)) / (sin (α - π / 4)) = - (Real.sqrt 14) / 2 := 
by 
  sorry

end trig_identity_l522_522159


namespace latus_rectum_parabola_l522_522133

theorem latus_rectum_parabola : ∀ x : ℝ, y = x^2 / 4 → latus_rectum y = 1 :=
sorry

end latus_rectum_parabola_l522_522133


namespace fibonacci_series_sum_l522_522293

noncomputable def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n + 1) + fib n

theorem fibonacci_series_sum :
  (∑' n, (fib n : ℝ) / 7^n) = (49 : ℝ) / 287 := 
by
  sorry

end fibonacci_series_sum_l522_522293


namespace ratio_of_larger_to_smaller_l522_522727

theorem ratio_of_larger_to_smaller (x y : ℝ) (h1 : x > y) (h2 : x + y = 7 * (x - y)) :
  x / y = 4 / 3 :=
by
  sorry

end ratio_of_larger_to_smaller_l522_522727


namespace carpet_dimensions_l522_522068

theorem carpet_dimensions
  (x y q : ℕ)
  (h_dim : y = 2 * x)
  (h_room1 : ((q^2 + 50^2) = (q * 2 - 50)^2 + (50 * 2 - q)^2))
  (h_room2 : ((q^2 + 38^2) = (q * 2 - 38)^2 + (38 * 2 - q)^2)) :
  x = 25 ∧ y = 50 :=
sorry

end carpet_dimensions_l522_522068


namespace quadratic_eq_with_roots_mean_l522_522853

theorem quadratic_eq_with_roots_mean
    (a b : ℝ)
    (h1 : (a + b) / 2 = 9)
    (h2 : real.sqrt (a * b) = 21) :
    (by sorry : x^2 - 18 * x + 441 = 0)

end quadratic_eq_with_roots_mean_l522_522853


namespace isosceles_trapezoid_inscribed_in_circle_permissible_values_of_k_l522_522983

theorem isosceles_trapezoid_inscribed_in_circle 
  (k : ℝ) (h1 : k > 1) :
  ∃ (α β : ℝ),
  α = Real.arccos (1 - 1/k) ∧ 
  β = Real.pi - α :=
sorry

theorem permissible_values_of_k 
  (k : ℝ) :
  k > 1 :=
by {
  sorry,
}

end isosceles_trapezoid_inscribed_in_circle_permissible_values_of_k_l522_522983


namespace expression_simplify_l522_522680

variable {a b : ℝ} (x : ℝ)
variable (h1 : x = a^(2/3) * b^(-1/2)) (h2 : a ≠ 0) (h3 : b ≠ 0)

theorem expression_simplify (x : ℝ) (h1 : x = a^(2/3) * b^(-1/2)) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  (x^3 - a^(-2/3) * b^(-1) * (a^2 + b^2) * x + b^(1/2)) / (b^(3/2) * x^2) = 0 := by
  sorry

end expression_simplify_l522_522680


namespace area_shaded_region_l522_522479

variables {A B O C D E F : Point}
variables {r : ℝ} (h_A : circle A r) (h_B : circle B r)
variables (h_r : r = 2) (h_O : midpoint O A B) (h_OA : dist O A = 2 * sqrt 2)
variables (h_OC_tangent : tangent OC A) (h_OD_tangent : tangent OD B)
variables (h_EF_tangent : common_tangent E F A B)

theorem area_shaded_region : 
  let region := shaded_area E C O D F in
  region_area region = 8 * sqrt 2 - 4 - π :=
sorry

end area_shaded_region_l522_522479


namespace value_of_b_plus_c_l522_522892

variable {a b c d : ℝ}

theorem value_of_b_plus_c (h1 : a + b = 4) (h2 : c + d = 5) (h3 : a + d = 2) : b + c = 7 :=
sorry

end value_of_b_plus_c_l522_522892


namespace final_price_of_pencil_l522_522367

-- Define the initial constants
def initialCost : ℝ := 4.00
def christmasDiscount : ℝ := 0.63
def seasonalDiscountRate : ℝ := 0.07
def finalDiscountRate : ℝ := 0.05
def taxRate : ℝ := 0.065

-- Define the steps of the problem concisely
def priceAfterChristmasDiscount := initialCost - christmasDiscount
def priceAfterSeasonalDiscount := priceAfterChristmasDiscount * (1 - seasonalDiscountRate)
def priceAfterFinalDiscount := priceAfterSeasonalDiscount * (1 - finalDiscountRate)
def finalPrice := priceAfterFinalDiscount * (1 + taxRate)

-- The theorem to be proven
theorem final_price_of_pencil :
  abs (finalPrice - 3.17) < 0.01 := by
  sorry

end final_price_of_pencil_l522_522367


namespace magnitude_of_T_l522_522643

def i : Complex := Complex.I

def T : Complex := (1 + i) ^ 18 - (1 - i) ^ 18

theorem magnitude_of_T : Complex.abs T = 1024 := by
  sorry

end magnitude_of_T_l522_522643


namespace quadratic_roots_m_eq_two_l522_522153

theorem quadratic_roots_m_eq_two (m : ℝ) (x₁ x₂ : ℝ) 
  (h_eq : m * x₁^2 - (m + 2) * x₁ + m / 4 = 0)
  (h1 : m * x₂^2 - (m + 2) * x₂ + m / 4 = 0) 
  (h2 : (x₁ ≠ x₂)) 
  (h3 : 1 / x₁ + 1 / x₂ = 4 * m) : 
  m = 2 :=
by sorry

end quadratic_roots_m_eq_two_l522_522153


namespace mangoes_in_basket_B_l522_522615

theorem mangoes_in_basket_B :
  ∀ (A C D E B : ℕ), 
    (A = 15) →
    (C = 20) →
    (D = 25) →
    (E = 35) →
    (5 * 25 = A + C + D + E + B) →
    (B = 30) :=
by
  intros A C D E B hA hC hD hE hSum
  sorry

end mangoes_in_basket_B_l522_522615


namespace evaluate_expression_l522_522907

theorem evaluate_expression 
  (a c : ℝ)
  (h : a + c = 9) :
  (a * (-1)^2 + (-1) + c) = 8 := 
by 
  sorry

end evaluate_expression_l522_522907


namespace polynomial_divisibility_l522_522591

theorem polynomial_divisibility (t : ℤ) : 
  (∀ x : ℤ, (5 * x^3 - 15 * x^2 + t * x - 20) ∣ (x - 2)) → (t = 20) → 
  ∀ x : ℤ, (5 * x^3 - 15 * x^2 + 20 * x - 20) ∣ (5 * x^2 + 5 * x + 5) :=
by
  intro h₁ h₂
  sorry

end polynomial_divisibility_l522_522591


namespace expression_never_prime_l522_522412

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem expression_never_prime (p : ℕ) (hp : is_prime p) : ¬ is_prime (p^2 + 20) := sorry

end expression_never_prime_l522_522412


namespace ethan_coconut_oil_per_candle_l522_522506

noncomputable def ounces_of_coconut_oil_per_candle (candles: ℕ) (total_weight: ℝ) (beeswax_per_candle: ℝ) : ℝ :=
(total_weight - candles * beeswax_per_candle) / candles

theorem ethan_coconut_oil_per_candle :
  ounces_of_coconut_oil_per_candle 7 63 8 = 1 :=
by
  sorry

end ethan_coconut_oil_per_candle_l522_522506


namespace train_length_correct_l522_522042

-- Let v_jogger be the speed of the jogger in m/s
-- Let v_train be the speed of the train in m/s
-- Let d_ahead be the distance ahead of the jogger in meters
-- Let t_pass be the time taken by the train to pass the jogger in seconds
-- Define the problem and the expected length of the train

def v_jogger_kmh := 9
def v_train_kmh := 45
def v_jogger := v_jogger_kmh * 1000 / 3600  -- Converting km/hr to m/s
def v_train := v_train_kmh * 1000 / 3600    -- Converting km/hr to m/s
def d_ahead := 200
def t_pass := 32
def l_train := 120

theorem train_length_correct :
  let relative_speed := v_train - v_jogger in
  let distance_covered := relative_speed * t_pass in
  distance_covered - d_ahead = l_train :=
  sorry

end train_length_correct_l522_522042


namespace probability_of_at_least_65_cents_heads_l522_522359

def coin_values : list ℕ := [1, 5, 10, 25, 50] -- in cents

def count_successful_scenarios : ℕ :=
  let scenarios := (2 ^ 3) + (2 ^ 1) in
  scenarios

theorem probability_of_at_least_65_cents_heads :
  (count_successful_scenarios / (2 ^ 5) : ℚ) = 5 / 16 :=
by
  sorry

end probability_of_at_least_65_cents_heads_l522_522359


namespace height_of_wall_l522_522585

theorem height_of_wall :
  let L_brick := 80
  let W_brick := 11.25
  let H_brick := 6
  let L_wall := 800
  let W_wall := 22.5
  let num_bricks := 2000
  let volume_brick := L_brick * W_brick * H_brick
  let total_volume_bricks := volume_brick * num_bricks
  let volume_wall := total_volume_bricks
  ∃ H_wall, volume_wall = L_wall * W_wall * H_wall ∧ H_wall = 600 :=
by
  let L_brick := 80
  let W_brick := 11.25
  let H_brick := 6
  let L_wall := 800
  let W_wall := 22.5
  let num_bricks := 2000
  let volume_brick := L_brick * W_brick * H_brick
  let total_volume_bricks := volume_brick * num_bricks
  let volume_wall := total_volume_bricks
  use 600
  have h1 : volume_wall = L_wall * W_wall * 600 := sorry
  exact ⟨h1, rfl⟩

end height_of_wall_l522_522585


namespace part1_correct_independence_P_AB_l522_522254

-- Defining the balls based on the description
inductive Ball
| red1 : Ball
| red1' : Ball
| red2 : Ball
| red2' : Ball
| red2'' : Ball
| red2''' : Ball
| blue1 : Ball
| blue2 : Ball
| blue2' : Ball

open Ball

-- Define the overall balls set
def balls : List Ball := [red1, red1', red2, red2', red2'', red2''', blue1, blue2, blue2']

-- Define the event for drawing one red and one of the specified condition
def draw_with_replacement : List (Ball × Ball) := do
  b1 ← balls,
  b2 ← balls,
  [(b1, b2)]

def sum_is_3 (b1 b2 : Ball) : Bool :=
  match b1, b2 with
  | red1, red2 => true
  | red1, red2' => true
  | red1, red2'' => true
  | red1, red2''' => true
  | red1', red2 => true
  | red1', red2' => true
  | red1', red2'' => true
  | red1', red2''' => true
  | blue1, red2 => true
  | blue1, red2' => true
  | blue1, red2'' => true
  | blue1, red2''' => true
  | red2, blue1 => true
  | red2', blue1 => true
  | red2'', blue1 => true
  | red2''', blue1 => true
  | _, _ => false

-- Calculate probability considering it's a uniform distribution
def drawing_probability (condition : Ball × Ball → Bool) : ℚ :=
  (draw_with_replacement.filter condition).length /. (draw_with_replacement.length : ℚ)

noncomputable
def part1_probability : ℚ :=
  drawing_probability (λ b => b.fst == red1 || b.fst == red1'  -- exactly one red ball
                       ∧ sum_is_3 b.fst b.snd ∧ b.fst ≠ b.snd)

-- Event definitions for Part 2
def event_A (b : Ball) : Bool := b == red1 || b == red1' || b == red2 || b == red2' || b == red2'' || b == red2'''

def event_B (b1 b2 : Ball) : Bool := (b1 == blue2 ∨ b1 == blue2' ∨ b1 == red2 ∨ b1 == red2' ∧ b2 == red1 || b2 == red1' || b2 == blue1)

noncomputable def P_A : ℚ :=
  (balls.filter event_A).length /. (balls.length : ℚ)

noncomputable def P_B : ℚ :=
  ((List.product balls balls).filter (λ b => event_B b.snd b.fst)).length /. ((balls.length * (balls.length - 1)) : ℚ)

noncomputable def P_AB : ℚ :=
  ((List.product balls balls).filter (λ b => event_A b.fst ∧ event_B b.fst b.snd)).length /. ((balls.length * (balls.length - 1)) : ℚ)

theorem part1_correct : part1_probability = 16 / 81 := sorry
theorem independence_P_AB : P_AB = P_A * P_B := sorry

end part1_correct_independence_P_AB_l522_522254


namespace pyramid_volume_calc_l522_522056

-- Define structures and values
variables (base_area triangle_area height volume : ℝ)

-- Given conditions as definitions in Lean
def square_base : Prop := True
def total_surface_area : Prop := 540 = base_area * (7/3)
def triangular_face_area : Prop := 4 * (1/3 * base_area) = 4 * triangle_area
def triangular_faces_relation : Prop := triangle_area = (1/3) * base_area
def volume_condition : Prop := volume = (1/3) * base_area * height
def evaluated_height : ℝ := 6.39

-- Volume is given 491.84 cubic units
noncomputable def given_volume : ℝ := 491.84

-- Proof statement to be proven in Lean
theorem pyramid_volume_calc : 
  square_base →
  total_surface_area →
  triangular_faces_relation →
  volume_condition →
  height = evaluated_height →
  volume = given_volume :=
begin
  intros _ _ _ _ h,
  rw h,
  sorry -- Proof may go here
end

end pyramid_volume_calc_l522_522056


namespace Dan_must_exceed_speed_l522_522771

theorem Dan_must_exceed_speed (distance : ℝ) (Cara_speed : ℝ) (delay : ℝ) (time_Cara : ℝ) (Dan_time : ℝ) : 
  distance = 120 ∧ Cara_speed = 30 ∧ delay = 1 ∧ time_Cara = distance / Cara_speed ∧ time_Cara = 4 ∧ Dan_time = time_Cara - delay ∧ Dan_time < 4 → 
  (distance / Dan_time) > 40 :=
by
  sorry

end Dan_must_exceed_speed_l522_522771


namespace sum_of_squares_of_cosines_l522_522486

theorem sum_of_squares_of_cosines :
  (Finset.range 181).sum (λ i, (Real.cos (i * 0.5 * Real.pi / 180)) ^ 2) = 90.5 := by
  -- Proof steps will go here.
  sorry

end sum_of_squares_of_cosines_l522_522486


namespace standard_equation_of_ellipse_no_line_l_l522_522909

-- The given conditions as definitions
def is_ellipse (x y a b : ℝ) := (x^2 / (a^2)) + (y^2 / (b^2)) = 1
def left_vertex (D : ℝ × ℝ) (a : ℝ) := D = (-a, 0)
def distance_short_axis (D : ℝ × ℝ) (endpoints : ℝ × ℝ) := 
  dist D endpoints = real.sqrt 5

-- Main theorem statements
theorem standard_equation_of_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : left_vertex (-2, 0) a) (h4 : is_ellipse (-2, 0) a b) 
  (h5 : distance_short_axis (-2, 0) (0, b)) : 
  is_ellipse 1 0 2 1 := sorry

theorem no_line_l (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : left_vertex (-2, 0) a) (h4 : is_ellipse (-2, 0) a b) 
  (h5 : distance_short_axis (-2, 0) (0, b)) :
  ¬ ∃ m A B : ℝ × ℝ, A.2 = -(3 * B.2) ∧ A = ((1, 0), (m * A.2 + 1)) ∧ is_ellipse A.1 A.2 a b ∧ is_ellipse B.1 B.2 a b := sorry

end standard_equation_of_ellipse_no_line_l_l522_522909


namespace probability_X_l522_522652

theorem probability_X (P : ℕ → ℚ) (h1 : P 1 = 1/10) (h2 : P 2 = 2/10) (h3 : P 3 = 3/10) (h4 : P 4 = 4/10) :
  P 2 + P 3 = 1/2 :=
by
  sorry

end probability_X_l522_522652


namespace max_value_expression_l522_522208

variable (n : ℕ)
variable (x : Fin n → ℝ)
variable (h : ∀ i, 0 ≤ x i ∧ x i ≤ Real.pi / 2)

theorem max_value_expression :
  (∑ i in Finset.univ, Real.sqrt (Real.sin (x i))) * 
  (∑ i in Finset.univ, Real.sqrt (Real.cos (x i))) ≤ n^2 / Real.sqrt 2 := sorry

end max_value_expression_l522_522208


namespace complex_solution_l522_522534

theorem complex_solution (z : ℂ) (h : (1 + 2 * complex.i) * z = 4 + 3 * complex.i) : 
  z = 2 - complex.i :=
by
  sorry

end complex_solution_l522_522534


namespace intersection_points_zero_l522_522953

theorem intersection_points_zero (a b c: ℝ) (h1: b^2 = a * c) (h2: a * c > 0) : 
  ∀ x: ℝ, ¬ (a * x^2 + b * x + c = 0) := 
by 
  sorry

end intersection_points_zero_l522_522953


namespace count_even_numbers_between_250_and_600_l522_522587

theorem count_even_numbers_between_250_and_600 : 
  ∃ n : ℕ, (n = 175 ∧ 
    ∀ k : ℕ, (250 < 2 * k ∧ 2 * k ≤ 600) ↔ (126 ≤ k ∧ k ≤ 300)) :=
by
  sorry

end count_even_numbers_between_250_and_600_l522_522587


namespace find_abc_l522_522129

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {p | ∃ e t : ℝ, p = (e, t, 1 / t) ∧ e ∈ ({-1, 1} : Set ℝ) ∧ t ≠ 0}

theorem find_abc (x y z : ℝ) :
  x + y + z = (1 / x + 1 / y + 1 / z) → 
  x^2 + y^2 + z^2 = (1 / x^2 + 1 / y^2 + 1 / z^2) →
  (x, y, z) ∈ solution_set := 
begin
  sorry
end

end find_abc_l522_522129


namespace compass_construction_sqrt2_l522_522574

/-- 
Given two points A and B with a distance of 1 unit, construct two points 
that are √2 units apart using only a compass.
-/
theorem compass_construction_sqrt2 (A B : ℝ × ℝ) (h : dist A B = 1) :
  ∃ C D : ℝ × ℝ, dist C D = real.sqrt 2 :=
sorry

end compass_construction_sqrt2_l522_522574


namespace area_of_quadrilateral_AFCH_l522_522661

noncomputable def area_of_AFCH : ℝ :=
  let AB := 9
  let BC := 5
  let EF := 3
  let FG := 10
  let small_area := BC * EF
  let large_area := AB * FG
  let ring_area := large_area - small_area
  let triangles_area := ring_area / 2
  in small_area + triangles_area

theorem area_of_quadrilateral_AFCH :
  area_of_AFCH = 52.5 :=
by
  sorry

end area_of_quadrilateral_AFCH_l522_522661


namespace josh_money_left_l522_522287

theorem josh_money_left (initial_amount : ℝ) (first_spend : ℝ) (second_spend : ℝ) 
  (h1 : initial_amount = 9) 
  (h2 : first_spend = 1.75) 
  (h3 : second_spend = 1.25) : 
  initial_amount - first_spend - second_spend = 6 := 
by 
  sorry

end josh_money_left_l522_522287


namespace analytical_expression_exists_a_max_1_l522_522650

-- Definitions of the problem conditions
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ∈ Ioo 0 1 then -x^3 + a * x
  else if x ∈ Ico -1 0 then x^3 - a * x
  else 0

lemma fx_even (a : ℝ) (x : ℝ) (h0 : x ∈ Ioo 0 1 ∪ Ico -1 0) :
  f a x = f a (-x) :=
begin
  sorry, -- Proof that f is even.
end

-- Proving the analytical expression.
theorem analytical_expression (a : ℝ) (x : ℝ) (h0 : x ∈ Ioo 0 1 ∪ Ico -1 0) :
  f a x = if x ∈ Ioo 0 1 then -x^3 + a * x else x^3 - a * x :=
begin
  sorry, -- Proof that this is the correct expression for f.
end

-- Proving the existence of a such that the maximum value is 1.
theorem exists_a_max_1 :
  ∃ a : ℝ, (∀ x ∈ Ioo 0 1, f a x ≤ 1) ∧ ∃ x ∈ Ioo 0 1, f a x = 1 :=
begin
  use (3 * real.cbrt 2 / 2),
  split,
  {
    intros x hx,
    sorry, -- Proof that for all x in (0,1], f(a, x) ≤ 1
  },
  {
    use (real.sqrt (3 * real.cbrt 2 / 2 / 3)),
    sorry, -- Proof that there exists an x such that f(a, x) = 1
  }
end

end analytical_expression_exists_a_max_1_l522_522650


namespace period_of_sine_3x_pipluspi_l522_522114

noncomputable def period_of_sine_function_coefficient (a : ℝ) : ℝ :=
  let standard_period := 2 * Real.pi
  in standard_period / a

theorem period_of_sine_3x_pipluspi : period_of_sine_function_coefficient 3 = 2 * Real.pi / 3 := by
  sorry

end period_of_sine_3x_pipluspi_l522_522114


namespace bisection_root_exists_l522_522065

def f (x : ℝ) : ℝ := x^5 + x - 3

theorem bisection_root_exists :
  ∃ x ∈ set.Icc (1 : ℝ) 2, f x = 0 :=
by sorry

end bisection_root_exists_l522_522065


namespace not_monotonically_decreasing_on_interval_l522_522071

def f (x : ℝ) : ℝ := x ^ 2 - 1

theorem not_monotonically_decreasing_on_interval :
  ¬ (∀ x y ∈ set.Ioo (-1) 1, x < y → f x ≥ f y) :=
sorry

end not_monotonically_decreasing_on_interval_l522_522071


namespace possible_values_of_expression_l522_522163

theorem possible_values_of_expression (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (v : ℤ), v ∈ ({5, 1, -3, -5} : Set ℤ) ∧ v = (Int.sign a + Int.sign b + Int.sign c + Int.sign d + Int.sign (a * b * c * d)) :=
by
  sorry

end possible_values_of_expression_l522_522163


namespace paperboy_delivery_l522_522047

/-- A paperboy delivers newspapers to 15 houses along Elm Street. He wishes to minimize his efforts but still meets his obligation by making sure he does not skip three consecutive houses and ensures that he delivers to at least one of the first three houses. Prove that the number of possible ways he can accomplish his deliveries is D_15, given the conditions. -/
theorem paperboy_delivery :
  let D : ℕ → ℕ :=
    λ n,
      if n = 1 then 2 else
      if n = 2 then 4 else
      if n = 3 then 7 else
      D (n-1) + D (n-2) + D (n-3)
  in D 15 = X := sorry

end paperboy_delivery_l522_522047


namespace part1_find_a_b_part2_inequality_l522_522204

theorem part1_find_a_b (f : ℝ → ℝ) (a b : ℝ) (h_f : ∀ x, f x = |2 * x + 1| + |x + a|) 
  (h_sol : ∀ x, f x ≤ 3 ↔ b ≤ x ∧ x ≤ 1) : 
  a = -1 ∧ b = -1 :=
sorry

theorem part2_inequality (m n : ℝ) (a : ℝ) (h_m : 0 < m) (h_n : 0 < n) 
  (h_eq : (1 / (2 * m)) + (2 / n) + 2 * a = 0) (h_a : a = -1) : 
  4 * m^2 + n^2 ≥ 4 :=
sorry

end part1_find_a_b_part2_inequality_l522_522204


namespace hexagon_segment_FX_length_l522_522352

-- Define the structure of the problem
structure Hexagon extends EuclideanGeometry.HasLength where
  A B C D E F X : Point
  AB_length : length A B = 2
  A_F : EuclideanGeometry.Hexagon A B C D E F

-- Main theorem statement
theorem hexagon_segment_FX_length {A B C D E F X : Point} [Hexagon] (h1 : length A B = 2) (h2 : length A X = 3 * length A B) :
    length F X = 2 * Real.sqrt 13 :=
  sorry

end hexagon_segment_FX_length_l522_522352


namespace martha_cakes_l522_522654

/--
If Martha has 18 small cakes and 3 children, and she divides the cakes equally among the children,
then each child gets 6 cakes.
-/
theorem martha_cakes (cakes : ℕ) (children : ℕ) (cakes_eq : cakes = 18) (children_eq : children = 3) :
  cakes / children = 6 :=
by {
  rw [cakes_eq, children_eq],
  norm_num,
}

end martha_cakes_l522_522654


namespace ratio_triangle_to_shaded_eq_1_to_pi_minus_2_l522_522058

-- Definitions for the square and circle

def radius : ℝ := 1
def diagonal (s : ℝ) : ℝ := s * Real.sqrt 2

def side_length_of_square : ℝ :=
  let diameter := 2 * radius
  diameter / Real.sqrt 2

def area_of_square (s : ℝ) : ℝ := s * s
def area_of_circle (r : ℝ) : ℝ := Real.pi * r * r
def area_of_triangle (base height : ℝ) : ℝ := (1/2) * base * height

def area_of_shaded_region : ℝ :=
  let circle_area := area_of_circle radius
  let square_area := area_of_square side_length_of_square
  circle_area - square_area

def ratio_of_triangle_to_shaded : ℝ :=
  let triangle_area := area_of_triangle side_length_of_square side_length_of_square
  triangle_area / area_of_shaded_region

-- Theorem to be proved
theorem ratio_triangle_to_shaded_eq_1_to_pi_minus_2 :
  ratio_of_triangle_to_shaded = 1 / (Real.pi - 2) :=
sorry

end ratio_triangle_to_shaded_eq_1_to_pi_minus_2_l522_522058


namespace least_constant_inequality_l522_522639

theorem least_constant_inequality (n : ℕ) (hn : n ≥ 2) (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i) :
    (∑ i j in Finset.univ.pair, x i * x j * (x i ^ 2 + x j ^ 2)) ≤ (1 / 8) * (∑ i, x i) ^ 4 := 
sorry

end least_constant_inequality_l522_522639


namespace intersection_x_coordinate_l522_522848

theorem intersection_x_coordinate (a b : ℝ) (h : a + b = 9) :
  let x := (5 - a) / (a - 3) in x = 1 :=
by
  assume a b : ℝ
  assume h : a + b = 9
  let x := (5 - a) / (a - 3)
  have hx : x = 1 := sorry
  exact hx

end intersection_x_coordinate_l522_522848


namespace correct_calculation_l522_522756

theorem correct_calculation :
  (∀ a b : ℝ, (4 * a * b)^2 = 16 * a^2 * b^2) ∧
  (∀ a : ℝ, a^2 * a^3 = a^5) ∧
  (∀ a : ℝ, a^2 + a^2 = 2 * a^2) ∧
  (∀ a b : ℝ, (-3 * a^3 * b)^2 = 9 * a^6 * b^2)
: option_D_correct :=
  sorry

end correct_calculation_l522_522756


namespace adjacent_cells_have_large_difference_l522_522982

def is_adjacent (i₁ j₁ i₂ j₂ : ℕ) : Prop :=
  (abs (i₁ - i₂) = 1 ∧ j₁ = j₂) ∨ (abs (j₁ - j₂) = 1 ∧ i₁ = i₂)

def valid_grid (grid : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, i < 8 → j < 8 → 1 ≤ grid i j ∧ grid i j ≤ 64 ∧ 
    (∀ i' j', i' < 8 → j' < 8 → (grid i' j' = grid i j → i' = i ∧ j' = j))

theorem adjacent_cells_have_large_difference (grid : ℕ → ℕ → ℕ) 
  (h_valid : valid_grid grid) : 
  ∃ i₁ j₁ i₂ j₂, i₁ < 8 ∧ j₁ < 8 ∧ i₂ < 8 ∧ j₂ < 8 ∧ is_adjacent i₁ j₁ i₂ j₂ ∧ 
  |grid i₁ j₁ - grid i₂ j₂| ≥ 5 :=
begin
  sorry
end

end adjacent_cells_have_large_difference_l522_522982


namespace solution_set_unique_l522_522968

theorem solution_set_unique (a b : ℝ) : 
  (∀ x y : ℝ, (x, y) = (2, 1) ↔ (ax + y = 2 ∧ x + by = 2)) → 
  a = 1/2 ∧ b = 0 :=
by sorry

end solution_set_unique_l522_522968


namespace lines_through_same_quadrants_l522_522992

theorem lines_through_same_quadrants (k b : ℝ) (hk : k ≠ 0):
    ∃ n, n ≥ 7 ∧ ∀ (f : Fin n → ℝ × ℝ), ∃ (i j : Fin n), i ≠ j ∧ 
    ((f i).1 > 0 ∧ (f j).1 > 0 ∧ (f i).2 > 0 ∧ (f j).2 > 0 ∨
     (f i).1 > 0 ∧ (f j).1 > 0 ∧ (f i).2 < 0 ∧ (f j).2 < 0 ∨
     (f i).1 > 0 ∧ (f j).1 > 0 ∧ (f i).2 = 0 ∧ (f j).2 = 0 ∨
     (f i).1 < 0 ∧ (f j).1 < 0 ∧ (f i).2 > 0 ∧ (f j).2 > 0 ∨
     (f i).1 < 0 ∧ (f j).1 < 0 ∧ (f i).2 < 0 ∧ (f j).2 < 0 ∨
     (f i).1 < 0 ∧ (f j).1 < 0 ∧ (f i).2 = 0 ∧ (f j).2 = 0) :=
by sorry

end lines_through_same_quadrants_l522_522992


namespace maximum_value_of_A_l522_522211

noncomputable def maximum_A (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  (∑ k in Finset.range n, real.sqrt (real.sin (x k))) * 
  (∑ k in Finset.range n, real.sqrt (real.cos (x k)))

theorem maximum_value_of_A (x : ℕ → ℝ) (n : ℕ) (h : ∀ k < n, x k ∈ Icc 0 (real.pi / 2)):
  maximum_A x n ≤ n^2 * real.sqrt 2⁻¹ ∧ 
  (∀ k < n, x k = real.pi / 4) → maximum_A x n = n^2 * real.sqrt 2⁻¹ :=
sorry

end maximum_value_of_A_l522_522211


namespace solve_x_l522_522505

theorem solve_x :
  ∀ (x y z w : ℤ),
    x = y + 7 →
    y = z + 15 →
    z = w + 25 →
    w = 65 →
    x = 112 :=
by
  intros x y z w
  intros h1 h2 h3 h4
  sorry

end solve_x_l522_522505


namespace josh_money_left_l522_522289

def initial_amount : ℝ := 9
def spent_on_drink : ℝ := 1.75
def spent_on_item : ℝ := 1.25

theorem josh_money_left : initial_amount - (spent_on_drink + spent_on_item) = 6 := by
  sorry

end josh_money_left_l522_522289


namespace find_m_value_l522_522919

noncomputable def m_value : ℝ := -sqrt 3 / 2

theorem find_m_value : 
  let f : ℝ → ℝ  := cos in
  let x1 := π / 2 in
  let x2 := 3 * π / 2 in
  (0 < x1) →
  (x1 < 2 * π) →
  (0 < x2) →
  (x2 < 2 * π) →
  f x1 = 0 →
  f x2 = 0 →
  ∃ x3 x4 : ℝ, x1 < x3 ∧ x3 < x4 ∧ x4 < x2 ∧ 
              2 * x3 = x1 + x4 ∧
              2 * x4 - x3 = x2 →
  ∀ y : ℝ, f y = m_value → y = x3 ∨ y = x4 :=
by
  sorry

end find_m_value_l522_522919


namespace hyperbola_equation_line_equation_through_focus_l522_522192

theorem hyperbola_equation
  (a b c : ℝ)
  (h_foci_shared : c = 5)
  (h_ellipse : (a = 7 ∧ b = √24))
  (h_asymptote_ratio : ∀ {x y : ℝ}, y = (4 / 3) * x ∨ y = -(4 / 3) * x) :
  ∃ (a' b' : ℝ), a' = 3 ∧ b' = 4 ∧ (c ^= a' ^ 2 + b' ^ 2) := 
by 
  sorry -- proof not required

theorem line_equation_through_focus
  (F : ℝ × ℝ)
  (h_focus : F = (5, 0))
  (inclination_angle : ℝ)
  (h_angle : inclination_angle = π / 3) :
  ∃ (m : ℝ), m = √3 ∧ ∀ x y , y = m * (x - 5) := 
by 
  sorry -- proof not required

end hyperbola_equation_line_equation_through_focus_l522_522192


namespace find_y_l522_522408

noncomputable def y : ℝ := Real.log 6 / Real.log 2 + 81

theorem find_y :
  8^(27:ℝ) + 8^(27:ℝ) + 8^(27:ℝ) + 8^(27:ℝ) + 8^(27:ℝ) + 8^(27:ℝ) = 2^y :=
sorry

end find_y_l522_522408


namespace exist_line_connecting_point_intersection_exist_parallel_line_through_point_exist_circumcircle_passing_points_l522_522019

-- Problem (a):

-- Given two lines l1 and l2 intersecting at a point P, and a point M, prove the existence of a line connecting M and P.
theorem exist_line_connecting_point_intersection (l1 l2 : Line) (P : Point) (M : Point) 
  (h_intersect : intersects l1 l2 P) : exists L : Line, connects L M P := sorry

-- Problem (b):

-- Given two pairs of lines (a1, a2) and (b1, b2) intersecting, defining a line l, and a point M,
-- prove the existence of a line through M parallel to l.
theorem exist_parallel_line_through_point (a1 a2 b1 b2 l : Line) (P : Point)
  (M : Point) (h_intersect_a : intersects a1 a2 P) (h_intersect_b : intersects b1 b2 P) 
  (h_defines_l : defines l P) : exists L : Line, parallel L l ∧ passes_through L M := sorry

-- Problem (c):

-- Given points A, B, C not collinear, defined by pairs of intersecting lines,
-- prove the existence of a circumcircle passing through A, B, C.
theorem exist_circumcircle_passing_points (a1 a2 h1 h2 c1 c2 : Line) 
  (A B C : Point) (h_intersect_A : intersects a1 a2 A) 
  (h_intersect_B : intersects h1 h2 B) (h_intersect_C : intersects c1 c2 C)
  (h_not_collinear : ¬ collinear A B C) : exists S : Circle, passes_through S A ∧ passes_through S B ∧ passes_through S C := sorry

end exist_line_connecting_point_intersection_exist_parallel_line_through_point_exist_circumcircle_passing_points_l522_522019


namespace coef_linear_term_expansion_l522_522266

theorem coef_linear_term_expansion : 
  let f := (1 : ℚ) + X
  let g := (X + 1/X) * f^5
  coeff g 1 = 11 := by
  sorry

end coef_linear_term_expansion_l522_522266


namespace problem_statements_l522_522414

theorem problem_statements :
  let S1 := ∀ (x : ℤ) (k : ℤ), x = 2 * k + 1 → (x % 2 = 1)
  let S2 := (∀ (x : ℝ), x > 2 → x > 1) 
            ∧ (∀ (x : ℝ), x > 1 → (x ≥ 2 ∨ x < 2)) 
  let S3 := ∀ (x : ℝ), ¬(∃ (x : ℝ), ∃ (y : ℝ), y = x^2 + 1 ∧ x = y)
  let S4 := ¬(∀ (x : ℝ), x > 1 → x^2 - x > 0) → (∃ (x : ℝ), x > 1 ∧ x^2 - x ≤ 0)
  (S1 ∧ S2 ∧ S3 ∧ ¬S4) := by
    sorry

end problem_statements_l522_522414


namespace min_glue_drops_l522_522816

-- Define the problem-specific constants
def num_stones : ℕ := 36
def target_mass : ℕ := 37

-- Define the original set of stone masses
def stone_masses : Finset ℕ := Finset.range (num_stones + 1)

-- Define the main theorem statement
theorem min_glue_drops : ∃ (d : ℕ), d = 9 ∧ 
  ∀ (set_of_pairs : Finset (Finset ℕ)), 
    (Finset.card set_of_pairs = d) → 
    (∀ (pair : Finset ℕ) in set_of_pairs, pair.card = 2) → 
    (∀ (stone_set : Finset ℕ), 
      (∀ (pair : Finset ℕ) in set_of_pairs, pair ⊆ stone_set) → 
      ¬ (stone_set.sum id = target_mass)) :=
sorry

end min_glue_drops_l522_522816


namespace relationship_a_b_l522_522250

-- Definitions of the two quadratic equations having a single common root
def has_common_root (a b : ℝ) : Prop :=
  ∃ t : ℝ, (t^2 + a * t + b = 0) ∧ (t^2 + b * t + a = 0)

-- Theorem stating the relationship between a and b
theorem relationship_a_b (a b : ℝ) (h : has_common_root a b) : a ≠ b → a + b + 1 = 0 :=
by sorry

end relationship_a_b_l522_522250


namespace passing_marks_l522_522017

theorem passing_marks (T P : ℝ) 
  (h1 : 0.30 * T = P - 60) 
  (h2 : 0.45 * T = P + 30) : 
  P = 240 := 
by
  sorry

end passing_marks_l522_522017


namespace area_ratio_and_angle_l522_522621

theorem area_ratio_and_angle (a : ℝ) (α : ℝ) (h1 : 0 < a)
  (hα1 : real.arctan 2 < α) (hα2 : α < real.pi / 2) :
  let S := a * a in
  let S_AMND := (a^2 / 2) + (a^2 / (2 * real.tan α)) in
  let S_BMNC := (a^2 / 2) * (1 - 1 / (real.tan α)) in
  S_AMND / S_BMNC = real.tan (α - real.pi / 4) / real.tan α :=
by {
  let S := a * a,
  let S_AMND := (a^2 / 2) + (a^2 / (2 * real.tan α)),
  let S_BMNC := (a^2 / 2) * (1 - 1 / (real.tan α)),
  sorry
}

end area_ratio_and_angle_l522_522621


namespace parabola_ratio_l522_522923

theorem parabola_ratio {p : ℝ} (hp : p > 0) :
  (let F := (p / 2, 0) in
   let A := (p / 6, sqrt (p*2 / 3))
   let B := (3 * p / 2, -sqrt (3 * p * 2)) in
  (abs ((fst A) + p / 2 - (fst F)) / abs ((fst B) + p / 2 - (fst F))) = 1 / 3) := sorry

end parabola_ratio_l522_522923


namespace geometric_sequence_sum_l522_522990

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
    (h1 : a 1 = 3)
    (h2 : a 4 = 24)
    (hn : ∀ n, a n = a 1 * q ^ (n - 1)) :
    (a 3 + a 4 + a 5 = 84) :=
by
  -- Proof will go here
  sorry

end geometric_sequence_sum_l522_522990


namespace cube_edge_length_l522_522712

-- Definitions based on given conditions
def paper_cost_per_kg : ℝ := 60
def paper_area_coverage_per_kg : ℝ := 20
def total_expenditure : ℝ := 1800
def surface_area_of_cube (a : ℝ) : ℝ := 6 * a^2

-- The main proof problem
theorem cube_edge_length :
  ∃ a : ℝ, surface_area_of_cube a = paper_area_coverage_per_kg * (total_expenditure / paper_cost_per_kg) ∧ a = 10 :=
by
  sorry

end cube_edge_length_l522_522712


namespace shape_D_is_symmetric_l522_522804

def IsSymmetricShape (original : Type) (shapes : List original) (symmetricShape : original) : Prop :=
  ∃ shape, shape ∈ shapes ∧ shape = symmetricShape

-- Define the original L-like shape and the list of shapes
constant original_L_shape : Type
constant shapes : List original_L_shape
constant symmetry_across_horizontal_line : original_L_shape

-- Define the shapes A, B, C, D, E
constant shape_A : original_L_shape
constant shape_B : original_L_shape
constant shape_C : original_L_shape
constant shape_D : original_L_shape
constant shape_E : original_L_shape

-- Assuming the shapes list includes A, B, C, D and E
axiom shapes_list : shapes = [shape_A, shape_B, shape_C, shape_D, shape_E]

-- The theorem states that shape_D is symmetric to the original L-like shape
theorem shape_D_is_symmetric :
  IsSymmetricShape original_L_shape shapes shape_D :=
by
  sorry

end shape_D_is_symmetric_l522_522804


namespace binom_20_6_l522_522087

theorem binom_20_6 : nat.choose 20 6 = 19380 := 
by 
  sorry

end binom_20_6_l522_522087


namespace triangle_is_right_l522_522598

theorem triangle_is_right (a b c : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_equations_share_root : ∃ α : ℝ, α^2 + 2*a*α + b^2 = 0 ∧ α^2 + 2*c*α - b^2 = 0) :
  a^2 = b^2 + c^2 :=
by sorry

end triangle_is_right_l522_522598


namespace smallest_n_where_a_n_is_neg_l522_522191

variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (d : ℝ)

-- Arithmetic sequence definition
def is_arithmetic_sequence : Prop := ∀ n, a (n + 1) = a n + d

-- Sum of the first n terms of the sequence
def sum_of_first_n_terms (n : ℕ) : ℝ := (n * (a 1 + a n)) / 2

-- Given conditions
axiom S_12_pos : sum_of_first_n_terms a 12 > 0
axiom S_13_neg : sum_of_first_n_terms a 13 < 0

-- The smallest n such that a_n < 0
theorem smallest_n_where_a_n_is_neg : ∃ n, a n < 0 ∧ ∀ m, m < n → a m ≥ 0 := 
  ∃ n, a n < 0 ∧ ∀ m, m < n → a m ≥ 0

end smallest_n_where_a_n_is_neg_l522_522191


namespace binom_20_6_l522_522089

theorem binom_20_6 : nat.choose 20 6 = 19380 := 
by 
  sorry

end binom_20_6_l522_522089


namespace inscribed_sphere_volume_l522_522556

theorem inscribed_sphere_volume
  (a : ℝ)
  (h_cube_surface_area : 6 * a^2 = 24) :
  (4 / 3) * Real.pi * (a / 2)^3 = (4 / 3) * Real.pi :=
by
  -- sorry to skip the actual proof
  sorry

end inscribed_sphere_volume_l522_522556


namespace min_value_of_sequence_l522_522879

theorem min_value_of_sequence :
  ∃ (a : ℕ → ℤ), a 1 = 0 ∧ (∀ n : ℕ, n ≥ 2 → |a n| = |a (n - 1) + 1|) ∧ (a 1 + a 2 + a 3 + a 4 = -2) :=
by
  sorry

end min_value_of_sequence_l522_522879


namespace line_divides_circle_union_equal_area_l522_522657

theorem line_divides_circle_union_equal_area :
  ∃ (a b c : ℕ), 
    (a.gcd b.gcd c = 1) ∧
    (a, b, c > 0) ∧
    (a^2 + b^2 + c^2 = 69) :=
by 
  sorry

end line_divides_circle_union_equal_area_l522_522657


namespace find_prime_pairs_l522_522514

theorem find_prime_pairs (p q n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hn : 0 < n) :
  p * (p + 1) + q * (q + 1) = n * (n + 1) ↔ (p = 3 ∧ q = 5 ∧ n = 6) ∨ (p = 5 ∧ q = 3 ∧ n = 6) ∨ (p = 2 ∧ q = 2 ∧ n = 3) :=
by
  sorry

end find_prime_pairs_l522_522514


namespace trigonometric_identity_l522_522082

theorem trigonometric_identity :
  sin (69 * Real.pi / 180) * cos (9 * Real.pi / 180) - sin (21 * Real.pi / 180) * cos (81 * Real.pi / 180) = Real.sqrt 3 / 2 :=
  sorry

end trigonometric_identity_l522_522082


namespace tetrahedron_edges_sum_l522_522298

theorem tetrahedron_edges_sum
  (a a1 b b1 c c1 : ℝ)
  (α β γ : ℝ)
  (hα : α ≤ π / 2)
  (hβ : β ≤ π / 2)
  (hγ : γ ≤ π / 2) :
  a * a1 * Real.cos α = b * b1 * Real.cos β + c * c1 * Real.cos γ
  ∨ b * b1 * Real.cos β = c * c1 * Real.cos γ + a * a1 * Real.cos α
  ∨ c * c1 * Real.cos γ = a * a1 * Real.cos α + b * b1 * Real.cos β :=
sorry

end tetrahedron_edges_sum_l522_522298


namespace negation_of_proposition_exists_negation_of_proposition_l522_522248

theorem negation_of_proposition : 
  (∀ x : ℝ, 2^x - 2*x - 2 ≥ 0) ↔ ¬(∀ x : ℝ, 2^x - 2*x - 2 ≥ 0) :=
by
  sorry

theorem exists_negation_of_proposition : 
  (¬(∀ x : ℝ, 2^x - 2*x - 2 ≥ 0)) ↔ ∃ x : ℝ, 2^x - 2*x - 2 < 0 :=
by
  sorry

end negation_of_proposition_exists_negation_of_proposition_l522_522248


namespace prob_club_then_diamond_then_heart_l522_522732

noncomputable def prob_first_card_club := 13 / 52
noncomputable def prob_second_card_diamond_given_first_club := 13 / 51
noncomputable def prob_third_card_heart_given_first_club_second_diamond := 13 / 50

noncomputable def overall_probability := 
  prob_first_card_club * 
  prob_second_card_diamond_given_first_club * 
  prob_third_card_heart_given_first_club_second_diamond

theorem prob_club_then_diamond_then_heart :
  overall_probability = 2197 / 132600 :=
by
  sorry

end prob_club_then_diamond_then_heart_l522_522732


namespace seating_arrangements_l522_522360

theorem seating_arrangements :
  let chairs := List.range 10 -- chairs numbered 1 through 10
  let men_women := List.range 10 -- alternating seating, half men and half women
  let married_couples := List.zip (List.range 5) (List.range 5)
  ∃ (m_w : List (ℕ × ℕ)), -- seating arrangement
    (m_w.length = 10) ∧ -- 10 seats
    (∀ (i j : ℕ), i ≠ j → m_w[i] ≠ m_w[j]) ∧ -- all positions are unique
    (∀ (i : ℕ), (m_w[i].fst + 1) % 2 = 0 → (m_w[(i + 1) % 10].fst + 1) % 2 ≠ 0 ∧ (m_w[(i + 9) % 10].fst + 1) % 2 ≠ 0) ∧ -- men and women alternate, no spouses adjacent or opposite
    (∀ (i : ℕ), ∃ k : ℕ, (i < 5 → m_w[i].fst = married_couples[k].fst ∧ ∃ j : ℕ, (i < 5 → m_w[j].fst = married_couples[k].snd))) → -- married couples
    (m_w.permutations.length = 480) := sorry

end seating_arrangements_l522_522360


namespace max_value_expression_l522_522299

noncomputable def a (φ : ℝ) : ℝ := 3 * Real.cos φ
noncomputable def b (φ : ℝ) : ℝ := 3 * Real.sin φ

theorem max_value_expression (φ θ : ℝ) : 
  ∃ c : ℝ, c = 3 * Real.cos (θ - φ) ∧ c ≤ 3 := by
  sorry

end max_value_expression_l522_522299


namespace find_x_in_interval_l522_522131

theorem find_x_in_interval : {x : ℝ | (x - 2) / (x - 4) ≥ 3} = set.Ioc 4 5 ∪ {5} :=
by
  sorry

end find_x_in_interval_l522_522131


namespace find_ratio_l522_522972

variable (A B C : ℝ) (a b c : ℝ)

axiom (A_eq_120 : A = Real.pi * (2 / 3)) -- 120 degrees in radians
axiom (b_eq_1 : b = 1)
axiom (area_eq_sqrt3 : (1 / 2) * b * c * Real.sin A = Real.sqrt 3)
axiom (law_of_cosines : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)
axiom (law_of_sines : b / Real.sin B = c / Real.sin C)

theorem find_ratio : 
  (b + c) / (Real.sin B + Real.sin C) = 2 * Real.sqrt 7 :=
sorry

end find_ratio_l522_522972


namespace coeff_x5_of_expansion_l522_522694

theorem coeff_x5_of_expansion : 
  (Polynomial.coeff ((Polynomial.C (1 : ℤ)) * (Polynomial.X ^ 2 - Polynomial.X - Polynomial.C 2) ^ 3) 5) = -3 := 
by sorry

end coeff_x5_of_expansion_l522_522694


namespace circumcenter_eq_orthocenter_l522_522715

theorem circumcenter_eq_orthocenter {A B C A' B' C' O O'} (h1 : radius_circumcircle_ABC = radius_circle_tangent_A'B'_C' A B C A' B' C' O O') :
  is_center_circumcircle O A B C ∧ is_orthocenter O A' B' C' :=
by
  sorry

end circumcenter_eq_orthocenter_l522_522715


namespace triangle_properties_equivalence_l522_522624

-- Define the given properties for the two triangles
variables {A B C A' B' C' : Type}

-- Triangle side lengths and properties
def triangles_equal (b b' c c' : ℝ) : Prop :=
  (b = b') ∧ (c = c')

def equivalent_side_lengths (a a' b b' c c' : ℝ) : Prop :=
  a = a'

def equivalent_medians (ma ma' b b' c c' a a' : ℝ) : Prop :=
  ma = ma'

def equivalent_altitudes (ha ha' Δ Δ' a a' : ℝ) : Prop :=
  ha = ha'

def equivalent_angle_bisectors (ta ta' b b' c c' a a' : ℝ) : Prop :=
  ta = ta'

def equivalent_circumradii (R R' a a' b b' c c' : ℝ) : Prop :=
  R = R'

def equivalent_areas (Δ Δ' b b' c c' A A' : ℝ) : Prop :=
  Δ = Δ'

-- Main theorem statement
theorem triangle_properties_equivalence
  (b b' c c' a a' ma ma' ha ha' ta ta' R R' Δ Δ' : ℝ)
  (A A' : ℝ)
  (eq_b : b = b')
  (eq_c : c = c') :
  equivalent_side_lengths a a' b b' c c' ∧ 
  equivalent_medians ma ma' b b' c c' a a' ∧ 
  equivalent_altitudes ha ha' Δ Δ' a a' ∧ 
  equivalent_angle_bisectors ta ta' b b' c c' a a' ∧ 
  equivalent_circumradii R R' a a' b b' c c' ∧ 
  equivalent_areas Δ Δ' b b' c c' A A'
:= by
  sorry

end triangle_properties_equivalence_l522_522624


namespace Tony_total_payment_l522_522634

-- Defining the cost of items
def lego_block_cost : ℝ := 250
def toy_sword_cost : ℝ := 120
def play_dough_cost : ℝ := 35

-- Quantities of each item
def total_lego_blocks : ℕ := 3
def total_toy_swords : ℕ := 5
def total_play_doughs : ℕ := 10

-- Quantities purchased on each day
def first_day_lego_blocks : ℕ := 2
def first_day_toy_swords : ℕ := 3
def second_day_lego_blocks : ℕ := total_lego_blocks - first_day_lego_blocks
def second_day_toy_swords : ℕ := total_toy_swords - first_day_toy_swords
def second_day_play_doughs : ℕ := total_play_doughs

-- Discounts and tax rates
def first_day_discount : ℝ := 0.20
def second_day_discount : ℝ := 0.10
def sales_tax : ℝ := 0.05

-- Calculating first day purchase amounts
def first_day_cost_before_discount : ℝ := (first_day_lego_blocks * lego_block_cost) + (first_day_toy_swords * toy_sword_cost)
def first_day_discount_amount : ℝ := first_day_cost_before_discount * first_day_discount
def first_day_cost_after_discount : ℝ := first_day_cost_before_discount - first_day_discount_amount
def first_day_sales_tax_amount : ℝ := first_day_cost_after_discount * sales_tax
def first_day_total_cost : ℝ := first_day_cost_after_discount + first_day_sales_tax_amount

-- Calculating second day purchase amounts
def second_day_cost_before_discount : ℝ := (second_day_lego_blocks * lego_block_cost) + (second_day_toy_swords * toy_sword_cost) + 
                                           (second_day_play_doughs * play_dough_cost)
def second_day_discount_amount : ℝ := second_day_cost_before_discount * second_day_discount
def second_day_cost_after_discount : ℝ := second_day_cost_before_discount - second_day_discount_amount
def second_day_sales_tax_amount : ℝ := second_day_cost_after_discount * sales_tax
def second_day_total_cost : ℝ := second_day_cost_after_discount + second_day_sales_tax_amount

-- Total cost
def total_cost : ℝ := first_day_total_cost + second_day_total_cost

-- Lean theorem statement
theorem Tony_total_payment : total_cost = 1516.20 := by
  sorry

end Tony_total_payment_l522_522634


namespace arithmetic_sequence_sum_example_l522_522027

variable {a : ℕ → ℝ} -- Defining a_n as a function from natural numbers to the reals
variable (S : ℕ → ℝ) -- Defining S_n as a function from natural numbers to the reals

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (a 0 + a (n - 1))

theorem arithmetic_sequence_sum_example
  (d : ℝ)
  (a3 : ℝ) (a6 : ℝ) (a9 : ℝ)
  (h_arith : is_arithmetic_sequence a d)
  (h_condition : a3 + a6 + a9 = 60) :
  sum_first_n_terms a 11 = 220 := 
by
  sorry

end arithmetic_sequence_sum_example_l522_522027


namespace simplify_complex_expression_l522_522353

noncomputable def ω : ℂ := (-1 + complex.I * real.sqrt 3) / 2
noncomputable def ω' : ℂ := (-1 - complex.I * real.sqrt 3) / 2

lemma ω_cube : ω^3 = 1 := by sorry
lemma ω'_cube : ω'^3 = 1 := by sorry

theorem simplify_complex_expression : (ω ^ 12) + (ω' ^ 12) = 2 := by
  have h1 : ω^3 = 1 := ω_cube
  have h2 : ω'^3 = 1 := ω'_cube
  sorry

end simplify_complex_expression_l522_522353


namespace left_pan_where_3_over_2_will_tip_l522_522613

variable {E : Type} [Nonempty E] [Fintype E]

-- Assume there are 10 elephants
constant elephants : Fin 10 → E

-- Here, we denote the weight of an elephant by some weight function, assumed as constant
constant weight : E → ℝ

-- Condition: If any four elephants are on the left pan and any three of the remainder are on the right pan, the left pan tips over
axiom tipping_condition : 
  ∀ (a b c d : Fin 10) (e f g : Fin 10), 
  weight (elephants a) + weight (elephants b) + weight (elephants c) + weight (elephants d) > 
  weight (elephants e) + weight (elephants f) + weight (elephants g)

theorem left_pan_where_3_over_2_will_tip : 
  ∀ (a b c d e: Fin 10), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e →
  weight (elephants a) + weight (elephants b) + weight (elephants c) > 
  weight (elephants d) + weight (elephants e) := 
by
  sorry

end left_pan_where_3_over_2_will_tip_l522_522613


namespace parallelogram_angles_parallelogram_area_l522_522665

variables {A B C D M : Point}
variables {AD_length : ℝ} (h_AD : AD_length = 1)
variables (is_midpoint : midpoint A D = M)
variables (angle_CM_AD : angle_between CM AD = 45)
variables (is_equidistant : equidistant B CM A)

theorem parallelogram_angles (p : is_parallelogram A B C D) :
  (angle_between AB AD = 75 ∧ angle_between BC AD = 105) :=
sorry

theorem parallelogram_area (p : is_parallelogram A B C D) :
  (area_parallelogram A B C D = (sqrt 3 + 1) / 4) :=
sorry

end parallelogram_angles_parallelogram_area_l522_522665


namespace find_sum_of_coefficients_l522_522704

-- Definitions from conditions
def is_parabola_passing_through (a b c x1 y1 x2 y2 : ℝ) : Prop :=
  (y1 = a * x1^2 + b * x1 + c) ∧ (y2 = a * x2^2 + b * x2 + c)

def has_max_value (f : ℝ → ℝ) (max : ℝ) : Prop :=
  ∃ x0, (∀ x, f x ≤ f x0) ∧ f x0 = max

 noncomputable def quadratic (a b c : ℝ) : ℝ → ℝ := λ x, a * x^2 + b * x + c

-- Problem statement
theorem find_sum_of_coefficients (a b c : ℝ) :
  is_parabola_passing_through a b c (-3) 0 3 0 →
  has_max_value (quadratic a b c) 36 →
  a + b + c = 32 :=
by
  intros h_pass h_max,
  sorry

end find_sum_of_coefficients_l522_522704


namespace log_exp_test_l522_522477

noncomputable def log_sqrt2_4 : ℝ := Real.log 4 / Real.log (Real.sqrt 2)
noncomputable def e_ln3 : ℝ := Real.exp (Real.log 3)
noncomputable def part_three : ℝ := (0.5 : ℝ)⁻²

theorem log_exp_test : log_sqrt2_4 + e_ln3 + part_three = 11 := by
  sorry

end log_exp_test_l522_522477


namespace flash_catch_ace_distance_l522_522066

noncomputable def flash_distance (y x v a : ℝ) (h1: x > 1) (h2: a > 0) : ℝ :=
  y + xv * (-(x - 1) * v + Real.sqrt((x - 1)^2 * v^2 + 2 * a * y)) / a

theorem flash_catch_ace_distance (y x v a : ℝ) (h1 : x > 1) (h2 : a > 0) : 
  flash_distance y x v a h1 h2 = y + xv * (-(x - 1) * v + Real.sqrt((x - 1)^2 * v^2 + 2 * a * y)) / a :=
by
  sorry

end flash_catch_ace_distance_l522_522066


namespace damaged_potatoes_l522_522463

theorem damaged_potatoes (initial_potatoes : ℕ) (weight_per_bag : ℕ) (price_per_bag : ℕ) (total_sales : ℕ) :
  initial_potatoes = 6500 →
  weight_per_bag = 50 →
  price_per_bag = 72 →
  total_sales = 9144 →
  ∃ damaged_potatoes : ℕ, damaged_potatoes = initial_potatoes - (total_sales / price_per_bag) * weight_per_bag ∧
                               damaged_potatoes = 150 :=
by
  intros _ _ _ _ 
  exact sorry

end damaged_potatoes_l522_522463


namespace distance_greater_than_one_probability_l522_522051

-- Define the side length of the square.
def side_length : ℝ := 4

-- Define the total area of the square.
def total_area : ℝ := side_length * side_length

-- Define the condition for the smaller square where the distance to all sides is greater than 1.
def smaller_side_length : ℝ := side_length - 2 * 1
def smaller_area : ℝ := smaller_side_length * smaller_side_length

-- Define the probability as the ratio of the areas.
def probability : ℝ := smaller_area / total_area

-- State the theorem that formalizes the given problem.
theorem distance_greater_than_one_probability :
  probability = 1 / 4 :=
by
  sorry

end distance_greater_than_one_probability_l522_522051


namespace shaded_region_area_correct_l522_522619

-- Definitions of given conditions
def side_length : ℝ := 3
def radius_small_hexagon : ℝ := side_length / 2
def area_large_hexagon : ℝ := (3 * Real.sqrt 3 / 2) * side_length ^ 2
def area_one_semicircle : ℝ := (1 / 2) * Real.pi * (radius_small_hexagon ^ 2)
def total_area_semicircles : ℝ := 6 * area_one_semicircle
def area_small_hexagon : ℝ := (3 * Real.sqrt 3 / 2) * (radius_small_hexagon ^ 2)

-- Definition for the shaded area we want to prove
def area_shaded_region : ℝ :=
  area_large_hexagon - total_area_semicircles - area_small_hexagon

-- The target theorem to prove
theorem shaded_region_area_correct :
  area_shaded_region = (81 * Real.sqrt 3 / 8) - (27 * Real.pi / 4) :=
  by
    -- Proof goes here
    sorry

end shaded_region_area_correct_l522_522619


namespace graphistan_maximum_k_l522_522229

noncomputable theory
open_locale classical

variables (V : Type) [fintype V] [decidable_eq V]

def out_degree (G : V → V → Prop) (v : V) : ℕ :=
fintype.card {w : V // G v w}

def in_degree (G : V → V → Prop) (v : V) : ℕ :=
fintype.card {w : V // G w v}

def strongly_connected (G : V → V → Prop) : Prop :=
∀ u v, ∃ (p : list V), u ∈ p ∧ v ∈ p ∧ ∀ i ∈ p.init, G i.head (i.tail.head)

theorem graphistan_maximum_k :
  ∀ (G : V → V → Prop),
  fintype.card V = 2011 →
  (∀ v, abs ((out_degree G v : ℤ) - (in_degree G v : ℤ)) ≤ 1005) →
  strongly_connected G :=
begin
  intros G h_card h_deg,
  sorry -- Proof goes here
end

end graphistan_maximum_k_l522_522229


namespace possible_values_expression_l522_522167

theorem possible_values_expression (a b c d : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) :
  ∃ (x : ℝ), x ∈ {5, 1, -3} ∧ x = (a / |a| + b / |b| + c / |c| + d / |d| + (abcd / |abcd|)) :=
by
  sorry

end possible_values_expression_l522_522167


namespace problem_1_problem_2_problem_3_l522_522567

-- Define the function f(x)
def f (x : ℝ) (k : ℝ) : ℝ :=
  x^2 + x⁻² + k * (x - x⁻¹)

-- Problem 1: Prove that k = 0 if f(x) is an even function
theorem problem_1 (k : ℝ) (h_even : ∀ x : ℝ, f (-x) k = f x k) : k = 0 :=
sorry

-- Problem 2: Prove that f(x) is monotonically increasing in the interval (1, +∞) when k = 0
theorem problem_2 (h_inc : ∀ x y : ℝ, 1 < x → 1 < y → x < y → f x 0 < f y 0) : true :=
sorry

-- Problem 3: Prove that the range of k is -4 ≤ k < -1 if the maximum value of g(x) = |f(x)| in the interval [1, |k|] is 2
theorem problem_3 (h_max : ∀ k : ℝ, (∀ (x : ℝ), 1 ≤ x → x ≤ |k| → abs (f x k) ≤ 2) → -4 ≤ k ∧ k < -1) : true :=
sorry

end problem_1_problem_2_problem_3_l522_522567


namespace correct_calculation_l522_522758

theorem correct_calculation (a b : ℝ) : (-3 * a^3 * b)^2 = 9 * a^6 * b^2 :=
by sorry

end correct_calculation_l522_522758


namespace inequality_lemma_l522_522309

theorem inequality_lemma (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b + b * c + c * a = 1) :
    (real.cbrt (1 / a + 6 * b) + real.cbrt (1 / b + 6 * c) + real.cbrt (1 / c + 6 * a)) ≤ 1 / (a * b * c) :=
sorry

end inequality_lemma_l522_522309


namespace ratio_h_r_bounds_l522_522753

theorem ratio_h_r_bounds
  {a b c h r : ℝ}
  (h_right_angle : a^2 + b^2 = c^2)
  (h_area1 : 1/2 * a * b = 1/2 * c * h)
  (h_area2 : 1/2 * (a + b + c) * r = 1/2 * a * b) :
  2 < h / r ∧ h / r ≤ 2.41 :=
by
  sorry

end ratio_h_r_bounds_l522_522753


namespace sequence_sum_2016_l522_522925

noncomputable def sequence (a : ℕ → ℚ) : Prop :=
a 1 = 1 / 2 ∧ ∀ n, a (n + 1) = 3 * a n + 1

def sum_sequence (S a : ℕ → ℚ) : Prop :=
S 0 = 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

theorem sequence_sum_2016 
  {a S : ℕ → ℚ}
  (h_seq : sequence a)
  (h_sum : sum_sequence S a) :
  S 2016 = (3 ^ 2016 - 2017) / 2 :=
sorry

end sequence_sum_2016_l522_522925


namespace seating_arrangement_no_two_adjacent_l522_522338

theorem seating_arrangement_no_two_adjacent :
  let chairs : Finset ℕ := {1, 2, 3, 4, 5, 6}
  let people : Finset ℕ := {1, 2, 3}
  let arrangements : Finset (Finset (Fin 6)) := { 
    s | s ⊆ chairs ∧ s.card = 3 ∧ ∀ (a ∈ s) (b ∈ s), a ≠ b + 1 ∧ b ≠ a + 1 
  }
  arrangements.card = 24 :=
by
  sorry

end seating_arrangement_no_two_adjacent_l522_522338


namespace color_ball_ratios_l522_522711

theorem color_ball_ratios (white_balls red_balls blue_balls : ℕ)
  (h_white : white_balls = 12)
  (h_red_ratio : 4 * red_balls = 3 * white_balls)
  (h_blue_ratio : 4 * blue_balls = 2 * white_balls) :
  red_balls = 9 ∧ blue_balls = 6 :=
by
  sorry

end color_ball_ratios_l522_522711


namespace sum_of_f_values_l522_522496

noncomputable def f : ℝ → ℝ := sorry -- Definition of f as it needs to satisfy the conditions set.

-- Stating the conditions for f:
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)
def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f(x + T) = f(x)

-- Main theorem statement
theorem sum_of_f_values (f : ℝ → ℝ) (h_odd : odd_function f) (h_periodic : periodic_function f 2) :
  f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 = 0 :=
sorry -- To be proven

end sum_of_f_values_l522_522496


namespace combination_20_6_l522_522092

theorem combination_20_6 : Nat.choose 20 6 = 38760 :=
by
  sorry

end combination_20_6_l522_522092


namespace exists_nice_coloring_with_three_consecutive_edges_same_color_l522_522877

open Classical -- for the axiom of choice

variables {F : Type} [Polyhedron F] -- assuming F is a convex polyhedron

def nice_coloring (F : Type) [Polyhedron F] : Prop :=
  ∃ (c : Edge F → Fin 3), 
    (∀ (v : Vertex F) (h : v ≠ A), 
      let edges := adjacent_edges v in
      ∃ (a b c : Edge F), 
      a ∈ edges ∧ b ∈ edges ∧ c ∈ edges ∧ 
      c a ≠ c b ∧ c b ≠ c c ∧ c c ≠ c a) ∧
    (number_of_nice_colorings c % 5 ≠ 0)

theorem exists_nice_coloring_with_three_consecutive_edges_same_color 
  (h_polyhedral : convex_polyhedron F)
  (h_degree_A : degree_vertex A = 5)
  (h_degree_other : ∀ (v : Vertex F), v ≠ A → degree_vertex v = 3)
  (h_nice_coloring_exists : ∃c, nice_coloring F) 
  (h_not_div_5 : ¬ (∑ (c : Edge F → Fin 3) in {c | nice_coloring F}, 1) % 5 = 0) :
  ∃ c : Edge F → Fin 3, nice_coloring F ∧ 
    ∃ (e1 e2 e3 : Edge F),
    (e1 ∈ adjacent_edges A ∧ e2 ∈ adjacent_edges A ∧ e3 ∈ adjacent_edges A ∧ 
      consecutive_edges e1 e2 ∧ consecutive_edges e2 e3 ∧ c e1 = c e2 ∧ c e2 = c e3) :=
sorry

end exists_nice_coloring_with_three_consecutive_edges_same_color_l522_522877


namespace price_difference_l522_522451

def P := ℝ

def Coupon_A_savings (P : ℝ) := 0.20 * P
def Coupon_B_savings : ℝ := 40
def Coupon_C_savings (P : ℝ) := 0.30 * (P - 120) + 20

def Coupon_A_geq_Coupon_B (P : ℝ) := Coupon_A_savings P ≥ Coupon_B_savings
def Coupon_A_geq_Coupon_C (P : ℝ) := Coupon_A_savings P ≥ Coupon_C_savings P

noncomputable def x : ℝ := 200
noncomputable def y : ℝ := 300

theorem price_difference (P : ℝ) (h1 : P > 120)
  (h2 : Coupon_A_geq_Coupon_B P)
  (h3 : Coupon_A_geq_Coupon_C P) :
  y - x = 100 := by
  sorry

end price_difference_l522_522451


namespace equilateral_configuration_l522_522116

theorem equilateral_configuration (C : Finset (EuclideanSpace ℝ (Fin 2))) 
  (h1 : 3 ≤ C.card)
  (h2 : ∀ x y ∈ C, x ≠ y → ∃ z ∈ C, x + z = 2 * (midpoint ℝ x y ∨ y + z = 2 * (midpoint ℝ x y))) :
  ∃ A B C : EuclideanSpace ℝ (Fin 2), A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ is_equilateral_triangle A B C :=
by {
  sorry
}

noncomputable def midpoint {V : Type*} [inner_product_space ℝ V] {x y : V} (a b : V) : V :=
1 / 2 * (a + b)

def is_equilateral_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := 
  dist A B = dist B C ∧ dist B C = dist C A

end equilateral_configuration_l522_522116


namespace tan_sum_deg_tan_23_22_deg_l522_522026

open Real

noncomputable def tan_deg (x : ℝ) : ℝ := tan (x * π / 180)

theorem tan_sum_deg {x y : ℝ} : tan (x + y) = (tan x + tan y) / (1 - tan x * tan y) :=
by sorry

theorem tan_23_22_deg :
  (1 + tan_deg 23) * (1 + tan_deg 22) = 2 :=
by 
  have h : tan_deg (23 + 22) = tan_deg 45, from congr_arg tan_deg rfl,
  rw [h, tan_sum_deg, tan_pi_over_four, add_div, one_mul, sub_mul, mul_comm (tan_deg 23) (tan_deg 22),
      div_one, sub_self, zero_div, add_zero],
  exact one_add_one_eq_two

end tan_sum_deg_tan_23_22_deg_l522_522026


namespace brick_length_is_20_l522_522783

noncomputable def courtyard_length : ℝ := 30 * 100 -- in cm
noncomputable def courtyard_width : ℝ := 16 * 100 -- in cm
noncomputable def total_bricks : ℝ := 24000
noncomputable def brick_width : ℝ := 10 -- in cm
noncomputable def courtyard_area : ℝ := courtyard_length * courtyard_width -- in cm²
noncomputable def total_brick_area : ℝ := total_bricks * (brick_width * 20) -- in cm²
theorem brick_length_is_20 :
  (∃ L : ℝ, courtyard_area = total_bricks * (brick_width * L) ∧ L = 20) :=
begin
  use 20,
  have H : courtyard_length * courtyard_width = 24000 * (10 * 20), by sorry,
  exact ⟨H, rfl⟩,
end

end brick_length_is_20_l522_522783


namespace minimum_distance_between_curves_l522_522852

noncomputable def distance_between_curves : ℝ :=
  let x₀ := (-(Real.log 3) - 7) / 3
  let f_x₀ := Real.exp (3 * x₀ + 7) - x₀
  in Real.sqrt 2 * (f_x₀ + (Real.log 3 + 7) / 3)

theorem minimum_distance_between_curves :
  distance_between_curves = Real.sqrt 2 * (8 + Real.log 3) / 3 :=
by
  -- Proof goes here
  sorry

end minimum_distance_between_curves_l522_522852


namespace arithmetic_sequence_probability_correct_l522_522145

noncomputable def arithmetic_sequence_probability : ℚ := 
  let total_ways := Nat.choose 5 3
  let arithmetic_sequences := 4
  (arithmetic_sequences : ℚ) / (total_ways : ℚ)

theorem arithmetic_sequence_probability_correct :
  arithmetic_sequence_probability = 0.4 := by
  unfold arithmetic_sequence_probability
  sorry

end arithmetic_sequence_probability_correct_l522_522145


namespace find_quadratic_expression_l522_522706

-- Define the quadratic function
def quadratic (a b c x : ℝ) := a * x^2 + b * x + c

-- Define conditions
def intersects_x_axis_at_A (a b c : ℝ) : Prop :=
  quadratic a b c (-2) = 0

def intersects_x_axis_at_B (a b c : ℝ) : Prop :=
  quadratic a b c (1) = 0

def has_maximum_value (a : ℝ) : Prop :=
  a < 0

-- Define the target function
def f_expr (x : ℝ) : ℝ := -x^2 - x + 2

-- The theorem to be proved
theorem find_quadratic_expression :
  ∃ a b c, 
    intersects_x_axis_at_A a b c ∧
    intersects_x_axis_at_B a b c ∧
    has_maximum_value a ∧
    ∀ x, quadratic a b c x = f_expr x :=
sorry

end find_quadratic_expression_l522_522706


namespace round_trip_time_l522_522368

theorem round_trip_time (current_speed : ℝ) (boat_speed_still : ℝ) (distance_upstream : ℝ) (total_time : ℝ) :
  current_speed = 4 → 
  boat_speed_still = 18 → 
  distance_upstream = 85.56 →
  total_time = 10 :=
by
  intros h_current h_boat h_distance
  sorry

end round_trip_time_l522_522368


namespace handshakes_7_boys_l522_522767

theorem handshakes_7_boys : Nat.choose 7 2 = 21 :=
by
  sorry

end handshakes_7_boys_l522_522767


namespace functional_equation_solution_l522_522130

theorem functional_equation_solution (f : ℕ+ → ℕ+) :
  (∀ n : ℕ+, f (f (f n)) + f (f n) + f n = 3 * n) →
  ∀ n : ℕ+, f n = n :=
by
  intro h
  sorry

end functional_equation_solution_l522_522130


namespace penny_purchase_exceeded_minimum_spend_l522_522269

theorem penny_purchase_exceeded_minimum_spend :
  let bulk_price_per_pound := 5
  let minimum_spend := 40
  let tax_per_pound := 1
  let total_paid := 240
  let total_cost_per_pound := bulk_price_per_pound + tax_per_pound
  let pounds_purchased := total_paid / total_cost_per_pound
  let minimum_pounds_to_spend := minimum_spend / bulk_price_per_pound
  pounds_purchased - minimum_pounds_to_spend = 32 :=
by
  -- The proof is omitted here as per the instructions.
  sorry

end penny_purchase_exceeded_minimum_spend_l522_522269


namespace packs_of_potato_fries_sold_l522_522123

-- Define the conditions
def P := 15
def P_price := 12
def S := 25
def S_price := 2
def G := 500
def N := 258
def F_price := 0.30

-- Prove the number of packs of potato fries sold
theorem packs_of_potato_fries_sold : 
  let R_p := P * P_price;
  let R_s := S * S_price;
  let raised := G - N;
  242 - (R_p + R_s) = F_price * 40 :=
by
  let Rp := P * P_price;
  let Rs := S * S_price;
  let raised := G - N;
  sorry

end packs_of_potato_fries_sold_l522_522123


namespace find_a_plus_b_l522_522958

theorem find_a_plus_b (a b : ℤ) (ha : a > 0) (hb : b > 0) (h : a^2 - b^4 = 2009) : a + b = 47 :=
by
  sorry

end find_a_plus_b_l522_522958


namespace smallest_sum_of_numbers_in_circle_l522_522778

-- Given conditions for the problem
theorem smallest_sum_of_numbers_in_circle 
    (n : ℕ) 
    (n = 2019) 
    (arr : Fin n → ℤ)
    (h1 : ∀ i, |arr i - arr ((i+1) % n)| ≥ 2)
    (h2 : ∀ i, arr i + arr ((i+1) % n) ≥ 6) 
    : ∃ sum, (sum = ∑ i in Finset.range n, arr i) ∧ sum = 6060 := 
sorry

end smallest_sum_of_numbers_in_circle_l522_522778


namespace quiz_score_of_dropped_student_l522_522422

theorem quiz_score_of_dropped_student (avg16 : ℝ) (avg15 : ℝ) (num_students : ℝ) (dropped_students : ℝ) (x : ℝ)
  (h1 : avg16 = 60.5) (h2 : avg15 = 64) (h3 : num_students = 16) (h4 : dropped_students = 1) :
  x = 60.5 * 16 - 64 * 15 :=
by
  sorry

end quiz_score_of_dropped_student_l522_522422


namespace percent_of_number_l522_522403

theorem percent_of_number (x : ℕ) : \frac{3}{8} * 160 = 0.6 := by 
sorry

end percent_of_number_l522_522403


namespace find_angle_B_l522_522999

def triangle_angles (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * real.cos B - b * real.cos A = c ∧ C = real.pi / 5

theorem find_angle_B (A B C a b c : ℝ) 
    (h : triangle_angles A B C a b c) : B = 3 * real.pi / 10 :=
by sorry

end find_angle_B_l522_522999


namespace find_angle_B_l522_522998

def triangle_angles (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * real.cos B - b * real.cos A = c ∧ C = real.pi / 5

theorem find_angle_B (A B C a b c : ℝ) 
    (h : triangle_angles A B C a b c) : B = 3 * real.pi / 10 :=
by sorry

end find_angle_B_l522_522998


namespace curve_symmetric_wrt_y_eq_x_l522_522908

-- Define the curve
def curve (x y : ℝ) : Prop :=
  x^2 * y + x * y^2 = 1

-- Define symmetry with respect to the line y=x
def symmetric_wrt_y_eq_x (P : ℝ × ℝ → Prop) : Prop :=
  ∀ x y, P (x, y) ↔ P (y, x)

-- State the theorem
theorem curve_symmetric_wrt_y_eq_x :
  symmetric_wrt_y_eq_x (λ (p : ℝ × ℝ), curve p.1 p.2) :=
sorry

end curve_symmetric_wrt_y_eq_x_l522_522908


namespace kylie_bracelets_count_l522_522290

theorem kylie_bracelets_count :
  (let num_necklaces_mon := 10 in
   let num_necklaces_tue := 2 in
   let num_earrings_wed := 7 in
   let beads_per_necklace := 20 in
   let beads_per_bracelet := 10 in
   let beads_per_earring := 5 in
   let total_beads := 325 in
   let total_necklaces := num_necklaces_mon + num_necklaces_tue in
   let beads_for_necklaces := total_necklaces * beads_per_necklace in
   let beads_for_earrings := num_earrings_wed * beads_per_earring in
   let beads_for_necklaces_and_earrings := beads_for_necklaces + beads_for_earrings in
   let beads_for_bracelets := total_beads - beads_for_necklaces_and_earrings in
   let num_bracelets := beads_for_bracelets / beads_per_bracelet in
   num_bracelets = 5) :=
sorry

end kylie_bracelets_count_l522_522290


namespace polynomial_factorization_proof_l522_522754

noncomputable def factorizable_binary_quadratic (m : ℚ) : Prop :=
  ∃ (a b : ℚ), (3*a - 5*b = 17) ∧ (a*b = -4) ∧ (m = 2*a + 3*b)

theorem polynomial_factorization_proof :
  ∀ (m : ℚ), factorizable_binary_quadratic m ↔ (m = 5 ∨ m = -58 / 15) :=
by
  sorry

end polynomial_factorization_proof_l522_522754


namespace solution_set_of_inequality_l522_522721

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - 5 * x + 6 ≤ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 3} :=
sorry

end solution_set_of_inequality_l522_522721


namespace smallest_area_of_square_l522_522002

theorem smallest_area_of_square {x_1 x_2 : ℝ} (h1 : x_1 + x_2 = 2) (h2 : x_1 * x_2 = -k) :
  ∃ (k : ℝ), 20 * (k + 1) = 20 * (57 - 4 * real.sqrt 841 + 1) - some_explanation :=
sorry

end smallest_area_of_square_l522_522002


namespace pascals_triangle_ratio_456_l522_522973

theorem pascals_triangle_ratio_456 (n : ℕ) :
  (∃ r : ℕ,
    (n.choose r * 5 = (n.choose (r + 1)) * 4) ∧
    ((n.choose (r + 1)) * 6 = (n.choose (r + 2)) * 5)) →
  n = 98 :=
sorry

end pascals_triangle_ratio_456_l522_522973


namespace percentage_error_equals_l522_522795

noncomputable def correct_fraction_calc : ℚ :=
  let num := (3/4 : ℚ) * 16 - (7/8 : ℚ) * 8
  let denom := (3/10 : ℚ) - (1/8 : ℚ)
  num / denom

noncomputable def incorrect_fraction_calc : ℚ :=
  let num := (3/4 : ℚ) * 16 - (7 / 8 : ℚ) * 8
  num * (3/5 : ℚ)

def percentage_error (correct incorrect : ℚ) : ℚ :=
  abs (correct - incorrect) / correct * 100

theorem percentage_error_equals :
  percentage_error correct_fraction_calc incorrect_fraction_calc = 89.47 :=
by
  sorry

end percentage_error_equals_l522_522795


namespace common_positive_divisors_count_l522_522938

-- To use noncomputable functions
noncomputable theory

open Nat

-- Define the two numbers
def num1 : ℕ := 9240
def num2 : ℕ := 13860

-- Define their greatest common divisor
def gcd_val : ℕ := gcd num1 num2

-- State the prime factorization of the gcd (this can be proven or assumed as a given condition for cleaner code)
def prime_factors_gcd := [(2, 2), (3, 1), (7, 1), (11, 1)]

-- Given the prime factorization, calculate the number of divisors
def number_of_divisors : ℕ := 
  prime_factors_gcd.foldr (λ (factor : ℕ × ℕ) acc, acc * (factor.snd + 1)) 1

-- The final theorem stating the number of common positive divisors of num1 and num2
theorem common_positive_divisors_count : number_of_divisors = 24 := by {
  -- Here would go the proof, which is not required in this task
  sorry
}

end common_positive_divisors_count_l522_522938


namespace divisible_by_g_squared_l522_522340

-- Define the gcd function and the necessary properties
noncomputable def gcd (a b : ℕ) : ℕ := sorry
noncomputable def gcd_three (a b c : ℕ) : ℕ := gcd a (gcd b c)

theorem divisible_by_g_squared (a b c : ℕ) :
  let g := gcd_three a b c in
  (∃ a' b' c' : ℕ, a = g * a' ∧ b = g * b' ∧ c = g * c' ∧ gcd_three a' b' c' = 1) →
  gcd (gcd (b * c) (a * c)) (a * b) ≥ g * g :=
by
  sorry

end divisible_by_g_squared_l522_522340


namespace line_points_sum_slope_and_intercept_l522_522600

-- Definition of the problem
theorem line_points_sum_slope_and_intercept (a b : ℝ) :
  (∀ x y : ℝ, (x = 2 ∧ y = 3) ∨ (x = 10 ∧ y = 19) → y = a * x + b) →
  a + b = 1 :=
by
  intro h
  sorry

end line_points_sum_slope_and_intercept_l522_522600


namespace possible_values_expression_l522_522172

-- Defining the main expression 
def main_expression (a b c d : ℝ) : ℝ :=
  (a / |a|) + (b / |b|) + (c / |c|) + (d / |d|) + (abcd / |abcd|)

-- The theorem that we need to prove
theorem possible_values_expression (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) : 
  main_expression a b c d ∈ {5, 1, -3} :=
sorry

end possible_values_expression_l522_522172


namespace probability_of_same_type_l522_522125

-- Definitions for the given conditions
def total_books : ℕ := 12 + 9
def novels : ℕ := 12
def biographies : ℕ := 9

-- Define the number of ways to pick any two books
def total_ways_to_pick_two_books : ℕ := Nat.choose total_books 2

-- Define the number of ways to pick two novels
def ways_to_pick_two_novels : ℕ := Nat.choose novels 2

-- Define the number of ways to pick two biographies
def ways_to_pick_two_biographies : ℕ := Nat.choose biographies 2

-- Define the number of ways to pick two books of the same type
def ways_to_pick_two_books_of_same_type : ℕ := ways_to_pick_two_novels + ways_to_pick_two_biographies

-- Calculate the probability
noncomputable def probability_same_type (total_ways ways_same_type : ℕ) : ℚ :=
  ways_same_type / total_ways

theorem probability_of_same_type :
  probability_same_type total_ways_to_pick_two_books ways_to_pick_two_books_of_same_type = 17 / 35 := by
  sorry

end probability_of_same_type_l522_522125


namespace leaf_distance_after_11_gusts_l522_522442

def distance_traveled (gusts : ℕ) (swirls : ℕ) (forward_per_gust : ℕ) (backward_per_swirl : ℕ) : ℕ :=
  (gusts * forward_per_gust) - (swirls * backward_per_swirl)

theorem leaf_distance_after_11_gusts :
  ∀ (forward_per_gust backward_per_swirl : ℕ),
  forward_per_gust = 5 →
  backward_per_swirl = 2 →
  distance_traveled 11 11 forward_per_gust backward_per_swirl = 33 :=
by
  intros forward_per_gust backward_per_swirl hfg hbs
  rw [hfg, hbs]
  unfold distance_traveled
  sorry

end leaf_distance_after_11_gusts_l522_522442


namespace smallest_yellow_marbles_l522_522801

def total_marbles (n : ℕ) := n

def blue_marbles (n : ℕ) := n / 3

def red_marbles (n : ℕ) := n / 4

def green_marbles := 6

def yellow_marbles (n : ℕ) := n - (blue_marbles n + red_marbles n + green_marbles)

theorem smallest_yellow_marbles (n : ℕ) (hn : n % 12 = 0) (blue : blue_marbles n = n / 3)
  (red : red_marbles n = n / 4) (green : green_marbles = 6) :
  yellow_marbles n = 4 ↔ n = 24 :=
by sorry

end smallest_yellow_marbles_l522_522801


namespace Hallie_earnings_l522_522578

theorem Hallie_earnings :
  let w := 10
  let hM := 7
  let tM := 18
  let hT := 5
  let tT := 12
  let hW := 7
  let tW := 20
  let mondayEarnings := hM * w + tM
  let tuesdayEarnings := hT * w + tT
  let wednesdayEarnings := hW * w + tW
  let totalEarnings := mondayEarnings + tuesdayEarnings + wednesdayEarnings
  totalEarnings = 240 := by {
    let w := 10
    let hM := 7
    let tM := 18
    let hT := 5
    let tT := 12
    let hW := 7
    let tW := 20
    let mondayEarnings := hM * w + tM
    let tuesdayEarnings := hT * w + tT
    let wednesdayEarnings := hW * w + tW
    let totalEarnings := mondayEarnings + tuesdayEarnings + wednesdayEarnings
    sorry
  }

end Hallie_earnings_l522_522578


namespace determinant_of_transformed_matrix_l522_522635

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, 4], ![3, 2]]

theorem determinant_of_transformed_matrix : Matrix.det (A * A - (3 : ℝ) • A) = 88 := by
  sorry

end determinant_of_transformed_matrix_l522_522635


namespace vector_magnitude_difference_l522_522227

theorem vector_magnitude_difference 
  (a b : ℝ^3) -- a and b are vectors in 3-dimensional space
  (h1 : ∥a∥ = 1) -- norm of vector a
  (h2 : ∥b∥ = 3) -- norm of vector b
  (h3 : real_inner a b = -1.5) -- inner product a·b = |a||b| cos(120°) = 1 * 3 * (-1/2) = -1.5
  : ∥a - b∥ = real.sqrt 13 :=
begin
  sorry

end vector_magnitude_difference_l522_522227


namespace common_divisors_9240_13860_l522_522941

def num_divisors (n : ℕ) : ℕ :=
  -- function to calculate the number of divisors (implementation is not provided here)
  sorry

theorem common_divisors_9240_13860 :
  let d := Nat.gcd 9240 13860
  d = 924 → num_divisors d = 24 := by
  intros d gcd_eq
  rw [gcd_eq]
  sorry

end common_divisors_9240_13860_l522_522941


namespace area_figure_pq_sum_l522_522040

-- Definitions for the problem, including required geometric properties
def side_length : ℝ := 3

-- Define the square area based on its side length
def square_area (s : ℝ) : ℝ := s * s

-- Define the function to calculate the area of an equilateral triangle
def triangle_area (s : ℝ) : ℝ := (s^2 * (Real.sin (2 * Real.pi / 3))) / 2

-- Define the pentagon area based on its composition by triangles
def pentagon_area (s : ℝ) : ℝ := 3 * triangle_area s / 2

-- Define total area of the figure
def total_area (s : ℝ) : ℝ :=
  let square_area_val := square_area s
  let pentagon_area_val := pentagon_area s
  (square_area_val + pentagon_area_val)

-- The problem is to translate the total_area to the form sqrt(p) + sqrt(q) 
-- and find p + q.
def p_q_sum (s : ℝ) (p q : ℝ) : Prop :=
  total_area s = Real.sqrt p + Real.sqrt q ∧ p + q = 108

-- Main theorem statement
theorem area_figure_pq_sum :
  ∃ p q : ℝ, p_q_sum side_length p q :=
sorry

end area_figure_pq_sum_l522_522040


namespace minimum_block_moves_sort_l522_522016

theorem minimum_block_moves_sort (start_seq end_seq : List ℕ) 
  (h_start : start_seq = [5, 4, 3, 2, 1]) 
  (h_end : end_seq = [1, 2, 3, 4, 5]) : 
  ∃ (n : ℕ), 
  n = 3 ∧ 
  (∀ (f : List ℕ → ℕ), 
   (f start_seq = end_seq) ⟶ 
   ∃ (seqs : List (List ℕ)), seqs.length = n ∧ 
     seqs.head = start_seq ∧ 
     seqs.last = end_seq ∧ 
     ∀ (i : ℕ), i < n - 1 → 
       ∃ (k : ℕ) (a b c : List ℕ), 
       k ≤ seqs.nth i.succ.length ∧ 
       a ++ b ++ c = seqs.nth i.succ ∧ 
       seqs.nth i.succ = a ++ c ++ b) :=
by sorry

end minimum_block_moves_sort_l522_522016


namespace value_of_f_neg_a_2016_l522_522188

def f (x : ℝ) : ℝ := if x ≤ 0 then x * (1 - x) else sorry

def a : ℕ → ℝ
| 0 => 1 / 2
| (n + 1) => 1 / (1 - a n)

theorem value_of_f_neg_a_2016 : f (-a 2016) = 2 := by
  have h1 : a 1 = 1 / 2 := rfl
  have h2 : a 2 = 2 := by simp [a, h1]
  have h3 : a 3 = -1 := by simp [a, h2]
  have h_periodicity : ∀ n, a (n + 3) = a n :=
    by intro n; induction n <;> simp [a, *, h1, h2, h3]
  have h2016 : a 2016 = a 3 := by exact_mod_cast (h_periodicity 671)
  have odd_f : ∀ x, f (-x) = -f x :=
    by sorry
  have f_at_neg1 : f (-1) = -(-1) * (1 + 1) := by sorry
  rw [h2016, odd_f, f_at_neg1]
  exact sorry

end value_of_f_neg_a_2016_l522_522188


namespace expected_cards_in_hand_l522_522470

theorem expected_cards_in_hand : 
  let factors : List ℕ := [1, 2, 7, 14, 11, 22, 77, 154, 13, 26, 91, 182, 143, 286, 1001, 2002]
  let perfect_square_condition (cards: List ℕ) : Prop := 
    ∃ subset: List ℕ, subset ≠ [] ∧ (subset.prod)%PerfectSquare
  let p : ℕ → ℚ
      | 0 => 1
      | k+1 => (16 - 2^k)/(16 - k) * (p k)
  (List.sum (List.iota 5 (p) : List ℚ) = 837 / 208) :=
begin
  sorry
end

end expected_cards_in_hand_l522_522470


namespace bisecting_lines_leq_sides_l522_522152

/--
For a convex n-sided polygon with no two sides parallel, the number of lines passing through an internal point O that can bisect the area of the polygon is at most n.
-/
theorem bisecting_lines_leq_sides (n : ℕ) (P : set (ℝ × ℝ))
  (O : ℝ × ℝ) : (convex P) → (∃n_sides : ℕ, n_sides = n ∧ P = polytope n) →
    (∀ (A B : ℝ × ℝ) (hA : A ∈ P) (hB : B ∈ P), A ≠ B → ∃ a b : ℝ, a * (A.1 - B.1) + b * (A.2 - B.2) = 0 → A = B) →
    (∃ k : ℕ, k ≤ n ∧ ∀ (l : ℝ), good_line_through O l → bisects_area P l) :=
begin
  sorry
end

end bisecting_lines_leq_sides_l522_522152


namespace median_of_set_l522_522952

theorem median_of_set {a : ℤ} {c : ℝ} (h1 : a ≠ 0) (h2 : 0 < c) (h3 : ac^2 = log c) (h4 : c < 1) :
  median {0, 1, a, c, 2c} = c :=
sorry

end median_of_set_l522_522952


namespace snooker_table_distance_l522_522433

theorem snooker_table_distance
    (side_length : ℝ)
    (bounces : ℕ)
    (total_distance : ℝ)
    (angle_property : ∀ (cushion : ℝ), True)
    (initial_point : ℝ × ℝ)
    (final_point : ℝ × ℝ) :
    side_length = 2 →
    bounces = 3 →
    total_distance = real.sqrt 52 →
    initial_point = (0, 0) →
    final_point = (6, 4) →
    total_distance = real.sqrt ((final_point.1 - initial_point.1) ^ 2 + (final_point.2 - initial_point.2) ^ 2) :=
by intros; sorry

end snooker_table_distance_l522_522433


namespace parallel_a_eq_2_perpendicular_a_eq_neg10_l522_522542

section
variable (a : ℝ)

def directional_vector : ℝ × ℝ × ℝ := (-1, 2, 5)
def normal_vector : ℝ × ℝ × ℝ := (2, -4, a)

-- Define the dot product
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3 

-- Define condition for parallelism: dot product is zero
def is_parallel (u v : ℝ × ℝ × ℝ) : Prop := dot_product u v = 0

-- Define condition for perpendicularity: vectors are proportional
def is_perpendicular (u v : ℝ × ℝ × ℝ) : Prop := ∃ k : ℝ, u.1 = k * v.1 ∧ u.2 = k * v.2 ∧ u.3 = k * v.3

theorem parallel_a_eq_2 : is_parallel directional_vector normal_vector → a = 2 := by
  sorry

theorem perpendicular_a_eq_neg10 : is_perpendicular directional_vector normal_vector → a = -10 := by
  sorry

end

end parallel_a_eq_2_perpendicular_a_eq_neg10_l522_522542


namespace jake_first_test_score_l522_522626

theorem jake_first_test_score 
  (avg_score : ℕ)
  (n_tests : ℕ)
  (second_test_extra : ℕ)
  (third_test_score : ℕ)
  (x : ℕ) : 
  avg_score = 75 → 
  n_tests = 4 → 
  second_test_extra = 10 → 
  third_test_score = 65 →
  (x + (x + second_test_extra) + third_test_score + third_test_score) / n_tests = avg_score →
  x = 80 := by
  intros h1 h2 h3 h4 h5
  sorry

end jake_first_test_score_l522_522626


namespace triangle_inradius_ratio_l522_522000

theorem triangle_inradius_ratio (AC BC AB : ℕ) (h₁ : AC = 3) (h₂ : BC = 4) (h₃ : AB = 5)
  (r_a r_b : ℝ) (D : Point) (h₄ : is_on_line D AB)
  (h₅ : bisects_angle C D) (h₆ : inradius_triangle ADC = r_a)
  (h₇ : inradius_triangle BCD = r_b) :
  r_a / r_b = (3 / 28) * (10 - real.sqrt 2) :=
sorry

end triangle_inradius_ratio_l522_522000


namespace triangle_area_rational_l522_522457

-- Define the conditions
def satisfies_eq (x y : ℤ) : Prop := x - y = 1

-- Define the points
variables (x1 y1 x2 y2 x3 y3 : ℤ)

-- Assume each point satisfies the equation
axiom point1 : satisfies_eq x1 y1
axiom point2 : satisfies_eq x2 y2
axiom point3 : satisfies_eq x3 y3

-- Statement that we need to prove
theorem triangle_area_rational :
  ∃ (area : ℚ), 
    ∃ (triangle_points : ∃ (x1 y1 x2 y2 x3 y3 : ℤ), satisfies_eq x1 y1 ∧ satisfies_eq x2 y2 ∧ satisfies_eq x3 y3), 
      true :=
sorry

end triangle_area_rational_l522_522457


namespace arithmetic_sequence_sum_l522_522263

variable {a : ℕ → ℝ} -- The arithmetic sequence {a_n} represented by a function a : ℕ → ℝ

/-- Given that the sum of some terms of an arithmetic sequence is 25, prove the sum of other terms -/
theorem arithmetic_sequence_sum (h : a 3 + a 4 + a 5 + a 6 + a 7 = 25) : a 2 + a 8 = 10 := by
    sorry

end arithmetic_sequence_sum_l522_522263


namespace product_of_primes_is_even_l522_522642

-- Define the conditions for P and Q to cover P, Q, P-Q, and P+Q being prime and positive
def is_prime (n : ℕ) : Prop := ¬ (n = 0 ∨ n = 1) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem product_of_primes_is_even {P Q : ℕ} (hP : is_prime P) (hQ : is_prime Q) 
  (hPQ_diff : is_prime (P - Q)) (hPQ_sum : is_prime (P + Q)) 
  (hPosP : P > 0) (hPosQ : Q > 0) 
  (hPosPQ_diff : P - Q > 0) (hPosPQ_sum : P + Q > 0) : 
  ∃ k : ℕ, P * Q * (P - Q) * (P + Q) = 2 * k := 
sorry

end product_of_primes_is_even_l522_522642


namespace common_positive_divisors_count_l522_522939

-- To use noncomputable functions
noncomputable theory

open Nat

-- Define the two numbers
def num1 : ℕ := 9240
def num2 : ℕ := 13860

-- Define their greatest common divisor
def gcd_val : ℕ := gcd num1 num2

-- State the prime factorization of the gcd (this can be proven or assumed as a given condition for cleaner code)
def prime_factors_gcd := [(2, 2), (3, 1), (7, 1), (11, 1)]

-- Given the prime factorization, calculate the number of divisors
def number_of_divisors : ℕ := 
  prime_factors_gcd.foldr (λ (factor : ℕ × ℕ) acc, acc * (factor.snd + 1)) 1

-- The final theorem stating the number of common positive divisors of num1 and num2
theorem common_positive_divisors_count : number_of_divisors = 24 := by {
  -- Here would go the proof, which is not required in this task
  sorry
}

end common_positive_divisors_count_l522_522939


namespace domain_f_l522_522148

def f (x : ℝ) : ℝ := (x + 2)^0 / (x + 1)

theorem domain_f :
  {x : ℝ | x ≠ -1 ∧ x ≠ -2} = {x : ℝ | (x + 2) ≠ 0 ∧ (x + 1) ≠ 0} :=
by
  sorry

end domain_f_l522_522148


namespace find_true_propositions_l522_522385

/-- Mathematical propositions about spatial vectors -/
def true_propositions_sequence : List Nat :=
  [2]

-- Proposition 1
def prop1 (u v : ℝ^3) : Prop :=
  (u ≠ 0 ∧ v ≠ 0 ∧ ¬ (u = k * v) for some k:ℝ) → ¬ (u ∥ v)

-- Proposition 2
def prop2 (u v : ℝ^3) : Prop :=
  (∥u∥ = ∥v∥ ∧ direction u = direction v) → u = v

-- Proposition 3
def prop3 (u v : ℝ^3) : Prop :=
  (u ∥ v ∧ ∥u∥ = ∥v∥) → u = v

-- Proposition 4
def prop4 (a b : ℝ^3) : Prop :=
  a ≠ b → ∥a∥ ≠ ∥b∥

/-- Proof problem for sequence numbers of all true propositions -/
theorem find_true_propositions :
  { n : ℕ // n ∈ true_propositions_sequence }
  :=
  sorry

end find_true_propositions_l522_522385


namespace integral_evaluation_l522_522815

noncomputable def integral_expression : ℝ :=
  ∫ x in -1..1, 2 * sqrt (1 - x^2) - sin x

theorem integral_evaluation : integral_expression = π :=
sorry

end integral_evaluation_l522_522815


namespace solve_for_x_l522_522895

namespace proof_problem

-- Define the operation a * b = 4 * a * b
def star (a b : ℝ) : ℝ := 4 * a * b

-- Given condition rewritten in terms of the operation star
def equation (x : ℝ) : Prop := star x x + star 2 x - star 2 4 = 0

-- The statement we intend to prove
theorem solve_for_x (x : ℝ) : equation x → (x = 2 ∨ x = -4) :=
by
  -- Proof omitted
  sorry

end proof_problem

end solve_for_x_l522_522895


namespace one_third_of_product_l522_522847

theorem one_third_of_product (a b c : ℕ) (h1 : a = 7) (h2 : b = 9) (h3 : c = 4) : (1 / 3 : ℚ) * (a * b * c : ℕ) = 84 := by
  sorry

end one_third_of_product_l522_522847


namespace arithmetic_sequence_sum_13_l522_522264

noncomputable def arithmetic_sum (n a d : ℕ) : ℕ :=
n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_13
    (a d : ℕ)
    (h : ∃ (a b c : ℕ), a_3 + a_5 + 2 * a_10 = 8)
    (arithmetic_seq : ∀ n : ℕ, a_n = a + (n - 1) * d):
  arithmetic_sum 13 a d = 26 :=
begin
  sorry
end

end arithmetic_sequence_sum_13_l522_522264


namespace tan_cos_sin_fraction_l522_522160

theorem tan_cos_sin_fraction (α : ℝ) (h : Real.tan α = -3) : 
  (Real.cos α + 2 * Real.sin α) / (Real.cos α - 3 * Real.sin α) = -1 / 2 := 
by
  sorry

end tan_cos_sin_fraction_l522_522160


namespace find_angle_C_l522_522275

namespace TriangleProblem

open Real

noncomputable theory

def is_valid_triangle (A B C a b c : ℝ) : Prop :=
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧ a > 0 ∧ b > 0 ∧ c > 0

def law_of_sines (A B C a b c : ℝ) : Prop :=
  a / sin A = b / sin B ∧ b / sin B = c / sin C

theorem find_angle_C
  (A B C a b c : ℝ)
  (h_triangle : is_valid_triangle A B C a b c)
  (h1 : A - C = π / 2)
  (h2 : a + c = sqrt 2 * b)
  (h_sines : law_of_sines A B C a b c) :
  C = π / 12 :=
begin
  sorry
end

end TriangleProblem

end find_angle_C_l522_522275


namespace four_digit_even_numbers_count_l522_522803

theorem four_digit_even_numbers_count : 
  let digits := {0, 1, 2, 3, 4, 5}
  in
    ∀ (n : ℕ), 
      (1000 ≤ n ∧ n < 10000 ∧ 
       (n % 10 = 0 ∨ n % 10 = 2 ∨ n % 10 = 4) ∧ 
       (∀ d ∈ digits, ∃! k, (n / 10^k) % 10 = d))
      ↔ False ∨ n = 156 := 
proof_helper sorry

end four_digit_even_numbers_count_l522_522803


namespace right_angle_case_acute_angle_case_obtuse_angle_case_l522_522426

-- Definitions
def circumcenter (O : Type) (A B C : Type) : Prop := sorry -- Definition of circumcenter.

def orthocenter (H : Type) (A B C : Type) : Prop := sorry -- Definition of orthocenter.

noncomputable def R : ℝ := sorry -- Circumradius of the triangle.

-- Conditions
variables {A B C O H : Type}
  (h_circumcenter : circumcenter O A B C)
  (h_orthocenter : orthocenter H A B C)

-- The angles α β γ represent the angles of triangle ABC.
variables {α β γ : ℝ}

-- Statements
-- Case 1: ∠C = 90°
theorem right_angle_case (h_angle_C : γ = 90) (h_H_eq_C : H = C) (h_AB_eq_2R : AB = 2 * R) : AH + BH >= AB := by
  sorry

-- Case 2: ∠C < 90°
theorem acute_angle_case (h_angle_C_lt_90 : γ < 90) : O_in_triangle_AHB := by
  sorry

-- Case 3: ∠C > 90°
theorem obtuse_angle_case (h_angle_C_gt_90 : γ > 90) : AH + BH > 2 * R := by
  sorry

end right_angle_case_acute_angle_case_obtuse_angle_case_l522_522426


namespace part_I_part_II_l522_522915

theorem part_I : 
  (∀ x : ℝ, |x - (2 : ℝ)| ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) :=
  sorry

theorem part_II :
  (∀ a b c : ℝ, a - 2 * b + c = 2 → a^2 + b^2 + c^2 ≥ 2 / 3) :=
  sorry

end part_I_part_II_l522_522915


namespace max_min_values_a_max_min_values_b_l522_522914

def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem max_min_values_a (x : ℝ) (h : x ∈ set.Icc (-2:ℝ) 0) :
  (∀ y, y ∈ set.Icc (-2:ℝ) 0 → f(y) ≤ f(-2)) ∧
  (∀ y, y ∈ set.Icc (-2:ℝ) 0 → f(y) ≥ f(0)) :=
by
  sorry

theorem max_min_values_b (x : ℝ) (h : x ∈ set.Icc (2:ℝ) 3) :
  (∀ y, y ∈ set.Icc (2:ℝ) 3 → f(y) ≤ f(3)) ∧
  (∀ y, y ∈ set.Icc (2:ℝ) 3 → f(y) ≥ f(2)) :=
by
  sorry

end max_min_values_a_max_min_values_b_l522_522914


namespace george_total_cost_l522_522146

noncomputable def movie_ticket_cost := 16

noncomputable def nachos_per_serving_cost := movie_ticket_cost / 2
noncomputable def total_nachos_cost := 2 * nachos_per_serving_cost

noncomputable def popcorn_per_bag_cost := nachos_per_serving_cost * 0.75
noncomputable def total_popcorn_cost := 3 * popcorn_per_bag_cost

noncomputable def soda_per_unit_cost := popcorn_per_bag_cost * 0.75
noncomputable def total_soda_cost := 4 * soda_per_unit_cost

noncomputable def combo_deal_cost := 7

noncomputable def total_food_bill_before_discount := total_nachos_cost + total_popcorn_cost + total_soda_cost

noncomputable def food_discount := total_food_bill_before_discount * 0.1
noncomputable def total_food_bill_after_discount := total_food_bill_before_discount - food_discount

noncomputable def total_cost_before_tax := movie_ticket_cost + total_food_bill_after_discount + combo_deal_cost

noncomputable def sales_tax := total_cost_before_tax * 0.05
noncomputable def total_cost_including_tax := total_cost_before_tax + sales_tax

theorem george_total_cost :
  total_cost_including_tax = 73.29 :=
by
  sorry

end george_total_cost_l522_522146


namespace speed_ratio_is_2_5_l522_522241

noncomputable def speed_ratio_proof (t_o : ℝ) (x : ℝ) (v_e v_o : ℝ) : ℝ :=
  if h : x > 2.5 then 2.5 else 0

theorem speed_ratio_is_2_5 (t_o : ℝ) (x : ℝ) (v_e v_o : ℝ)
  (h_to : t_o = 10)
  (h_te : ∀ x, x > 2.5 → t_e = 7 - x)
  (h_dist_eq : ∀ x, x > 2.5 → v_e * 2 = v_o * (x + 2))
  (h_speed_rel : ∀ x, x > 2.5 → v_e * (7 - x) = v_o * 10) :
  speed_ratio_proof t_o x v_e v_o = 2.5 := by
  sorry

end speed_ratio_is_2_5_l522_522241


namespace side_b_value_area_of_triangle_l522_522276

-- Define the given conditions of the triangle
variable {A B C : ℝ}
variable {a b c : ℝ}
variable (triangle_ABC : Type) [Triangle triangle_ABC]

-- Conditions provided
hypothesis (h1 : a = 3)
hypothesis (h2 : cos A = sqrt 6 / 3)
hypothesis (h3 : B = A + (π / 2))

-- Prove: the length of side b
theorem side_b_value : b = 3 * sqrt 2 := sorry

-- Prove: the area of the triangle
theorem area_of_triangle : area triangle_ABC = (3 * sqrt 2) / 2 := sorry

end side_b_value_area_of_triangle_l522_522276


namespace main_theorem_l522_522109

noncomputable def problem_statement : Prop :=
  let a := cos^2 (π / 9)
  let b := cos^2 (2 * π / 9)
  let c := cos^2 (4 * π / 9)
  cubic_polynomial : (x - a) * (x - b) * (x - c) = x^3 - (9 / 2) * x^2 + (27 / 16) * x - (1 / 16)
  ∧ sqrt((3 - a) * (3 - b) * (3 - c)) = (5 * sqrt(5)) / 4

theorem main_theorem : problem_statement :=
  by
    sorry

end main_theorem_l522_522109


namespace binom_20_6_l522_522088

theorem binom_20_6 : nat.choose 20 6 = 19380 := 
by 
  sorry

end binom_20_6_l522_522088


namespace ratio_of_X_to_Y_l522_522978

theorem ratio_of_X_to_Y (total_respondents : ℕ) (preferred_X : ℕ)
    (h_total : total_respondents = 250)
    (h_X : preferred_X = 200) :
    preferred_X / (total_respondents - preferred_X) = 4 := by
  sorry

end ratio_of_X_to_Y_l522_522978


namespace binom_coeff_equality_sum_of_coeffs_l522_522551

-- First proof: Proving n = 10 when C_n^3 = C_n^7
theorem binom_coeff_equality (n : ℕ) (h : nat.choose n 3 = nat.choose n 7) : n = 10 :=
sorry

-- Second proof: Proving the sum of coefficients for n = 10
theorem sum_of_coeffs (h : 10 = 10) : (-1)^3 * nat.choose 10 3 + (-1)^7 * nat.choose 10 7 = -240 :=
sorry

end binom_coeff_equality_sum_of_coeffs_l522_522551


namespace inconsistency_proof_l522_522614

-- Let TotalBoys be the number of boys, which is 120
def TotalBoys := 120

-- Let AverageMarks be the average marks obtained by 120 boys, which is 40
def AverageMarks := 40

-- Let PassedBoys be the number of boys who passed, which is 125
def PassedBoys := 125

-- Let AverageMarksFailed be the average marks of failed boys, which is 15
def AverageMarksFailed := 15

-- We need to prove the inconsistency
theorem inconsistency_proof :
  ∀ (P : ℝ), 
    (TotalBoys * AverageMarks = PassedBoys * P + (TotalBoys - PassedBoys) * AverageMarksFailed) →
    False :=
by
  intro P h
  sorry

end inconsistency_proof_l522_522614


namespace pyramid_volume_l522_522772

-- Given conditions
def lateral_edge_angle := 45
def diagonal_section_area (S : ℝ) := S

-- Volume calculation theorem
theorem pyramid_volume (S : ℝ) (hS : S ≥ 0) :
  ∃ V : ℝ, V = (2 / 3) * S * real.sqrt S :=
by
  sorry

end pyramid_volume_l522_522772


namespace mary_initial_flour_l522_522325

theorem mary_initial_flour (F_total F_add F_initial : ℕ) 
  (h_total : F_total = 9)
  (h_add : F_add = 6)
  (h_initial : F_initial = F_total - F_add) :
  F_initial = 3 :=
sorry

end mary_initial_flour_l522_522325


namespace difference_abc_cba_l522_522546

theorem difference_abc_cba {a b c : ℕ} (h : a = c + 2) :
  let num1 := 100 * a + 10 * b + c,
      num2 := 100 * c + 10 * b + a
  in num1 - num2 = 198 :=
by
  sorry

end difference_abc_cba_l522_522546


namespace exist_m_n_for_epsilon_l522_522292

theorem exist_m_n_for_epsilon 
  (h k : ℕ) (h_pos : 0 < h) (k_pos : 0 < k) (ε : ℝ) (ε_pos : 0 < ε) : 
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ ε < |h * real.sqrt m - k * real.sqrt n| ∧ |h * real.sqrt m - k * real.sqrt n| < 2 * ε :=
by 
  sorry

end exist_m_n_for_epsilon_l522_522292


namespace correct_calculation_l522_522759

theorem correct_calculation (a b : ℝ) : (-3 * a^3 * b)^2 = 9 * a^6 * b^2 :=
by sorry

end correct_calculation_l522_522759


namespace intersection_A_B_l522_522927

def A : Set ℝ := { x : ℝ | -1 < x ∧ x < 3 }
def B : Set ℝ := { x : ℝ | x < 2 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  sorry

end intersection_A_B_l522_522927


namespace no_solution_natural_gt_one_l522_522835

theorem no_solution_natural_gt_one (x : ℕ) (h : 1 < x) : 
  (∑ k in (range (x^2 + 1)), (1 / (x + k : ℝ)) ≠ 1) :=
sorry

end no_solution_natural_gt_one_l522_522835


namespace range_of_a_l522_522571

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = Real.sqrt x) :
  (f a < f (a + 1)) ↔ a ∈ Set.Ici (-1) :=
by
  sorry

end range_of_a_l522_522571


namespace ab_cd_not_prime_l522_522648

theorem ab_cd_not_prime {a b c d : ℕ} (a_gt_b : a > b) (b_gt_c : b > c) (c_gt_d : c > d)
  (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d) 
  (H : a * c + b * d = (b + d + a - c) * (b + d - a + c)) : 
  ¬Nat.prime (a * b + c * d) := 
sorry

end ab_cd_not_prime_l522_522648


namespace mode_of_scores_is_97_l522_522350

open List

def scores : List ℕ := [
  65, 65,
  71, 73, 73, 76,
  80, 80, 84, 84, 88, 88, 88,
  92, 92, 95, 97, 97, 97, 97,
  101, 101, 101, 104, 106,
  110, 110, 110
]

theorem mode_of_scores_is_97 : ∃ m, m = 97 ∧ (∀ x, count x scores ≤ count 97 scores) :=
by
  sorry

end mode_of_scores_is_97_l522_522350


namespace jake_spent_half_day_on_monday_l522_522627

def jake_watching_show (x : ℝ) : Prop :=
  let monday := 24 * x in
  let tuesday := 4 in
  let wednesday := 6 in
  let thursday := (1/2) * (monday + tuesday + wednesday) in
  let friday := 19 in
  let total := monday + tuesday + wednesday + thursday + friday in
  total = 52

theorem jake_spent_half_day_on_monday (x: ℝ) (h: jake_watching_show x) : x = 1/2 :=
begin
  sorry
end

end jake_spent_half_day_on_monday_l522_522627


namespace common_divisors_count_l522_522945

-- Define the numbers
def num1 := 9240
def num2 := 13860

-- Define the gcd of the numbers
def gcdNum := Nat.gcd num1 num2

-- Prove the number of divisors of the gcd is 48
theorem common_divisors_count : (Nat.divisors gcdNum).card = 48 :=
by
  -- Normally we would provide a detailed proof here
  sorry

end common_divisors_count_l522_522945


namespace find_region_area_l522_522850

def fractional_part (x : ℝ) : ℝ := x - x.floor

noncomputable def region_area : ℝ :=
  let condition (x y : ℝ) : Prop := 
    x ≥ 0 ∧ y ≥ 0 ∧ 100 * fractional_part x ≥ x.floor + y.floor
  in 1717

theorem find_region_area (x y : ℝ) (h : x ≥ 0 ∧ y ≥ 0 ∧ 100 * fractional_part x ≥ x.floor + y.floor) : 
  region_area = 1717 :=
sorry

end find_region_area_l522_522850


namespace max_values_of_vasya_l522_522664

theorem max_values_of_vasya (P : ℕ → ℕ → ℕ) (k : ℕ) (n : ℕ) (a b : ℕ) 
  (hP : ∀ x : ℕ, ∃ P_x : ℕ → ℕ, is_quadratic_trinomial P_x) :
  ∃ m ≤ 20, ∀ i ≤ m, ∃ j, P j (k + i) = a * i + b := by sorry

end max_values_of_vasya_l522_522664


namespace linear_equation_validity_l522_522009

def is_linear_eq_one_var (eq : String) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ eq = (a.toString ++ "*x + " ++ b.toString ++ " = 0")

theorem linear_equation_validity :
  is_linear_eq_one_var "x = 3" :=
begin
  use [1, -3],
  split,
  { norm_num },
  { refl }
end

end linear_equation_validity_l522_522009


namespace possible_values_expression_l522_522170

theorem possible_values_expression (a b c d : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) :
  ∃ (x : ℝ), x ∈ {5, 1, -3} ∧ x = (a / |a| + b / |b| + c / |c| + d / |d| + (abcd / |abcd|)) :=
by
  sorry

end possible_values_expression_l522_522170


namespace range_of_m_l522_522223

noncomputable def p (m : ℝ) : Prop :=
  (m > 2)

noncomputable def q (m : ℝ) : Prop :=
  (m > 1)

theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) → (1 < m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l522_522223


namespace new_determinant_l522_522491

-- Given the condition that the determinant of the original matrix is 12
def original_determinant (x y z w : ℝ) : Prop :=
  x * w - y * z = 12

-- Proof that the determinant of the new matrix equals the expected result
theorem new_determinant (x y z w : ℝ) (h : original_determinant x y z w) :
  (2 * x + z) * w - (2 * y - w) * z = 24 + z * w + w * z := by
  sorry

end new_determinant_l522_522491


namespace probability_adjacent_Alice_Bob_l522_522980

-- Defining the conditions
def actors := {"Alice", "Bob", "Actor3", "Actor4", "Actor5", "Actor6", "Actor7"}
def grid_rows := 3
def grid_cols := 3
def total_seats := grid_rows * grid_cols
def empty_seat := 1
def seated_actors := (total_seats - empty_seat)

-- Question and proof goal
theorem probability_adjacent_Alice_Bob :
  let total_arrangements := (total_seats.choose seated_actors) * seated_actors.factorial in
  let adjacencies := 16 in  -- total adjacency pairs based on solution steps
  let favorable_positions := 2 * adjacencies * (6 - 2).factorial in
  favorable_positions / total_arrangements = 1 / 79 := 
sorry

end probability_adjacent_Alice_Bob_l522_522980


namespace asymptote_equation_l522_522207

theorem asymptote_equation {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  (a + Real.sqrt (a^2 + b^2) = 2 * b) →
  (4 * x = 3 * y) ∨ (4 * x = -3 * y) :=
by
  sorry

end asymptote_equation_l522_522207


namespace jerry_wants_to_raise_average_l522_522630

theorem jerry_wants_to_raise_average :
  let average_first_3_tests := 85
  let total_first_3_tests := 3 * average_first_3_tests
  let score_fourth_test := 93
  let total_after_fourth_test := total_first_3_tests + score_fourth_test
  let new_average := (total_after_fourth_test : ℤ) / 4
  new_average - average_first_3_tests = 2 :=
by
  let average_first_3_tests := 85
  let total_first_3_tests := 3 * average_first_3_tests
  let score_fourth_test := 93
  let total_after_fourth_test := total_first_3_tests + score_fourth_test
  let new_average := (total_after_fourth_test : ℤ) / 4
  have h : new_average - average_first_3_tests = 2 := by
    sorry
  exact h

end jerry_wants_to_raise_average_l522_522630


namespace quadrilateral_simson_euler_l522_522255

noncomputable def is_cyclic_quadrilateral {K : Type*} [field K] (A B C D : K) : Prop :=
  |A| = |B| ∧ |C| = |D| ∧ |A| = |C| ∧ |B| = |D|

noncomputable def simson_perpendicular_euler {K : Type*} [field K] [is_cyclic : is_cyclic_quadrilateral A B C D] : Prop :=
  (ab + ac + ad + bc + bd + cd) = 0

theorem quadrilateral_simson_euler {K : Type*} [field K] (A B C D : K)
  (h_cyclic : is_cyclic_quadrilateral A B C D)
  (h_perpendicular : simson_perpendicular_euler A B C D):
  simson_perpendicular_euler B A C D :=
sorry

end quadrilateral_simson_euler_l522_522255


namespace find_x_for_ffx_equal_fx_l522_522310

def f(x : ℝ) : ℝ := x^3 - 4 * x

theorem find_x_for_ffx_equal_fx :
  ∀ x : ℝ, f(f(x)) = f(x) → x = 0 ∨ x = 2 ∨ x = -2 :=
by
  -- proof will go here
  sorry

end find_x_for_ffx_equal_fx_l522_522310


namespace range_of_b_min_value_a_add_b_min_value_ab_l522_522533

theorem range_of_b (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : b > 1 := sorry

theorem min_value_a_add_b (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : a + b ≥ 8 := sorry

theorem min_value_ab (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : a * b ≥ 16 := sorry

end range_of_b_min_value_a_add_b_min_value_ab_l522_522533


namespace average_total_matches_l522_522769

-- Define the conditions
namespace AverageRuns

variable (matches1 matches2 : ℕ) (avg1 avg2 : ℕ)

def totalRuns (matches avg : ℕ) : ℕ := matches * avg

theorem average_total_matches
  (h1 : matches1 = 10)
  (h2 : matches2 = 10)
  (h3 : avg1 = 40)
  (h4 : avg2 = 30) :
  (totalRuns matches1 avg1 + totalRuns matches2 avg2) / (matches1 + matches2) = 35 := by
  sorry

end AverageRuns

end average_total_matches_l522_522769


namespace grover_total_profit_l522_522230

-- Definitions based on conditions
def original_price : ℝ := 10
def discount_first_box : ℝ := 0.20
def discount_second_box : ℝ := 0.30
def discount_third_box : ℝ := 0.40
def packs_first_box : ℕ := 20
def packs_second_box : ℕ := 30
def packs_third_box : ℕ := 40
def masks_per_pack : ℕ := 5
def price_per_mask_first_box : ℝ := 0.75
def price_per_mask_second_box : ℝ := 0.85
def price_per_mask_third_box : ℝ := 0.95

-- Computations
def cost_first_box := original_price - (discount_first_box * original_price)
def cost_second_box := original_price - (discount_second_box * original_price)
def cost_third_box := original_price - (discount_third_box * original_price)

def total_cost := cost_first_box + cost_second_box + cost_third_box

def revenue_first_box := packs_first_box * masks_per_pack * price_per_mask_first_box
def revenue_second_box := packs_second_box * masks_per_pack * price_per_mask_second_box
def revenue_third_box := packs_third_box * masks_per_pack * price_per_mask_third_box

def total_revenue := revenue_first_box + revenue_second_box + revenue_third_box

def total_profit := total_revenue - total_cost

-- Proof statement
theorem grover_total_profit : total_profit = 371.5 := by
  sorry

end grover_total_profit_l522_522230


namespace evaluate_f_difference_l522_522959

def f (x : ℝ) : ℝ := x^6 - 2 * x^4 + 7 * x

theorem evaluate_f_difference :
  f 3 - f (-3) = 42 := by
  sorry

end evaluate_f_difference_l522_522959


namespace find_four_digit_number_l522_522763

theorem find_four_digit_number (A B C : ℕ) : 
  let x := 1000 * A + 100 * B + 10 * C + 9 in
  let y := 1000 * A + 100 * B + 10 * C + 6 in
  y + 57 = 1823 → x = 1769 := 
by {
  intros A B C h,
  unfold x y,
  have h1 : 1000 * A + 100 * B + 10 * C + 6 + 57 = 1823 := h,
  have h2 : 1000 * A + 100 * B + 10 * C + 63 = 1823 := by rw [← h1],
  have h3 : 1000 * A + 100 * B + 10 * C = 1760 := by linarith,
  have h4 : 1000 * A + 100 * B + 10 * C + 9 = 1769 := by linarith,
  exact h4,
}

end find_four_digit_number_l522_522763


namespace law_firm_more_than_two_years_l522_522611

theorem law_firm_more_than_two_years (p_second p_not_first : ℝ) : 
  p_second = 0.30 →
  p_not_first = 0.60 →
  ∃ p_more_than_two_years : ℝ, p_more_than_two_years = 0.30 :=
by
  intros h1 h2
  use (p_not_first - p_second)
  rw [h1, h2]
  norm_num
  done

end law_firm_more_than_two_years_l522_522611


namespace height_difference_l522_522956

theorem height_difference (B_height A_height : ℝ) (h : A_height = 0.6 * B_height) :
  (B_height - A_height) / A_height * 100 = 66.67 := 
sorry

end height_difference_l522_522956


namespace solution_is_correct_l522_522684

-- Define the conditions of the problem.
variable (x y z : ℝ)

-- The system of equations given in the problem
def system_of_equations (x y z : ℝ) :=
  (1/x + 1/(y+z) = 6/5) ∧
  (1/y + 1/(x+z) = 3/4) ∧
  (1/z + 1/(x+y) = 2/3)

-- The desired solution
def solution (x y z : ℝ) := x = 2 ∧ y = 3 ∧ z = 1

-- The theorem to prove
theorem solution_is_correct (h : system_of_equations x y z) : solution x y z :=
sorry

end solution_is_correct_l522_522684


namespace relationships_with_correlation_l522_522371

def has_correlation (relationship : string) : Prop := sorry

def relationship_1 : string := "Great teachers produce great students"
def relationship_2 : string := "The relationship between the volume of a sphere and its radius"
def relationship_3 : string := "The relationship between apple production and climate"
def relationship_4 : string := "Crows cawing is a bad omen"
def relationship_5 : string := "The relationship between the diameter of the cross-section and the height of the same type of tree"
def relationship_6 : string := "The relationship between a student and their student ID number"

theorem relationships_with_correlation :
  has_correlation relationship_1 ∧
  has_correlation relationship_3 ∧
  has_correlation relationship_5 :=
sorry

end relationships_with_correlation_l522_522371


namespace sqrt_product_cos_squared_l522_522107

theorem sqrt_product_cos_squared :
  let x1 := cos (π / 9) ^ 2
  let x2 := cos (2 * π / 9) ^ 2
  let x3 := cos (4 * π / 9) ^ 2
  512 * x1 ^ 9 - 2304 * x1 ^ 7 + 3360 * x1 ^ 5 - 1680 * x1 ^ 3 + 315 * x1 - 9 = 0
  ∧ 512 * x2 ^ 9 - 2304 * x2 ^ 7 + 3360 * x2 ^ 5 - 1680 * x2 ^ 3 + 315 * x2 - 9 = 0
  ∧ 512 * x3 ^ 9 - 2304 * x3 ^ 7 + 3360 * x3 ^ 5 - 1680 * x3 ^ 3 + 315 * x3 - 9 = 0
  → sqrt ((3 - x1) * (3 - x2) * (3 - x3)) = 9 * sqrt 3 / 8 :=
by
  sorry

end sqrt_product_cos_squared_l522_522107


namespace xy_max_value_l522_522873

theorem xy_max_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 4 * y = 12) :
  xy <= 9 := by
  sorry

end xy_max_value_l522_522873


namespace correct_relation_l522_522218

def A : Set ℝ := { x | x > 1 }

theorem correct_relation : 2 ∈ A := by
  -- Proof would go here
  sorry

end correct_relation_l522_522218


namespace remainder_of_division_l522_522392

theorem remainder_of_division (x y R : ℕ) 
  (h1 : y = 1782)
  (h2 : y - x = 1500)
  (h3 : y = 6 * x + R) :
  R = 90 :=
by
  sorry

end remainder_of_division_l522_522392


namespace consecutive_vertices_in_heptagon_l522_522438

theorem consecutive_vertices_in_heptagon 
    (N : ℕ)
    (hN : N > 2)
    (polygon : polygon N)
    (heptagons : polygon_subdivided_into_heptagons polygon) : 
  ∃ (v1 v2 v3 v4 : ℕ), consecutive_vertices polygon heptagons v1 v2 v3 v4 :=
sorry

end consecutive_vertices_in_heptagon_l522_522438


namespace possible_values_of_expression_l522_522161

theorem possible_values_of_expression (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (v : ℤ), v ∈ ({5, 1, -3, -5} : Set ℤ) ∧ v = (Int.sign a + Int.sign b + Int.sign c + Int.sign d + Int.sign (a * b * c * d)) :=
by
  sorry

end possible_values_of_expression_l522_522161


namespace find_a6_l522_522725

-- Defining the conditions of the problem
def a1 := 2
def S3 := 12

-- Defining the necessary arithmetic sequence properties
def Sn (a1 d : ℕ) (n : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2
def an (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

-- Proof statement in Lean
theorem find_a6 (d : ℕ) (a1_val S3_val : ℕ) (h1 : a1_val = 2) (h2 : S3_val = 12) 
    (h3 : 3 * (2 * a1_val + (3 - 1) * d) / 2 = S3_val) : an a1_val d 6 = 12 :=
by 
  -- omitted proof
  sorry

end find_a6_l522_522725


namespace cosine_sum_equiv_l522_522701

theorem cosine_sum_equiv :
  (∃ a b c d : ℕ, 
    a = 4 ∧ b = 6 ∧ c = 3 ∧ d = 1 ∧ 
    ∀ x : ℝ, cos (2 * x) + cos (4 * x) + cos (8 * x) + cos (10 * x) = a * (cos (b * x) * cos (c * x) * cos (d * x))) → 4 + 6 + 3 + 1 = 14 :=
by
  intro h
  cases h with a h1
  cases h1 with b h2
  cases h2 with c h3
  cases h3 with d h4
  cases h4 with ha h5
  cases h5 with hb h6
  cases h6 with hc hd
  exact rfl

end cosine_sum_equiv_l522_522701


namespace omega_range_min_diff_zeros_l522_522200

noncomputable def omega : ℝ := sorry

def f (ω x : ℝ) : ℝ := 2 * sin (ω * x)

-- Problem 1
theorem omega_range (ω : ℝ) (h_pos : ω > 0)
  (h_mono : ∀ x y, x ∈ Icc (-π / 4) (2 * π / 3) → y ∈ Icc (-π / 4) (2 * π / 3) → x < y → f ω x < f ω y) :
  0 < ω ∧ ω ≤ 3 / 4 :=
sorry

def g (x : ℝ) : ℝ := 2 * sin (2 * (x + π / 6)) + 1

-- Problem 2
theorem min_diff_zeros (a b : ℝ) : 
  (∀ x ∈ Icc a b, g x = 0) → b - a = 43 * π / 3 :=
sorry

end omega_range_min_diff_zeros_l522_522200


namespace base_with_final_digit_one_l522_522141

theorem base_with_final_digit_one (n : ℕ) (h : n = 576) :
  (∃! b : ℕ, 2 ≤ b ∧ b ≤ 9 ∧ (n - 1) % b = 0) :=
by
  have h1 : n - 1 = 575 := sorry -- This handles the calculation 576 - 1 = 575
  have h2 : ∃! b : ℕ, 2 ≤ b ∧ b ≤ 9 ∧ 575 % b = 0 := sorry -- Handle the unique base condition
  exact h2

end base_with_final_digit_one_l522_522141


namespace range_of_m_for_z_in_second_quadrant_l522_522246

theorem range_of_m_for_z_in_second_quadrant (m : ℝ) :
  let z := (m - 2) + (m + 1) * Complex.I in
  (z.re < 0 ∧ z.im > 0) ↔ (-1 < m ∧ m < 2) :=
by
  let z := (m - 2) + (m + 1) * Complex.I
  sorry

end range_of_m_for_z_in_second_quadrant_l522_522246


namespace calculation_result_l522_522028

theorem calculation_result : (1000 * 7 / 10 * 17 * 5^2 = 297500) :=
by sorry

end calculation_result_l522_522028


namespace similar_triangles_l522_522224

theorem similar_triangles
  (x0 x1 x2 : ℝ)
  (hA : x1 ≠ x0)
  (hB : x2 ≠ x0)
  (C : ℝ × ℝ := (0, x0 * x1))
  (D : ℝ × ℝ := (0, x0 * x2)):
  ∆(0, 0) (x1, 0) ≅ ∆(0, 0) (x2, 0) :=
begin
  -- Definitions of vertices of the triangles
  let A := (x1, 0),
  let B := (x2, 0),

  -- Definitions of the similarity conditions
  have hOC_OA : C.2 / A.1 = x0,
    from (mul_div_cancel' x0 (ne_of_lt (by linarith))).symm,
  have hOD_OB : D.2 / B.1 = x0,
    from (mul_div_cancel' x0 (ne_of_lt (by linarith))).symm,

  -- Prove that triangles are similar by side ratios
  -- Refer to angle similarity criterion (AA)
  sorry
end

end similar_triangles_l522_522224


namespace find_MO_l522_522257

variables (K L M N O : Type)
variables [linear_ordered_field K] [linear_ordered_field L] [linear_ordered_field M] [linear_ordered_field N] [linear_ordered_field O]
variables 
  (convex_quadrilateral : ∀ (A B C D : K), convex A B C D)
  (perpendicular_MN_KM : ∀ (A B : K), ∃ (x : A), ∀ (y : B), perp x y)
  (perpendicular_KL_LN : ∀ (A B : K), ∃ (x : A), ∀ (y : B), perp x y) 
  (MN_eq : MN = 65)
  (KL_eq : KL = 28)
  (perpendicular_L_KN : ∀ (A B : K), ∃ (O : Type), ∀ (B : Type), perp O B)
  (KO_eq : KO = 8)

theorem find_MO :
  MO = 90 :=
by
  sorry

end find_MO_l522_522257


namespace sine_condition_necessary_but_not_sufficient_for_side_equality_l522_522623

-- Define the main conditions of the problem
variables {A B C : ℝ} -- Angles in radians
variables {a b c : ℝ} -- Side lengths opposite to angles A, B, and C respectively

-- The condition related to the sine function
axiom sin_2A_eq_sin_2B : sin (2 * A) = sin (2 * B)

-- The triangle angle condition
axiom triangle_angle_sum : A + B + C = π

-- The Law of Sines in triangles
axiom law_of_sines : sin A / a = sin B / b ∧ sin B / b = sin C / c

-- Statement of the theorem
theorem sine_condition_necessary_but_not_sufficient_for_side_equality :
  sin_2A_eq_sin_2B → 
  (! necessary_but_not_sufficient (a = b) (sin (2 * A) = sin (2 * B))) := sorry

end sine_condition_necessary_but_not_sufficient_for_side_equality_l522_522623


namespace sum_f_240_equals_768_l522_522523

-- Function \( f \) definition as per the conditions.
def f (n : ℕ) : ℤ :=
  if ∃ k : ℕ, k * k = n then 0
  else Int.floor (1 / (Real.fract (Real.sqrt n.toReal)))

-- Statement of the proof problem.
theorem sum_f_240_equals_768 :
  (∑ k in Finset.range 241, f k) = 768 := sorry

end sum_f_240_equals_768_l522_522523


namespace circles_tangent_and_secant_l522_522228

theorem circles_tangent_and_secant 
  {P Q A B M N : Point}
  (h_intersection : (circle1 : Circle).intersect (circle2 : Circle) = {P, Q})
  (h_tangent : is_tangent (line_through A B) circle1)
  (h_tangent2 : is_tangent (line_through A B) circle2)
  (h_parallel : line_parallel (line_through M N) (line_through A B)) :
  (MA / NB) = (QA / QB) :=
sorry

end circles_tangent_and_secant_l522_522228


namespace magicians_trick_l522_522740

def card_pairs : list (ℕ × ℕ) :=
  [(1, 24), (2, 23), (3, 22), (4, 21), (5, 20), (6, 19), (7, 18), (8, 17), 
   (9, 16), (10, 15), (11, 14), (12, 13)]

def identify_added_card (chosen_cards : list ℕ) (pre_agreed_pairs : list (ℕ × ℕ)) : Prop :=
  ∃ a b, (a, b) ∈ pre_agreed_pairs ∧ a ∈ chosen_cards ∧ b ∈ chosen_cards

theorem magicians_trick (chosen_cards : list ℕ) (returned_cards : list ℕ)
  (h_chosen_len : chosen_cards.length = 13) (h_returned_len : returned_cards.length = 2)
  (pre_agreed_pairs : list (ℕ × ℕ) := card_pairs) :
  ∃ added_card ∈ (chosen_cards.filter (λ card, card ∉ returned_cards)),
    ∀ shuffled_cards, shuffled_cards.length = 3 →
    shuffled_cards ~ (returned_cards ++ [added_card]) →
    ∃ a b, (a, b) ∈ pre_agreed_pairs ∧ [a, b] ⊆ shuffled_cards :=
by sorry

end magicians_trick_l522_522740


namespace log_xy_l522_522950

theorem log_xy (x y : ℝ) (h1 : log (x * y^5) = 2) (h2 : log (x^3 * y) = 2) : log (x * y) = 6 / 7 := 
by
  sorry

end log_xy_l522_522950


namespace find_number_l522_522044

noncomputable def number_divided_by_seven_is_five_fourteen (x : ℝ) : Prop :=
  x / 7 = 5 / 14

theorem find_number (x : ℝ) (h : number_divided_by_seven_is_five_fourteen x) : x = 2.5 :=
by
  sorry

end find_number_l522_522044


namespace value_of_f_prime_at_1_l522_522202

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem value_of_f_prime_at_1 : deriv f 1 = 1 :=
by
  sorry

end value_of_f_prime_at_1_l522_522202


namespace marbles_p_plus_q_l522_522841

theorem marbles_p_plus_q :
  ∃ p q : ℕ, p.gcd q = 1 ∧
             (∀ a b n_black : ℕ, 
               a + b = 36 ∧
               (∃ k : ℕ, k * k = 25 ∧ k ≤ a ∧ k ≤ b) ∧
               n_black = k →
               p + q = 493 ∧ 
               ∃ white_prob : ℚ, 
                 white_prob = (a - n_black) / a * (b - n_black) / b ∧ 
                 white_prob.num = p ∧ white_prob.denom = q) :=
begin
  sorry,
end

end marbles_p_plus_q_l522_522841


namespace max_chips_can_be_removed_l522_522334

theorem max_chips_can_be_removed (initial_chips : (Fin 10) × (Fin 10) → ℕ) 
  (condition : ∀ i j, initial_chips (i, j) = 1) : 
    ∃ removed_chips : ℕ, removed_chips = 90 :=
by
  sorry

end max_chips_can_be_removed_l522_522334


namespace range_of_b_min_value_a_add_b_min_value_ab_l522_522532

theorem range_of_b (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : b > 1 := sorry

theorem min_value_a_add_b (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : a + b ≥ 8 := sorry

theorem min_value_ab (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : a * b ≥ 16 := sorry

end range_of_b_min_value_a_add_b_min_value_ab_l522_522532


namespace cevian_and_circumcircle_inequality_l522_522647

variables {A B C D E F G A' B' C' : Point}
variable [triangle ABC : Triangle]
variable [centroid G ABC : Centroid]
variable [intersection_of_extended_cevians D E F G ABC AGD BGE CGF]
variable [intersection_of_circumcircle A' B' C' D E F ABC]

theorem cevian_and_circumcircle_inequality :
  (A'D / DA) + (B'E / EB) + (C'F / FC) ≥ 1 := 
sorry

end cevian_and_circumcircle_inequality_l522_522647


namespace ratio_of_areas_l522_522386

variables (A B C D O : Type) [MetricSpace O]
variables [InCircleSphere O A B C D] [Perpendicular (line AC) (line BD)]
variables (M N : O) [Midpoint M B C] [Midpoint N C D]

theorem ratio_of_areas (ABCD OMCN : Quadrilateral) :
    inscribed_in_circle ABCD O →
    perpendicular_diagonals ABCD →
    midpoint M B C →
    midpoint N C D →
    area OMCN = (1 / 4) * area ABCD :=
sorry

end ratio_of_areas_l522_522386


namespace possible_values_of_expression_l522_522179

noncomputable def sign (x : ℝ) : ℝ :=
if x > 0 then 1 else -1

theorem possible_values_of_expression
  (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := (sign a) + (sign b) + (sign c) + (sign d) + (sign (a * b * c * d))
  in expr ∈ {5, 1, -1, -5} :=
by
  let expr := (sign a) + (sign b) + (sign c) + (sign d) + (sign (a * b * c * d))
  sorry

end possible_values_of_expression_l522_522179


namespace problem1_problem2_l522_522083

theorem problem1 : 27^((2:ℝ)/(3:ℝ)) - 2^(Real.logb 2 3) * Real.logb 2 (1/8) = 18 := 
by
  sorry -- proof omitted

theorem problem2 : 1/(Real.sqrt 5 - 2) - (Real.sqrt 5 + 2)^0 - Real.sqrt ((2 - Real.sqrt 5)^2) = 2*(Real.sqrt 5 - 1) := 
by
  sorry -- proof omitted

end problem1_problem2_l522_522083


namespace gcf_addition_l522_522854

theorem gcf_addition (a b gcf : ℕ) (h1 : a = 198) (h2 : b = 396)
  (h_gcf : Nat.gcd a b = gcf) : gcf + 4 * a = 990 :=
by
  have ha : a = 198 := h1
  have hgcf : gcf = 198 := by
    rw [←h1, ←h2, Nat.gcd, Nat.gcd_rec]
    rfl
  rw [hgcf, ha]
  norm_num
  sorry

end gcf_addition_l522_522854


namespace part1_part2_l522_522651

variables {m x : ℝ}

def f (a k : ℝ) (x : ℝ) := a^x + k * a^(-x)

theorem part1 (h₀ : ∀ x : ℝ, f a k (-x) = -f a k x)
              (h₁ : a > 0) 
              (h₂ : a ≠ 1) 
              (h₃ : f a k 1 = 3 / 2) :
  a = 2 ∧ k = -1 :=
sorry

theorem part2 {a k : ℝ} 
              (ha : a = 2) 
              (hk : k = -1) 
              (hx : x ∈ set.Icc (1 / 2) 2) : 
  (2 * sqrt 2) < m ↔ f 2 (-1) (2 * x^2) + f 2 (-1) (1 - m * x) < 0 :=
sorry

end part1_part2_l522_522651


namespace imaginary_part_of_complex_number_l522_522196

open Complex

theorem imaginary_part_of_complex_number :
  let z := (1 + 2 * I) / (1 - I) ^ 2
  Im(z) = 1 / 2 :=
by
  sorry

end imaginary_part_of_complex_number_l522_522196


namespace chan_family_problem_cubic_root_problem_angle_sum_problem_prime_number_sum_problem_l522_522655

theorem chan_family_problem (a : ℕ) 
  (H_prime : Prime a) 
  (H_grandsons : 64 + a^2 = 16*a + 1) : 
  a = 7 := by sorry

theorem cubic_root_problem (a b : ℕ) 
  (H_eq : (a : ℝ)/7 = (2 + real.sqrt b)^(1/3) + (2 - real.sqrt b)^(1/3)) : 
  b = 5 := by sorry

theorem angle_sum_problem (C : ℝ) 
  (H_angles : 6 + 5 + C + C + C + C + 20 = 180) : 
  C = 35 := by sorry

theorem prime_number_sum_problem (d C P_1 P_2 P_3 P_4 P_5 P_6 : ℕ)
  (H_prime_series : [P_1, P_2, ..., P_6] = [2, 3, 5, 7, 11, 13])
  (H_sum : P_1 + P_2 + P_3 + P_4 = P_5 + P_6) :
  d = 6 := by sorry

end chan_family_problem_cubic_root_problem_angle_sum_problem_prime_number_sum_problem_l522_522655


namespace determine_m_l522_522314

-- Definitions and conditions
def U : set ℕ := {1, 2, 3, 4}
def A (m : ℕ) : set ℕ := {x | x^2 - 5 * x + m = 0 ∧ x ∈ U}
def C_U (s : set ℕ) : set ℕ := U \ s

-- Theorem statement
theorem determine_m (m : ℕ) (h : C_U (A m) = {1, 4}) : m = 6 :=
sorry

end determine_m_l522_522314


namespace log_xy_l522_522949

theorem log_xy (x y : ℝ) (h1 : log (x * y^5) = 2) (h2 : log (x^3 * y) = 2) : log (x * y) = 6 / 7 := 
by
  sorry

end log_xy_l522_522949


namespace max_value_of_quadratic_l522_522005

theorem max_value_of_quadratic : ∃ x, f x = 10 :=
  let f (x : ℝ) : ℝ := -4 * x^2 + 12 * x + 1
  ∃ x, f x = 10 :=
by
  -- Skip the full proof:
  sorry

end max_value_of_quadratic_l522_522005


namespace find_square_l522_522544

theorem find_square (q x : ℝ) 
  (h1 : x + q = 74) 
  (h2 : x + 2 * q^2 = 180) : 
  x = 66 :=
by {
  sorry
}

end find_square_l522_522544


namespace number_of_valid_3_digit_numbers_l522_522933

def is_even (n : ℕ) : Prop := n % 2 = 0

def valid_3_digit_numbers_count : ℕ :=
  let digits := [(4, 8), (8, 4), (6, 6)]
  digits.length * 9

theorem number_of_valid_3_digit_numbers : valid_3_digit_numbers_count = 27 :=
by
  sorry

end number_of_valid_3_digit_numbers_l522_522933


namespace reflection_line_l522_522396

open Function

-- Define the vertices of triangle DEF
def D : ℝ × ℝ := (-3, 2)
def E : ℝ × ℝ := (1, 4)
def F : ℝ × ℝ := (-5, -1)

-- Define the reflected vertices of triangle DEF
def D' : ℝ × ℝ := (-11, 2)
def E' : ℝ × ℝ := (-9, 4)
def F' : ℝ × ℝ := (-15, -1)

-- Axiom stating the condition for reflection with respect to the vertical line
axiom reflection_condition (A A' : ℝ × ℝ) : ∃ x, (fst A + fst A') / 2 = x

theorem reflection_line :
  (reflection_condition D D') ∧ (reflection_condition E E') ∧ (reflection_condition F F') →
  ∃ L, L = -7 := 
by
  sorry

end reflection_line_l522_522396


namespace divide_figure_into_three_parts_l522_522826

noncomputable theory
open_locale classical

-- Definition of splitting into identical parts in Lean 4
def can_be_divided_into_three_identical_parts (figure : set (ℕ × ℕ)) : Prop :=
  ∃ (P Q R : set (ℕ × ℕ)), 
  P ∪ Q ∪ R = figure ∧ 
  P ∩ Q = ∅ ∧ Q ∩ R = ∅ ∧ R ∩ P = ∅ ∧ 
  ∃ (a b c : ℕ), P ≈ Q ∧ Q ≈ R ∧ |P| = a ∧ |Q| = b ∧ |R| = c ∧ a = b ∧ b = c

-- Problem Statement in Lean 4
theorem divide_figure_into_three_parts (figure : set (ℕ × ℕ)) :
  can_be_divided_into_three_identical_parts figure :=
sorry

end divide_figure_into_three_parts_l522_522826


namespace polynomial_solution_l522_522138

noncomputable def p (x : ℝ) := 2 * Real.sqrt 3 * x^4 - 6

theorem polynomial_solution (x : ℝ) : 
  (p (x^4) - p (x^4 - 3) = (p x)^3 - 18) :=
by
  sorry

end polynomial_solution_l522_522138


namespace zero_in_interval_l522_522389

def f (x : ℝ) : ℝ := 3^x - 1 / (Real.sqrt x + 1) - 6

theorem zero_in_interval : f 0 < 0 → f 1 < 0 → f 2 > 0 → ∃ x, (1 < x ∧ x < 2) ∧ f x = 0 :=
by
  sorry

end zero_in_interval_l522_522389


namespace maximum_guaranteed_money_l522_522401

-- Define the conditions:
def card_values : List ℕ := List.range 100 |>.map (λ x => x + 1)
def available_card_values (x : ℕ) : Bool := x ∈ card_values

-- The theorem: proving the maximum guaranteed amount
theorem maximum_guaranteed_money : 
  Σ (strategy : ℕ → ℕ), ∀ (card_val : ℕ), card_val ∈ card_values → strategy card_val = 50 → ∑ val in card_values.filter (λ v, strategy v = 50) = 2550 :=
sorry

end maximum_guaranteed_money_l522_522401


namespace space_diagonals_of_polyhedron_l522_522439

theorem space_diagonals_of_polyhedron 
  (V E : ℕ) (faces : ℕ) 
  (triangular_faces quadrilateral_faces : ℕ)
  (hV : V = 30) 
  (hE : E = 58) 
  (h_faces : faces = 36) 
  (h_triangular_faces : triangular_faces = 26) 
  (h_quadrilateral_faces : quadrilateral_faces = 10) :
  let total_pairs := nat.choose V 2,
      face_diagonals := triangular_faces * 0 + quadrilateral_faces * 2,
      space_diagonals := total_pairs - E - face_diagonals in
  space_diagonals = 357 :=
by
  sorry

end space_diagonals_of_polyhedron_l522_522439


namespace cluster_pairs_l522_522247

def f1 (x : ℝ) : ℝ := sin x * cos x
def f2 (x : ℝ) : ℝ := 2 * sin (x + π / 4)
def f3 (x : ℝ) : ℝ := sin x + sqrt 3 * cos x
def f4 (x : ℝ) : ℝ := sqrt 2 * sin (2 * x) + 1

def same_cluster (f g : ℝ → ℝ) : Prop :=
  (∃ T : ℝ, ∀ x : ℝ, f (x + T) = g x) ∧
  (∃ A : ℝ, ∀ x : ℝ, f x = A * g x)

theorem cluster_pairs :
  same_cluster f2 f3 :=
sorry

end cluster_pairs_l522_522247


namespace eighteen_ab_eq_PbQ2a_l522_522294

variable {a b : ℕ}

def P (a : ℕ) := 2^a
def Q (b : ℕ) := 3^b

theorem eighteen_ab_eq_PbQ2a (a b : ℕ) : 18^(a * b) = P a b * Q b^(2 * a) :=
by
  sorry

end eighteen_ab_eq_PbQ2a_l522_522294


namespace min_val_f_l522_522373

noncomputable def f (x : ℝ) : ℝ := Math.sin x + Math.cos x - Math.sin x * Math.cos x

def t (x : ℝ) : ℝ := Math.sqrt 2 * Math.sin (x + Real.pi / 4)

theorem min_val_f : ∃ x : ℝ, f x = -1/2 - Real.sqrt 2 := by
  have h1 : t x = Math.sin x + Math.cos x := sorry
  have h2 : Math.sin x * Math.cos x = (t x ^ 2 - 1) / 2 := sorry
  sorry

end min_val_f_l522_522373


namespace female_democrats_count_l522_522023

theorem female_democrats_count :
  ∃ (F : ℕ) (M : ℕ),
    F + M = 750 ∧
    (F / 2) + (M / 4) = 250 ∧
    1 / 3 * 750 = 250 ∧
    F / 2 = 125 := sorry

end female_democrats_count_l522_522023


namespace combined_tax_rate_l522_522631

theorem combined_tax_rate (John_income Ingrid_income : ℝ)
  (John_tax_rate Ingrid_tax_rate : ℝ)
  (John_income_val : John_income = 58000)
  (Ingrid_income_val : Ingrid_income = 72000)
  (John_tax_rate_val : John_tax_rate = 0.30)
  (Ingrid_tax_rate_val : Ingrid_tax_rate = 0.40) :
  (John_tax_rate * John_income + Ingrid_tax_rate * Ingrid_income) / 
  (John_income + Ingrid_income) * 100 = 35.54 :=
by {
  rw [John_income_val, Ingrid_income_val, John_tax_rate_val, Ingrid_tax_rate_val],
  sorry
}

end combined_tax_rate_l522_522631


namespace triangle_side_sum_l522_522222

theorem triangle_side_sum (a b c : ℝ) (angle_A : ℝ) (area : ℝ)
  (h_a : a = 2) (h_angle_A : angle_A = 60) (h_area : area = sqrt 3)
  (h_area_eq : area = 1/2 * b * c * sin (angle_A * real.pi / 180)) :
  b + c = 4 :=
by
  sorry

end triangle_side_sum_l522_522222


namespace coloring_scheme_satisfies_conditions_l522_522104

noncomputable def coloring (x y : ℤ) : string :=
  if x = y then if x % 2 = 0 then "black" else "white"
  else "red"

theorem coloring_scheme_satisfies_conditions :
  (∀ y : ℤ, ∃ x, coloring x y = "black") ∧
  (∀ y : ℤ, ∃ x, coloring x y = "white") ∧
  (∀ y : ℤ, ∃ x, coloring x y = "red") ∧
  (∀ (A B C : (ℤ × ℤ)), coloring A.1 A.2 = "white" → coloring B.1 B.2 = "red" → coloring C.1 C.2 = "black" → 
  ∃ D : (ℤ × ℤ), coloring D.1 D.2 = "red" ∧ 
  D.1 = A.1 + C.1 - B.1 ∧ D.2 = A.2 + C.2 - B.2) :=
begin
  sorry
end

end coloring_scheme_satisfies_conditions_l522_522104


namespace binomial_parameters_l522_522904

theorem binomial_parameters
  (n : ℕ) (p : ℚ)
  (hE : n * p = 12) (hD : n * p * (1 - p) = 2.4) :
  n = 15 ∧ p = 4 / 5 :=
by
  sorry

end binomial_parameters_l522_522904


namespace prob_X_eq_2_ex_X_eq_4_l522_522876

noncomputable def binomial_distribution : Type := sorry

axiom binomial_6_2_3 : binomial_distribution := sorry

theorem prob_X_eq_2 : P(X = 2) = 20 / 243 := by
  sorry

theorem ex_X_eq_4 : E(X) = 4 := by
  sorry

end prob_X_eq_2_ex_X_eq_4_l522_522876


namespace homogeneous_polynomial_unique_form_l522_522703

noncomputable def f (x y : ℝ) : ℝ := sorry

theorem homogeneous_polynomial_unique_form (n : ℕ) 
  (h_homogeneous : ∀ (x y : ℝ), f (x / (x + y)) (y / (x + y)) = ((x + y)^(-n)) * f x y)
  (h_initial : f 1 0 = 1)
  (h_relation : ∀ (a b c : ℝ), f (a + b) c + f (b + c) a + f (c + a) b = 0) :
  ∀ (x y : ℝ), f x y = (x - 2 * y) * (x + y)^(n - 1) := 
by
  sorry

end homogeneous_polynomial_unique_form_l522_522703


namespace no_integer_satisfies_inequality_l522_522498

theorem no_integer_satisfies_inequality :
  ∀ x : ℤ, (30 < x ∧ x < 90) → log 10 (x - 30) + log 10 (90 - x) < 1 → false :=
by 
  intro x h1 h2
  sorry

end no_integer_satisfies_inequality_l522_522498


namespace trigonometric_identity_l522_522504

-- Definitions of trigonometric values used in the problem
def sin_45 : ℝ := Real.sin (Real.pi / 4)
def cos_15 : ℝ := Real.cos (Real.pi / 12)
def cos_225 : ℝ := Real.cos (5 * Real.pi / 4)
def sin_165 : ℝ := Real.sin (11 * Real.pi / 12)

theorem trigonometric_identity :
  sin_45 * cos_15 + cos_225 * sin_165 = 1 / 2 :=
by
  sorry

end trigonometric_identity_l522_522504


namespace somu_one_fifth_age_back_l522_522687

theorem somu_one_fifth_age_back {S F Y : ℕ}
  (h1 : S = 16)
  (h2 : S = F / 3)
  (h3 : S - Y = (F - Y) / 5) :
  Y = 8 :=
by
  sorry

end somu_one_fifth_age_back_l522_522687


namespace part1_part2_l522_522313

-- Definitions based on the given problem conditions
variables {α β t x1 x2 : ℝ}
axiom quadratic_roots : 2 * α^2 - t * α - 2 = 0 ∧ 2 * β^2 - t * β - 2 = 0
axiom alpha_lt_beta : α < β
axiom in_interval : x1 ∈ set.Icc α β ∧ x2 ∈ set.Icc α β

-- Lean 4 statement for Part (1)
theorem part1 (hα : 2 * α^2 - t * α - 2 = 0) (hβ : 2 * β^2 - t * β - 2 = 0)
  (hαβ : α < β) (hx1 : x1 ∈ set.Icc α β) (hx2 : x2 ∈ set.Icc α β) : 
  4 * x1 * x2 - t * (x1 + x2) - 4 < 0 :=
sorry

-- Function definition and additional variables for Part (2)
def f (x : ℝ) := (4 * x - t) / (x^2 + 1)
noncomputable def f_max := max (f α) (f β)
noncomputable def f_min := min (f α) (f β)
def g (t : ℝ) := f_max - f_min

-- Lean 4 statement for Part (2)
theorem part2 (hα : 2 * α^2 - t * α - 2 = 0) (hβ : 2 * β^2 - t * β - 2 = 0)
  (hαβ : α < β) : g t = 4 :=
sorry

end part1_part2_l522_522313


namespace nancy_threw_out_2_carrots_l522_522326

theorem nancy_threw_out_2_carrots :
  ∀ (x : ℕ), 12 - x + 21 = 31 → x = 2 :=
by
  sorry

end nancy_threw_out_2_carrots_l522_522326


namespace angle_QSR_79_l522_522988

noncomputable def angles_problem (P Q R S T: Type) [AffineGeometry P] : Prop :=
  let angle_PQR := 90
  let angle_QRT := 158
  let angle_PRS := angle_QRS
  let angle_QRP := 180 - angle_QRT
  let angle_QRS := angle_PRS
  let x := angle_QRS
  let angle_QSR := 90 - x
  angle_QSR = 79

theorem angle_QSR_79 (P Q R S T : Type) [AffineGeometry P] : angles_problem P Q R S T :=
  sorry

end angle_QSR_79_l522_522988


namespace number_of_comedies_l522_522838

variables (T a : ℝ)
variables (dramas thrillers scifi comedies action_movies : ℝ)

-- Define the total number of movies rented
def total_movies_rented := T

-- Conditions
def comedies_percentage : ℝ := 0.48
def action_movies_percentage : ℝ := 0.16
def remaining_percentage : ℝ := 1 - comedies_percentage - action_movies_percentage

def dramas_count := 3 * a
def thrillers_count := 2 * dramas_count
def scifi_count := a

-- The equation representing the total rentals
noncomputable def total_rentals_equation : Prop := 
  T = comedies_percentage * T + action_movies_percentage * T + dramas_count + thrillers_count + scifi_count

-- Number of comedies in terms of T
noncomputable def comedies_in_terms_of_T : ℝ := comedies_percentage * T

-- To prove: 
-- Prove number of comedies in terms of a
theorem number_of_comedies (a : ℝ) : comedies_in_terms_of_T = (40 / 3) * a :=
by sorry

end number_of_comedies_l522_522838


namespace possible_values_expression_l522_522174

-- Defining the main expression 
def main_expression (a b c d : ℝ) : ℝ :=
  (a / |a|) + (b / |b|) + (c / |c|) + (d / |d|) + (abcd / |abcd|)

-- The theorem that we need to prove
theorem possible_values_expression (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) : 
  main_expression a b c d ∈ {5, 1, -3} :=
sorry

end possible_values_expression_l522_522174


namespace airport_exchange_rate_frac_l522_522013

variable (euros_received : ℕ) (euros : ℕ) (official_exchange_rate : ℕ) (dollars_received : ℕ)

theorem airport_exchange_rate_frac (h1 : euros = 70) (h2 : official_exchange_rate = 5) (h3 : dollars_received = 10) :
  (euros_received * dollars_received) = (euros * official_exchange_rate) →
  euros_received = 5 / 7 :=
  sorry

end airport_exchange_rate_frac_l522_522013


namespace houses_distance_l522_522744

theorem houses_distance (num_houses : ℕ) (total_length : ℝ) (at_both_ends : Bool) 
  (h1: num_houses = 6) (h2: total_length = 11.5) (h3: at_both_ends = true) : 
  total_length / (num_houses - 1) = 2.3 := 
by
  sorry

end houses_distance_l522_522744


namespace product_of_c_l522_522503

noncomputable def points_distance (c : ℝ) : ℝ :=
  real.sqrt ((3*c - 1)^2 + (c + 5 - 4)^2)

theorem product_of_c : (points_distance c = 5) → 
  (∀ x y : ℝ, (x = (2 + real.sqrt 234) / 10 ∨ x = (2 - real.sqrt 234) / 10) ∧ 
  (y = (2 + real.sqrt 234) / 10 ∨ y = (2 - real.sqrt 234) / 10) → 
  x * y = -2.3) :=
by
  intros h x y hx hy
  sorry

end product_of_c_l522_522503


namespace number_of_rose_bushes_l522_522079

-- Definitions derived from the conditions
def semi_major_axis := 15
def semi_minor_axis := 8
def spacing_distance := 1

-- The mathematically equivalent proof problem
theorem number_of_rose_bushes : 
  let a := semi_major_axis in
  let b := semi_minor_axis in
  let d := spacing_distance in
  let circumference := 2 * Real.pi * Real.sqrt ((a^2 + b^2) / 2) in
  let num_bushes := circumference / d in
  Real.ceil num_bushes = 76 :=
by
  sorry

end number_of_rose_bushes_l522_522079


namespace reporters_not_cover_politics_l522_522020

-- Define the total number of reporters as 100% for simplicity.
variable (total_reporters : ℝ) (percent_local_politics : ℝ) (percent_non_local_politics : ℝ)

-- Given conditions
def condition_1 := percent_local_politics = 12 / 100
def condition_2 := percent_non_local_politics = 40 / 100

-- Question: What percent of the reporters do not cover politics?
def question := ∀ P : ℝ, (0.60 * P = percent_local_politics * 100) → (100 - P * 100)

-- Correct answer: 80%
def correct_answer := 80

-- Proof that percent of the reporters who do not cover politics equals to 80% given the conditions
theorem reporters_not_cover_politics : 
  total_reporters = 100 → 
  condition_1 → 
  condition_2 → 
  (question percent_local_politics percent_non_local_politics) = correct_answer := 
by
  intros h1 h2 h3
  sorry

end reporters_not_cover_politics_l522_522020


namespace no_solution_for_sum_eq_one_l522_522833

theorem no_solution_for_sum_eq_one (x : ℕ) (hx : x > 1) : 
  (∑ k in finset.range (x^2 + 1), (1 : ℝ) / (x + k)) ≠ 1 :=
by
  sorry

end no_solution_for_sum_eq_one_l522_522833


namespace maria_percentage_paid_l522_522723

def P : ℝ := 800
def sale_price (P : ℝ) : ℝ := 0.80 * P
def maria_price (SP : ℝ) : ℝ := 0.90 * SP
def percentage_paid (maria_price : ℝ) (P : ℝ) : ℝ := (maria_price / P) * 100

theorem maria_percentage_paid :
  let SP := sale_price P in
  let final_price := maria_price SP in
  percentage_paid final_price P = 72 :=
by
  sorry

end maria_percentage_paid_l522_522723


namespace part1_part2_l522_522553

-- Define f's properties and conditions.
variable (f : ℝ → ℝ)
variable (h1 : ∀ x : ℝ, 0 < x → f x > 1 / x^2)
variable (h2 : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x)
variable (h3 : ∀ x : ℝ, 0 < x → (f x)^2 * f (f x - 1 / x^2) = (f 1)^3)

-- State the first part: Proving f(1) = 2
theorem part1 : f 1 = 2 := 
by
  sorry

-- Define the specific function provided
def specific_f (x : ℝ) : ℝ := 2 / x^2

-- Prove that the specific function satisfies all conditions
theorem part2 :
  (∀ x : ℝ, 0 < x → specific_f x > 1 / x^2) ∧
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → specific_f y < specific_f x) ∧
  (∀ x : ℝ, 0 < x → (specific_f x)^2 * specific_f (specific_f x - 1 / x^2) = (specific_f 1)^3) := 
by
  sorry

end part1_part2_l522_522553


namespace range_of_k_l522_522142

theorem range_of_k (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 2 * x + 6 * k < 0) → k < -real.sqrt 6 / 6 :=
by
  sorry

end range_of_k_l522_522142


namespace minimum_paper_toys_is_eight_l522_522663

noncomputable def minimum_paper_toys (s_boats: ℕ) (s_planes: ℕ) : ℕ :=
  s_boats * 8 + s_planes * 6

theorem minimum_paper_toys_is_eight :
  ∀ (s_boats s_planes : ℕ), s_boats >= 1 → minimum_paper_toys s_boats s_planes = 8 → s_planes = 0 :=
by
  intros s_boats s_planes h_boats h_eq
  have h1: s_boats * 8 + s_planes * 6 = 8 := h_eq
  sorry

end minimum_paper_toys_is_eight_l522_522663


namespace length_AB_l522_522212

section ParabolaProblem

-- Given the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- A point P on the parabola
def point_on_parabola (x₀ y₀ : ℝ) : Prop := parabola x₀ y₀

-- Define the midpoint R of PQ with P(x₀, y₀) and Q(x₀, 0)
def midpoint (x₀ y₀ x y : ℝ) : Prop := x = x₀ ∧ y = y₀ / 2

-- Define the trajectory D of the midpoint R
def trajectory (x y : ℝ) : Prop := y^2 = x

-- The focus of the parabola C is at (1, 0)
def focus := (1 : ℝ, 0 : ℝ)

-- Equation of line l passing through the focus with slope 1
def line_l (x y : ℝ) : Prop := y = x - 1

-- Intersection of line l and trajectory D
def intersection (x y : ℝ) : Prop := (line_l x y) ∧ (trajectory x y)

theorem length_AB : 
  ∀ x₀ y₀ x₁ y₁ x₂ y₂ : ℝ, 
  point_on_parabola x₀ y₀ →
  midpoint x₀ y₀ x₁ y₁ →
  trajectory x₁ y₁ →
  intersection x₁ y₁ →
  intersection x₂ y₂ →
  |x₁ - x₂| = real.sqrt 10 :=
by sorry

end ParabolaProblem

end length_AB_l522_522212


namespace find_common_ratio_l522_522549

open Nat

noncomputable def geometric_common_ratio (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n : ℕ, n > 0 → a n * a (n + 4) = 9^(n + 1))

-- Define the sequence a_n as a geometric sequence with ratio q
noncomputable def is_geometric (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a n = a 1 * q^(n - 1)

-- Our Goal
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  geometric_common_ratio a q →
  is_geometric a q →
  (q = 3 ∨ q equals -3) :=
by
  sorry

end find_common_ratio_l522_522549


namespace evaluate_floor_abs_neg57_8_l522_522124

theorem evaluate_floor_abs_neg57_8 : Int.floor (Real.abs (-57.8)) = 57 := by
  sorry

end evaluate_floor_abs_neg57_8_l522_522124


namespace train_crossing_time_l522_522454

theorem train_crossing_time
    (length_train : ℕ) (length_bridge : ℕ) (speed_train : ℕ)
    (h_train : length_train = 100)
    (h_bridge : length_bridge = 300)
    (h_speed : speed_train = 40)
    : (length_train + length_bridge) / speed_train = 10 := 
by
  rw [h_train, h_bridge, h_speed]
  norm_num
  sorry

end train_crossing_time_l522_522454


namespace seq_a2018_l522_522926

noncomputable def sequence : ℕ+ → ℚ
| ⟨1, _⟩ := 3 / 4
| ⟨n + 1, h⟩ := 1 - (1 / sequence ⟨n, Nat.succ_pos n⟩)

theorem seq_a2018 : sequence ⟨2018, by norm_num⟩ = -1 / 3 :=
sorry

end seq_a2018_l522_522926


namespace bisection_termination_condition_l522_522963

variable (a b : ℝ) (f : ℝ → ℝ)

-- Conditions: a < b, f has a unique root in (a, b), accuracy = 0.001
axiom unique_root_in_interval : ∃! x, a < x ∧ x < b ∧ f x = 0
def accuracy : ℝ := 0.001

-- Prove that n = ⌈ log2 ((b - a) / 0.001) ⌉ is the termination condition
theorem bisection_termination_condition :
  ∃ (n : ℕ), n = ⌈ real.log2 ((b - a) / accuracy) ⌉ := sorry

end bisection_termination_condition_l522_522963


namespace area_trapezoid_AFGE_l522_522689

theorem area_trapezoid_AFGE (A B C D E F G : Point)
  (h_rect : is_rectangle A B C D)
  (h_area_rect : area A B C D = 70)
  (h_F_on_BC : F ∈ line_segment B C)
  (h_D_mid_EG : midpoint D E G) :
  area A F G E = 70 := 
by sorry

end area_trapezoid_AFGE_l522_522689


namespace jack_finishes_in_16_days_l522_522279

noncomputable def pages_in_book : ℕ := 285
noncomputable def weekday_reading_rate : ℕ := 23
noncomputable def weekend_reading_rate : ℕ := 35
noncomputable def weekdays_per_week : ℕ := 5
noncomputable def weekends_per_week : ℕ := 2
noncomputable def weekday_skipped : ℕ := 1
noncomputable def weekend_skipped : ℕ := 1

noncomputable def pages_per_week : ℕ :=
  (weekdays_per_week - weekday_skipped) * weekday_reading_rate + 
  (weekends_per_week - weekend_skipped) * weekend_reading_rate

noncomputable def weeks_needed : ℕ :=
  pages_in_book / pages_per_week

noncomputable def pages_left_after_weeks : ℕ :=
  pages_in_book % pages_per_week

noncomputable def extra_days_needed (pages_left : ℕ) : ℕ :=
  if pages_left > weekend_reading_rate then 2
  else if pages_left > weekday_reading_rate then 2
  else 1

noncomputable def total_days_needed : ℕ :=
  weeks_needed * 7 + extra_days_needed (pages_left_after_weeks)

theorem jack_finishes_in_16_days : total_days_needed = 16 := by
  sorry

end jack_finishes_in_16_days_l522_522279


namespace geometric_sequence_nec_not_suff_l522_522545

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n ≠ 0 → (a (n + 1) / a n) = (a (n + 2) / a (n + 1))

noncomputable def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 3) = a (n + 1) * a (n + 2)

theorem geometric_sequence_nec_not_suff (a : ℕ → ℝ) (hn : ∀ n : ℕ, a n ≠ 0) : 
  (is_geometric_sequence a → satisfies_condition a) ∧ ¬(satisfies_condition a → is_geometric_sequence a) :=
by
  sorry

end geometric_sequence_nec_not_suff_l522_522545


namespace ellipse_no_k_exists_l522_522197

-- Define the ellipse and its properties
noncomputable def ellipse_equation (a b : ℝ) (h : a > b ∧ b > 0) (e : ℝ) 
  (h_e : e = (real.sqrt 2) / 2) (l_minor : ℝ) (h_minor : 2 * b = l_minor) : Prop := 
  let c := real.sqrt (a^2 - b^2) in
  a = real.sqrt 2 ∧ b = 1 ∧ c = 1 ∧ (a^2 = b^2 + c^2) ∧ e = c / a ∧ 
  (∀ x y, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ (x^2 / 2) + y^2 = 1)

-- Collinearity condition for vectors OP, OQ and AB
noncomputable def collinearity_condition (k : ℝ) 
  (h_k : k < - (real.sqrt 2) / 2 ∨ k > (real.sqrt 2) / 2) : Prop :=
  let P := (λ x : ℝ, (x, k * x + real.sqrt 2)) in
  ∀ x1 x2 y1 y2, x1 + x2 = -(4 * real.sqrt 2 * k) / (1 + 2 * k^2) ∧ 
                  y1 + y2 = k * (x1 + x2) + 2 * real.sqrt 2 ↔
  (x1 + x2) = - (real.sqrt 2) * (y1 + y2) ∧ 
  ¬ (P x1 + P x2) collinear ((real.sqrt 2, 0), (0,1))

-- Main theorem
theorem ellipse_no_k_exists 
  (a b : ℝ) (h : a > b ∧ b > 0) (e : ℝ) (h_e : e = (real.sqrt 2) / 2) (l_minor : ℝ) (h_minor : 2 * b = l_minor)
  (k : ℝ) (h_k : k < - (real.sqrt 2) / 2 ∨ k > (real.sqrt 2) / 2) :
  ellipse_equation a b h e h_e l_minor h_minor ∧ collinearity_condition k h_k → false := 
sorry

end ellipse_no_k_exists_l522_522197


namespace find_t_l522_522382

theorem find_t (t : ℕ) (x y x0 y0 x1 y1 x2 y2 : ℕ)
    (h1 : (x0, y0) = (0, 6))
    (h2 : (x1, y1) = (5, 21))
    (h3 : (x2, y2) = (10, 36))
    (h_eq : ∀ {x y}, (x ≠ 0 -> (x * y1 = y * x1)) -> 
                   (x ≠ x1 -> (x * y2 = y * x2)) -> 
                   (x ≠ x2 -> (x * (y2 - y1) = y * (x2 - x1)))) : 
    t = 3 * 40 + 6 := begin
  sorry
end

end find_t_l522_522382


namespace largest_prime_factor_133_is_19_l522_522010

def largest_prime_factor (n: ℕ) : ℕ :=
  sorry

-- Define the given numbers
def num1 := 45
def num2 := 65
def num3 := 91
def num4 := 85
def num5 := 133

-- Define the largest prime factors for the given numbers
def lpf_num1 := 5
def lpf_num2 := 13
def lpf_num3 := 13
def lpf_num4 := 17
def lpf_num5 := 19

-- Define a condition that ensures the calculated largest prime factor is correct
axiom lpf_correct1 : largest_prime_factor num1 = lpf_num1
axiom lpf_correct2 : largest_prime_factor num2 = lpf_num2
axiom lpf_correct3 : largest_prime_factor num3 = lpf_num3
axiom lpf_correct4 : largest_prime_factor num4 = lpf_num4
axiom lpf_correct5 : largest_prime_factor num5 = lpf_num5

-- Define the proof statement
theorem largest_prime_factor_133_is_19 : 
  ∀ (a: ℕ), (a = 133) -> (largest_prime_factor a = lpf_num5) ∧ (lpf_num5 > lpf_num1) ∧ (lpf_num5 > lpf_num2) ∧ (lpf_num5 > lpf_num3) ∧ (lpf_num5 > lpf_num4) :=
by
  intro a ha
  rw ha
  exact ⟨lpf_correct5, sorry, sorry, sorry, sorry⟩ -- Needs proof of comparisons

end largest_prime_factor_133_is_19_l522_522010


namespace find_range_of_a_l522_522965

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^3) / 3 - (a / 2) * x^2 + x + 1

def is_monotonically_decreasing_in (a : ℝ) (x : ℝ) : Prop := 
  ∀ s t : ℝ, (s ∈ Set.Ioo (3 / 2) 4) ∧ (t ∈ Set.Ioo (3 / 2) 4) ∧ s < t → 
  f a t ≤ f a s

theorem find_range_of_a :
  ∀ a : ℝ, is_monotonically_decreasing_in a x → 
  a ∈ Set.Ici (17/4)
:= sorry

end find_range_of_a_l522_522965


namespace zero_of_f_in_interval_l522_522730

noncomputable def f (x : ℝ) : ℝ := Real.log (x / 2) - (1 / x)

theorem zero_of_f_in_interval : ∃ x ∈ Ioo 2 3, f x = 0 :=
sorry

end zero_of_f_in_interval_l522_522730


namespace dress_designs_l522_522785

theorem dress_designs (colors patterns accessories : ℕ) (h_colors : colors = 5) (h_patterns : patterns = 4) (h_accessories : accessories = 2) : 
  colors * patterns * accessories = 40 :=
by 
  rw [h_colors, h_patterns, h_accessories]
  norm_num

end dress_designs_l522_522785


namespace eval_expression_l522_522476

theorem eval_expression :
  - (2:ℝ)⁻¹ * (-8) - real.sqrt 9 - abs (-4) = -3 :=
by
  sorry

end eval_expression_l522_522476


namespace determine_ellipse_equation_l522_522187

noncomputable def ellipse_equation (a b : ℝ) (h1 : a > 0) (h2 : 4 * a^2 = 5 * b^2)
  (h3 : a ≠ 0) : (ℝ × ℝ) → ℝ := 
  λ P, (P.1^2 / a^2) + (P.2^2 / b^2) - 1

theorem determine_ellipse_equation : 
  ∃ (a b : ℝ) (h1 : a > 0) (h2 : 4 * a^2 = 5 * b^2), 
  ellipse_equation a b h1 h2 (-5, 4) = 0 ∧ 
  (a^2 = 45 ∧ b^2 = 36) :=
begin
  use [√45, √36],
  rw [←mul_self_inj_of_nonneg (sqrt_nonneg _) (sqrt_nonneg _)],
  rw [sqrt_sq (le_of_lt (by norm_num : (0 : ℝ) < 45)),
      sqrt_sq (le_of_lt (by norm_num : (0 : ℝ) < 36))],
  split,
  { rw [ellipse_equation, sq_sqrt, sq_sqrt];
    norm_num,
    linarith },
  { split; norm_num }
end

end determine_ellipse_equation_l522_522187


namespace second_smallest_four_digit_pascal_l522_522751

theorem second_smallest_four_digit_pascal :
  ∃ (n k : ℕ), (1000 < Nat.choose n k) ∧ (Nat.choose n k = 1001) :=
by
  sorry

end second_smallest_four_digit_pascal_l522_522751


namespace least_positive_integer_l522_522748

theorem least_positive_integer (n : ℕ) (h₁ : n % 3 = 0) (h₂ : n % 4 = 1) (h₃ : n % 5 = 2) : n = 57 :=
by
  -- sorry to skip the proof
  sorry

end least_positive_integer_l522_522748


namespace problem_l522_522912

-- Definitions and conditions
variables (a b x1 x2 : ℝ)
noncomputable def f (x : ℝ) : ℝ := (x - a) * Real.log x

-- Hypotheses
hypothesis ha : a > 1
hypothesis hx1 : x1 > 0
hypothesis hx2 : x2 > x1
hypothesis root_x1 : f a b x1 = b
hypothesis root_x2 : f a b x2 = b

-- Goal
theorem problem (a > 1) (b x1 x2 : ℝ)
  (x1 > 0) (x2 > x1)
  (root_x1 : f a b x1 = b)
  (root_x2 : f a b x2 = b) :
  x2 - x1 ≤ b * (1 / (Real.log a) - 1 / (1 - a)) + (a - 1) :=
sorry

end problem_l522_522912


namespace sufficient_condition_for_P_l522_522593

noncomputable def increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

theorem sufficient_condition_for_P (f : ℝ → ℝ) (t : ℝ) 
  (h_inc : increasing f) (h_val1 : f (-1) = -4) (h_val2 : f 2 = 2) :
  (∀ x, (x ∈ {x | -1 - t < x ∧ x < 2 - t}) → x < -1) → t ≥ 3 :=
by
  sorry

end sufficient_condition_for_P_l522_522593


namespace triangle_side_length_count_l522_522828

theorem triangle_side_length_count : 
  let sides := (8, 5)
  ∃ (count : ℕ), count = ∑ i in (finset.Ico 4 13), 1
  sorry

end triangle_side_length_count_l522_522828


namespace part_1_l522_522918

variable (a : ℝ) (f : ℝ → ℝ)

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

def f (x : ℝ) : ℝ := 2 * x - 2 * (a + 1) * Real.exp x + a * Real.exp (2 * x)

theorem part_1 : is_monotonically_increasing (f a) → a = 1 := 
  sorry

end part_1_l522_522918


namespace centroid_for_segment_pass_l522_522970

open TopologicalSpace

variable {α : Type}

structure triangle (α : Type) :=
(A B C : α)

variables {t : triangle α}
variables (A B C M N : α)
variables [HasVectorSpace (point α)]
variables [HasDiv Real (point α)]

def segment1 := segment_len B M / segment_len M A
def segment2 := segment_len C N / segment_len N A

theorem centroid_for_segment_pass
  (hAB : LiesOnSegment A B M)
  (hAC : LiesOnSegment A C N)
  (hCondition : segment1 + segment2 = 1):
  PassesThroughCentroid t M N :=
sorry

end centroid_for_segment_pass_l522_522970


namespace students_spend_185_minutes_in_timeout_l522_522320

variable (tR tF tS t_total : ℕ)

-- Conditions
def running_timeouts : ℕ := 5
def food_timeouts : ℕ := 5 * running_timeouts - 1
def swearing_timeouts : ℕ := food_timeouts / 3
def total_timeouts : ℕ := running_timeouts + food_timeouts + swearing_timeouts
def timeout_duration : ℕ := 5

-- Total time spent in time-out
def total_timeout_minutes : ℕ := total_timeouts * timeout_duration

theorem students_spend_185_minutes_in_timeout :
  total_timeout_minutes = 185 :=
by
  -- The answer is directly given by the conditions and the correct answer identified.
  sorry

end students_spend_185_minutes_in_timeout_l522_522320


namespace minimum_positive_period_l522_522709

-- Given function definition
def given_function (x : ℝ) : ℝ :=
  3 * Real.sin (2 * x - 3 * Real.pi / 4)

-- Definition of the minimum positive period
def period := Real.pi

-- Theorem statement
theorem minimum_positive_period :
  ∃ T > 0, ∀ x, given_function (x + T) = given_function x ∧ T = period :=
by
  sorry

end minimum_positive_period_l522_522709


namespace rectangle_two_diagonal_corners_l522_522789

theorem rectangle_two_diagonal_corners {rect : Type} (corner : rect) (domino : rect → (fin 2 → fin 2 → Prop)) :
  (∃ (n : ℕ), ∀ (d : rect), domino d (fin.mk 2 n) ∧ domino d (fin.mk 1 n)) →
  (∀ (d₁ d₂ : rect), d₁ ≠ d₂ → ¬(∃ p : rect, domino d₁ p ∧ domino d₂ p)) →
  ∃ corners : finset rect, corners.card = 2 ∧ (∀ d : rect, ∃ c ∈ corners, domino d c) :=
by
  sorry

end rectangle_two_diagonal_corners_l522_522789


namespace same_graph_eq2_eq3_different_graph_eq1_eq2_eq3_l522_522411

noncomputable def eq1 (x : ℝ) := x - 2
noncomputable def eq2 (x : ℝ) := (x - 2) * (x - 2) / (x + 2)
noncomputable def eq3 (x : ℝ) := (x - 2) * (x - 2)

theorem same_graph_eq2_eq3 :
  ∀ x : ℝ, x ≠ -2 -> (eq2 x = eq3 x / (x + 2)) :=
by
  intros
  unfold eq2 eq3
  rw [div_eq_mul_inv]
  rw [mul_comm (x - 2) (x - 2)]
  rw [←mul_assoc]
  rw [mul_inv_cancel] ;
  try { ring }
  assumption

theorem different_graph_eq1_eq2_eq3 :
  ∃ x : ℝ, eq1 x ≠ eq2 x ∧ eq1 x ≠ eq3 x / (x + 2) :=
by
  use 0
  unfold eq1 eq2 eq3
  simp
  split
  all_goals { norm_num }
  done

end same_graph_eq2_eq3_different_graph_eq1_eq2_eq3_l522_522411


namespace team_points_behind_l522_522620

theorem team_points_behind :
  let max_points : ℕ := 7 in
  let dulce_points : ℕ := 5 in
  let combined_points : ℕ := max_points + dulce_points in
  let val_points : ℕ := 4 * combined_points in
  let team_points : ℕ := max_points + dulce_points + val_points in
  let opponents_points : ℕ := 80 in
  opponents_points - team_points = 20 :=
by
  sorry

end team_points_behind_l522_522620


namespace researchers_distribution_l522_522038

theorem researchers_distribution : 
  ∃ (dist_schemes : ℕ), dist_schemes = 36 ∧
  ∀ (R : Type) (S : Type), 
    (R = Fin 4) → (S = Fin 3) →
    (∀ (r : R) (s : S), ∃ (groups : Finset (Finset R)), groups.card = 3 ∧ 
    (∀ g ∈ groups, ∃ s ∈ (Finset.univ : Finset S), g.card ≥ 1)) :=
by
  sorry

end researchers_distribution_l522_522038


namespace math_problem_l522_522251

theorem math_problem
  (x : ℕ) (y : ℕ)
  (h1 : x = (Finset.range (60 + 1 + 1) \ Finset.range 50).sum id)
  (h2 : y = ((Finset.range (60 + 1) \ Finset.range 50).filter (λ n => n % 2 = 0)).card)
  (h3 : x + y = 611) :
  (Finset.range (60 + 1 + 1) \ Finset.range 50).sum id = 605 ∧
  ((Finset.range (60 + 1) \ Finset.range 50).filter (λ n => n % 2 = 0)).card = 6 := 
by
  sorry

end math_problem_l522_522251


namespace find_f_sqrt8_sqrt2_l522_522825

def f (x y : ℝ) : ℝ :=
  (x^2 - y^2) / (Int.floor (x^2 / 2) * 2 - Int.floor (y^2 / 2) * 2)

theorem find_f_sqrt8_sqrt2 (z : ℤ) (hz : z > 0) (h : (√8)^2 + (√2)^2 = z^2) :
  f (√8) (√2) = 1 :=
by
  sorry

end find_f_sqrt8_sqrt2_l522_522825


namespace polynomial_form_l522_522510

noncomputable def fourth_degree_polynomial : Type := 
  {p : ℝ → ℝ // (∀ x : ℝ, p x = p (-x)) ∧
               (∀ x : ℝ, p x ≥ 0) ∧ 
               (p 0 = 1) ∧ 
               ∃ x1 x2 : ℝ, (x1 ≠ 0 ∨ x2 ≠ 0) ∧ |x1 - x2| = 2 ∧ 
                            (∀ x : ℝ, p x ≥ p x1 ∧ p x ≥ p x2)}

theorem polynomial_form (p : fourth_degree_polynomial) : 
  ∃ a : ℝ, (0 < a ∧ a ≤ 1) ∧ 
           ∀ x : ℝ, subtype.val p x = a * (x^2 - 1) ^ 2 + 1 - a := 
sorry

end polynomial_form_l522_522510


namespace inequality_sqrt_ab_l522_522157

theorem inequality_sqrt_ab :
  ∀ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c →
  sqrt (a * b * (a + b)) + sqrt (b * c * (b + c)) + sqrt (c * a * (c + a)) >
  sqrt ((a + b) * (b + c) * (c + a)) :=
by
  intros a b c ha hb hc
  sorry

end inequality_sqrt_ab_l522_522157


namespace coral_third_week_pages_l522_522101

theorem coral_third_week_pages :
  let total_pages := 600
  let week1_read := total_pages / 2
  let remaining_after_week1 := total_pages - week1_read
  let week2_read := remaining_after_week1 * 0.30
  let remaining_after_week2 := remaining_after_week1 - week2_read
  remaining_after_week2 = 210 :=
by
  sorry

end coral_third_week_pages_l522_522101


namespace monotonicity_intervals_max_m_value_l522_522565

noncomputable def f (x : ℝ) : ℝ :=  (3 / 2) * x^2 - 3 * Real.log x

theorem monotonicity_intervals :
  (∀ x > (1:ℝ), ∃ ε > (0:ℝ), ∀ y, x < y → y < x + ε → f x < f y)
  ∧ (∀ x, (0:ℝ) < x → x < (1:ℝ) → ∃ ε > (0:ℝ), ∀ y, x - ε < y → y < x → f y < f x) :=
by sorry

theorem max_m_value (m : ℤ) (h : ∀ x > (1:ℝ), f (x * Real.log x + 2 * x - 1) > f (↑m * (x - 1))) :
  m ≤ 4 :=
by sorry

end monotonicity_intervals_max_m_value_l522_522565


namespace sufficient_but_not_necessary_condition_l522_522303

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x^2 - 1 = 0 → x^3 - x = 0) ∧ ¬ (x^3 - x = 0 → x^2 - 1 = 0) := by
  sorry

end sufficient_but_not_necessary_condition_l522_522303


namespace extremal_graph_eq_turan_l522_522866

theorem extremal_graph_eq_turan (r n : ℕ) (G : SimpleGraph (Fin n)) [DecidableRel G.Adj] 
(h_r : r > 1) (h_ex : G.edgeCount = ex(n, completeGraph r)) 
(h_not_contains_Kr : ¬ G.Contains (completeGraph r)) : 
  G = turanGraph (r-1) n := 
sorry

end extremal_graph_eq_turan_l522_522866


namespace josh_money_left_l522_522286

theorem josh_money_left (initial_amount : ℝ) (first_spend : ℝ) (second_spend : ℝ) 
  (h1 : initial_amount = 9) 
  (h2 : first_spend = 1.75) 
  (h3 : second_spend = 1.25) : 
  initial_amount - first_spend - second_spend = 6 := 
by 
  sorry

end josh_money_left_l522_522286


namespace sum_of_diagonals_approx_l522_522041

-- Definitions for the given problem
def inscribed_hexagon (a b c d e f : Point) : Prop :=
  are_cyclic {a, b, c, d, e, f}

def sides_lengths (a b c d e f : Point) : Prop :=
  dist a b = 40 ∧ dist b c = 100 ∧ dist c d = 100 ∧ dist d e = 100 ∧ dist e f = 100 ∧ dist f a = 100

-- The tuple of conditions and the proof
theorem sum_of_diagonals_approx (a b c d e f : Point)
  (h1 : inscribed_hexagon a b c d e f)
  (h2 : sides_lengths a b c d e f) :
  dist a c + dist a d + dist a e ≈ 376.22 :=
by
  sorry

end sum_of_diagonals_approx_l522_522041


namespace heating_time_correct_l522_522629

def initial_temp : ℤ := 20

def desired_temp : ℤ := 100

def heating_rate : ℤ := 5

def time_to_heat (initial desired rate : ℤ) : ℤ :=
  (desired - initial) / rate

theorem heating_time_correct :
  time_to_heat initial_temp desired_temp heating_rate = 16 :=
by
  sorry

end heating_time_correct_l522_522629


namespace num_perfect_cubes_between_l522_522936

theorem num_perfect_cubes_between :
  let lower_bound := 3^7 + 1 
  let upper_bound := 3^{15} + 1 
  let smallest_perfect_cube_above_lower := 13^3
  let largest_perfect_cube_below_upper := 243^3
  lower_bound < smallest_perfect_cube_above_lower ∧ lower_bound < upper_bound ∧ upper_bound > largest_perfect_cube_below_upper → 
  (∀ n : ℕ, (smallest_perfect_cube_above_lower ≤ n * n * n ∧ n * n * n ≤ largest_perfect_cube_below_upper) ↔ (13 ≤ n ∧ n ≤ 243)) → 
  ∃ (k : ℕ), k = 243 - 13 + 1 ∧ k = 231 :=
by {
  sorry
}

end num_perfect_cubes_between_l522_522936


namespace common_ratio_is_one_eighth_l522_522991

-- Define the geometric sequence terms a1 and a2
def a1 : ℝ := 64
def a2 : ℝ := 8

-- Define the common ratio q and state that it is equal to a2 / a1
def q : ℝ := a2 / a1

-- The theorem to prove that q is 1/8 given the conditions
theorem common_ratio_is_one_eighth (h1 : a1 = 64) (h2 : a2 = 8) : q = 1 / 8 :=
by
  sorry

end common_ratio_is_one_eighth_l522_522991


namespace find_m_l522_522884

open Nat

def is_arithmetic (a : ℕ → ℤ) (n : ℕ) : Prop := ∀ i < n - 1, a (i + 2) - a (i + 1) = a (i + 1) - a i
def is_geometric (a : ℕ → ℤ) (n : ℕ) : Prop := ∀ i ≥ n, a (i + 1) * a n = a i * a (n + 1)
def sum_prod_condition (a : ℕ → ℤ) (m : ℕ) : Prop := a m + a (m + 1) + a (m + 2) = a m * a (m + 1) * a (m + 2)

theorem find_m (a : ℕ → ℤ)
  (h1 : a 3 = -1)
  (h2 : a 7 = 4)
  (h3 : is_arithmetic a 6)
  (h4 : is_geometric a 5) :
  ∃ m : ℕ, m = 1 ∨ m = 3 ∧ sum_prod_condition a m := sorry

end find_m_l522_522884


namespace price_increase_equivalence_l522_522277

theorem price_increase_equivalence (P : ℝ) : 
  let increase_35 := P * 1.35
  let increase_40 := increase_35 * 1.40
  let increase_20 := increase_40 * 1.20
  let final_increase := increase_20
  final_increase = P * 2.268 :=
by
  -- proof skipped
  sorry

end price_increase_equivalence_l522_522277


namespace factorial_fraction_l522_522595

theorem factorial_fraction (n : ℕ) (hn : (99! : ℚ) / (101! - 99!) = 1 / n) : n = 10099 :=
sorry

end factorial_fraction_l522_522595


namespace ratio_A_to_B_investment_l522_522799

variable (A B C : Type) [Field A] [Field B] [Field C]
variable (investA investB investC profit total_profit : A) 

-- Conditions
axiom A_invests_some_times_as_B : ∃ n : A, investA = n * investB
axiom B_invests_two_thirds_of_C : investB = (2/3) * investC
axiom total_profit_statement : total_profit = 3300
axiom B_share_statement : profit = 600

-- Theorem: Ratio of A's investment to B's investment is 3:1
theorem ratio_A_to_B_investment : ∃ n : A, investA = 3 * investB :=
sorry

end ratio_A_to_B_investment_l522_522799


namespace maximum_value_of_A_l522_522210

noncomputable def maximum_A (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  (∑ k in Finset.range n, real.sqrt (real.sin (x k))) * 
  (∑ k in Finset.range n, real.sqrt (real.cos (x k)))

theorem maximum_value_of_A (x : ℕ → ℝ) (n : ℕ) (h : ∀ k < n, x k ∈ Icc 0 (real.pi / 2)):
  maximum_A x n ≤ n^2 * real.sqrt 2⁻¹ ∧ 
  (∀ k < n, x k = real.pi / 4) → maximum_A x n = n^2 * real.sqrt 2⁻¹ :=
sorry

end maximum_value_of_A_l522_522210


namespace loaned_books_value_l522_522417

-- Definitions for conditions
def initial_books : ℕ := 75
def end_books : ℕ := 57
def return_rate : ℝ := 0.70

-- Definition for loaned books
def loaned_books : ℝ := initial_books - end_books / (1 - return_rate)

-- Theorem stating the asked question
theorem loaned_books_value : loaned_books = 60 := by
  sorry

end loaned_books_value_l522_522417


namespace minimum_value_l522_522856

noncomputable def f (y : ℝ) := y^2 + 9 * y + 81 / y^3

theorem minimum_value : ∃ y > 0, f y = 39 ∧ ∀ z > 0, f z ≥ 39 :=
by {
    have h := real.sqrt_pos.mpr (by norm_num : (6:ℝ) > 0),
    sorry
}

end minimum_value_l522_522856


namespace probability_four_ones_in_fifteen_dice_l522_522508

noncomputable def diceProbability : ℝ :=
  (coe (Nat.choose 15 4) : ℝ) * (1/6)^4 * (5/6)^11

theorem probability_four_ones_in_fifteen_dice : abs (diceProbability - 0.202) < 0.001 := 
  sorry

end probability_four_ones_in_fifteen_dice_l522_522508


namespace domain_of_sqrt_function_l522_522559

theorem domain_of_sqrt_function : {x : ℝ | 2 - x ≥ 0} = set.Iic 2 := 
sorry

end domain_of_sqrt_function_l522_522559


namespace tenth_term_arithmetic_sequence_l522_522003

theorem tenth_term_arithmetic_sequence :
  ∀ (a₁ a₃₀ : ℕ) (d : ℕ) (n : ℕ), a₁ = 3 → a₃₀ = 89 → n = 10 → 
  (a₃₀ - a₁) / 29 = d → a₁ + (n - 1) * d = 30 :=
by
  intros a₁ a₃₀ d n h₁ h₃₀ hn hd
  sorry

end tenth_term_arithmetic_sequence_l522_522003


namespace average_speed_with_stoppages_l522_522014

theorem average_speed_with_stoppages
    (D : ℝ) -- distance the train travels
    (T_no_stop : ℝ := D / 250) -- time taken to cover the distance without stoppages
    (T_with_stop : ℝ := 2 * T_no_stop) -- total time with stoppages
    : (D / T_with_stop) = 125 := 
by sorry

end average_speed_with_stoppages_l522_522014


namespace transform_complex_l522_522075

-- Define the initial complex number and transformations
def initial_complex : ℂ := -4 + 6 * complex.I

def rotate_60ccw (z : ℂ) : ℂ :=
  z * (1 / 2 + (real.sqrt 3) / 2 * complex.I)

def dilate_2 (z : ℂ) : ℂ :=
  2 * z

-- Define the final goal
theorem transform_complex :
  let rotated := rotate_60ccw initial_complex
      dilated := dilate_2 rotated
  in dilated = -22 + (6 - 4 * real.sqrt 3) * complex.I :=
by
  sorry

end transform_complex_l522_522075
