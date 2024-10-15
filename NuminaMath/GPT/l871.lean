import Mathlib

namespace NUMINAMATH_GPT_angle_x_is_36_l871_87178

theorem angle_x_is_36
    (x : ℝ)
    (h1 : 7 * x + 3 * x = 360)
    (h2 : 8 * x ≤ 360) :
    x = 36 := 
by {
  sorry
}

end NUMINAMATH_GPT_angle_x_is_36_l871_87178


namespace NUMINAMATH_GPT_probability_defective_unit_l871_87113

theorem probability_defective_unit (T : ℝ) 
  (P_A : ℝ := 9 / 1000) 
  (P_B : ℝ := 1 / 50) 
  (output_ratio_A : ℝ := 0.4)
  (output_ratio_B : ℝ := 0.6) : 
  (P_A * output_ratio_A + P_B * output_ratio_B) = 0.0156 :=
by
  sorry

end NUMINAMATH_GPT_probability_defective_unit_l871_87113


namespace NUMINAMATH_GPT_dot_product_sum_eq_fifteen_l871_87199

-- Define the vectors a, b, and c
def vec_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vec_b (y : ℝ) : ℝ × ℝ := (1, y)
def vec_c : ℝ × ℝ := (3, -6)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Conditions from the problem
def cond_perpendicular (x : ℝ) : Prop :=
  dot_product (vec_a x) vec_c = 0

def cond_parallel (y : ℝ) : Prop :=
  1 / 3 = y / -6

-- Lean statement for the problem
theorem dot_product_sum_eq_fifteen (x y : ℝ)
  (h1 : cond_perpendicular x) 
  (h2 : cond_parallel y) :
  dot_product (vec_a x + vec_b y) vec_c = 15 :=
sorry

end NUMINAMATH_GPT_dot_product_sum_eq_fifteen_l871_87199


namespace NUMINAMATH_GPT_unique_solution_exists_l871_87142

theorem unique_solution_exists (n m k : ℕ) :
  n = m^3 ∧ n = 1000 * m + k ∧ 0 ≤ k ∧ k < 1000 ∧ (1000 * m ≤ m^3 ∧ m^3 < 1000 * (m + 1)) → n = 32768 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_exists_l871_87142


namespace NUMINAMATH_GPT_inequality_abc_l871_87140

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b + b * c + c * a ≤ 1) :
  a + b + c + Real.sqrt 3 ≥ 8 * a * b * c * (1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_abc_l871_87140


namespace NUMINAMATH_GPT_exactly_one_pair_probability_l871_87103

def four_dice_probability : ℚ :=
  sorry  -- Here we skip the actual computation and proof

theorem exactly_one_pair_probability : four_dice_probability = 5/9 := by {
  -- Placeholder for proof, explanation, and calculation
  sorry
}

end NUMINAMATH_GPT_exactly_one_pair_probability_l871_87103


namespace NUMINAMATH_GPT_intersection_points_l871_87101

noncomputable def even_function (f : ℝ → ℝ) :=
  ∀ x, f (-x) = f x

def monotonically_increasing (f : ℝ → ℝ) :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

theorem intersection_points (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_mono_inc : monotonically_increasing f)
  (h_sign_change : f 1 * f 2 < 0) :
  ∃! x1 x2 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 :=
sorry

end NUMINAMATH_GPT_intersection_points_l871_87101


namespace NUMINAMATH_GPT_cookies_per_person_l871_87106

-- Definitions based on conditions
def cookies_total : ℕ := 144
def people_count : ℕ := 6

-- The goal is to prove the number of cookies per person
theorem cookies_per_person : cookies_total / people_count = 24 :=
by
  sorry

end NUMINAMATH_GPT_cookies_per_person_l871_87106


namespace NUMINAMATH_GPT_max_value_f_on_interval_l871_87158

def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 1

theorem max_value_f_on_interval : 
  ∀ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f x ≤ 15 :=
by
  sorry

end NUMINAMATH_GPT_max_value_f_on_interval_l871_87158


namespace NUMINAMATH_GPT_mean_of_xyz_l871_87185

theorem mean_of_xyz (a b c d e f g x y z : ℝ)
  (h1 : (a + b + c + d + e + f + g) / 7 = 48)
  (h2 : (a + b + c + d + e + f + g + x + y + z) / 10 = 55) :
  (x + y + z) / 3 = 71.33333333333333 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_xyz_l871_87185


namespace NUMINAMATH_GPT_sum_of_first_5n_l871_87147

theorem sum_of_first_5n (n : ℕ) (h : (3 * n) * (3 * n + 1) / 2 = n * (n + 1) / 2 + 270) : (5 * n) * (5 * n + 1) / 2 = 820 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_5n_l871_87147


namespace NUMINAMATH_GPT_find_third_number_l871_87131

theorem find_third_number (first_number second_number third_number : ℕ) 
  (h1 : first_number = 200)
  (h2 : first_number + second_number + third_number = 500)
  (h3 : second_number = 2 * third_number) :
  third_number = 100 := sorry

end NUMINAMATH_GPT_find_third_number_l871_87131


namespace NUMINAMATH_GPT_keiko_jogging_speed_l871_87151

variable (s : ℝ) -- Keiko's jogging speed
variable (b : ℝ) -- radius of the inner semicircle
variable (L_inner : ℝ := 200 + 2 * Real.pi * b) -- total length of the inner track
variable (L_outer : ℝ := 200 + 2 * Real.pi * (b + 8)) -- total length of the outer track
variable (t_inner : ℝ := L_inner / s) -- time to jog the inside edge
variable (t_outer : ℝ := L_outer / s) -- time to jog the outside edge
variable (time_difference : ℝ := 48) -- time difference between jogging inside and outside edges

theorem keiko_jogging_speed : L_inner = 200 + 2 * Real.pi * b →
                           L_outer = 200 + 2 * Real.pi * (b + 8) →
                           t_outer = t_inner + 48 →
                           s = Real.pi / 3 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_keiko_jogging_speed_l871_87151


namespace NUMINAMATH_GPT_tony_slices_remaining_l871_87100

def slices_remaining : ℕ :=
  let total_slices := 22
  let slices_per_sandwich := 2
  let daily_sandwiches := 1
  let extra_sandwiches := 1
  let total_sandwiches := 7 * daily_sandwiches + extra_sandwiches
  total_slices - total_sandwiches * slices_per_sandwich

theorem tony_slices_remaining :
  slices_remaining = 6 :=
by
  -- Here, the theorem proof would go, but it is omitted as per instructions
  sorry

end NUMINAMATH_GPT_tony_slices_remaining_l871_87100


namespace NUMINAMATH_GPT_g_g_2_eq_78652_l871_87161

def g (x : ℝ) : ℝ := 4 * x^3 - 3 * x + 1

theorem g_g_2_eq_78652 : g (g 2) = 78652 := by
  sorry

end NUMINAMATH_GPT_g_g_2_eq_78652_l871_87161


namespace NUMINAMATH_GPT_range_of_a_l871_87194

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - (a^2 + a) * x + a^3 > 0 ↔ (x < a^2 ∨ x > a)) → (0 ≤ a ∧ a ≤ 1) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_range_of_a_l871_87194


namespace NUMINAMATH_GPT_lcm_fractions_l871_87116

theorem lcm_fractions (x : ℕ) (hx : x ≠ 0) : 
  (∀ (a b c : ℕ), (a = 4*x ∧ b = 5*x ∧ c = 6*x) → (Nat.lcm (Nat.lcm a b) c = 60 * x)) :=
by
  sorry

end NUMINAMATH_GPT_lcm_fractions_l871_87116


namespace NUMINAMATH_GPT_vector_BC_coordinates_l871_87144

-- Define the given vectors
def vec_AB : ℝ × ℝ := (2, -1)
def vec_AC : ℝ × ℝ := (-4, 1)

-- Define the vector subtraction
def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

-- Define the vector BC as the result of the subtraction
def vec_BC : ℝ × ℝ := vec_sub vec_AC vec_AB

-- State the theorem
theorem vector_BC_coordinates : vec_BC = (-6, 2) := by
  sorry

end NUMINAMATH_GPT_vector_BC_coordinates_l871_87144


namespace NUMINAMATH_GPT_hallie_number_of_paintings_sold_l871_87188

/-- 
Hallie is an artist. She wins an art contest, and she receives a $150 prize. 
She sells some of her paintings for $50 each. 
She makes a total of $300 from her art. 
How many paintings did she sell?
-/
theorem hallie_number_of_paintings_sold 
    (prize : ℕ)
    (price_per_painting : ℕ)
    (total_earnings : ℕ)
    (prize_eq : prize = 150)
    (price_eq : price_per_painting = 50)
    (total_eq : total_earnings = 300) :
    (total_earnings - prize) / price_per_painting = 3 :=
by
  sorry

end NUMINAMATH_GPT_hallie_number_of_paintings_sold_l871_87188


namespace NUMINAMATH_GPT_problem_statement_l871_87118

noncomputable def f (x : ℝ) : ℝ := ∫ t in -x..x, Real.cos t

theorem problem_statement : f (f (Real.pi / 4)) = 2 * Real.sin (Real.sqrt 2) := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l871_87118


namespace NUMINAMATH_GPT_anna_money_left_l871_87124

theorem anna_money_left : 
  let initial_money := 10.0
  let gum_cost := 3.0 -- 3 packs at $1.00 each
  let chocolate_cost := 5.0 -- 5 bars at $1.00 each
  let cane_cost := 1.0 -- 2 canes at $0.50 each
  let total_spent := gum_cost + chocolate_cost + cane_cost
  let money_left := initial_money - total_spent
  money_left = 1.0 := by
  sorry

end NUMINAMATH_GPT_anna_money_left_l871_87124


namespace NUMINAMATH_GPT_no_adjacent_standing_probability_l871_87173

noncomputable def probability_no_adjacent_standing : ℚ := 
  let total_outcomes := 2 ^ 10
  let favorable_outcomes := 123
  favorable_outcomes / total_outcomes

theorem no_adjacent_standing_probability :
  probability_no_adjacent_standing = 123 / 1024 := by
  sorry

end NUMINAMATH_GPT_no_adjacent_standing_probability_l871_87173


namespace NUMINAMATH_GPT_lcm_of_ratio_and_hcf_l871_87117

theorem lcm_of_ratio_and_hcf (a b : ℕ) (h1 : a = 3 * 8) (h2 : b = 4 * 8) (h3 : Nat.gcd a b = 8) : Nat.lcm a b = 96 :=
  sorry

end NUMINAMATH_GPT_lcm_of_ratio_and_hcf_l871_87117


namespace NUMINAMATH_GPT_depth_second_project_l871_87183

def volume (depth length breadth : ℝ) : ℝ := depth * length * breadth

theorem depth_second_project (D : ℝ) : 
  (volume 100 25 30 = volume D 20 50) → D = 75 :=
by 
  sorry

end NUMINAMATH_GPT_depth_second_project_l871_87183


namespace NUMINAMATH_GPT_volume_of_cube_l871_87129

theorem volume_of_cube (A : ℝ) (s V : ℝ) 
  (hA : A = 150) 
  (h_surface_area : A = 6 * s^2) 
  (h_side_length : s = 5) :
  V = s^3 →
  V = 125 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_cube_l871_87129


namespace NUMINAMATH_GPT_fractional_eq_has_positive_root_m_value_l871_87164

-- Define the conditions and the proof goal
theorem fractional_eq_has_positive_root_m_value (m x : ℝ) (h1 : x - 2 ≠ 0) (h2 : 2 - x ≠ 0) (h3 : ∃ x > 0, (m / (x - 2)) = ((1 - x) / (2 - x)) - 3) : m = 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_fractional_eq_has_positive_root_m_value_l871_87164


namespace NUMINAMATH_GPT_Janet_previous_movie_length_l871_87163

theorem Janet_previous_movie_length (L : ℝ) (H1 : 1.60 * L = 1920 / 100) : L / 60 = 0.20 :=
by
  sorry

end NUMINAMATH_GPT_Janet_previous_movie_length_l871_87163


namespace NUMINAMATH_GPT_abs_diff_of_sum_and_product_l871_87137

theorem abs_diff_of_sum_and_product (x y : ℝ) (h1 : x + y = 20) (h2 : x * y = 96) : |x - y| = 4 := 
by
  sorry

end NUMINAMATH_GPT_abs_diff_of_sum_and_product_l871_87137


namespace NUMINAMATH_GPT_isosceles_triangle_base_angles_l871_87162

theorem isosceles_triangle_base_angles (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A = B ∨ B = C ∨ C = A) (h₃ : A = 80 ∨ B = 80 ∨ C = 80) :
  A = 50 ∨ B = 50 ∨ C = 50 ∨ A = 80 ∨ B = 80 ∨ C = 80 := 
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_angles_l871_87162


namespace NUMINAMATH_GPT_A_three_two_l871_87111

def A : ℕ → ℕ → ℕ
| 0, n => n + 1
| m+1, 0 => A m 2
| m+1, n+1 => A m (A (m + 1) n)

theorem A_three_two : A 3 2 = 5 := 
by 
  sorry

end NUMINAMATH_GPT_A_three_two_l871_87111


namespace NUMINAMATH_GPT_decagon_not_divided_properly_l871_87190

theorem decagon_not_divided_properly :
  ∀ (n m : ℕ),
  (∃ black white : Finset ℕ, ∀ b ∈ black, ∀ w ∈ white,
    (b + w = 10) ∧ (b % 3 = 0) ∧ (w % 3 = 0)) →
  n - m = 10 → (n % 3 = 0) ∧ (m % 3 = 0) → 10 % 3 = 0 → False :=
by
  sorry

end NUMINAMATH_GPT_decagon_not_divided_properly_l871_87190


namespace NUMINAMATH_GPT_songs_after_operations_l871_87160

-- Definitions based on conditions
def initialSongs : ℕ := 15
def deletedSongs : ℕ := 8
def addedSongs : ℕ := 50

-- Problem statement to be proved
theorem songs_after_operations : initialSongs - deletedSongs + addedSongs = 57 :=
by
  sorry

end NUMINAMATH_GPT_songs_after_operations_l871_87160


namespace NUMINAMATH_GPT_perfect_square_trinomial_m_l871_87152

theorem perfect_square_trinomial_m (m : ℝ) :
  (∃ a : ℝ, (x^2 + 2 * (m - 1) * x + 4) = (x + a)^2) → (m = 3 ∨ m = -1) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_m_l871_87152


namespace NUMINAMATH_GPT_variance_of_temperatures_l871_87135

def temperatures : List ℕ := [28, 21, 22, 26, 28, 25]

noncomputable def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

noncomputable def variance (l : List ℕ) : ℚ :=
  let m := mean l
  (l.map (λ x => (x - m)^2)).sum / l.length

theorem variance_of_temperatures : variance temperatures = 22 / 3 := 
by
  sorry

end NUMINAMATH_GPT_variance_of_temperatures_l871_87135


namespace NUMINAMATH_GPT_sum_of_base5_numbers_l871_87141

-- Definitions for the numbers in base 5
def n1_base5 := (1 * 5^2 + 3 * 5^1 + 2 * 5^0 : ℕ)
def n2_base5 := (2 * 5^2 + 1 * 5^1 + 4 * 5^0 : ℕ)
def n3_base5 := (3 * 5^2 + 4 * 5^1 + 1 * 5^0 : ℕ)

-- Sum the numbers in base 10
def sum_base10 := n1_base5 + n2_base5 + n3_base5

-- Define the base 5 value of the sum
def sum_base5 := 
  -- Convert the sum to base 5
  1 * 5^3 + 2 * 5^2 + 4 * 5^1 + 2 * 5^0

-- The theorem we want to prove
theorem sum_of_base5_numbers :
    (132 + 214 + 341 : ℕ) = 1242 := by
    sorry

end NUMINAMATH_GPT_sum_of_base5_numbers_l871_87141


namespace NUMINAMATH_GPT_precise_approximate_classification_l871_87126

def data_points : List String := ["Xiao Ming bought 5 books today",
                                  "The war in Afghanistan cost the United States $1 billion per month in 2002",
                                  "Relevant departments predict that in 2002, the sales of movies in DVD format will exceed those of VHS tapes for the first time, reaching $9.5 billion",
                                  "The human brain has 10,000,000,000 cells",
                                  "Xiao Hong scored 92 points on this test",
                                  "The Earth has more than 1.5 trillion tons of coal reserves"]

def is_precise (data : String) : Bool :=
  match data with
  | "Xiao Ming bought 5 books today" => true
  | "The war in Afghanistan cost the United States $1 billion per month in 2002" => true
  | "Relevant departments predict that in 2002, the sales of movies in DVD format will exceed those of VHS tapes for the first time, reaching $9.5 billion" => true
  | "Xiao Hong scored 92 points on this test" => true
  | _ => false

def is_approximate (data : String) : Bool :=
  match data with
  | "The human brain has 10,000,000,000 cells" => true
  | "The Earth has more than 1.5 trillion tons of coal reserves" => true
  | _ => false

theorem precise_approximate_classification :
  (data_points.filter is_precise = ["Xiao Ming bought 5 books today",
                                    "The war in Afghanistan cost the United States $1 billion per month in 2002",
                                    "Relevant departments predict that in 2002, the sales of movies in DVD format will exceed those of VHS tapes for the first time, reaching $9.5 billion",
                                    "Xiao Hong scored 92 points on this test"]) ∧
  (data_points.filter is_approximate = ["The human brain has 10,000,000,000 cells",
                                        "The Earth has more than 1.5 trillion tons of coal reserves"]) :=
by sorry

end NUMINAMATH_GPT_precise_approximate_classification_l871_87126


namespace NUMINAMATH_GPT_sally_reads_10_pages_on_weekdays_l871_87175

def sallyReadsOnWeekdays (x : ℕ) (total_pages : ℕ) (weekdays : ℕ) (weekend_days : ℕ) (weekend_pages : ℕ) : Prop :=
  (weekdays + weekend_days * weekend_pages = total_pages) → (weekdays * x = total_pages - weekend_days * weekend_pages)

theorem sally_reads_10_pages_on_weekdays :
  sallyReadsOnWeekdays 10 180 10 4 20 :=
by
  intros h
  sorry  -- proof to be filled in

end NUMINAMATH_GPT_sally_reads_10_pages_on_weekdays_l871_87175


namespace NUMINAMATH_GPT_floor_e_eq_2_l871_87187

noncomputable def e_approx : ℝ := 2.71828

theorem floor_e_eq_2 : ⌊e_approx⌋ = 2 :=
sorry

end NUMINAMATH_GPT_floor_e_eq_2_l871_87187


namespace NUMINAMATH_GPT_minimum_value_at_zero_l871_87191

noncomputable def f (x : ℝ) : ℝ := (x - 1) * Real.exp (x - 1)

theorem minimum_value_at_zero : ∀ x : ℝ, f 0 ≤ f x :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_at_zero_l871_87191


namespace NUMINAMATH_GPT_find_x_l871_87120

-- Define the vectors a and b
def a : ℝ × ℝ := (-2, 3)
def b (x : ℝ) : ℝ × ℝ := (x, -3)

-- Define the parallel condition between (b - a) and b
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u  = (k * v.1, k * v.2)

-- The problem statement in Lean 4
theorem find_x (x : ℝ) (h : parallel (b x - a) (b x)) : x = 2 := 
  sorry

end NUMINAMATH_GPT_find_x_l871_87120


namespace NUMINAMATH_GPT_cubic_eq_has_real_roots_l871_87168

theorem cubic_eq_has_real_roots (K : ℝ) (hK : K ≠ 0) : ∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) * (x - 3) :=
by sorry

end NUMINAMATH_GPT_cubic_eq_has_real_roots_l871_87168


namespace NUMINAMATH_GPT_factorization_of_polynomial_l871_87154

theorem factorization_of_polynomial : 
  ∀ (x : ℝ), 18 * x^3 + 9 * x^2 + 3 * x = 3 * x * (6 * x^2 + 3 * x + 1) :=
by sorry

end NUMINAMATH_GPT_factorization_of_polynomial_l871_87154


namespace NUMINAMATH_GPT_increase_by_multiplication_l871_87193

theorem increase_by_multiplication (n : ℕ) (h : n = 14) : (15 * n) - n = 196 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_increase_by_multiplication_l871_87193


namespace NUMINAMATH_GPT_second_route_time_l871_87115

-- Defining time for the first route with all green lights
def R_green : ℕ := 10

-- Defining the additional time added by each red light
def per_red_light : ℕ := 3

-- Defining total time for the first route with all red lights
def R_red : ℕ := R_green + 3 * per_red_light

-- Defining the second route time plus the difference
def S : ℕ := R_red - 5

theorem second_route_time : S = 14 := by
  sorry

end NUMINAMATH_GPT_second_route_time_l871_87115


namespace NUMINAMATH_GPT_point_T_coordinates_l871_87132

-- Definition of a point in 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Definition of a square with specific points O, P, Q, R
structure Square where
  O : Point
  P : Point
  Q : Point
  R : Point

-- Condition: O is the origin
def O : Point := {x := 0, y := 0}

-- Condition: Q is at (3, 3)
def Q : Point := {x := 3, y := 3}

-- Assuming the function area_triang for calculating the area of a triangle given three points
def area_triang (A B C : Point) : ℝ :=
  0.5 * abs ((B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x))

-- Assuming the function area_square for calculating the area of a square given the length of the side
def area_square (s : ℝ) : ℝ := s * s

-- Coordinates of point P and R since it's a square with sides parallel to axis
def P : Point := {x := 3, y := 0}
def R : Point := {x := 0, y := 3}

-- Definition of the square OPQR
def OPQR : Square := {O := O, P := P, Q := Q, R := R}

-- Length of the side of square OPQR
def side_length : ℝ := 3

-- Area of the square OPQR
def square_area : ℝ := area_square side_length

-- Twice the area of the square OPQR
def required_area : ℝ := 2 * square_area

-- Point T that needs to be proven
def T : Point := {x := 3, y := 12}

-- The main theorem to prove
theorem point_T_coordinates (T : Point) : area_triang P Q T = required_area → T = {x := 3, y := 12} :=
by
  sorry

end NUMINAMATH_GPT_point_T_coordinates_l871_87132


namespace NUMINAMATH_GPT_interest_rate_is_five_percent_l871_87170

-- Define the principal amount P and the interest rate r.
variables (P : ℝ) (r : ℝ)

-- Define the conditions given in the problem
def simple_interest_condition : Prop := P * r * 2 = 40
def compound_interest_condition : Prop := P * (1 + r)^2 - P = 41

-- Define the goal statement to prove
theorem interest_rate_is_five_percent (h1 : simple_interest_condition P r) (h2 : compound_interest_condition P r) : r = 0.05 :=
sorry

end NUMINAMATH_GPT_interest_rate_is_five_percent_l871_87170


namespace NUMINAMATH_GPT_prime_sum_mod_eighth_l871_87176

theorem prime_sum_mod_eighth (p1 p2 p3 p4 p5 p6 p7 p8 : ℕ) 
  (h₁ : p1 = 2) 
  (h₂ : p2 = 3) 
  (h₃ : p3 = 5) 
  (h₄ : p4 = 7) 
  (h₅ : p5 = 11) 
  (h₆ : p6 = 13) 
  (h₇ : p7 = 17) 
  (h₈ : p8 = 19) : 
  ((p1 + p2 + p3 + p4 + p5 + p6 + p7) % p8) = 1 :=
by
  sorry

end NUMINAMATH_GPT_prime_sum_mod_eighth_l871_87176


namespace NUMINAMATH_GPT_green_peaches_in_each_basket_l871_87102

theorem green_peaches_in_each_basket (G : ℕ) 
  (h1 : ∀ B : ℕ, B = 15) 
  (h2 : ∀ R : ℕ, R = 19) 
  (h3 : ∀ P : ℕ, P = 345) 
  (h_eq : 345 = 15 * (19 + G)) : 
  G = 4 := by
  sorry

end NUMINAMATH_GPT_green_peaches_in_each_basket_l871_87102


namespace NUMINAMATH_GPT_smallest_positive_multiple_l871_87155

theorem smallest_positive_multiple (a : ℕ) (h : 17 * a % 53 = 7) : 17 * a = 544 :=
sorry

end NUMINAMATH_GPT_smallest_positive_multiple_l871_87155


namespace NUMINAMATH_GPT_sale_in_second_month_l871_87125

theorem sale_in_second_month 
  (sale_first_month: ℕ := 2500)
  (sale_third_month: ℕ := 3540)
  (sale_fourth_month: ℕ := 1520)
  (average_sale: ℕ := 2890)
  (total_sales: ℕ := 11560) :
  sale_first_month + sale_third_month + sale_fourth_month + (sale_second_month: ℕ) = total_sales → 
  sale_second_month = 4000 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_sale_in_second_month_l871_87125


namespace NUMINAMATH_GPT_intersection_A_B_l871_87166

open Set

def A : Set ℝ := {1, 2, 1/2}
def B : Set ℝ := {y | ∃ x ∈ A, y = x^2}

theorem intersection_A_B : A ∩ B = { 1 } := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l871_87166


namespace NUMINAMATH_GPT_N_eq_M_union_P_l871_87148

def M : Set ℝ := {x : ℝ | ∃ (n : ℤ), x = n}
def N : Set ℝ := {x : ℝ | ∃ (n : ℤ), x = n / 2}
def P : Set ℝ := {x : ℝ | ∃ (n : ℤ), x = n + 1 / 2}

theorem N_eq_M_union_P : N = M ∪ P :=
  sorry

end NUMINAMATH_GPT_N_eq_M_union_P_l871_87148


namespace NUMINAMATH_GPT_krista_driving_hours_each_day_l871_87150

-- Define the conditions as constants
def road_trip_days : ℕ := 3
def jade_hours_per_day : ℕ := 8
def total_hours : ℕ := 42

-- Define the function to calculate Krista's hours per day
noncomputable def krista_hours_per_day : ℕ :=
  (total_hours - road_trip_days * jade_hours_per_day) / road_trip_days

-- State the theorem to prove Krista drove 6 hours each day
theorem krista_driving_hours_each_day : krista_hours_per_day = 6 := by
  sorry

end NUMINAMATH_GPT_krista_driving_hours_each_day_l871_87150


namespace NUMINAMATH_GPT_inequality_AM_GM_l871_87104

variable {a b c d : ℝ}
variable (h₁ : 0 < a)
variable (h₂ : 0 < b)
variable (h₃ : 0 < c)
variable (h₄ : 0 < d)

theorem inequality_AM_GM :
  (c / a * (8 * b + c) + d / b * (8 * c + d) + a / c * (8 * d + a) + b / d * (8 * a + b)) ≥ 9 * (a + b + c + d) :=
sorry

end NUMINAMATH_GPT_inequality_AM_GM_l871_87104


namespace NUMINAMATH_GPT_total_height_of_three_buildings_l871_87108

theorem total_height_of_three_buildings :
  let h1 := 600
  let h2 := 2 * h1
  let h3 := 3 * (h1 + h2)
  h1 + h2 + h3 = 7200 :=
by
  sorry

end NUMINAMATH_GPT_total_height_of_three_buildings_l871_87108


namespace NUMINAMATH_GPT_jake_fewer_peaches_than_steven_l871_87138

theorem jake_fewer_peaches_than_steven :
  ∀ (jill steven jake : ℕ),
    jill = 87 →
    steven = jill + 18 →
    jake = jill + 13 →
    steven - jake = 5 :=
by
  intros jill steven jake hjill hsteven hjake
  sorry

end NUMINAMATH_GPT_jake_fewer_peaches_than_steven_l871_87138


namespace NUMINAMATH_GPT_ilya_defeats_dragon_l871_87180

noncomputable def prob_defeat : ℝ := 1 / 4 * 2 + 1 / 3 * 1 + 5 / 12 * 0

theorem ilya_defeats_dragon : prob_defeat = 1 := sorry

end NUMINAMATH_GPT_ilya_defeats_dragon_l871_87180


namespace NUMINAMATH_GPT_georgie_window_ways_l871_87172

theorem georgie_window_ways (n : Nat) (h : n = 8) :
  let ways := n * (n - 1)
  ways = 56 := by
  sorry

end NUMINAMATH_GPT_georgie_window_ways_l871_87172


namespace NUMINAMATH_GPT_convex_octagon_min_obtuse_l871_87121

-- Define a type for a polygon (here specifically an octagon)
structure Polygon (n : ℕ) :=
(vertices : ℕ)
(convex : Prop)

-- Define that an octagon is a specific polygon with 8 vertices
def octagon : Polygon 8 :=
{ vertices := 8,
  convex := sorry }

-- Define the predicate for convex polygons
def is_convex (poly : Polygon 8) : Prop := poly.convex

-- Defining the statement that a convex octagon has at least 5 obtuse interior angles
theorem convex_octagon_min_obtuse (poly : Polygon 8) (h : is_convex poly) : ∃ (n : ℕ), n = 5 :=
sorry

end NUMINAMATH_GPT_convex_octagon_min_obtuse_l871_87121


namespace NUMINAMATH_GPT_directrix_parabola_l871_87165

theorem directrix_parabola (x y : ℝ) (h : x^2 = 8 * y) : y = -2 :=
sorry

end NUMINAMATH_GPT_directrix_parabola_l871_87165


namespace NUMINAMATH_GPT_cookies_left_l871_87130

theorem cookies_left (total_cookies : ℕ) (total_neighbors : ℕ) (cookies_per_neighbor : ℕ) (sarah_cookies : ℕ)
  (h1 : total_cookies = 150)
  (h2 : total_neighbors = 15)
  (h3 : cookies_per_neighbor = 10)
  (h4 : sarah_cookies = 12) :
  total_cookies - ((total_neighbors - 1) * cookies_per_neighbor + sarah_cookies) = 8 :=
by
  simp [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_cookies_left_l871_87130


namespace NUMINAMATH_GPT_complement_of_intersection_l871_87146

open Set

def A : Set ℝ := {x | x^2 - x - 6 ≤ 0}
def B : Set ℝ := {x | x^2 + x - 6 > 0}
def S : Set ℝ := univ -- S is the set of all real numbers

theorem complement_of_intersection :
  S \ (A ∩ B) = { x : ℝ | x ≤ 2 } ∪ { x : ℝ | 3 < x } :=
by
  sorry

end NUMINAMATH_GPT_complement_of_intersection_l871_87146


namespace NUMINAMATH_GPT_binomial_theorem_example_l871_87167

theorem binomial_theorem_example 
  (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
  (h1 : (2 - 1)^5 = a_0 + a_1 * 1 + a_2 * 1^2 + a_3 * 1^3 + a_4 * 1^4 + a_5 * 1^5)
  (h2 : (2 - (-1))^5 = a_0 - a_1 + a_2 * (-1)^2 - a_3 * (-1)^3 + a_4 * (-1)^4 - a_5 * (-1)^5)
  (h3 : a_5 = -1) :
  (a_0 + a_2 + a_4 : ℤ) / (a_1 + a_3 : ℤ) = -61 / 60 := 
sorry

end NUMINAMATH_GPT_binomial_theorem_example_l871_87167


namespace NUMINAMATH_GPT_red_marbles_l871_87197

theorem red_marbles (R B : ℕ) (h₁ : B = R + 24) (h₂ : B = 5 * R) : R = 6 := by
  sorry

end NUMINAMATH_GPT_red_marbles_l871_87197


namespace NUMINAMATH_GPT_sarah_calculate_profit_l871_87134

noncomputable def sarah_total_profit (hot_day_price : ℚ) (regular_day_price : ℚ) (cost_per_cup : ℚ) (cups_per_day : ℕ) (hot_days : ℕ) (total_days : ℕ) : ℚ := 
  let hot_day_revenue := hot_day_price * cups_per_day * hot_days
  let regular_day_revenue := regular_day_price * cups_per_day * (total_days - hot_days)
  let total_revenue := hot_day_revenue + regular_day_revenue
  let total_cost := cost_per_cup * cups_per_day * total_days
  total_revenue - total_cost

theorem sarah_calculate_profit : 
  let hot_day_price := (20951704545454546 : ℚ) / 10000000000000000
  let regular_day_price := hot_day_price / 1.25
  let cost_per_cup := 75 / 100
  let cups_per_day := 32
  let hot_days := 4
  let total_days := 10
  sarah_total_profit hot_day_price regular_day_price cost_per_cup cups_per_day hot_days total_days = (34935102 : ℚ) / 10000000 :=
by
  sorry

end NUMINAMATH_GPT_sarah_calculate_profit_l871_87134


namespace NUMINAMATH_GPT_volume_increase_l871_87153

theorem volume_increase (L B H : ℝ) :
  let L_new := 1.25 * L
  let B_new := 0.85 * B
  let H_new := 1.10 * H
  (L_new * B_new * H_new) = 1.16875 * (L * B * H) := 
by
  sorry

end NUMINAMATH_GPT_volume_increase_l871_87153


namespace NUMINAMATH_GPT_glasses_per_pitcher_l871_87127

def total_glasses : Nat := 30
def num_pitchers : Nat := 6

theorem glasses_per_pitcher : total_glasses / num_pitchers = 5 := by
  sorry

end NUMINAMATH_GPT_glasses_per_pitcher_l871_87127


namespace NUMINAMATH_GPT_cards_per_page_l871_87133

theorem cards_per_page 
  (new_cards old_cards pages : ℕ) 
  (h1 : new_cards = 8)
  (h2 : old_cards = 10)
  (h3 : pages = 6) : (new_cards + old_cards) / pages = 3 := 
by 
  sorry

end NUMINAMATH_GPT_cards_per_page_l871_87133


namespace NUMINAMATH_GPT_inequality_a_b_c_l871_87143

theorem inequality_a_b_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^3 / (a^2 + a * b + b^2)) + (b^3 / (b^2 + b * c + c^2)) + (c^3 / (c^2 + c * a + a^2)) ≥ (a + b + c) / 3 :=
sorry

end NUMINAMATH_GPT_inequality_a_b_c_l871_87143


namespace NUMINAMATH_GPT_donny_remaining_money_l871_87145

theorem donny_remaining_money :
  let initial_amount := 78
  let kite_cost := 8
  let frisbee_cost := 9
  initial_amount - (kite_cost + frisbee_cost) = 61 :=
by
  sorry

end NUMINAMATH_GPT_donny_remaining_money_l871_87145


namespace NUMINAMATH_GPT_zoo_tickets_total_cost_l871_87119

-- Define the given conditions
def num_children := 6
def num_adults := 10
def cost_child_ticket := 10
def cost_adult_ticket := 16

-- Calculate the expected total cost
def total_cost := 220

-- State the theorem
theorem zoo_tickets_total_cost :
  num_children * cost_child_ticket + num_adults * cost_adult_ticket = total_cost :=
by
  sorry

end NUMINAMATH_GPT_zoo_tickets_total_cost_l871_87119


namespace NUMINAMATH_GPT_all_numbers_positive_l871_87186

noncomputable def condition (a : Fin 9 → ℝ) : Prop :=
  ∀ (S : Finset (Fin 9)), S.card = 4 → S.sum (a : Fin 9 → ℝ) < (Finset.univ \ S).sum (a : Fin 9 → ℝ)

theorem all_numbers_positive (a : Fin 9 → ℝ) (h : condition a) : ∀ i, 0 < a i :=
by
  sorry

end NUMINAMATH_GPT_all_numbers_positive_l871_87186


namespace NUMINAMATH_GPT_three_segments_form_triangle_l871_87196

def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem three_segments_form_triangle :
  ¬ can_form_triangle 2 3 6 ∧
  ¬ can_form_triangle 2 4 6 ∧
  ¬ can_form_triangle 2 2 4 ∧
    can_form_triangle 6 6 6 :=
by
  repeat {sorry}

end NUMINAMATH_GPT_three_segments_form_triangle_l871_87196


namespace NUMINAMATH_GPT_chef_initial_eggs_l871_87136

-- Define the conditions
def eggs_in_fridge := 10
def eggs_per_cake := 5
def cakes_made := 10

-- Prove that the number of initial eggs is 60
theorem chef_initial_eggs : (eggs_per_cake * cakes_made + eggs_in_fridge) = 60 :=
by
  sorry

end NUMINAMATH_GPT_chef_initial_eggs_l871_87136


namespace NUMINAMATH_GPT_paper_fold_ratio_l871_87169

theorem paper_fold_ratio (paper_side : ℕ) (fold_fraction : ℚ) (cut_fraction : ℚ)
  (thin_section_width thick_section_width : ℕ) (small_width large_width : ℚ)
  (P_small P_large : ℚ) (ratio : ℚ) :
  paper_side = 6 →
  fold_fraction = 1 / 3 →
  cut_fraction = 2 / 3 →
  thin_section_width = 2 →
  thick_section_width = 4 →
  small_width = 2 →
  large_width = 16 / 3 →
  P_small = 2 * (6 + small_width) →
  P_large = 2 * (6 + large_width) →
  ratio = P_small / P_large →
  ratio = 12 / 17 :=
by
  sorry

end NUMINAMATH_GPT_paper_fold_ratio_l871_87169


namespace NUMINAMATH_GPT_trains_clear_in_correct_time_l871_87110

noncomputable def time_to_clear (length1 length2 : ℝ) (speed1_kmph speed2_kmph : ℝ) : ℝ :=
  let speed1_mps := speed1_kmph * 1000 / 3600
  let speed2_mps := speed2_kmph * 1000 / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := length1 + length2
  total_distance / relative_speed

-- The lengths of the trains
def length1 : ℝ := 151
def length2 : ℝ := 165

-- The speeds of the trains in km/h
def speed1_kmph : ℝ := 80
def speed2_kmph : ℝ := 65

-- The correct answer
def correct_time : ℝ := 7.844

theorem trains_clear_in_correct_time :
  time_to_clear length1 length2 speed1_kmph speed2_kmph = correct_time :=
by
  -- Skipping proof
  sorry

end NUMINAMATH_GPT_trains_clear_in_correct_time_l871_87110


namespace NUMINAMATH_GPT_polygon_sides_eq_five_l871_87149

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem polygon_sides_eq_five (n : ℕ) (h : n - number_of_diagonals n = 0) : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_eq_five_l871_87149


namespace NUMINAMATH_GPT_solve_eq_2_pow_x_plus_3_pow_y_eq_z_sq_l871_87159

theorem solve_eq_2_pow_x_plus_3_pow_y_eq_z_sq (x y z : ℕ) :
  ((x = 3 ∧ y = 0 ∧ z = 3) ∨ (x = 0 ∧ y = 1 ∧ z = 2) ∨ (x = 4 ∧ y = 2 ∧ z = 5)) →
  2^x + 3^y = z^2 :=
by
  sorry

end NUMINAMATH_GPT_solve_eq_2_pow_x_plus_3_pow_y_eq_z_sq_l871_87159


namespace NUMINAMATH_GPT_lizard_problem_theorem_l871_87123

def lizard_problem : Prop :=
  ∃ (E W S : ℕ), 
  E = 3 ∧ 
  W = 3 * E ∧ 
  S = 7 * W ∧ 
  (S + W) - E = 69

theorem lizard_problem_theorem : lizard_problem :=
by
  sorry

end NUMINAMATH_GPT_lizard_problem_theorem_l871_87123


namespace NUMINAMATH_GPT_average_salary_l871_87174

theorem average_salary (A B C D E : ℕ) (hA : A = 8000) (hB : B = 5000) (hC : C = 14000) (hD : D = 7000) (hE : E = 9000) :
  (A + B + C + D + E) / 5 = 8800 :=
by
  -- the proof will be inserted here
  sorry

end NUMINAMATH_GPT_average_salary_l871_87174


namespace NUMINAMATH_GPT_carnival_wait_time_l871_87192

theorem carnival_wait_time :
  ∀ (T : ℕ), 4 * 60 = 4 * 30 + T + 4 * 15 → T = 60 :=
by
  intro T
  intro h
  sorry

end NUMINAMATH_GPT_carnival_wait_time_l871_87192


namespace NUMINAMATH_GPT_day_of_week_50th_day_of_year_N_minus_1_l871_87171

def day_of_week (d : ℕ) (first_day : ℕ) : ℕ :=
  (first_day + d - 1) % 7

theorem day_of_week_50th_day_of_year_N_minus_1 
  (N : ℕ) 
  (day_250_N : ℕ) 
  (day_150_N_plus_1 : ℕ) 
  (h1 : day_250_N = 3)  -- 250th day of year N is Wednesday (3rd day of week, 0 = Sunday)
  (h2 : day_150_N_plus_1 = 3) -- 150th day of year N+1 is also Wednesday (3rd day of week, 0 = Sunday)
  : day_of_week 50 (day_of_week 1 ((day_of_week 1 day_250_N - 1 + 250) % 365 - 1 + 366)) = 6 := 
sorry

-- Explanation:
-- day_of_week function calculates the day of the week given the nth day of the year and the first day of the year.
-- Given conditions that 250th day of year N and 150th day of year N+1 are both Wednesdays (represented by 3 assuming Sunday = 0).
-- We need to derive that the 50th day of year N-1 is a Saturday (represented by 6 assuming Sunday = 0).

end NUMINAMATH_GPT_day_of_week_50th_day_of_year_N_minus_1_l871_87171


namespace NUMINAMATH_GPT_current_population_l871_87179

theorem current_population (initial_population deaths_leaving_percentage : ℕ) (current_population : ℕ) :
  initial_population = 3161 → deaths_leaving_percentage = 5 →
  deaths_leaving_percentage / 100 * initial_population + deaths_leaving_percentage * (initial_population - deaths_leaving_percentage / 100 * initial_population) / 100 = initial_population - current_population →
  current_population = 2553 :=
 by
  sorry

end NUMINAMATH_GPT_current_population_l871_87179


namespace NUMINAMATH_GPT_find_a_l871_87195

theorem find_a (a x : ℝ) : 
  ((x + a)^2 / (3 * x + 65) = 2) 
  ∧ (∃ x1 x2 : ℝ,  x1 ≠ x2 ∧ (x1 = x2 + 22 ∨ x2 = x1 + 22 )) 
  → a = 3 := 
sorry

end NUMINAMATH_GPT_find_a_l871_87195


namespace NUMINAMATH_GPT_pascal_triangle_fifth_number_l871_87177

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end NUMINAMATH_GPT_pascal_triangle_fifth_number_l871_87177


namespace NUMINAMATH_GPT_system_of_equations_l871_87128

theorem system_of_equations (x y : ℝ) (h1 : 3 * x + 210 = 5 * y) (h2 : 10 * y - 10 * x = 100) :
    (3 * x + 210 = 5 * y) ∧ (10 * y - 10 * x = 100) := by
  sorry

end NUMINAMATH_GPT_system_of_equations_l871_87128


namespace NUMINAMATH_GPT_cos_C_value_l871_87105

theorem cos_C_value (a b c : ℝ) (A B C : ℝ)
  (h1 : a * Real.cos B + b * Real.cos A = 3 * c * Real.cos C)
  (h2 : 0 < A ∧ A < π)
  (h3 : 0 < B ∧ B < π)
  (h4 : 0 < C ∧ C < π)
  (h5 : A + B + C = π)
  : Real.cos C = (Real.sqrt 10) / 10 :=
sorry

end NUMINAMATH_GPT_cos_C_value_l871_87105


namespace NUMINAMATH_GPT_edmonton_to_red_deer_distance_l871_87156

noncomputable def distance_from_Edmonton_to_Calgary (speed time: ℝ) : ℝ :=
  speed * time

theorem edmonton_to_red_deer_distance :
  let speed := 110
  let time := 3
  let distance_Calgary_RedDeer := 110
  let distance_Edmonton_Calgary := distance_from_Edmonton_to_Calgary speed time
  let distance_Edmonton_RedDeer := distance_Edmonton_Calgary - distance_Calgary_RedDeer
  distance_Edmonton_RedDeer = 220 :=
by
  sorry

end NUMINAMATH_GPT_edmonton_to_red_deer_distance_l871_87156


namespace NUMINAMATH_GPT_Jed_older_than_Matt_l871_87112

-- Definitions of ages and conditions
def Jed_current_age : ℕ := sorry
def Matt_current_age : ℕ := sorry
axiom condition1 : Jed_current_age + 10 = 25
axiom condition2 : Jed_current_age + Matt_current_age = 20

-- Proof statement
theorem Jed_older_than_Matt : Jed_current_age - Matt_current_age = 10 :=
by
  sorry

end NUMINAMATH_GPT_Jed_older_than_Matt_l871_87112


namespace NUMINAMATH_GPT_equal_probability_after_adding_balls_l871_87122

theorem equal_probability_after_adding_balls :
  let initial_white := 2
  let initial_yellow := 3
  let added_white := 4
  let added_yellow := 3
  let total_white := initial_white + added_white
  let total_yellow := initial_yellow + added_yellow
  let total_balls := total_white + total_yellow
  (total_white / total_balls) = (total_yellow / total_balls) := by
  sorry

end NUMINAMATH_GPT_equal_probability_after_adding_balls_l871_87122


namespace NUMINAMATH_GPT_xiao_ming_total_evaluation_score_l871_87182

theorem xiao_ming_total_evaluation_score 
  (regular midterm final : ℤ) (weight_regular weight_midterm weight_final : ℕ)
  (h1 : regular = 80)
  (h2 : midterm = 90)
  (h3 : final = 85)
  (h_weight_regular : weight_regular = 3)
  (h_weight_midterm : weight_midterm = 3)
  (h_weight_final : weight_final = 4) :
  (regular * weight_regular + midterm * weight_midterm + final * weight_final) /
    (weight_regular + weight_midterm + weight_final) = 85 :=
by
  sorry

end NUMINAMATH_GPT_xiao_ming_total_evaluation_score_l871_87182


namespace NUMINAMATH_GPT_evaluate_polynomial_at_3_using_horners_method_l871_87157

def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

theorem evaluate_polynomial_at_3_using_horners_method : f 3 = 1641 := by
 sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_3_using_horners_method_l871_87157


namespace NUMINAMATH_GPT_selection_methods_l871_87198

-- Define the number of students and lectures.
def numberOfStudents : Nat := 6
def numberOfLectures : Nat := 5

-- Define the problem as proving the number of selection methods equals 5^6.
theorem selection_methods : (numberOfLectures ^ numberOfStudents) = 15625 := by
  -- Include the proper mathematical equivalence statement
  sorry

end NUMINAMATH_GPT_selection_methods_l871_87198


namespace NUMINAMATH_GPT_base8_perfect_square_b_zero_l871_87189

-- Define the base 8 representation and the perfect square condition
def base8_to_decimal (a b : ℕ) : ℕ := 512 * a + 64 + 8 * b + 4

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- The main theorem stating that if the number in base 8 is a perfect square, then b = 0
theorem base8_perfect_square_b_zero (a b : ℕ) (h₀ : a ≠ 0) 
  (h₁ : is_perfect_square (base8_to_decimal a b)) : b = 0 :=
sorry

end NUMINAMATH_GPT_base8_perfect_square_b_zero_l871_87189


namespace NUMINAMATH_GPT_hyperbola_center_l871_87139

def is_midpoint (x1 y1 x2 y2 xc yc : ℝ) : Prop :=
  xc = (x1 + x2) / 2 ∧ yc = (y1 + y2) / 2

theorem hyperbola_center :
  is_midpoint 2 (-3) (-4) 5 (-1) 1 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_center_l871_87139


namespace NUMINAMATH_GPT_find_integer_n_l871_87114

theorem find_integer_n :
  ∃ (n : ℤ), -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * Real.pi / 180) = Real.cos (675 * Real.pi / 180) ∧ n = 45 :=
sorry

end NUMINAMATH_GPT_find_integer_n_l871_87114


namespace NUMINAMATH_GPT_numbers_not_expressed_l871_87107

theorem numbers_not_expressed (a b : ℕ) (hb : 0 < b) (ha : 0 < a) :
 ∀ n : ℕ, (¬ ∃ a b : ℕ, n = a / b + (a + 1) / (b + 1) ∧ 0 < b ∧ 0 < a) ↔ (n = 1 ∨ ∃ m : ℕ, n = 2^m + 2) := 
by 
  sorry

end NUMINAMATH_GPT_numbers_not_expressed_l871_87107


namespace NUMINAMATH_GPT_dot_product_AB_AC_dot_product_AB_BC_l871_87109

-- The definition of equilateral triangle with side length 6
structure EquilateralTriangle (A B C : Type*) :=
  (side_len : ℝ)
  (angle_ABC : ℝ)
  (angle_BCA : ℝ)
  (angle_CAB : ℝ)
  (AB_len : ℝ)
  (AC_len : ℝ)
  (BC_len : ℝ)
  (AB_eq_AC : AB_len = AC_len)
  (AB_eq_BC : AB_len = BC_len)
  (cos_ABC : ℝ)
  (cos_BCA : ℝ)
  (cos_CAB : ℝ)

-- Given an equilateral triangle with side length 6 where the angles are defined,
-- we can define the specific triangle
noncomputable def triangleABC (A B C : Type*) : EquilateralTriangle A B C :=
{ side_len := 6,
  angle_ABC := 120,
  angle_BCA := 60,
  angle_CAB := 60,
  AB_len := 6,
  AC_len := 6,
  BC_len := 6,
  AB_eq_AC := rfl,
  AB_eq_BC := rfl,
  cos_ABC := -0.5,
  cos_BCA := 0.5,
  cos_CAB := 0.5 }

-- Prove the dot product of vectors AB and AC
theorem dot_product_AB_AC (A B C : Type*) 
  (T : EquilateralTriangle A B C) : 
  (T.AB_len * T.AC_len * T.cos_BCA) = 18 :=
by sorry

-- Prove the dot product of vectors AB and BC
theorem dot_product_AB_BC (A B C : Type*) 
  (T : EquilateralTriangle A B C) : 
  (T.AB_len * T.BC_len * T.cos_ABC) = -18 :=
by sorry

end NUMINAMATH_GPT_dot_product_AB_AC_dot_product_AB_BC_l871_87109


namespace NUMINAMATH_GPT_smallestThreeDigitNumberWithPerfectSquare_l871_87181

def isThreeDigitNumber (a : ℕ) : Prop := 100 ≤ a ∧ a ≤ 999

def formsPerfectSquare (a : ℕ) : Prop := ∃ n : ℕ, 1001 * a + 1 = n * n

theorem smallestThreeDigitNumberWithPerfectSquare :
  ∀ a : ℕ, isThreeDigitNumber a → formsPerfectSquare a → a = 183 :=
by
sorry

end NUMINAMATH_GPT_smallestThreeDigitNumberWithPerfectSquare_l871_87181


namespace NUMINAMATH_GPT_Dave_tiles_210_square_feet_l871_87184

theorem Dave_tiles_210_square_feet
  (ratio_charlie_dave : ℕ := 5 / 7)
  (total_area : ℕ := 360)
  : ∀ (work_done_by_dave : ℕ), work_done_by_dave = 210 :=
by
  sorry

end NUMINAMATH_GPT_Dave_tiles_210_square_feet_l871_87184
