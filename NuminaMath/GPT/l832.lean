import Mathlib

namespace NUMINAMATH_GPT_arithmetic_sqrt_of_4_eq_2_l832_83227

theorem arithmetic_sqrt_of_4_eq_2 (x : ℕ) (h : x^2 = 4) : x = 2 :=
sorry

end NUMINAMATH_GPT_arithmetic_sqrt_of_4_eq_2_l832_83227


namespace NUMINAMATH_GPT_side_length_of_octagon_l832_83257

-- Define the conditions
def is_octagon (n : ℕ) := n = 8
def perimeter (p : ℕ) := p = 72

-- Define the problem statement
theorem side_length_of_octagon (n p l : ℕ) 
  (h1 : is_octagon n) 
  (h2 : perimeter p) 
  (h3 : p / n = l) :
  l = 9 := 
  sorry

end NUMINAMATH_GPT_side_length_of_octagon_l832_83257


namespace NUMINAMATH_GPT_rectangle_side_length_along_hypotenuse_l832_83288

-- Define the right triangle with given sides
def triangle_PQR (PR PQ QR : ℝ) : Prop := 
  PR^2 + PQ^2 = QR^2

-- Condition: Right triangle PQR with PR = 9 and PQ = 12
def PQR : Prop := triangle_PQR 9 12 (Real.sqrt (9^2 + 12^2))

-- Define the property of the rectangle
def rectangle_condition (x : ℝ) (s : ℝ) : Prop := 
  (3 / (Real.sqrt (9^2 + 12^2))) = (x / 9) ∧ s = ((9 - x) * (Real.sqrt (9^2 + 12^2)) / 9)

-- Main theorem
theorem rectangle_side_length_along_hypotenuse : 
  PQR ∧ (∃ x, rectangle_condition x 12) → (∃ s, s = 12) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_rectangle_side_length_along_hypotenuse_l832_83288


namespace NUMINAMATH_GPT_problem200_squared_minus_399_composite_l832_83225

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  ¬ is_prime n

theorem problem200_squared_minus_399_composite : is_composite (200^2 - 399) :=
sorry

end NUMINAMATH_GPT_problem200_squared_minus_399_composite_l832_83225


namespace NUMINAMATH_GPT_oranges_after_selling_l832_83233

-- Definitions derived from the conditions
def oranges_picked := 37
def oranges_sold := 10
def oranges_left := 27

-- The theorem to prove that Joan is left with 27 oranges
theorem oranges_after_selling (h : oranges_picked - oranges_sold = oranges_left) : oranges_left = 27 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_oranges_after_selling_l832_83233


namespace NUMINAMATH_GPT_train_speed_l832_83267

noncomputable def trainLength : ℕ := 400
noncomputable def timeToCrossPole : ℕ := 20

theorem train_speed : (trainLength / timeToCrossPole) = 20 := by
  sorry

end NUMINAMATH_GPT_train_speed_l832_83267


namespace NUMINAMATH_GPT_find_divisor_l832_83202

theorem find_divisor (n x y z a b c : ℕ) (h1 : 63 = n * x + a) (h2 : 91 = n * y + b) (h3 : 130 = n * z + c) (h4 : a + b + c = 26) : n = 43 :=
sorry

end NUMINAMATH_GPT_find_divisor_l832_83202


namespace NUMINAMATH_GPT_unique_peg_placement_l832_83226

noncomputable def peg_placement := 
  ∃! f : (Fin 6 → Fin 6 → Option (Fin 5)), 
    (∀ i j, f i j = some 0 → (∀ k, k ≠ i → f k j ≠ some 0) ∧ (∀ l, l ≠ j → f i l ≠ some 0)) ∧  -- Yellow pegs
    (∀ i j, f i j = some 1 → (∀ k, k ≠ i → f k j ≠ some 1) ∧ (∀ l, l ≠ j → f i l ≠ some 1)) ∧  -- Red pegs
    (∀ i j, f i j = some 2 → (∀ k, k ≠ i → f k j ≠ some 2) ∧ (∀ l, l ≠ j → f i l ≠ some 2)) ∧  -- Green pegs
    (∀ i j, f i j = some 3 → (∀ k, k ≠ i → f k j ≠ some 3) ∧ (∀ l, l ≠ j → f i l ≠ some 3)) ∧  -- Blue pegs
    (∀ i j, f i j = some 4 → (∀ k, k ≠ i → f k j ≠ some 4) ∧ (∀ l, l ≠ j → f i l ≠ some 4)) ∧  -- Orange pegs
    (∃! i j, f i j = some 0) ∧
    (∃! i j, f i j = some 1) ∧
    (∃! i j, f i j = some 2) ∧
    (∃! i j, f i j = some 3) ∧
    (∃! i j, f i j = some 4)
    
theorem unique_peg_placement : peg_placement :=
sorry

end NUMINAMATH_GPT_unique_peg_placement_l832_83226


namespace NUMINAMATH_GPT_Doug_age_l832_83232

theorem Doug_age (Q J D : ℕ) (h1 : Q = J + 6) (h2 : J = D - 3) (h3 : Q = 19) : D = 16 := by
  sorry

end NUMINAMATH_GPT_Doug_age_l832_83232


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l832_83280

theorem ratio_of_x_to_y (x y : ℝ) (h : y = 0.20 * x) : x / y = 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_x_to_y_l832_83280


namespace NUMINAMATH_GPT_solve_system_of_equations_l832_83204

theorem solve_system_of_equations (x y z : ℝ) :
  (2 * x^2 / (1 + x^2) = y) →
  (2 * y^2 / (1 + y^2) = z) →
  (2 * z^2 / (1 + z^2) = x) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1)) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l832_83204


namespace NUMINAMATH_GPT_james_oranges_l832_83260

-- Define the problem conditions
variables (o a : ℕ) -- o is number of oranges, a is number of apples

-- Condition: James bought apples and oranges over a seven-day week
def days_week := o + a = 7

-- Condition: The total cost must be a whole number of dollars (divisible by 100 cents)
def total_cost := 65 * o + 40 * a ≡ 0 [MOD 100]

-- We need to prove: James bought 4 oranges
theorem james_oranges (o a : ℕ) (h_days_week : days_week o a) (h_total_cost : total_cost o a) : o = 4 :=
sorry

end NUMINAMATH_GPT_james_oranges_l832_83260


namespace NUMINAMATH_GPT_problem1_problem2_l832_83263

def vector_dot (v1 v2 : ℝ × ℝ) : ℝ := 
  v1.1 * v2.1 + v1.2 * v2.2

def perpendicular (v1 v2 : ℝ × ℝ) : Prop := 
  vector_dot v1 v2 = 0

def parallel (v1 v2 : ℝ × ℝ) : Prop := 
  v1.1 * v2.2 = v1.2 * v2.1

-- Given vectors in the problem
def a : ℝ × ℝ := (-3, 1)
def b : ℝ × ℝ := (1, -2)
def c : ℝ × ℝ := (1, -1)
def n (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)
def v : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)

-- Problem 1: Find k when n is perpendicular to v
theorem problem1 (k : ℝ) : perpendicular (n k) v → k = 5 / 3 := 
by sorry

-- Problem 2: Find k when n is parallel to c + k * b
theorem problem2 (k : ℝ) : parallel (n k) (c.1 + k * b.1, c.2 + k * b.2) → k = -1 / 3 := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_l832_83263


namespace NUMINAMATH_GPT_difference_between_wins_and_losses_l832_83282

noncomputable def number_of_wins (n m : ℕ) : Prop :=
  0 ≤ n ∧ 0 ≤ m ∧ n + m ≤ 42 ∧ n + (42 - n - m) / 2 = 30 / 1

theorem difference_between_wins_and_losses (n m : ℕ) (h : number_of_wins n m) : n - m = 18 :=
sorry

end NUMINAMATH_GPT_difference_between_wins_and_losses_l832_83282


namespace NUMINAMATH_GPT_price_of_each_sundae_l832_83292

theorem price_of_each_sundae (A B : ℝ) (x y z : ℝ) (hx : 200 * x = 80) (hy : A = y) (hz : y = 0.40)
  (hxy : A - 80 = z) (hyz : 200 * z = B) : y = 0.60 :=
by
  sorry

end NUMINAMATH_GPT_price_of_each_sundae_l832_83292


namespace NUMINAMATH_GPT_max_men_with_all_amenities_marrried_l832_83217

theorem max_men_with_all_amenities_marrried :
  let total_men := 100
  let married_men := 85
  let men_with_TV := 75
  let men_with_radio := 85
  let men_with_AC := 70
  (∀ s : Finset ℕ, s.card ≤ total_men) →
  (∀ s : Finset ℕ, s.card ≤ married_men) →
  (∀ s : Finset ℕ, s.card ≤ men_with_TV) →
  (∀ s : Finset ℕ, s.card ≤ men_with_radio) →
  (∀ s : Finset ℕ, s.card ≤ men_with_AC) →
  (∀ s : Finset ℕ, s.card ≤ min married_men (min men_with_TV (min men_with_radio men_with_AC))) :=
by
  intros
  sorry

end NUMINAMATH_GPT_max_men_with_all_amenities_marrried_l832_83217


namespace NUMINAMATH_GPT_m_perp_n_α_perp_β_l832_83243

variables {Plane Line : Type}
variables (α β : Plane) (m n : Line)

def perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry

-- Problem 1:
axiom m_perp_α : perpendicular_to_plane m α
axiom n_perp_β : perpendicular_to_plane n β
axiom α_perp_β : perpendicular_planes α β

theorem m_perp_n : perpendicular_lines m n :=
sorry

-- Problem 2:
axiom m_perp_n' : perpendicular_lines m n
axiom m_perp_α' : perpendicular_to_plane m α
axiom n_perp_β' : perpendicular_to_plane n β

theorem α_perp_β' : perpendicular_planes α β :=
sorry

end NUMINAMATH_GPT_m_perp_n_α_perp_β_l832_83243


namespace NUMINAMATH_GPT_sum_of_angles_subtended_by_arcs_l832_83219

theorem sum_of_angles_subtended_by_arcs
  (A B X Y C : Type)
  (arc_AX arc_XC : ℝ)
  (h1 : arc_AX = 58)
  (h2 : arc_XC = 62)
  (R S : ℝ)
  (hR : R = arc_AX / 2)
  (hS : S = arc_XC / 2) :
  R + S = 60 :=
by
  rw [hR, hS, h1, h2]
  norm_num

end NUMINAMATH_GPT_sum_of_angles_subtended_by_arcs_l832_83219


namespace NUMINAMATH_GPT_rectangle_hall_length_l832_83212

variable (L B : ℝ)

theorem rectangle_hall_length (h1 : B = (2 / 3) * L) (h2 : L * B = 2400) : L = 60 :=
by sorry

end NUMINAMATH_GPT_rectangle_hall_length_l832_83212


namespace NUMINAMATH_GPT_abs_ab_eq_2_sqrt_111_l832_83262

theorem abs_ab_eq_2_sqrt_111 (a b : ℝ) (h1 : b^2 - a^2 = 25) (h2 : a^2 + b^2 = 49) : |a * b| = 2 * Real.sqrt 111 := sorry

end NUMINAMATH_GPT_abs_ab_eq_2_sqrt_111_l832_83262


namespace NUMINAMATH_GPT_cows_eat_grass_l832_83240

theorem cows_eat_grass (ha_per_cow_per_week : ℝ) (ha_grow_per_week : ℝ) :
  (∀ (weeks_cows_weeks_ha : ℕ × ℕ × ℕ × ℕ), weeks_cows_weeks_ha = (2, 3, 2, 2) →
    (2 : ℝ) = 3 * 2 * ha_per_cow_per_week - 2 * ha_grow_per_week) → 
  (∀ (weeks_cows_weeks_ha : ℕ × ℕ × ℕ × ℕ), weeks_cows_weeks_ha = (4, 2, 4, 2) →
    (2 : ℝ) = 2 * 4 * ha_per_cow_per_week - 4 * ha_grow_per_week) → 
  ∃ (cows : ℕ), (6 : ℝ) = cows * 6 * ha_per_cow_per_week - 6 * ha_grow_per_week ∧ cows = 3 :=
sorry

end NUMINAMATH_GPT_cows_eat_grass_l832_83240


namespace NUMINAMATH_GPT_total_flour_required_l832_83247

-- Definitions specified based on the given conditions
def flour_already_put_in : ℕ := 10
def flour_needed : ℕ := 2

-- Lean 4 statement to prove the total amount of flour required by the recipe
theorem total_flour_required : (flour_already_put_in + flour_needed) = 12 :=
by
  sorry

end NUMINAMATH_GPT_total_flour_required_l832_83247


namespace NUMINAMATH_GPT_right_triangle_inequality_l832_83269

theorem right_triangle_inequality {a b c : ℝ} (h₁ : a^2 + b^2 = c^2) : a + b ≤ c * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_right_triangle_inequality_l832_83269


namespace NUMINAMATH_GPT_trajectory_equation_l832_83278

def fixed_point : ℝ × ℝ := (1, 2)

def moving_point (x y : ℝ) : ℝ × ℝ := (x, y)

def dot_product (p1 p2 : ℝ × ℝ) : ℝ :=
p1.1 * p2.1 + p1.2 * p2.2

theorem trajectory_equation (x y : ℝ) (h : dot_product (moving_point x y) fixed_point = 4) :
  x + 2 * y - 4 = 0 :=
sorry

end NUMINAMATH_GPT_trajectory_equation_l832_83278


namespace NUMINAMATH_GPT_circle_circumference_l832_83205

theorem circle_circumference (a b : ℝ) (h1 : a = 9) (h2 : b = 12) :
  ∃ c : ℝ, c = 15 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_circle_circumference_l832_83205


namespace NUMINAMATH_GPT_smallest_integer_in_range_l832_83297

theorem smallest_integer_in_range :
  ∃ n : ℕ, 
  1 < n ∧ 
  n % 3 = 2 ∧ 
  n % 5 = 2 ∧ 
  n % 7 = 2 ∧ 
  90 < n ∧ n < 119 :=
sorry

end NUMINAMATH_GPT_smallest_integer_in_range_l832_83297


namespace NUMINAMATH_GPT_digits_with_five_or_seven_is_5416_l832_83272

/-- The total number of four-digit positive integers. -/
def total_four_digit_integers : ℕ := 9000

/-- The number of four-digit integers without the digits 5 or 7. -/
def no_five_seven_integers : ℕ := 3584

/-- The number of four-digit positive integers that have at least one digit that is a 5 or a 7. -/
def digits_with_five_or_seven : ℕ :=
  total_four_digit_integers - no_five_seven_integers

/-- Proof that the number of four-digit integers with at least one digit that is a 5 or 7 is 5416. -/
theorem digits_with_five_or_seven_is_5416 :
  digits_with_five_or_seven = 5416 :=
by
  sorry

end NUMINAMATH_GPT_digits_with_five_or_seven_is_5416_l832_83272


namespace NUMINAMATH_GPT_max_value_of_vector_dot_product_l832_83270

theorem max_value_of_vector_dot_product :
  ∀ (x y : ℝ), (-2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2) → (2 * x - y ≤ 4) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_max_value_of_vector_dot_product_l832_83270


namespace NUMINAMATH_GPT_length_of_each_song_l832_83216

-- Conditions
def first_side_songs : Nat := 6
def second_side_songs : Nat := 4
def total_length_of_tape : Nat := 40

-- Definition of length of each song
def total_songs := first_side_songs + second_side_songs

-- Question: Prove that each song is 4 minutes long
theorem length_of_each_song (h1 : first_side_songs = 6) 
                            (h2 : second_side_songs = 4) 
                            (h3 : total_length_of_tape = 40) 
                            (h4 : total_songs = first_side_songs + second_side_songs) : 
  total_length_of_tape / total_songs = 4 :=
by
  sorry

end NUMINAMATH_GPT_length_of_each_song_l832_83216


namespace NUMINAMATH_GPT_fraction_meaningful_iff_l832_83284

theorem fraction_meaningful_iff (x : ℝ) : (∃ y, y = 1 / (x - 1)) ↔ x ≠ 1 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_meaningful_iff_l832_83284


namespace NUMINAMATH_GPT_tensor_12_9_l832_83274

def tensor (a b : ℚ) : ℚ := a + (4 * a) / (3 * b)

theorem tensor_12_9 : tensor 12 9 = 13 + 7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_tensor_12_9_l832_83274


namespace NUMINAMATH_GPT_consecutive_even_numbers_average_35_greatest_39_l832_83279

-- Defining the conditions of the problem
def average_of_even_numbers (n : ℕ) (S : ℕ) : ℕ := (n * S + (2 * n * (n - 1)) / 2) / n

-- Main statement to be proven
theorem consecutive_even_numbers_average_35_greatest_39 : 
  ∃ (n : ℕ), average_of_even_numbers n (38 - (n - 1) * 2) = 35 ∧ (38 - (n - 1) * 2) + (n - 1) * 2 = 38 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_even_numbers_average_35_greatest_39_l832_83279


namespace NUMINAMATH_GPT_seating_arrangement_l832_83244

theorem seating_arrangement (x y : ℕ) (h1 : 9 * x + 7 * y = 61) : x = 6 :=
by 
  sorry

end NUMINAMATH_GPT_seating_arrangement_l832_83244


namespace NUMINAMATH_GPT_years_of_school_eq_13_l832_83277

/-- Conditions definitions -/
def cost_per_semester : ℕ := 20000
def semesters_per_year : ℕ := 2
def total_cost : ℕ := 520000

/-- Derived definitions from conditions -/
def cost_per_year := cost_per_semester * semesters_per_year
def number_of_years := total_cost / cost_per_year

/-- Proof that number of years equals 13 given the conditions -/
theorem years_of_school_eq_13 : number_of_years = 13 :=
by sorry

end NUMINAMATH_GPT_years_of_school_eq_13_l832_83277


namespace NUMINAMATH_GPT_triangle_area_right_angled_l832_83246

theorem triangle_area_right_angled (a : ℝ) (h₁ : 0 < a) (h₂ : a < 24) :
  let b := 24
  let c := 48 - a
  (a^2 + b^2 = c^2) → (1/2) * a * b = 216 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_right_angled_l832_83246


namespace NUMINAMATH_GPT_inequality_solution_l832_83248

theorem inequality_solution {x : ℝ} (h : -2 < (x^2 - 18*x + 24) / (x^2 - 4*x + 8) ∧ (x^2 - 18*x + 24) / (x^2 - 4*x + 8) < 2) : 
  x ∈ Set.Ioo (-2 : ℝ) (10 / 3) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l832_83248


namespace NUMINAMATH_GPT_olivia_pieces_of_paper_l832_83296

theorem olivia_pieces_of_paper (initial_pieces : ℕ) (used_pieces : ℕ) (pieces_left : ℕ) 
  (h1 : initial_pieces = 81) (h2 : used_pieces = 56) : 
  pieces_left = 81 - 56 :=
by
  sorry

end NUMINAMATH_GPT_olivia_pieces_of_paper_l832_83296


namespace NUMINAMATH_GPT_problem_solution_l832_83285

/-- Let ⌊x⌋ denote the greatest integer less than or equal to x. Prove
    that the number of real solutions to the equation x² - 2⌊x⌋ - 3 = 0 is 3. -/
theorem problem_solution : ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, x^2 - 2 * ⌊x⌋ - 3 = 0 := 
sorry

end NUMINAMATH_GPT_problem_solution_l832_83285


namespace NUMINAMATH_GPT_total_cost_is_130_l832_83249

-- Defining the number of each type of pet
def n_puppies : ℕ := 2
def n_kittens : ℕ := 2
def n_parakeets : ℕ := 3

-- Defining the cost of one parakeet
def c_parakeet : ℕ := 10

-- Defining the cost of one puppy and one kitten based on the conditions
def c_puppy : ℕ := 3 * c_parakeet
def c_kitten : ℕ := 2 * c_parakeet

-- Defining the total cost of all pets
def total_cost : ℕ :=
  (n_puppies * c_puppy) + (n_kittens * c_kitten) + (n_parakeets * c_parakeet)

-- Lean theorem stating that the total cost is 130 dollars
theorem total_cost_is_130 : total_cost = 130 := by
  -- The proof will be filled in here.
  sorry

end NUMINAMATH_GPT_total_cost_is_130_l832_83249


namespace NUMINAMATH_GPT_problem_l832_83266

def f (x : ℤ) : ℤ := 7 * x - 3

theorem problem : f (f (f 3)) = 858 := by
  sorry

end NUMINAMATH_GPT_problem_l832_83266


namespace NUMINAMATH_GPT_jack_paid_20_l832_83293

-- Define the conditions
def numberOfSandwiches : Nat := 3
def costPerSandwich : Nat := 5
def changeReceived : Nat := 5

-- Define the total cost
def totalCost : Nat := numberOfSandwiches * costPerSandwich

-- Define the amount paid
def amountPaid : Nat := totalCost + changeReceived

-- Prove that the amount paid is 20
theorem jack_paid_20 : amountPaid = 20 := by
  -- You may assume the steps and calculations here, only providing the statement
  sorry

end NUMINAMATH_GPT_jack_paid_20_l832_83293


namespace NUMINAMATH_GPT_range_of_a_l832_83256

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x| ≥ a * x) → -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l832_83256


namespace NUMINAMATH_GPT_selling_price_l832_83223

theorem selling_price (cost_price : ℝ) (loss_percentage : ℝ) : 
    cost_price = 1600 → loss_percentage = 0.15 → 
    (cost_price - (loss_percentage * cost_price)) = 1360 :=
by
  intros h_cp h_lp
  rw [h_cp, h_lp]
  norm_num

end NUMINAMATH_GPT_selling_price_l832_83223


namespace NUMINAMATH_GPT_betty_berries_july_five_l832_83203
open Nat

def betty_bear_berries : Prop :=
  ∃ (b : ℕ), (5 * b + 100 = 150) ∧ (b + 40 = 50)

theorem betty_berries_july_five : betty_bear_berries :=
  sorry

end NUMINAMATH_GPT_betty_berries_july_five_l832_83203


namespace NUMINAMATH_GPT_minimum_disks_needed_l832_83229

-- Define the conditions
def total_files : ℕ := 25
def disk_capacity : ℝ := 2.0
def files_06MB : ℕ := 5
def size_06MB_file : ℝ := 0.6
def files_10MB : ℕ := 10
def size_10MB_file : ℝ := 1.0
def files_03MB : ℕ := total_files - files_06MB - files_10MB
def size_03MB_file : ℝ := 0.3

-- Define the theorem that needs to be proved
theorem minimum_disks_needed : 
    ∃ (disks: ℕ), disks = 10 ∧ 
    (5 * size_06MB_file + 10 * size_10MB_file + 10 * size_03MB_file) ≤ disks * disk_capacity := 
by
  sorry

end NUMINAMATH_GPT_minimum_disks_needed_l832_83229


namespace NUMINAMATH_GPT_printer_x_time_l832_83258

-- Define the basic parameters given in the problem
def job_time_printer_y := 12
def job_time_printer_z := 8
def ratio := 10 / 3

-- Work rates of the printers
def work_rate_y := 1 / job_time_printer_y
def work_rate_z := 1 / job_time_printer_z

-- Combined work rate and total time for printers Y and Z
def combined_work_rate_y_z := work_rate_y + work_rate_z
def time_printers_y_z := 1 / combined_work_rate_y_z

-- Given ratio relation
def time_printer_x := ratio * time_printers_y_z

-- Mathematical statement to prove: time it takes for printer X to do the job alone
theorem printer_x_time : time_printer_x = 16 := by
  sorry

end NUMINAMATH_GPT_printer_x_time_l832_83258


namespace NUMINAMATH_GPT_arithmetic_sequence_8th_term_is_71_l832_83298

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_8th_term_is_71_l832_83298


namespace NUMINAMATH_GPT_log_expression_as_product_l832_83237

noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_expression_as_product (A m n p : ℝ) (hm : 0 < m) (hn : 0 < n) (hp : 0 < p) (hA : 0 < A) :
  log m A * log n A + log n A * log p A + log p A * log m A =
  log A (m * n * p) * log p A * log n A * log m A :=
by
  sorry

end NUMINAMATH_GPT_log_expression_as_product_l832_83237


namespace NUMINAMATH_GPT_part_I_part_II_l832_83215

noncomputable def f (x a : ℝ) := 2 * |x - 1| - a
noncomputable def g (x m : ℝ) := - |x + m|

theorem part_I (a : ℝ) : 
  (∃! x : ℤ, x = -3 ∧ g x 3 > -1) → m = 3 := 
sorry

theorem part_II (m : ℝ) : 
  (∀ x : ℝ, f x a > g x m) → a < 4 := 
sorry

end NUMINAMATH_GPT_part_I_part_II_l832_83215


namespace NUMINAMATH_GPT_total_seeds_eaten_correct_l832_83207

-- Define the number of seeds each player ate
def seeds_first_player : ℕ := 78
def seeds_second_player : ℕ := 53
def seeds_third_player (seeds_second_player : ℕ) : ℕ := seeds_second_player + 30

-- Define the total seeds eaten
def total_seeds_eaten (seeds_first_player seeds_second_player seeds_third_player : ℕ) : ℕ :=
  seeds_first_player + seeds_second_player + seeds_third_player

-- Statement of the theorem
theorem total_seeds_eaten_correct : total_seeds_eaten seeds_first_player seeds_second_player (seeds_third_player seeds_second_player) = 214 :=
by
  sorry

end NUMINAMATH_GPT_total_seeds_eaten_correct_l832_83207


namespace NUMINAMATH_GPT_sum_first10PrimesGT50_eq_732_l832_83250

def first10PrimesGT50 := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

theorem sum_first10PrimesGT50_eq_732 :
  first10PrimesGT50.sum = 732 := by
  sorry

end NUMINAMATH_GPT_sum_first10PrimesGT50_eq_732_l832_83250


namespace NUMINAMATH_GPT_minneapolis_st_louis_temperature_l832_83214

theorem minneapolis_st_louis_temperature (N M L : ℝ) (h1 : M = L + N)
                                         (h2 : M - 7 = L + N - 7)
                                         (h3 : L + 5 = L + 5)
                                         (h4 : (M - 7) - (L + 5) = |(L + N - 7) - (L + 5)|) :
  ∃ (N1 N2 : ℝ), (|N - 12| = 4) ∧ N1 = 16 ∧ N2 = 8 ∧ N1 * N2 = 128 :=
by {
  sorry
}

end NUMINAMATH_GPT_minneapolis_st_louis_temperature_l832_83214


namespace NUMINAMATH_GPT_profit_share_difference_l832_83261

theorem profit_share_difference (P : ℝ) (hP : P = 1000) 
  (rX rY : ℝ) (hRatio : rX / rY = (1/2) / (1/3)) : 
  let total_parts := (1/2) + (1/3)
  let value_per_part := P / total_parts
  let x_share := (1/2) * value_per_part
  let y_share := (1/3) * value_per_part
  x_share - y_share = 200 := by 
  sorry

end NUMINAMATH_GPT_profit_share_difference_l832_83261


namespace NUMINAMATH_GPT_four_digit_flippies_div_by_4_l832_83206

def is_flippy (n : ℕ) : Prop := 
  let digits := [4, 6]
  n / 1000 ∈ digits ∧
  (n / 100 % 10) ∈ digits ∧
  ((n / 10 % 10) = if (n / 100 % 10) = 4 then 6 else 4) ∧
  (n % 10) = if (n / 1000) = 4 then 6 else 4

def is_divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

theorem four_digit_flippies_div_by_4 : 
  ∃! n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ is_flippy n ∧ is_divisible_by_4 n :=
by
  sorry

end NUMINAMATH_GPT_four_digit_flippies_div_by_4_l832_83206


namespace NUMINAMATH_GPT_maximal_points_coloring_l832_83235

/-- Given finitely many points in the plane where no three points are collinear,
which are colored either red or green, such that any monochromatic triangle
contains at least one point of the other color in its interior, the maximal number
of such points is 8. -/
theorem maximal_points_coloring (points : Finset (ℝ × ℝ))
  (h_no_three_collinear : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p3 ∈ points →
    p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
    ¬ ∃ k b, ∀ p ∈ [p1, p2, p3], p.2 = k * p.1 + b)
  (colored : (ℝ × ℝ) → Prop)
  (h_coloring : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p3 ∈ points →
    colored p1 = colored p2 → colored p2 = colored p3 →
    ∃ p, p ∈ points ∧ colored p ≠ colored p1) :
  points.card ≤ 8 :=
sorry

end NUMINAMATH_GPT_maximal_points_coloring_l832_83235


namespace NUMINAMATH_GPT_penny_purchase_exceeded_minimum_spend_l832_83213

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

end NUMINAMATH_GPT_penny_purchase_exceeded_minimum_spend_l832_83213


namespace NUMINAMATH_GPT_chocolate_bars_per_box_l832_83236

theorem chocolate_bars_per_box (total_chocolate_bars boxes : ℕ) (h1 : total_chocolate_bars = 710) (h2 : boxes = 142) : total_chocolate_bars / boxes = 5 := by
  sorry

end NUMINAMATH_GPT_chocolate_bars_per_box_l832_83236


namespace NUMINAMATH_GPT_select_subset_divisible_by_n_l832_83254

theorem select_subset_divisible_by_n (n : ℕ) (h : n > 0) (l : List ℤ) (hl : l.length = 2 * n - 1) :
  ∃ s : Finset ℤ, s.card = n ∧ (s.sum id) % n = 0 := 
sorry

end NUMINAMATH_GPT_select_subset_divisible_by_n_l832_83254


namespace NUMINAMATH_GPT_divides_a_square_minus_a_and_a_cube_minus_a_l832_83228

theorem divides_a_square_minus_a_and_a_cube_minus_a (a : ℤ) : 
  (2 ∣ a^2 - a) ∧ (3 ∣ a^3 - a) :=
by
  sorry

end NUMINAMATH_GPT_divides_a_square_minus_a_and_a_cube_minus_a_l832_83228


namespace NUMINAMATH_GPT_hyperbola_eccentricity_sqrt2_l832_83299

noncomputable def isHyperbolaPerpendicularAsymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  let asymptote1 := (1/a : ℝ)
  let asymptote2 := (-1/b : ℝ)
  asymptote1 * asymptote2 = -1

theorem hyperbola_eccentricity_sqrt2 (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  isHyperbolaPerpendicularAsymptotes a b ha hb →
  let e := Real.sqrt (1 + (b^2 / a^2))
  e = Real.sqrt 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_sqrt2_l832_83299


namespace NUMINAMATH_GPT_find_n_values_l832_83265

theorem find_n_values (n : ℕ) (h : ∃ k : ℕ, n^2 - 19 * n + 91 = k^2) : n = 9 ∨ n = 10 :=
sorry

end NUMINAMATH_GPT_find_n_values_l832_83265


namespace NUMINAMATH_GPT_jelly_cost_l832_83220

theorem jelly_cost (B J : ℕ) 
  (h1 : 15 * (6 * B + 7 * J) = 315) 
  (h2 : 0 ≤ B) 
  (h3 : 0 ≤ J) : 
  15 * J * 7 = 315 := 
sorry

end NUMINAMATH_GPT_jelly_cost_l832_83220


namespace NUMINAMATH_GPT_find_scalars_l832_83295

noncomputable def N : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![-2, 0]]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![0, 1]]

theorem find_scalars (r s : ℤ) (h_r : r = 3) (h_s : s = -8) :
    N * N = r • N + s • I :=
by
  rw [h_r, h_s]
  sorry

end NUMINAMATH_GPT_find_scalars_l832_83295


namespace NUMINAMATH_GPT_lucy_times_three_ago_l832_83268

  -- Defining the necessary variables and conditions
  def lucy_age_now : ℕ := 50
  def lovely_age (x : ℕ) : ℕ := 20  -- The age of Lovely when x years has passed
  
  -- Statement of the problem
  theorem lucy_times_three_ago {x : ℕ} : 
    (lucy_age_now - x = 3 * (lovely_age x - x)) → (lucy_age_now + 10 = 2 * (lovely_age x + 10)) → x = 5 := 
  by
  -- Proof is omitted
  sorry
  
end NUMINAMATH_GPT_lucy_times_three_ago_l832_83268


namespace NUMINAMATH_GPT_percentage_increase_equal_price_l832_83286

/-
A merchant has selected two items to be placed on sale, one of which currently sells for 20 percent less than the other.
He wishes to raise the price of the cheaper item so that the two items are equally priced.
By what percentage must he raise the price of the less expensive item?
-/
theorem percentage_increase_equal_price (P: ℝ) : (P > 0) → 
  (∀ cheap_item, cheap_item = 0.80 * P → ((P - cheap_item) / cheap_item) * 100 = 25) :=
by
  intro P_pos
  intro cheap_item
  intro h
  sorry

end NUMINAMATH_GPT_percentage_increase_equal_price_l832_83286


namespace NUMINAMATH_GPT_Eric_test_score_l832_83245

theorem Eric_test_score (n : ℕ) (old_avg new_avg : ℚ) (Eric_score : ℚ) :
  n = 22 →
  old_avg = 84 →
  new_avg = 85 →
  Eric_score = (n * new_avg) - ((n - 1) * old_avg) →
  Eric_score = 106 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_Eric_test_score_l832_83245


namespace NUMINAMATH_GPT_points_per_right_answer_l832_83239

variable (p : ℕ)
variable (total_problems : ℕ := 25)
variable (wrong_problems : ℕ := 3)
variable (score : ℤ := 85)

theorem points_per_right_answer :
  (total_problems - wrong_problems) * p - wrong_problems = score -> p = 4 :=
  sorry

end NUMINAMATH_GPT_points_per_right_answer_l832_83239


namespace NUMINAMATH_GPT_sphere_hemisphere_radius_relationship_l832_83210

theorem sphere_hemisphere_radius_relationship (r : ℝ) (R : ℝ) (π : ℝ) (h : 0 < π):
  (4 / 3) * π * R^3 = (2 / 3) * π * r^3 →
  r = 3 * (2^(1/3 : ℝ)) →
  R = 3 :=
by
  sorry

end NUMINAMATH_GPT_sphere_hemisphere_radius_relationship_l832_83210


namespace NUMINAMATH_GPT_number_of_blue_eyed_students_in_k_class_l832_83287

-- Definitions based on the given conditions
def total_students := 40
def blond_hair_to_blue_eyes_ratio := 2.5
def students_with_both := 8
def students_with_neither := 5

-- We need to prove that the number of blue-eyed students is 10
theorem number_of_blue_eyed_students_in_k_class 
  (x : ℕ)  -- number of blue-eyed students
  (H1 : total_students = 40)
  (H2 : ∀ x, blond_hair_to_blue_eyes_ratio * x = number_of_blond_students)
  (H3 : students_with_both = 8)
  (H4 : students_with_neither = 5)
  : x = 10 :=
sorry

end NUMINAMATH_GPT_number_of_blue_eyed_students_in_k_class_l832_83287


namespace NUMINAMATH_GPT_greenfield_academy_math_count_l832_83208

theorem greenfield_academy_math_count (total_players taking_physics both_subjects : ℕ) 
(h_total: total_players = 30) 
(h_physics: taking_physics = 15) 
(h_both: both_subjects = 3) : 
∃ taking_math : ℕ, taking_math = 21 :=
by
  sorry

end NUMINAMATH_GPT_greenfield_academy_math_count_l832_83208


namespace NUMINAMATH_GPT_combined_weight_difference_l832_83271

def chemistry_weight : ℝ := 7.125
def geometry_weight : ℝ := 0.625
def calculus_weight : ℝ := -5.25
def biology_weight : ℝ := 3.755

theorem combined_weight_difference :
  (chemistry_weight - calculus_weight) - (geometry_weight + biology_weight) = 7.995 :=
by
  sorry

end NUMINAMATH_GPT_combined_weight_difference_l832_83271


namespace NUMINAMATH_GPT_marked_price_l832_83211

theorem marked_price (x : ℝ) (purchase_price : ℝ) (selling_price : ℝ) (profit_margin : ℝ) 
  (h_purchase_price : purchase_price = 100)
  (h_profit_margin : profit_margin = 0.2)
  (h_selling_price : selling_price = purchase_price * (1 + profit_margin))
  (h_price_relation : 0.8 * x = selling_price) : 
  x = 150 :=
by sorry

end NUMINAMATH_GPT_marked_price_l832_83211


namespace NUMINAMATH_GPT_last_digit_2_pow_1000_last_digit_3_pow_1000_last_digit_7_pow_1000_l832_83276

-- Define the cycle period used in the problem
def cycle_period_2 := [2, 4, 8, 6]
def cycle_period_3 := [3, 9, 7, 1]
def cycle_period_7 := [7, 9, 3, 1]

-- Define a function to get the last digit from the cycle for given n
def last_digit_from_cycle (cycle : List ℕ) (n : ℕ) : ℕ :=
  let cycle_length := cycle.length
  cycle.get! ((n % cycle_length) - 1)

-- Problem statements
theorem last_digit_2_pow_1000 : last_digit_from_cycle cycle_period_2 1000 = 6 := sorry
theorem last_digit_3_pow_1000 : last_digit_from_cycle cycle_period_3 1000 = 1 := sorry
theorem last_digit_7_pow_1000 : last_digit_from_cycle cycle_period_7 1000 = 1 := sorry

end NUMINAMATH_GPT_last_digit_2_pow_1000_last_digit_3_pow_1000_last_digit_7_pow_1000_l832_83276


namespace NUMINAMATH_GPT_discount_equivalence_l832_83231

variable (Original_Price : ℝ)

theorem discount_equivalence (h1 : Real) (h2 : Real) :
  (h1 = 0.5 * Original_Price) →
  (h2 = 0.7 * h1) →
  (Original_Price - h2) / Original_Price = 0.65 :=
by
  intros
  sorry

end NUMINAMATH_GPT_discount_equivalence_l832_83231


namespace NUMINAMATH_GPT_find_value_of_expression_l832_83291

theorem find_value_of_expression
  (a b c d : ℝ) (h₀ : a ≥ 0) (h₁ : b ≥ 0) (h₂ : c ≥ 0) (h₃ : d ≥ 0)
  (h₄ : a / (b + c + d) = b / (a + c + d))
  (h₅ : b / (a + c + d) = c / (a + b + d))
  (h₆ : c / (a + b + d) = d / (a + b + c))
  (h₇ : d / (a + b + c) = a / (b + c + d)) :
  (a + b) / (c + d) + (b + c) / (a + d) + (c + d) / (a + b) + (d + a) / (b + c) = 4 :=
by sorry

end NUMINAMATH_GPT_find_value_of_expression_l832_83291


namespace NUMINAMATH_GPT_problem1_problem2_l832_83241

-- Definitions and conditions
def A (m : ℝ) : Set ℝ := { x | m ≤ x ∧ x ≤ m + 1 }
def B : Set ℝ := { x | x < -6 ∨ x > 1 }

-- (Ⅰ) Problem statement: Prove that if A ∩ B = ∅, then -6 ≤ m ≤ 0.
theorem problem1 (m : ℝ) : A m ∩ B = ∅ ↔ -6 ≤ m ∧ m ≤ 0 := 
by
  sorry

-- (Ⅱ) Problem statement: Prove that if A ⊆ B, then m < -7 or m > 1.
theorem problem2 (m : ℝ) : A m ⊆ B ↔ m < -7 ∨ m > 1 := 
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l832_83241


namespace NUMINAMATH_GPT_smallest_term_of_sequence_l832_83224

-- Define the sequence a_n
def a (n : ℕ) : ℤ := 3 * n^2 - 28 * n

-- The statement that the 5th term is the smallest in the sequence
theorem smallest_term_of_sequence : ∀ n : ℕ, a 5 ≤ a n := by
  sorry

end NUMINAMATH_GPT_smallest_term_of_sequence_l832_83224


namespace NUMINAMATH_GPT_radius_of_circumscribed_circle_l832_83221

theorem radius_of_circumscribed_circle 
    (r : ℝ)
    (h : 8 * r = π * r^2) : 
    r = 8 / π :=
by
    sorry

end NUMINAMATH_GPT_radius_of_circumscribed_circle_l832_83221


namespace NUMINAMATH_GPT_smallest_sum_arith_geo_sequence_l832_83200

theorem smallest_sum_arith_geo_sequence 
  (A B C D: ℕ) 
  (h1: A > 0) 
  (h2: B > 0) 
  (h3: C > 0) 
  (h4: D > 0)
  (h5: 2 * B = A + C)
  (h6: B * D = C * C)
  (h7: 3 * C = 4 * B) : 
  A + B + C + D = 43 := 
sorry

end NUMINAMATH_GPT_smallest_sum_arith_geo_sequence_l832_83200


namespace NUMINAMATH_GPT_largest_gcd_of_sum_1729_l832_83283

theorem largest_gcd_of_sum_1729 (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1729) :
  ∃ g, g = Nat.gcd x y ∧ g = 247 := sorry

end NUMINAMATH_GPT_largest_gcd_of_sum_1729_l832_83283


namespace NUMINAMATH_GPT_kristin_runs_n_times_faster_l832_83201

theorem kristin_runs_n_times_faster (D K S : ℝ) (n : ℝ) 
  (h1 : K = n * S) 
  (h2 : 12 * D / K = 4 * D / S) : 
  n = 3 :=
by
  sorry

end NUMINAMATH_GPT_kristin_runs_n_times_faster_l832_83201


namespace NUMINAMATH_GPT_hotdogs_sold_correct_l832_83273

def initial_hotdogs : ℕ := 99
def remaining_hotdogs : ℕ := 97
def sold_hotdogs : ℕ := initial_hotdogs - remaining_hotdogs

theorem hotdogs_sold_correct : sold_hotdogs = 2 := by
  sorry

end NUMINAMATH_GPT_hotdogs_sold_correct_l832_83273


namespace NUMINAMATH_GPT_find_f2_f5_sum_l832_83255

theorem find_f2_f5_sum
  (f : ℤ → ℤ)
  (a b : ℤ)
  (h1 : f 1 = 4)
  (h2 : ∀ z : ℤ, f z = 3 * z + 6)
  (h3 : ∀ x y : ℤ, f (x + y) = f x + f y + a * x * y + b) :
  f 2 + f 5 = 33 :=
sorry

end NUMINAMATH_GPT_find_f2_f5_sum_l832_83255


namespace NUMINAMATH_GPT_chromium_percentage_new_alloy_l832_83281

variable (w1 w2 : ℝ) (cr1 cr2 : ℝ)

theorem chromium_percentage_new_alloy (h_w1 : w1 = 15) (h_w2 : w2 = 30) (h_cr1 : cr1 = 0.12) (h_cr2 : cr2 = 0.08) :
  (cr1 * w1 + cr2 * w2) / (w1 + w2) * 100 = 9.33 := by
  sorry

end NUMINAMATH_GPT_chromium_percentage_new_alloy_l832_83281


namespace NUMINAMATH_GPT_largest_inscribed_triangle_area_l832_83251

-- Definition of the conditions
def radius : ℝ := 10
def diameter : ℝ := 2 * radius

-- The theorem to be proven
theorem largest_inscribed_triangle_area (r : ℝ) (D : ℝ) (h : D = 2 * r) : 
  ∃ (A : ℝ), A = 100 := by
  have base := D
  have height := r
  have area := (1 / 2) * base * height
  use area
  sorry

end NUMINAMATH_GPT_largest_inscribed_triangle_area_l832_83251


namespace NUMINAMATH_GPT_base_7_to_10_of_23456_l832_83259

theorem base_7_to_10_of_23456 : 
  (2 * 7 ^ 4 + 3 * 7 ^ 3 + 4 * 7 ^ 2 + 5 * 7 ^ 1 + 6 * 7 ^ 0) = 6068 :=
by sorry

end NUMINAMATH_GPT_base_7_to_10_of_23456_l832_83259


namespace NUMINAMATH_GPT_number_of_apples_l832_83230

theorem number_of_apples (A : ℝ) (h : 0.75 * A * 0.5 + 0.25 * A * 0.1 = 40) : A = 100 :=
by
  sorry

end NUMINAMATH_GPT_number_of_apples_l832_83230


namespace NUMINAMATH_GPT_career_preference_representation_l832_83209

noncomputable def male_to_female_ratio : ℕ × ℕ := (2, 3)
noncomputable def total_students := male_to_female_ratio.1 + male_to_female_ratio.2
noncomputable def students_prefer_career := 2
noncomputable def full_circle_degrees := 360

theorem career_preference_representation :
  (students_prefer_career / total_students : ℚ) * full_circle_degrees = 144 := by
  sorry

end NUMINAMATH_GPT_career_preference_representation_l832_83209


namespace NUMINAMATH_GPT_symmetric_line_eq_l832_83294

/-- 
Given two circles O: x^2 + y^2 = 4 and C: x^2 + y^2 + 4x - 4y + 4 = 0, 
prove the equation of the line l such that the two circles are symmetric 
with respect to line l is x - y + 2 = 0.
-/
theorem symmetric_line_eq {x y : ℝ} :
  (∀ x y : ℝ, (x^2 + y^2 = 4) → (x^2 + y^2 + 4*x - 4*y + 4 = 0)) → (∀ x y : ℝ, (x - y + 2 = 0)) :=
  sorry

end NUMINAMATH_GPT_symmetric_line_eq_l832_83294


namespace NUMINAMATH_GPT_custom_op_value_l832_83218

-- Define the custom operation (a \$ b)
def custom_op (a b : Int) : Int := a * (b - 1) + a * b

-- Main theorem to prove the equivalence
theorem custom_op_value : custom_op 5 (-3) = -35 := by
  sorry

end NUMINAMATH_GPT_custom_op_value_l832_83218


namespace NUMINAMATH_GPT_incorrect_positional_relationship_l832_83252

-- Definitions for the geometric relationships
def line := Type
def plane := Type

def parallel (l : line) (α : plane) : Prop := sorry
def perpendicular (l : line) (α : plane) : Prop := sorry
def subset (l : line) (α : plane) : Prop := sorry
def distinct (l m : line) : Prop := l ≠ m

-- Given conditions
variables (l m : line) (α : plane)

-- Theorem statement: prove that D is incorrect given the conditions
theorem incorrect_positional_relationship
  (h_distinct : distinct l m)
  (h_parallel_l_α : parallel l α)
  (h_parallel_m_α : parallel m α) :
  ¬ (parallel l m) :=
sorry

end NUMINAMATH_GPT_incorrect_positional_relationship_l832_83252


namespace NUMINAMATH_GPT_chocolate_chips_per_member_l832_83290

/-
Define the problem conditions:
-/
def family_members := 4
def batches_choc_chip := 3
def cookies_per_batch_choc_chip := 12
def chips_per_cookie_choc_chip := 2
def batches_double_choc_chip := 2
def cookies_per_batch_double_choc_chip := 10
def chips_per_cookie_double_choc_chip := 4

/-
State the theorem to be proved:
-/
theorem chocolate_chips_per_member : 
  let total_choc_chip_cookies := batches_choc_chip * cookies_per_batch_choc_chip
  let total_choc_chips_choc_chip := total_choc_chip_cookies * chips_per_cookie_choc_chip
  let total_double_choc_chip_cookies := batches_double_choc_chip * cookies_per_batch_double_choc_chip
  let total_choc_chips_double_choc_chip := total_double_choc_chip_cookies * chips_per_cookie_double_choc_chip
  let total_choc_chips := total_choc_chips_choc_chip + total_choc_chips_double_choc_chip
  let chips_per_member := total_choc_chips / family_members
  chips_per_member = 38 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_chips_per_member_l832_83290


namespace NUMINAMATH_GPT_lally_internet_days_l832_83253

-- Definitions based on the conditions
def cost_per_day : ℝ := 0.5
def debt_limit : ℝ := 5
def initial_payment : ℝ := 7
def initial_balance : ℝ := 0

-- Proof problem statement
theorem lally_internet_days : ∀ (d : ℕ), 
  (initial_balance + initial_payment - cost_per_day * d ≤ debt_limit) -> (d = 14) :=
sorry

end NUMINAMATH_GPT_lally_internet_days_l832_83253


namespace NUMINAMATH_GPT_tangent_circle_exists_l832_83238
open Set

-- Definitions of given point, line, and circle
variables {Point : Type*} {Line : Type*} {Circle : Type*} 
variables (M : Point) (l : Line) (S : Circle)
variables (center_S : Point) (radius_S : ℝ)

-- Conditions of the problem
variables (touches_line : Circle → Line → Prop) (touches_circle : Circle → Circle → Prop)
variables (passes_through : Circle → Point → Prop) (center_of : Circle → Point)
variables (radius_of : Circle → ℝ)

-- Existence theorem to prove
theorem tangent_circle_exists 
  (given_tangent_to_line : Circle → Line → Bool)
  (given_tangent_to_circle : Circle → Circle → Bool)
  (given_passes_through : Circle → Point → Bool):
  ∃ (Ω : Circle), 
    given_tangent_to_line Ω l ∧
    given_tangent_to_circle Ω S ∧
    given_passes_through Ω M :=
sorry

end NUMINAMATH_GPT_tangent_circle_exists_l832_83238


namespace NUMINAMATH_GPT_joan_exam_time_difference_l832_83222

theorem joan_exam_time_difference :
  ∀ (E_time M_time E_questions M_questions : ℕ),
  E_time = 60 →
  M_time = 90 →
  E_questions = 30 →
  M_questions = 15 →
  (M_time / M_questions) - (E_time / E_questions) = 4 :=
by
  intros E_time M_time E_questions M_questions hE_time hM_time hE_questions hM_questions
  sorry

end NUMINAMATH_GPT_joan_exam_time_difference_l832_83222


namespace NUMINAMATH_GPT_initial_concentration_of_hydrochloric_acid_l832_83264

theorem initial_concentration_of_hydrochloric_acid
  (initial_mass : ℕ)
  (drained_mass : ℕ)
  (added_concentration : ℕ)
  (final_concentration : ℕ)
  (total_mass : ℕ)
  (initial_concentration : ℕ) :
  initial_mass = 300 ∧ drained_mass = 25 ∧ added_concentration = 80 ∧ final_concentration = 25 ∧ total_mass = 300 →
  (275 * initial_concentration / 100 + 20 = 75) →
  initial_concentration = 20 :=
by
  intros h_eq h_new_solution
  -- Rewriting the data given in h_eq and solving h_new_solution
  rcases h_eq with ⟨h_initial_mass, h_drained_mass, h_added_concentration, h_final_concentration, h_total_mass⟩
  sorry

end NUMINAMATH_GPT_initial_concentration_of_hydrochloric_acid_l832_83264


namespace NUMINAMATH_GPT_perfect_score_l832_83234

theorem perfect_score (P : ℕ) (h : 3 * P = 63) : P = 21 :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_perfect_score_l832_83234


namespace NUMINAMATH_GPT_figure_square_count_l832_83242

theorem figure_square_count (f : ℕ → ℕ)
  (h0 : f 0 = 2)
  (h1 : f 1 = 8)
  (h2 : f 2 = 18)
  (h3 : f 3 = 32) :
  f 100 = 20402 :=
sorry

end NUMINAMATH_GPT_figure_square_count_l832_83242


namespace NUMINAMATH_GPT_maximum_area_of_triangle_OAB_l832_83289

noncomputable def maximum_area_triangle (a b : ℝ) : ℝ :=
  if 2 * a + b = 5 ∧ a > 0 ∧ b > 0 then (1 / 2) * a * b else 0

theorem maximum_area_of_triangle_OAB : 
  (∀ (a b : ℝ), 2 * a + b = 5 ∧ a > 0 ∧ b > 0 → (1 / 2) * a * b ≤ 25 / 16) :=
by
  sorry

end NUMINAMATH_GPT_maximum_area_of_triangle_OAB_l832_83289


namespace NUMINAMATH_GPT_u_less_than_v_l832_83275

noncomputable def f (u : ℝ) := (u + u^2 + u^3 + u^4 + u^5 + u^6 + u^7 + u^8) + 10 * u^9
noncomputable def g (v : ℝ) := (v + v^2 + v^3 + v^4 + v^5 + v^6 + v^7 + v^8 + v^9 + v^10) + 10 * v^11

theorem u_less_than_v
  (u v : ℝ)
  (hu : f u = 8)
  (hv : g v = 8) :
  u < v := 
sorry

end NUMINAMATH_GPT_u_less_than_v_l832_83275
