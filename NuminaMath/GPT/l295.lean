import Mathlib

namespace NUMINAMATH_GPT_inversely_varies_y_l295_29572

theorem inversely_varies_y (x y : ℕ) (k : ℕ) (h₁ : 7 * y = k / x^3) (h₂ : y = 8) (h₃ : x = 2) : 
  y = 1 :=
by
  sorry

end NUMINAMATH_GPT_inversely_varies_y_l295_29572


namespace NUMINAMATH_GPT_circle_radius_one_l295_29558

-- Define the circle equation as a hypothesis
def circle_equation (x y : ℝ) : Prop :=
  16 * x^2 + 32 * x + 16 * y^2 - 48 * y + 68 = 0

-- The goal is to prove the radius of the circle defined above
theorem circle_radius_one :
  ∃ r : ℝ, r = 1 ∧ ∀ x y : ℝ, circle_equation x y → (x + 1)^2 + (y - 1.5)^2 = r^2 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_one_l295_29558


namespace NUMINAMATH_GPT_total_points_l295_29534

theorem total_points (total_players : ℕ) (paige_points : ℕ) (other_points : ℕ) (points_per_other_player : ℕ) :
  total_players = 5 →
  paige_points = 11 →
  points_per_other_player = 6 →
  other_points = (total_players - 1) * points_per_other_player →
  paige_points + other_points = 35 :=
by
  intro h_total_players h_paige_points h_points_per_other_player h_other_points
  sorry

end NUMINAMATH_GPT_total_points_l295_29534


namespace NUMINAMATH_GPT_students_met_goal_l295_29542

def money_needed_per_student : ℕ := 450
def number_of_students : ℕ := 6
def collective_expenses : ℕ := 3000
def amount_raised_day1 : ℕ := 600
def amount_raised_day2 : ℕ := 900
def amount_raised_day3 : ℕ := 400
def days_remaining : ℕ := 4
def half_of_first_three_days : ℕ :=
  (amount_raised_day1 + amount_raised_day2 + amount_raised_day3) / 2

def total_needed : ℕ :=
  money_needed_per_student * number_of_students + collective_expenses
def total_raised : ℕ :=
  amount_raised_day1 + amount_raised_day2 + amount_raised_day3 + (half_of_first_three_days * days_remaining)

theorem students_met_goal : total_raised >= total_needed := by
  sorry

end NUMINAMATH_GPT_students_met_goal_l295_29542


namespace NUMINAMATH_GPT_minimum_m_plus_n_l295_29563

theorem minimum_m_plus_n (m n : ℕ) (h1 : 98 * m = n ^ 3) (h2 : 0 < m) (h3 : 0 < n) : m + n = 42 :=
sorry

end NUMINAMATH_GPT_minimum_m_plus_n_l295_29563


namespace NUMINAMATH_GPT_sum_of_squares_of_medians_triangle_13_14_15_l295_29584

noncomputable def sum_of_squares_of_medians (a b c : ℝ) : ℝ :=
  (3 / 4) * (a^2 + b^2 + c^2)

theorem sum_of_squares_of_medians_triangle_13_14_15 :
  sum_of_squares_of_medians 13 14 15 = 442.5 :=
by
  -- By calculation using the definition of sum_of_squares_of_medians
  -- and substituting the given side lengths.
  -- Detailed proof steps are omitted
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_medians_triangle_13_14_15_l295_29584


namespace NUMINAMATH_GPT_no_polyhedron_with_area_ratio_ge_two_l295_29529

theorem no_polyhedron_with_area_ratio_ge_two (n : ℕ) (areas : Fin n → ℝ)
  (h : ∀ (i j : Fin n), i < j → (areas j) / (areas i) ≥ 2) : False := by
  sorry

end NUMINAMATH_GPT_no_polyhedron_with_area_ratio_ge_two_l295_29529


namespace NUMINAMATH_GPT_min_reciprocal_sum_l295_29576

theorem min_reciprocal_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (hS : S 2019 = 4038) 
  (h_seq : ∀ n, S n = (n * (a 1 + a n)) / 2) :
  ∃ m, m = 4 ∧ (∀ i, i = 9 → ∀ j, j = 2011 → 
  a i + a j = 4 ∧ m = min (1 / a i + 9 / a j) 4) :=
by sorry

end NUMINAMATH_GPT_min_reciprocal_sum_l295_29576


namespace NUMINAMATH_GPT_eccentricity_of_given_ellipse_l295_29551

noncomputable def ellipse_eccentricity (φ : Real) : Real :=
  let x := 3 * Real.cos φ
  let y := 5 * Real.sin φ
  let a := 5
  let b := 3
  let c := Real.sqrt (a * a - b * b)
  c / a

theorem eccentricity_of_given_ellipse (φ : Real) :
  ellipse_eccentricity φ = 4 / 5 :=
sorry

end NUMINAMATH_GPT_eccentricity_of_given_ellipse_l295_29551


namespace NUMINAMATH_GPT_mul_inv_mod_391_l295_29503

theorem mul_inv_mod_391 (a : ℤ) (ha : 143 * a % 391 = 1) : a = 28 := by
  sorry

end NUMINAMATH_GPT_mul_inv_mod_391_l295_29503


namespace NUMINAMATH_GPT_modular_inverse_expression_l295_29538

-- Definitions of the inverses as given in the conditions
def inv_7_mod_77 : ℤ := 11
def inv_13_mod_77 : ℤ := 6

-- The main theorem stating the equivalence
theorem modular_inverse_expression :
  (3 * inv_7_mod_77 + 9 * inv_13_mod_77) % 77 = 10 :=
by
  sorry

end NUMINAMATH_GPT_modular_inverse_expression_l295_29538


namespace NUMINAMATH_GPT_log10_two_bounds_l295_29589

theorem log10_two_bounds
  (h1 : 10 ^ 3 = 1000)
  (h2 : 10 ^ 4 = 10000)
  (h3 : 2 ^ 10 = 1024)
  (h4 : 2 ^ 12 = 4096) :
  1 / 4 < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < 0.4 := 
sorry

end NUMINAMATH_GPT_log10_two_bounds_l295_29589


namespace NUMINAMATH_GPT_digit_B_divisibility_l295_29511

theorem digit_B_divisibility :
  ∃ B : ℕ, B < 10 ∧
    (∃ n : ℕ, 658274 * 10 + B = 2 * n) ∧
    (∃ m : ℕ, 6582740 + B = 4 * m) ∧
    (B = 0 ∨ B = 5) ∧
    (∃ k : ℕ, 658274 * 10 + B = 7 * k) ∧
    (∃ p : ℕ, 6582740 + B = 8 * p) :=
sorry

end NUMINAMATH_GPT_digit_B_divisibility_l295_29511


namespace NUMINAMATH_GPT_part1_part2_l295_29527

-- Definitions of sets A and B
def A : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }
def B (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ 3 - 2 * a }

-- Part 1: Prove that (complement of A union B = Universal Set) implies a in (-∞, 0]
theorem part1 (U : Set ℝ) (hU : (Aᶜ ∪ B a) = U) : a ≤ 0 := sorry

-- Part 2: Prove that (A intersection B = B) implies a in [1/2, ∞)
theorem part2 (h : (A ∩ B a) = B a) : 1/2 ≤ a := sorry

end NUMINAMATH_GPT_part1_part2_l295_29527


namespace NUMINAMATH_GPT_determine_m_l295_29521

variables (m x : ℝ)
noncomputable def f (x : ℝ) := x^2 - 3*x + m
noncomputable def g (x : ℝ) := x^2 - 3*x + 5*m

theorem determine_m (h : 3 * f 5 = 2 * g 5) : m = 10 / 7 :=
by
  sorry

end NUMINAMATH_GPT_determine_m_l295_29521


namespace NUMINAMATH_GPT_infinite_points_in_region_l295_29519

theorem infinite_points_in_region : 
  ∀ x y : ℚ, 0 < x → 0 < y → x + 2 * y ≤ 6 → ¬(∃ n : ℕ, ∀ x y : ℚ, 0 < x → 0 < y → x + 2 * y ≤ 6 → sorry) :=
sorry

end NUMINAMATH_GPT_infinite_points_in_region_l295_29519


namespace NUMINAMATH_GPT_determine_digits_l295_29579

theorem determine_digits :
  ∃ (A B C D : ℕ), 
    1000 ≤ 1000 * A + 100 * B + 10 * C + D ∧ 
    1000 * A + 100 * B + 10 * C + D ≤ 9999 ∧ 
    1000 ≤ 1000 * C + 100 * B + 10 * A + D ∧ 
    1000 * C + 100 * B + 10 * A + D ≤ 9999 ∧ 
    (1000 * A + 100 * B + 10 * C + D) * D = 1000 * C + 100 * B + 10 * A + D ∧ 
    A = 2 ∧ B = 1 ∧ C = 7 ∧ D = 8 :=
by
  sorry

end NUMINAMATH_GPT_determine_digits_l295_29579


namespace NUMINAMATH_GPT_profit_calculation_correct_l295_29531

def main_actor_fee : ℕ := 500
def supporting_actor_fee : ℕ := 100
def extra_fee : ℕ := 50
def main_actor_food : ℕ := 10
def supporting_actor_food : ℕ := 5
def remaining_member_food : ℕ := 3
def post_production_cost : ℕ := 850
def revenue : ℕ := 10000

def total_actor_fees : ℕ := 2 * main_actor_fee + 3 * supporting_actor_fee + extra_fee
def total_food_cost : ℕ := 2 * main_actor_food + 4 * supporting_actor_food + 44 * remaining_member_food
def total_equipment_rental : ℕ := 2 * (total_actor_fees + total_food_cost)
def total_cost : ℕ := total_actor_fees + total_food_cost + total_equipment_rental + post_production_cost
def profit : ℕ := revenue - total_cost

theorem profit_calculation_correct : profit = 4584 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_profit_calculation_correct_l295_29531


namespace NUMINAMATH_GPT_largest_value_l295_29504

noncomputable def largest_possible_4x_3y (x y : ℝ) : ℝ :=
  4 * x + 3 * y

theorem largest_value (x y : ℝ) :
  x^2 + y^2 = 16 * x + 8 * y + 8 → (∃ x y, largest_possible_4x_3y x y = 9.64) :=
by
  sorry

end NUMINAMATH_GPT_largest_value_l295_29504


namespace NUMINAMATH_GPT_ed_lighter_than_al_l295_29548

theorem ed_lighter_than_al :
  let Al := Ben + 25
  let Ben := Carl - 16
  let Ed := 146
  let Carl := 175
  Al - Ed = 38 :=
by
  sorry

end NUMINAMATH_GPT_ed_lighter_than_al_l295_29548


namespace NUMINAMATH_GPT_valid_pic4_valid_pic5_l295_29583

-- Define the type for grid coordinates
structure Coord where
  x : ℕ
  y : ℕ

-- Define the function to check if two coordinates are adjacent by side
def adjacent (a b : Coord) : Prop :=
  (a.x = b.x ∧ (a.y = b.y + 1 ∨ a.y = b.y - 1)) ∨
  (a.y = b.y ∧ (a.x = b.x + 1 ∨ a.x = b.x - 1))

-- Define the coordinates for the pictures №4 and №5
def pic4_coords : List (ℕ × Coord) :=
  [(1, ⟨0, 0⟩), (2, ⟨1, 0⟩), (4, ⟨2, 0⟩), (3, ⟨0, 1⟩),
   (5, ⟨1, 1⟩), (6, ⟨2, 1⟩), (7, ⟨2, 2⟩), (8, ⟨1, 3⟩)]

def pic5_coords : List (ℕ × Coord) :=
  [(1, ⟨0, 0⟩), (2, ⟨0, 1⟩), (3, ⟨0, 2⟩), (4, ⟨0, 3⟩), (5, ⟨1, 3⟩)]

-- Define the validity condition for a picture
def valid_picture (coords : List (ℕ × Coord)) : Prop :=
  ∀ (n : ℕ) (c1 c2 : Coord), (n, c1) ∈ coords → (n + 1, c2) ∈ coords → adjacent c1 c2

-- The theorem to prove that pictures №4 and №5 are valid configurations
theorem valid_pic4 : valid_picture pic4_coords := sorry

theorem valid_pic5 : valid_picture pic5_coords := sorry

end NUMINAMATH_GPT_valid_pic4_valid_pic5_l295_29583


namespace NUMINAMATH_GPT_find_q_l295_29549

noncomputable def expr (a b c : ℝ) := a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2

noncomputable def lhs (a b c : ℝ) := (a - b) * (b - c) * (c - a)

theorem find_q (a b c : ℝ) : expr a b c = lhs a b c * 1 := by
  sorry

end NUMINAMATH_GPT_find_q_l295_29549


namespace NUMINAMATH_GPT_last_three_digits_of_7_pow_103_l295_29570

theorem last_three_digits_of_7_pow_103 : (7^103) % 1000 = 614 := by
  sorry

end NUMINAMATH_GPT_last_three_digits_of_7_pow_103_l295_29570


namespace NUMINAMATH_GPT_part_one_part_two_l295_29596

def f (x : ℝ) := |x + 2|

theorem part_one (x : ℝ) : 2 * f x < 4 - |x - 1| ↔ -7 / 3 < x ∧ x < -1 := sorry

theorem part_two (m n : ℝ) (x a : ℝ) (h : m > 0) (h : n > 0) (h : m + n = 1) :
  (|x - a| - f x ≤ 1/m + 1/n) ↔ (-6 ≤ a ∧ a ≤ 2) := sorry

end NUMINAMATH_GPT_part_one_part_two_l295_29596


namespace NUMINAMATH_GPT_expression_in_scientific_notation_l295_29567

-- Conditions
def billion : ℝ := 10^9
def a : ℝ := 20.8

-- Statement
theorem expression_in_scientific_notation : a * billion = 2.08 * 10^10 := by
  sorry

end NUMINAMATH_GPT_expression_in_scientific_notation_l295_29567


namespace NUMINAMATH_GPT_g_inv_eq_l295_29591

def g (x : ℝ) : ℝ := 2 * x ^ 2 + 3 * x - 5

theorem g_inv_eq (x : ℝ) (g_inv : ℝ → ℝ) (h_inv : ∀ y, g (g_inv y) = y ∧ g_inv (g y) = y) :
  (x = ( -1 + Real.sqrt 11 ) / 2) ∨ (x = ( -1 - Real.sqrt 11 ) / 2) :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_g_inv_eq_l295_29591


namespace NUMINAMATH_GPT_goose_eggs_at_pond_l295_29573

noncomputable def total_goose_eggs (E : ℝ) : Prop :=
  (5 / 12) * (5 / 16) * (5 / 9) * (3 / 7) * E = 84

theorem goose_eggs_at_pond : 
  ∃ E : ℝ, total_goose_eggs E ∧ E = 678 :=
by
  use 678
  dsimp [total_goose_eggs]
  sorry

end NUMINAMATH_GPT_goose_eggs_at_pond_l295_29573


namespace NUMINAMATH_GPT_revenue_decrease_1_percent_l295_29556

variable (T C : ℝ)  -- Assumption: T and C are real numbers representing the original tax and consumption

noncomputable def original_revenue : ℝ := T * C
noncomputable def new_tax_rate : ℝ := T * 0.90
noncomputable def new_consumption : ℝ := C * 1.10
noncomputable def new_revenue : ℝ := new_tax_rate T * new_consumption C

theorem revenue_decrease_1_percent :
  new_revenue T C = 0.99 * original_revenue T C := by
  sorry

end NUMINAMATH_GPT_revenue_decrease_1_percent_l295_29556


namespace NUMINAMATH_GPT_coordinates_of_point_on_x_axis_l295_29522

theorem coordinates_of_point_on_x_axis (m : ℤ) 
  (h : 2 * m + 8 = 0) : (m + 5, 2 * m + 8) = (1, 0) :=
sorry

end NUMINAMATH_GPT_coordinates_of_point_on_x_axis_l295_29522


namespace NUMINAMATH_GPT_total_cages_used_l295_29599

def num_puppies : Nat := 45
def num_adult_dogs : Nat := 30
def num_kittens : Nat := 25

def puppies_sold : Nat := 39
def adult_dogs_sold : Nat := 15
def kittens_sold : Nat := 10

def cage_capacity_puppies : Nat := 3
def cage_capacity_adult_dogs : Nat := 2
def cage_capacity_kittens : Nat := 2

def remaining_puppies : Nat := num_puppies - puppies_sold
def remaining_adult_dogs : Nat := num_adult_dogs - adult_dogs_sold
def remaining_kittens : Nat := num_kittens - kittens_sold

def cages_for_puppies : Nat := (remaining_puppies + cage_capacity_puppies - 1) / cage_capacity_puppies
def cages_for_adult_dogs : Nat := (remaining_adult_dogs + cage_capacity_adult_dogs - 1) / cage_capacity_adult_dogs
def cages_for_kittens : Nat := (remaining_kittens + cage_capacity_kittens - 1) / cage_capacity_kittens

def total_cages : Nat := cages_for_puppies + cages_for_adult_dogs + cages_for_kittens

-- Theorem stating the final goal
theorem total_cages_used : total_cages = 18 := by
  sorry

end NUMINAMATH_GPT_total_cages_used_l295_29599


namespace NUMINAMATH_GPT_average_sleep_time_l295_29510

def sleep_times : List ℕ := [10, 9, 10, 8, 8]

theorem average_sleep_time : (sleep_times.sum / sleep_times.length) = 9 := by
  sorry

end NUMINAMATH_GPT_average_sleep_time_l295_29510


namespace NUMINAMATH_GPT_max_minus_min_l295_29505

noncomputable def f (x : ℝ) := if x > 0 then (x - 1) ^ 2 else (x + 1) ^ 2

theorem max_minus_min (n m : ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ (-1 / 2) → n ≤ f x ∧ f x ≤ m) →
  m - n = 1 :=
by { sorry }

end NUMINAMATH_GPT_max_minus_min_l295_29505


namespace NUMINAMATH_GPT_factor_x4_minus_64_l295_29543

theorem factor_x4_minus_64 :
  ∀ (x : ℝ), (x^4 - 64) = (x^2 - 8) * (x^2 + 8) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_factor_x4_minus_64_l295_29543


namespace NUMINAMATH_GPT_find_g_25_l295_29586

noncomputable def g (x : ℝ) : ℝ := sorry

axiom h₁ : ∀ (x y : ℝ), x > 0 → y > 0 → g (x / y) = (y / x) * g x
axiom h₂ : g 50 = 4

theorem find_g_25 : g 25 = 4 / 25 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_g_25_l295_29586


namespace NUMINAMATH_GPT_problem1_problem2_l295_29575

-- Problem 1
theorem problem1 (x : ℝ) : x * (x - 1) - 3 * (x - 1) = 0 → (x = 1) ∨ (x = 3) :=
by sorry

-- Problem 2
theorem problem2 (x : ℝ) : x^2 + 2*x - 1 = 0 → (x = -1 + Real.sqrt 2) ∨ (x = -1 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l295_29575


namespace NUMINAMATH_GPT_sequence_sum_l295_29582

theorem sequence_sum (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n : ℕ, S (n + 1) = (n + 1) * (n + 1) - 1)
  (ha : ∀ n : ℕ, a (n + 1) = S (n + 1) - S n) :
  a 1 + a 3 + a 5 + a 7 + a 9 = 44 :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_l295_29582


namespace NUMINAMATH_GPT_tire_cost_l295_29569

theorem tire_cost (total_cost : ℝ) (num_tires : ℕ)
    (h1 : num_tires = 8) (h2 : total_cost = 4) : 
    total_cost / num_tires = 0.50 := 
by
  sorry

end NUMINAMATH_GPT_tire_cost_l295_29569


namespace NUMINAMATH_GPT_square_value_is_10000_l295_29554
noncomputable def squareValue : Real := 6400000 / 400 / 1.6

theorem square_value_is_10000 : squareValue = 10000 :=
  by
  -- The proof is based on the provided steps, which will be omitted here.
  sorry

end NUMINAMATH_GPT_square_value_is_10000_l295_29554


namespace NUMINAMATH_GPT_totalPears_l295_29559

-- Define the number of pears picked by Sara and Sally
def saraPears : ℕ := 45
def sallyPears : ℕ := 11

-- Statement to prove
theorem totalPears : saraPears + sallyPears = 56 :=
by
  sorry

end NUMINAMATH_GPT_totalPears_l295_29559


namespace NUMINAMATH_GPT_ratio_first_to_second_l295_29545

theorem ratio_first_to_second (A B C : ℕ) (h1 : A + B + C = 98) (h2 : B = 30) (h3 : B / C = 5 / 8) : A / B = 2 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_first_to_second_l295_29545


namespace NUMINAMATH_GPT_distance_center_to_point_l295_29500

theorem distance_center_to_point : 
  let center := (2, 3)
  let point  := (5, -2)
  let distance := Real.sqrt ((5 - 2)^2 + (-2 - 3)^2)
  distance = Real.sqrt 34 := by
  sorry

end NUMINAMATH_GPT_distance_center_to_point_l295_29500


namespace NUMINAMATH_GPT_custom_star_calc_l295_29524

-- defining the custom operation "*"
def custom_star (a b : ℤ) : ℤ :=
  a * b - (b-1) * b

-- providing the theorem statement
theorem custom_star_calc : custom_star 2 (-3) = -18 :=
  sorry

end NUMINAMATH_GPT_custom_star_calc_l295_29524


namespace NUMINAMATH_GPT_a_eq_zero_l295_29512

theorem a_eq_zero (a b : ℤ) (h : ∀ n : ℕ, ∃ k : ℤ, 2^n * a + b = k^2) : a = 0 :=
sorry

end NUMINAMATH_GPT_a_eq_zero_l295_29512


namespace NUMINAMATH_GPT_avg_of_xyz_l295_29541

-- Define the given condition
def given_condition (x y z : ℝ) := 
  (5 / 2) * (x + y + z) = 20

-- Define the question (and the proof target) using the given conditions.
theorem avg_of_xyz (x y z : ℝ) (h : given_condition x y z) : 
  (x + y + z) / 3 = 8 / 3 :=
sorry

end NUMINAMATH_GPT_avg_of_xyz_l295_29541


namespace NUMINAMATH_GPT_enrique_commission_l295_29536

def commission_earned (suits_sold: ℕ) (suit_price: ℝ) (shirts_sold: ℕ) (shirt_price: ℝ) 
                      (loafers_sold: ℕ) (loafers_price: ℝ) (commission_rate: ℝ) : ℝ :=
  let total_sales := (suits_sold * suit_price) + (shirts_sold * shirt_price) + (loafers_sold * loafers_price)
  total_sales * commission_rate

theorem enrique_commission :
  commission_earned 2 700 6 50 2 150 0.15 = 300 := by
  sorry

end NUMINAMATH_GPT_enrique_commission_l295_29536


namespace NUMINAMATH_GPT_eighth_term_of_arithmetic_sequence_l295_29560

theorem eighth_term_of_arithmetic_sequence :
  ∀ (a : ℕ → ℤ),
  (a 1 = 11) →
  (a 2 = 8) →
  (a 3 = 5) →
  (∃ (d : ℤ), ∀ n, a (n + 1) = a n + d) →
  a 8 = -10 :=
by
  intros a h1 h2 h3 arith
  sorry

end NUMINAMATH_GPT_eighth_term_of_arithmetic_sequence_l295_29560


namespace NUMINAMATH_GPT_total_students_l295_29561

theorem total_students (absent_percent : ℝ) (present_students : ℕ) (total_students : ℝ) :
  absent_percent = 0.14 → present_students = 43 → total_students * (1 - absent_percent) = present_students → total_students = 50 := 
by
  intros
  sorry

end NUMINAMATH_GPT_total_students_l295_29561


namespace NUMINAMATH_GPT_sum_D_E_F_l295_29581

theorem sum_D_E_F (D E F : ℤ) (h : ∀ x, x^3 + D * x^2 + E * x + F = (x + 3) * x * (x - 4)) : 
  D + E + F = -13 :=
by
  sorry

end NUMINAMATH_GPT_sum_D_E_F_l295_29581


namespace NUMINAMATH_GPT_additional_cars_needed_to_make_multiple_of_8_l295_29592

theorem additional_cars_needed_to_make_multiple_of_8 (current_cars : ℕ) (rows_of_cars : ℕ) (next_multiple : ℕ)
  (h1 : current_cars = 37)
  (h2 : rows_of_cars = 8)
  (h3 : next_multiple = 40)
  (h4 : next_multiple ≥ current_cars)
  (h5 : next_multiple % rows_of_cars = 0) :
  (next_multiple - current_cars) = 3 :=
by { sorry }

end NUMINAMATH_GPT_additional_cars_needed_to_make_multiple_of_8_l295_29592


namespace NUMINAMATH_GPT_x_y_iff_pos_l295_29513

theorem x_y_iff_pos (x y : ℝ) : x + y > |x - y| ↔ x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_GPT_x_y_iff_pos_l295_29513


namespace NUMINAMATH_GPT_difference_high_low_score_l295_29533

theorem difference_high_low_score :
  ∀ (num_innings : ℕ) (total_runs : ℕ) (exc_total_runs : ℕ) (high_score : ℕ) (low_score : ℕ),
  num_innings = 46 →
  total_runs = 60 * 46 →
  exc_total_runs = 58 * 44 →
  high_score = 194 →
  total_runs - exc_total_runs = high_score + low_score →
  high_score - low_score = 180 :=
by
  intros num_innings total_runs exc_total_runs high_score low_score h_innings h_total h_exc_total h_high_sum h_difference
  sorry

end NUMINAMATH_GPT_difference_high_low_score_l295_29533


namespace NUMINAMATH_GPT_math_problem_l295_29515

-- Definition of ⊕
def opp (a b : ℝ) : ℝ := a * b + a - b

-- Definition of ⊗
def tensor (a b : ℝ) : ℝ := (a * b) + a - b

theorem math_problem (a b : ℝ) :
  opp a b + tensor (b - a) b = b^2 - b := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l295_29515


namespace NUMINAMATH_GPT_probability_of_white_crows_remain_same_l295_29595

theorem probability_of_white_crows_remain_same (a b c d : ℕ) (h1 : a + b = 50) (h2 : c + d = 50) 
  (ha1 : a > 0) (h3 : b ≥ a) (h4 : d ≥ c - 1) :
  ((b - a) * (d - c) + a + b) / (50 * 51) > (bc + ad) / (50 * 51)
:= by
  -- We need to show that the probability of the number of white crows on the birch remaining the same 
  -- is greater than the probability of it changing.
  sorry

end NUMINAMATH_GPT_probability_of_white_crows_remain_same_l295_29595


namespace NUMINAMATH_GPT_p_minus_q_value_l295_29514

theorem p_minus_q_value (p q : ℝ) (h1 : (x - 4) * (x + 4) = 24 * x - 96) (h2 : x^2 - 24 * x + 80 = 0) (h3 : p = 20) (h4 : q = 4) : p - q = 16 :=
by
  sorry

end NUMINAMATH_GPT_p_minus_q_value_l295_29514


namespace NUMINAMATH_GPT_divisible_by_3_l295_29526

theorem divisible_by_3 (x y : ℤ) (h : (x^2 + y^2) % 3 = 0) : x % 3 = 0 ∧ y % 3 = 0 :=
sorry

end NUMINAMATH_GPT_divisible_by_3_l295_29526


namespace NUMINAMATH_GPT_installation_cost_l295_29547

theorem installation_cost (P I : ℝ) (h₁ : 0.80 * P = 12500)
  (h₂ : 18400 = 1.15 * (12500 + 125 + I)) :
  I = 3375 :=
by
  sorry

end NUMINAMATH_GPT_installation_cost_l295_29547


namespace NUMINAMATH_GPT_apples_harvested_l295_29546

theorem apples_harvested (weight_juice weight_restaurant weight_per_bag sales_price total_sales : ℤ) 
  (h1 : weight_juice = 90) 
  (h2 : weight_restaurant = 60) 
  (h3 : weight_per_bag = 5) 
  (h4 : sales_price = 8) 
  (h5 : total_sales = 408) : 
  (weight_juice + weight_restaurant + (total_sales / sales_price) * weight_per_bag = 405) :=
by
  sorry

end NUMINAMATH_GPT_apples_harvested_l295_29546


namespace NUMINAMATH_GPT_susan_avg_speed_l295_29528

theorem susan_avg_speed 
  (speed1 : ℕ)
  (distance1 : ℕ)
  (speed2 : ℕ)
  (distance2 : ℕ)
  (no_stops : Prop) 
  (H1 : speed1 = 15)
  (H2 : distance1 = 40)
  (H3 : speed2 = 60)
  (H4 : distance2 = 20)
  (H5 : no_stops) :
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let avg_speed := total_distance / total_time
  avg_speed = 20 := by
  sorry

end NUMINAMATH_GPT_susan_avg_speed_l295_29528


namespace NUMINAMATH_GPT_imaginary_part_of_z_is_2_l295_29588

noncomputable def z : ℂ := (3 * Complex.I + 1) / (1 - Complex.I)

theorem imaginary_part_of_z_is_2 : z.im = 2 := 
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_imaginary_part_of_z_is_2_l295_29588


namespace NUMINAMATH_GPT_exists_x_y_not_divisible_by_3_l295_29578

theorem exists_x_y_not_divisible_by_3 (k : ℕ) (hk : 0 < k) :
  ∃ (x y : ℤ), (x^2 + 2 * y^2 = 3^k) ∧ (¬ (x % 3 = 0)) ∧ (¬ (y % 3 = 0)) :=
sorry

end NUMINAMATH_GPT_exists_x_y_not_divisible_by_3_l295_29578


namespace NUMINAMATH_GPT_min_value_ab_l295_29502

theorem min_value_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (a / 2) + b = 1) :
  (1 / a) + (1 / b) = (3 / 2) + Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_min_value_ab_l295_29502


namespace NUMINAMATH_GPT_sugar_percentage_after_additions_l295_29590

noncomputable def initial_solution_volume : ℝ := 440
noncomputable def initial_water_percentage : ℝ := 0.88
noncomputable def initial_kola_percentage : ℝ := 0.08
noncomputable def initial_sugar_percentage : ℝ := 1 - initial_water_percentage - initial_kola_percentage
noncomputable def sugar_added : ℝ := 3.2
noncomputable def water_added : ℝ := 10
noncomputable def kola_added : ℝ := 6.8

noncomputable def initial_sugar_amount := initial_sugar_percentage * initial_solution_volume
noncomputable def new_sugar_amount := initial_sugar_amount + sugar_added
noncomputable def new_solution_volume := initial_solution_volume + sugar_added + water_added + kola_added

noncomputable def final_sugar_percentage := (new_sugar_amount / new_solution_volume) * 100

theorem sugar_percentage_after_additions :
    final_sugar_percentage = 4.52 :=
by
    sorry

end NUMINAMATH_GPT_sugar_percentage_after_additions_l295_29590


namespace NUMINAMATH_GPT_jose_land_division_l295_29540

/-- Let the total land Jose bought be 20000 square meters. Let Jose divide this land equally among himself and his four siblings. Prove that the land Jose will have after dividing it is 4000 square meters. -/
theorem jose_land_division : 
  let total_land := 20000
  let numberOfPeople := 5
  total_land / numberOfPeople = 4000 := by
sorry

end NUMINAMATH_GPT_jose_land_division_l295_29540


namespace NUMINAMATH_GPT_frogs_moving_l295_29550

theorem frogs_moving (initial_frogs tadpoles mature_frogs pond_capacity frogs_to_move : ℕ)
  (h1 : initial_frogs = 5)
  (h2 : tadpoles = 3 * initial_frogs)
  (h3 : mature_frogs = (2 * tadpoles) / 3)
  (h4 : pond_capacity = 8)
  (h5 : frogs_to_move = (initial_frogs + mature_frogs) - pond_capacity) :
  frogs_to_move = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_frogs_moving_l295_29550


namespace NUMINAMATH_GPT_train_speed_is_72_kmh_l295_29566

noncomputable def train_length : ℝ := 110
noncomputable def bridge_length : ℝ := 175
noncomputable def crossing_time : ℝ := 14.248860091192705

theorem train_speed_is_72_kmh :
  (train_length + bridge_length) / crossing_time * 3.6 = 72 := by
  sorry

end NUMINAMATH_GPT_train_speed_is_72_kmh_l295_29566


namespace NUMINAMATH_GPT_max_value_of_expression_l295_29507

theorem max_value_of_expression (x y z : ℝ) (h : 0 < x) (h' : 0 < y) (h'' : 0 < z) (hxyz : x * y * z = 1) :
  (∃ s, s = x ∧ ∃ t, t = y ∧ ∃ u, u = z ∧ 
  (x^2 * y / (x + y) + y^2 * z / (y + z) + z^2 * x / (z + x) ≤ 3 / 2)) :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l295_29507


namespace NUMINAMATH_GPT_min_value_96_l295_29509

noncomputable def min_value (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_xyz : x * y * z = 32) : ℝ :=
x^2 + 4 * x * y + 4 * y^2 + 2 * z^2

theorem min_value_96 (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_xyz : x * y * z = 32) :
  min_value x y z h_pos h_xyz = 96 :=
sorry

end NUMINAMATH_GPT_min_value_96_l295_29509


namespace NUMINAMATH_GPT_students_paid_half_l295_29568

theorem students_paid_half (F H : ℕ) 
  (h1 : F + H = 25)
  (h2 : 50 * F + 25 * H = 1150) : 
  H = 4 := by
  sorry

end NUMINAMATH_GPT_students_paid_half_l295_29568


namespace NUMINAMATH_GPT_total_bricks_used_l295_29597

-- Definitions for conditions
def num_courses_per_wall : Nat := 10
def num_bricks_per_course : Nat := 20
def num_complete_walls : Nat := 5
def incomplete_wall_missing_courses : Nat := 3

-- Lean statement to prove the mathematically equivalent problem
theorem total_bricks_used : 
  (num_complete_walls * (num_courses_per_wall * num_bricks_per_course) + 
  ((num_courses_per_wall - incomplete_wall_missing_courses) * num_bricks_per_course)) = 1140 :=
by
  sorry

end NUMINAMATH_GPT_total_bricks_used_l295_29597


namespace NUMINAMATH_GPT_part1_part2_l295_29565

-- Define the function f(x) = |x - 1| + |x - 2|
def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2)

-- Prove the statement about f(x) and the inequality
theorem part1 : { x : ℝ | (2 / 3) ≤ x ∧ x ≤ 4 } ⊆ { x : ℝ | f x ≤ x + 1 } :=
sorry

-- State k = 1 as the minimum value of f(x)
def k : ℝ := 1

-- Prove the non-existence of positive a and b satisfying the given conditions
theorem part2 : ¬ ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ 2 * a + b = k ∧ (1 / a + 2 / b = 4) :=
sorry

end NUMINAMATH_GPT_part1_part2_l295_29565


namespace NUMINAMATH_GPT_average_age_of_girls_l295_29587

theorem average_age_of_girls (total_students : ℕ) (avg_age_boys : ℕ) (num_girls : ℕ) (avg_age_school : ℚ) 
  (h1 : total_students = 604) 
  (h2 : avg_age_boys = 12) 
  (h3 : num_girls = 151) 
  (h4 : avg_age_school = 11.75) : 
  (total_age_of_girls / num_girls) = 11 :=
by
  -- Definitions
  let num_boys := total_students - num_girls
  let total_age := avg_age_school * total_students
  let total_age_boys := avg_age_boys * num_boys
  let total_age_girls := total_age - total_age_boys
  -- Proof goal
  have : total_age_of_girls = total_age_girls := sorry
  have : total_age_of_girls / num_girls = 11 := sorry
  sorry

end NUMINAMATH_GPT_average_age_of_girls_l295_29587


namespace NUMINAMATH_GPT_jonas_shoes_l295_29564

theorem jonas_shoes (socks pairs_of_pants t_shirts shoes : ℕ) (new_socks : ℕ) (h1 : socks = 20) (h2 : pairs_of_pants = 10) (h3 : t_shirts = 10) (h4 : new_socks = 35 ∧ (socks + new_socks = 35)) :
  shoes = 35 :=
by
  sorry

end NUMINAMATH_GPT_jonas_shoes_l295_29564


namespace NUMINAMATH_GPT_Christina_weekly_distance_l295_29539

/-- 
Prove that Christina covered 74 kilometers that week given the following conditions:
1. Christina walks 7km to school every day from Monday to Friday.
2. She returns home covering the same distance each day.
3. Last Friday, she had to pass by her friend, which is another 2km away from the school in the opposite direction from home.
-/
theorem Christina_weekly_distance : 
  let distance_to_school := 7
  let days_school := 5
  let extra_distance_Friday := 2
  let daily_distance := 2 * distance_to_school
  let total_distance_from_Monday_to_Thursday := 4 * daily_distance
  let distance_on_Friday := daily_distance + 2 * extra_distance_Friday
  total_distance_from_Monday_to_Thursday + distance_on_Friday = 74 := 
by
  sorry

end NUMINAMATH_GPT_Christina_weekly_distance_l295_29539


namespace NUMINAMATH_GPT_inequality_exponentiation_l295_29585

theorem inequality_exponentiation (a b c : ℝ) (ha : 0 < a) (hab : a < b) (hb : b < 1) (hc : c > 1) : 
  a * b^c > b * a^c := 
sorry

end NUMINAMATH_GPT_inequality_exponentiation_l295_29585


namespace NUMINAMATH_GPT_pies_calculation_l295_29552

-- Definition: Number of ingredients per pie
def ingredients_per_pie (apples total_apples pies : ℤ) : ℤ := total_apples / pies

-- Definition: Number of pies that can be made with available ingredients 
def pies_from_ingredients (ingredient_amount per_pie : ℤ) : ℤ := ingredient_amount / per_pie

-- Hypothesis
theorem pies_calculation (apples_per_pie pears_per_pie apples pears pies : ℤ) 
  (h1: ingredients_per_pie apples 12 pies = 4)
  (h2: ingredients_per_pie apples 6 pies = 2)
  (h3: pies_from_ingredients 36 4 = 9)
  (h4: pies_from_ingredients 18 2 = 9): 
  pies = 9 := 
sorry

end NUMINAMATH_GPT_pies_calculation_l295_29552


namespace NUMINAMATH_GPT_triangle_angle_A_l295_29520

theorem triangle_angle_A (a c C A : Real) (h1 : a = 1) (h2 : c = Real.sqrt 3) (h3 : C = 2 * Real.pi / 3) 
(h4 : Real.sin A = 1 / 2) : A = Real.pi / 6 :=
sorry

end NUMINAMATH_GPT_triangle_angle_A_l295_29520


namespace NUMINAMATH_GPT_band_fundraising_goal_exceed_l295_29562

theorem band_fundraising_goal_exceed
    (goal : ℕ)
    (basic_wash_cost deluxe_wash_cost premium_wash_cost cookie_cost : ℕ)
    (basic_wash_families deluxe_wash_families premium_wash_families sold_cookies : ℕ)
    (total_earnings : ℤ) :
    
    goal = 150 →
    basic_wash_cost = 5 →
    deluxe_wash_cost = 8 →
    premium_wash_cost = 12 →
    cookie_cost = 2 →
    basic_wash_families = 10 →
    deluxe_wash_families = 6 →
    premium_wash_families = 2 →
    sold_cookies = 30 →
    total_earnings = 
        (basic_wash_cost * basic_wash_families +
         deluxe_wash_cost * deluxe_wash_families +
         premium_wash_cost * premium_wash_families +
         cookie_cost * sold_cookies : ℤ) →
    (goal : ℤ) - total_earnings = -32 :=
by
  intros h_goal h_basic h_deluxe h_premium h_cookie h_basic_fam h_deluxe_fam h_premium_fam h_sold_cookies h_total_earnings
  sorry

end NUMINAMATH_GPT_band_fundraising_goal_exceed_l295_29562


namespace NUMINAMATH_GPT_max_T_n_at_2_l295_29516

noncomputable def geom_seq (a n : ℕ) : ℕ :=
  a * 2 ^ n

noncomputable def S_n (a n : ℕ) : ℕ :=
  a * (2 ^ n - 1)

noncomputable def T_n (a n : ℕ) : ℕ :=
  (17 * S_n a n - S_n a (2 * n)) / geom_seq a n

theorem max_T_n_at_2 (a : ℕ) : (∀ n > 0, T_n a n ≤ T_n a 2) :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_max_T_n_at_2_l295_29516


namespace NUMINAMATH_GPT_pieces_picked_by_olivia_l295_29577

-- Define the conditions
def picked_by_edward : ℕ := 3
def total_picked : ℕ := 19

-- Prove the number of pieces picked up by Olivia
theorem pieces_picked_by_olivia (O : ℕ) (h : O + picked_by_edward = total_picked) : O = 16 :=
by sorry

end NUMINAMATH_GPT_pieces_picked_by_olivia_l295_29577


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l295_29535

theorem common_ratio_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_nonzero : d ≠ 0) 
  (h_geom : (a 1)^2 = a 0 * a 2) :
  (a 2) / (a 0) = 3 / 2 := 
sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l295_29535


namespace NUMINAMATH_GPT_batsman_average_after_25th_innings_l295_29518

theorem batsman_average_after_25th_innings :
  ∃ A : ℝ, 
    (∀ s : ℝ, s = 25 * A + 62.5 → 24 * A + 95 = s) →
    A + 2.5 = 35 :=
by
  sorry

end NUMINAMATH_GPT_batsman_average_after_25th_innings_l295_29518


namespace NUMINAMATH_GPT_decreasing_function_range_a_l295_29553

noncomputable def f (a x : ℝ) : ℝ := -x^3 + x^2 + a * x

theorem decreasing_function_range_a (a : ℝ) :
  (∀ x : ℝ, deriv (f a) x ≤ 0) ↔ a ≤ -(1/3) :=
by
  -- This is a placeholder for the proof.
  sorry

end NUMINAMATH_GPT_decreasing_function_range_a_l295_29553


namespace NUMINAMATH_GPT_second_section_area_l295_29555

theorem second_section_area 
  (sod_area_per_square : ℕ := 4)
  (total_squares : ℕ := 1500)
  (first_section_length : ℕ := 30)
  (first_section_width : ℕ := 40)
  (total_area_needed : ℕ := total_squares * sod_area_per_square)
  (first_section_area : ℕ := first_section_length * first_section_width) :
  total_area_needed = first_section_area + 4800 := 
by 
  sorry

end NUMINAMATH_GPT_second_section_area_l295_29555


namespace NUMINAMATH_GPT_fill_cistern_7_2_hours_l295_29523

theorem fill_cistern_7_2_hours :
  let R_fill := 1 / 4
  let R_empty := 1 / 9
  R_fill - R_empty = 5 / 36 →
  1 / (R_fill - R_empty) = 7.2 := 
by
  intros
  sorry

end NUMINAMATH_GPT_fill_cistern_7_2_hours_l295_29523


namespace NUMINAMATH_GPT_percentage_failed_in_hindi_l295_29571

theorem percentage_failed_in_hindi (P_E : ℝ) (P_H_and_E : ℝ) (P_P : ℝ) (H : ℝ) : 
  P_E = 0.5 ∧ P_H_and_E = 0.25 ∧ P_P = 0.5 → H = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_failed_in_hindi_l295_29571


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_geometric_sequence_properties_l295_29574

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) : ℕ :=
  2 * n - 1

-- Define the sum of the first n terms of {a_n}
def S (n : ℕ) : ℕ :=
  n ^ 2

-- Prove the nth term and the sum of the first n terms of {a_n}
theorem arithmetic_sequence_properties (n : ℕ) :
  a n = 2 * n - 1 ∧ S n = n ^ 2 :=
by sorry

-- Define the geometric sequence {b_n}
def b (n : ℕ) : ℕ :=
  2 ^ (2 * n - 1)

-- Define the sum of the first n terms of {b_n}
def T (n : ℕ) : ℕ :=
  (2 ^ n * (4 ^ n - 1)) / 3

-- Prove the nth term and the sum of the first n terms of {b_n}
theorem geometric_sequence_properties (n : ℕ) (a4 S4 : ℕ) (q : ℕ)
  (h_a4 : a4 = a 4)
  (h_S4 : S4 = S 4)
  (h_q : q ^ 2 - (a4 + 1) * q + S4 = 0) :
  b n = 2 ^ (2 * n - 1) ∧ T n = (2 ^ n * (4 ^ n - 1)) / 3 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_properties_geometric_sequence_properties_l295_29574


namespace NUMINAMATH_GPT_cost_per_lunch_is_7_l295_29530

-- Definitions of the conditions
def total_children := 35
def total_chaperones := 5
def janet := 1
def additional_lunches := 3
def total_cost := 308

-- Calculate the total number of lunches
def total_lunches : Int :=
  total_children + total_chaperones + janet + additional_lunches

-- Statement to prove that the cost per lunch is 7
theorem cost_per_lunch_is_7 : total_cost / total_lunches = 7 := by
  sorry

end NUMINAMATH_GPT_cost_per_lunch_is_7_l295_29530


namespace NUMINAMATH_GPT_largest_fraction_l295_29517

theorem largest_fraction :
  let A := (5 : ℚ) / 11
  let B := (7 : ℚ) / 15
  let C := (29 : ℚ) / 59
  let D := (200 : ℚ) / 399
  let E := (251 : ℚ) / 501
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end NUMINAMATH_GPT_largest_fraction_l295_29517


namespace NUMINAMATH_GPT_unique_zero_f_x1_minus_2x2_l295_29501

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x

-- Condition a ≥ 0
variable (a : ℝ) (a_nonneg : 0 ≤ a)

-- Define the first part of the problem
theorem unique_zero_f : ∃! x, f a x = 0 :=
  sorry

-- Variables for the second part of the problem
variable (x₁ x₂ : ℝ)
variable (cond : f a x₁ = g a x₁ - g a x₂)

-- Define the second part of the problem
theorem x1_minus_2x2 : x₁ - 2 * x₂ ≥ 1 - 2 * Real.log 2 :=
  sorry

end NUMINAMATH_GPT_unique_zero_f_x1_minus_2x2_l295_29501


namespace NUMINAMATH_GPT_remainder_9_minus_n_plus_n_plus_5_mod_8_l295_29537

theorem remainder_9_minus_n_plus_n_plus_5_mod_8 (n : ℤ) : 
  ((9 - n) + (n + 5)) % 8 = 6 := by
  sorry

end NUMINAMATH_GPT_remainder_9_minus_n_plus_n_plus_5_mod_8_l295_29537


namespace NUMINAMATH_GPT_total_heads_is_46_l295_29544

noncomputable def total_heads (hens cows : ℕ) : ℕ :=
  hens + cows

def num_feet_hens (num_hens : ℕ) : ℕ :=
  2 * num_hens

def num_cows (total_feet feet_hens_per_cow feet_cow_per_cow : ℕ) : ℕ :=
  (total_feet - feet_hens_per_cow) / feet_cow_per_cow

theorem total_heads_is_46 (num_hens : ℕ) (total_feet : ℕ)
  (hen_feet cow_feet hen_head cow_head : ℕ)
  (num_heads : ℕ) :
  num_hens = 24 →
  total_feet = 136 →
  hen_feet = 2 →
  cow_feet = 4 →
  hen_head = 1 →
  cow_head = 1 →
  num_heads = total_heads num_hens (num_cows total_feet (num_feet_hens num_hens) cow_feet) →
  num_heads = 46 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_heads_is_46_l295_29544


namespace NUMINAMATH_GPT_find_Δ_l295_29598

-- Define the constants and conditions
variables (Δ p : ℕ)
axiom condition1 : Δ + p = 84
axiom condition2 : (Δ + p) + p = 153

-- State the theorem
theorem find_Δ : Δ = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_Δ_l295_29598


namespace NUMINAMATH_GPT_value_of_d_l295_29508

theorem value_of_d (y : ℝ) (d : ℝ) (h1 : y > 0) (h2 : (4 * y) / 20 + (3 * y) / d = 0.5 * y) : d = 10 :=
by
  sorry

end NUMINAMATH_GPT_value_of_d_l295_29508


namespace NUMINAMATH_GPT_sequence_solution_l295_29594

theorem sequence_solution (a : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ (m n : ℕ), 0 < m → 0 < n → |a n - a m| ≤ (2 * m * n) / (m ^ 2 + n ^ 2)) :
  ∀ (n : ℕ), a n = 1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_solution_l295_29594


namespace NUMINAMATH_GPT_probability_of_red_ball_l295_29506

theorem probability_of_red_ball :
  let total_balls := 9
  let red_balls := 6
  let probability := (red_balls : ℚ) / total_balls
  probability = (2 : ℚ) / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_red_ball_l295_29506


namespace NUMINAMATH_GPT_ji_hoon_original_answer_l295_29532

-- Define the conditions: Ji-hoon's mistake
def ji_hoon_mistake (x : ℝ) := x - 7 = 0.45

-- The theorem statement
theorem ji_hoon_original_answer (x : ℝ) (h : ji_hoon_mistake x) : x * 7 = 52.15 :=
by
  sorry

end NUMINAMATH_GPT_ji_hoon_original_answer_l295_29532


namespace NUMINAMATH_GPT_isosceles_triangle_angle_l295_29593

theorem isosceles_triangle_angle {x : ℝ} (hx0 : 0 < x) (hx1 : x < 90) (hx2 : 2 * x = 180 / 7) : x = 180 / 7 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_angle_l295_29593


namespace NUMINAMATH_GPT_initial_oil_amounts_l295_29580

-- Definitions related to the problem
variables (A0 B0 C0 : ℝ)
variables (x : ℝ)

-- Conditions given in the problem
def bucketC_initial := C0 = 48
def transferA_to_B := x = 64 ∧ 64 = (2/3 * A0)
def transferB_to_C := x = 64 ∧ 64 = ((4/5 * (B0 + 1/3 * A0)) * (1/5 + 1))

-- Proof statement to show the solutions
theorem initial_oil_amounts (A0 B0 : ℝ) (C0 x : ℝ) 
  (h1 : bucketC_initial C0)
  (h2 : transferA_to_B A0 x)
  (h3 : transferB_to_C B0 A0 x) :
  A0 = 96 ∧ B0 = 48 :=
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_initial_oil_amounts_l295_29580


namespace NUMINAMATH_GPT_quadratic_root_square_condition_l295_29557

theorem quadratic_root_square_condition (p q r : ℝ) 
  (h1 : ∃ α β : ℝ, α + β = -q / p ∧ α * β = r / p ∧ β = α^2) : p - 4 * q ≥ 0 :=
sorry

end NUMINAMATH_GPT_quadratic_root_square_condition_l295_29557


namespace NUMINAMATH_GPT_fraction_integer_l295_29525

theorem fraction_integer (x y : ℤ) (h₁ : ∃ k : ℤ, 3 * x + 4 * y = 5 * k) : ∃ m : ℤ, 4 * x - 3 * y = 5 * m :=
by
  sorry

end NUMINAMATH_GPT_fraction_integer_l295_29525
