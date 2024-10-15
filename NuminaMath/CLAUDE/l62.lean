import Mathlib

namespace NUMINAMATH_CALUDE_alex_paper_distribution_l62_6242

/-- The number of ways to distribute n distinct items to m recipients,
    where each recipient can receive multiple items. -/
def distribution_ways (n m : ℕ) : ℕ := m^n

/-- The problem statement -/
theorem alex_paper_distribution :
  distribution_ways 5 10 = 100000 := by
  sorry

end NUMINAMATH_CALUDE_alex_paper_distribution_l62_6242


namespace NUMINAMATH_CALUDE_max_distinct_letters_exists_table_with_11_letters_l62_6260

/-- Represents a 5x5 table of letters -/
def LetterTable := Fin 5 → Fin 5 → Char

/-- Checks if a row contains at most 3 different letters -/
def rowValid (table : LetterTable) (row : Fin 5) : Prop :=
  (Finset.image (λ col => table row col) Finset.univ).card ≤ 3

/-- Checks if a column contains at most 3 different letters -/
def colValid (table : LetterTable) (col : Fin 5) : Prop :=
  (Finset.image (λ row => table row col) Finset.univ).card ≤ 3

/-- Checks if the entire table is valid -/
def tableValid (table : LetterTable) : Prop :=
  (∀ row, rowValid table row) ∧ (∀ col, colValid table col)

/-- Counts the number of different letters in the table -/
def distinctLetters (table : LetterTable) : ℕ :=
  (Finset.image (λ (row, col) => table row col) (Finset.univ.product Finset.univ)).card

/-- The main theorem stating that the maximum number of distinct letters is 11 -/
theorem max_distinct_letters :
  ∀ (table : LetterTable), tableValid table → distinctLetters table ≤ 11 :=
sorry

/-- There exists a valid table with exactly 11 distinct letters -/
theorem exists_table_with_11_letters :
  ∃ (table : LetterTable), tableValid table ∧ distinctLetters table = 11 :=
sorry

end NUMINAMATH_CALUDE_max_distinct_letters_exists_table_with_11_letters_l62_6260


namespace NUMINAMATH_CALUDE_waste_after_ten_years_l62_6295

/-- Calculates the amount of waste after n years given an initial amount and growth rate -/
def wasteAmount (a : ℝ) (b : ℝ) (n : ℕ) : ℝ :=
  a * (1 + b) ^ n

/-- Theorem: The amount of waste after 10 years is a(1+b)^10 -/
theorem waste_after_ten_years (a b : ℝ) :
  wasteAmount a b 10 = a * (1 + b) ^ 10 := by
  sorry

#check waste_after_ten_years

end NUMINAMATH_CALUDE_waste_after_ten_years_l62_6295


namespace NUMINAMATH_CALUDE_total_pencils_l62_6275

theorem total_pencils (drawer : Real) (desk_initial : Real) (pencil_case : Real) (dan_added : Real)
  (h1 : drawer = 43.5)
  (h2 : desk_initial = 19.25)
  (h3 : pencil_case = 8.75)
  (h4 : dan_added = 16) :
  drawer + desk_initial + pencil_case + dan_added = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l62_6275


namespace NUMINAMATH_CALUDE_triangle_theorem_l62_6274

noncomputable section

variables {a b c : ℝ} {A B C : ℝ}

def triangle_area (a b c : ℝ) : ℝ := (1/4) * Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))

theorem triangle_theorem 
  (h1 : b^2 + c^2 - a^2 = a*c*(Real.cos C) + c^2*(Real.cos A))
  (h2 : 0 < A ∧ A < π)
  (h3 : 0 < B ∧ B < π)
  (h4 : 0 < C ∧ C < π)
  (h5 : A + B + C = π)
  (h6 : a*Real.sin B = b*Real.sin A)
  (h7 : b*Real.sin C = c*Real.sin B)
  (h8 : triangle_area a b c = 25*(Real.sqrt 3)/4)
  (h9 : a = 5) :
  A = π/3 ∧ Real.sin B + Real.sin C = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l62_6274


namespace NUMINAMATH_CALUDE_ball_count_in_box_l62_6225

theorem ball_count_in_box (n : ℕ) (yellow_count : ℕ) (prob_yellow : ℚ) : 
  yellow_count = 9 → prob_yellow = 3/10 → (yellow_count : ℚ) / n = prob_yellow → n = 30 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_in_box_l62_6225


namespace NUMINAMATH_CALUDE_jake_weight_proof_l62_6278

/-- Jake's present weight in pounds -/
def jake_weight : ℝ := 108

/-- Jake's sister's weight in pounds -/
def sister_weight : ℝ := 48

/-- The combined weight of Jake and his sister in pounds -/
def combined_weight : ℝ := 156

theorem jake_weight_proof :
  (jake_weight - 12 = 2 * sister_weight) ∧
  (jake_weight + sister_weight = combined_weight) →
  jake_weight = 108 :=
by sorry

end NUMINAMATH_CALUDE_jake_weight_proof_l62_6278


namespace NUMINAMATH_CALUDE_orange_juice_mixture_fraction_l62_6211

/-- Represents the fraction of orange juice in a mixture of two pitchers -/
def orange_juice_fraction (capacity1 capacity2 : ℚ) (fraction1 fraction2 : ℚ) : ℚ :=
  (capacity1 * fraction1 + capacity2 * fraction2) / (capacity1 + capacity2)

/-- Theorem stating that the fraction of orange juice in the given mixture is 17/52 -/
theorem orange_juice_mixture_fraction :
  orange_juice_fraction 500 800 (1/4) (3/8) = 17/52 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_mixture_fraction_l62_6211


namespace NUMINAMATH_CALUDE_james_brothers_count_l62_6230

def market_value : ℝ := 500000
def selling_price : ℝ := market_value * 1.2
def revenue_after_taxes : ℝ := selling_price * 0.9
def share_per_person : ℝ := 135000

theorem james_brothers_count :
  ∃ (n : ℕ), (revenue_after_taxes / (n + 1 : ℝ) = share_per_person) ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_james_brothers_count_l62_6230


namespace NUMINAMATH_CALUDE_tan_2y_value_l62_6294

theorem tan_2y_value (x y : ℝ) 
  (h : Real.sin (x - y) * Real.cos x - Real.cos (x - y) * Real.sin x = 3/5) : 
  Real.tan (2 * y) = 24/7 ∨ Real.tan (2 * y) = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_2y_value_l62_6294


namespace NUMINAMATH_CALUDE_chocolate_division_l62_6204

/-- Represents the number of chocolate pieces Maria has after a given number of days -/
def chocolatePieces (days : ℕ) : ℕ :=
  9 + 8 * days

theorem chocolate_division :
  (chocolatePieces 3 = 25) ∧
  (∀ n : ℕ, chocolatePieces n ≠ 2014) :=
by sorry

end NUMINAMATH_CALUDE_chocolate_division_l62_6204


namespace NUMINAMATH_CALUDE_elsa_final_marbles_l62_6247

/-- Calculates the final number of marbles Elsa has at the end of the day. -/
def elsas_marbles (initial : ℕ) (lost_breakfast : ℕ) (given_to_susie : ℕ) (received_from_mom : ℕ) : ℕ :=
  initial - lost_breakfast - given_to_susie + received_from_mom + 2 * given_to_susie

/-- Theorem stating that Elsa ends up with 54 marbles given the conditions of the problem. -/
theorem elsa_final_marbles :
  elsas_marbles 40 3 5 12 = 54 :=
by sorry

end NUMINAMATH_CALUDE_elsa_final_marbles_l62_6247


namespace NUMINAMATH_CALUDE_circle_of_students_l62_6224

theorem circle_of_students (n : ℕ) (h : n > 0) :
  (∃ (a b : ℕ), a < n ∧ b < n ∧ a = 6 ∧ b = 16 ∧ (b - a) * 2 + 2 = n) →
  n = 22 :=
by sorry

end NUMINAMATH_CALUDE_circle_of_students_l62_6224


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l62_6217

/-- The volume of a sphere inscribed in a right circular cylinder -/
theorem inscribed_sphere_volume (h : ℝ) (d : ℝ) (h_pos : h > 0) (d_pos : d > 0) :
  let r : ℝ := d / 2
  let cylinder_volume : ℝ := π * r^2 * h
  let sphere_volume : ℝ := (4/3) * π * r^3
  h = 12 ∧ d = 10 → sphere_volume = (500/3) * π := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l62_6217


namespace NUMINAMATH_CALUDE_john_weight_on_bar_l62_6296

/-- The weight John can put on the bar given the weight bench capacity, safety margin, and his own weight -/
def weight_on_bar (bench_capacity : ℝ) (safety_margin : ℝ) (john_weight : ℝ) : ℝ :=
  bench_capacity * (1 - safety_margin) - john_weight

/-- Theorem stating the weight John can put on the bar -/
theorem john_weight_on_bar :
  weight_on_bar 1000 0.2 250 = 550 := by
  sorry

end NUMINAMATH_CALUDE_john_weight_on_bar_l62_6296


namespace NUMINAMATH_CALUDE_total_profit_is_2034_l62_6245

/-- Represents a group of piglets with their selling and feeding information -/
structure PigletGroup where
  count : Nat
  sellingPrice : Nat
  sellingTime : Nat
  initialFeedCost : Nat
  initialFeedTime : Nat
  laterFeedCost : Nat
  laterFeedTime : Nat

/-- Calculates the profit for a single piglet group -/
def groupProfit (group : PigletGroup) : Int :=
  group.count * group.sellingPrice - 
  group.count * (group.initialFeedCost * group.initialFeedTime + 
                 group.laterFeedCost * group.laterFeedTime)

/-- The farmer's piglet groups -/
def pigletGroups : List PigletGroup := [
  ⟨3, 375, 11, 13, 8, 15, 3⟩,
  ⟨4, 425, 14, 14, 5, 16, 9⟩,
  ⟨2, 475, 18, 15, 10, 18, 8⟩,
  ⟨1, 550, 20, 20, 20, 20, 0⟩
]

/-- Theorem stating the total profit is $2034 -/
theorem total_profit_is_2034 : 
  (pigletGroups.map groupProfit).sum = 2034 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_2034_l62_6245


namespace NUMINAMATH_CALUDE_determinant_transformation_l62_6288

theorem determinant_transformation (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = 9 →
  Matrix.det !![2*p, 5*p + 4*q; 2*r, 5*r + 4*s] = 72 := by
  sorry

end NUMINAMATH_CALUDE_determinant_transformation_l62_6288


namespace NUMINAMATH_CALUDE_sum_of_squares_modulo_prime_sum_of_squares_zero_modulo_prime_1mod4_sum_of_squares_nonzero_modulo_prime_3mod4_l62_6272

theorem sum_of_squares_modulo_prime (p n : ℤ) (hp : Prime p) (hp5 : p > 5) :
  ∃ x y : ℤ, x % p ≠ 0 ∧ y % p ≠ 0 ∧ (x^2 + y^2) % p = n % p :=
sorry

theorem sum_of_squares_zero_modulo_prime_1mod4 (p : ℤ) (hp : Prime p) (hp1 : p % 4 = 1) :
  ∃ x y : ℤ, x % p ≠ 0 ∧ y % p ≠ 0 ∧ (x^2 + y^2) % p = 0 :=
sorry

theorem sum_of_squares_nonzero_modulo_prime_3mod4 (p : ℤ) (hp : Prime p) (hp3 : p % 4 = 3) :
  ∀ x y : ℤ, x % p ≠ 0 → y % p ≠ 0 → (x^2 + y^2) % p ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_squares_modulo_prime_sum_of_squares_zero_modulo_prime_1mod4_sum_of_squares_nonzero_modulo_prime_3mod4_l62_6272


namespace NUMINAMATH_CALUDE_water_bottles_problem_l62_6276

theorem water_bottles_problem (initial_bottles : ℕ) : 
  (3 * (initial_bottles - 3) = 21) → initial_bottles = 10 := by
  sorry

end NUMINAMATH_CALUDE_water_bottles_problem_l62_6276


namespace NUMINAMATH_CALUDE_checkerboard_matching_sum_l62_6299

/-- Row-wise numbering function -/
def f (i j : ℕ) : ℕ := 19 * (i - 1) + j

/-- Column-wise numbering function -/
def g (i j : ℕ) : ℕ := 15 * (j - 1) + i

/-- The set of pairs (i, j) where the numbers match in both systems -/
def matching_squares : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => f p.1 p.2 = g p.1 p.2)
    (Finset.product (Finset.range 15) (Finset.range 19))

theorem checkerboard_matching_sum :
  (matching_squares.sum fun p => f p.1 p.2) = 668 := by
  sorry


end NUMINAMATH_CALUDE_checkerboard_matching_sum_l62_6299


namespace NUMINAMATH_CALUDE_club_sports_intersection_l62_6234

/-- Given a club with 310 members, where 138 play tennis, 255 play baseball,
    and 11 play no sports, prove that 94 people play both tennis and baseball. -/
theorem club_sports_intersection (total : ℕ) (tennis : ℕ) (baseball : ℕ) (no_sport : ℕ)
    (h_total : total = 310)
    (h_tennis : tennis = 138)
    (h_baseball : baseball = 255)
    (h_no_sport : no_sport = 11) :
    tennis + baseball - (total - no_sport) = 94 := by
  sorry

end NUMINAMATH_CALUDE_club_sports_intersection_l62_6234


namespace NUMINAMATH_CALUDE_beaver_group_size_l62_6221

/-- The number of beavers in the first group -/
def first_group_size : ℕ := 20

/-- The time taken by the first group to build the dam (in hours) -/
def first_group_time : ℕ := 3

/-- The number of beavers in the second group -/
def second_group_size : ℕ := 12

/-- The time taken by the second group to build the dam (in hours) -/
def second_group_time : ℕ := 5

/-- Theorem stating that the first group size is 20 beavers -/
theorem beaver_group_size :
  first_group_size * first_group_time = second_group_size * second_group_time :=
by sorry

end NUMINAMATH_CALUDE_beaver_group_size_l62_6221


namespace NUMINAMATH_CALUDE_complex_modulus_evaluation_l62_6212

theorem complex_modulus_evaluation :
  Complex.abs (3 / 4 - 5 * Complex.I + (1 + 3 * Complex.I)) = Real.sqrt 113 / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_evaluation_l62_6212


namespace NUMINAMATH_CALUDE_sixth_power_of_z_l62_6277

theorem sixth_power_of_z (z : ℂ) : z = (Real.sqrt 3 - Complex.I) / 2 → z^6 = -1 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_of_z_l62_6277


namespace NUMINAMATH_CALUDE_least_consecutive_primes_l62_6213

/-- Definition of the sequence x_n -/
def x (a b n : ℕ) : ℚ :=
  (a^n - 1) / (b^n - 1)

/-- Main theorem statement -/
theorem least_consecutive_primes (a b : ℕ) (h1 : a > b) (h2 : b > 1) :
  ∃ d : ℕ, d = 3 ∧
  (∀ n : ℕ, ¬(Prime (x a b n) ∧ Prime (x a b (n+1)) ∧ Prime (x a b (n+2)))) ∧
  (∀ d' : ℕ, d' < d →
    ∃ a' b' n' : ℕ, a' > b' ∧ b' > 1 ∧
      Prime (x a' b' n') ∧ Prime (x a' b' (n'+1)) ∧
      (d' = 2 → Prime (x a' b' (n'+2)))) :=
sorry

end NUMINAMATH_CALUDE_least_consecutive_primes_l62_6213


namespace NUMINAMATH_CALUDE_no_charming_numbers_l62_6227

/-- A two-digit positive integer is charming if it equals the sum of the square of its tens digit
and the product of its digits. -/
def IsCharming (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = a^2 + a * b

/-- There are no charming two-digit positive integers. -/
theorem no_charming_numbers : ¬∃ n : ℕ, IsCharming n := by
  sorry

end NUMINAMATH_CALUDE_no_charming_numbers_l62_6227


namespace NUMINAMATH_CALUDE_cube_opposite_face_l62_6271

-- Define a cube face
inductive Face : Type
| A | B | C | D | E | F

-- Define the property of being adjacent
def adjacent (x y : Face) : Prop := sorry

-- Define the property of sharing an edge
def sharesEdge (x y : Face) : Prop := sorry

-- Define the property of being opposite
def opposite (x y : Face) : Prop := sorry

-- Theorem statement
theorem cube_opposite_face :
  -- Conditions
  (sharesEdge Face.B Face.A) →
  (adjacent Face.C Face.B) →
  (¬ adjacent Face.C Face.A) →
  (sharesEdge Face.D Face.A) →
  (sharesEdge Face.D Face.F) →
  -- Conclusion
  (opposite Face.C Face.E) := by
sorry

end NUMINAMATH_CALUDE_cube_opposite_face_l62_6271


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l62_6249

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_k_value :
  let a : ℝ × ℝ := (6, 2)
  let b : ℝ × ℝ := (-2, k)
  parallel a b → k = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l62_6249


namespace NUMINAMATH_CALUDE_cubic_equation_natural_roots_l62_6244

/-- The cubic equation has three natural number roots if and only if p = 76 -/
theorem cubic_equation_natural_roots (p : ℝ) : 
  (∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (5 * (x : ℝ)^3 - 5*(p+1)*(x : ℝ)^2 + (71*p-1)*(x : ℝ) + 1 = 66*p) ∧
    (5 * (y : ℝ)^3 - 5*(p+1)*(y : ℝ)^2 + (71*p-1)*(y : ℝ) + 1 = 66*p) ∧
    (5 * (z : ℝ)^3 - 5*(p+1)*(z : ℝ)^2 + (71*p-1)*(z : ℝ) + 1 = 66*p)) ↔
  p = 76 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_natural_roots_l62_6244


namespace NUMINAMATH_CALUDE_renovation_project_materials_l62_6287

theorem renovation_project_materials (sand dirt cement gravel stone : ℝ) 
  (h1 : sand = 0.17)
  (h2 : dirt = 0.33)
  (h3 : cement = 0.17)
  (h4 : gravel = 0.25)
  (h5 : stone = 0.08) :
  sand + dirt + cement + gravel + stone = 1 := by
  sorry

end NUMINAMATH_CALUDE_renovation_project_materials_l62_6287


namespace NUMINAMATH_CALUDE_existence_of_sum_equality_l62_6216

theorem existence_of_sum_equality (A : Set ℕ) 
  (h : ∀ n : ℕ, ∃ m ∈ A, n ≤ m ∧ m < n + 100) :
  ∃ a b c d : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a + b = c + d :=
sorry

end NUMINAMATH_CALUDE_existence_of_sum_equality_l62_6216


namespace NUMINAMATH_CALUDE_root_in_interval_l62_6291

def f (x : ℝ) := x^3 + x^2 - 2*x - 2

theorem root_in_interval :
  f 1 = -2 →
  f 1.5 = 0.65 →
  f 1.25 = -0.984 →
  f 1.375 = -0.260 →
  f 1.4375 = 0.162 →
  f 1.40625 = -0.054 →
  ∃ x, x > 1.3 ∧ x < 1.5 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l62_6291


namespace NUMINAMATH_CALUDE_box_dimensions_l62_6284

theorem box_dimensions (a b c : ℝ) 
  (h1 : a + c = 17) 
  (h2 : a + b = 13) 
  (h3 : b + c = 20) 
  (h4 : a < b) 
  (h5 : b < c) : 
  a = 5 ∧ b = 8 ∧ c = 12 := by
sorry

end NUMINAMATH_CALUDE_box_dimensions_l62_6284


namespace NUMINAMATH_CALUDE_roots_equal_condition_l62_6246

theorem roots_equal_condition (m : ℝ) : 
  (∃! x : ℝ, (x * (x - 1) - (m + 1)) / ((x - 1) * (m - 1)) = x / m) ↔ m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_roots_equal_condition_l62_6246


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l62_6236

theorem quadratic_equation_solution (x y z t : ℝ) :
  x^2 + y^2 + z^2 + t^2 = x*(y + z + t) → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l62_6236


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l62_6298

theorem multiply_and_simplify (x : ℝ) : 
  (x^4 + 49*x^2 + 2401) * (x^2 - 49) = x^6 - 117649 := by
sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l62_6298


namespace NUMINAMATH_CALUDE_complex_subtraction_l62_6209

theorem complex_subtraction : (5 - 3*I) - (2 + 7*I) = 3 - 10*I := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l62_6209


namespace NUMINAMATH_CALUDE_intersection_in_square_l62_6269

-- Define the trajectory function
def trajectory (x : ℝ) : ℝ :=
  (((x^5 - 2013)^5 - 2013)^5 - 2013)^5

-- Define the radar line function
def radar_line (x : ℝ) : ℝ :=
  x + 2013

-- Define the function for the difference between trajectory and radar line
def intersection_function (x : ℝ) : ℝ :=
  trajectory x - radar_line x

-- Theorem statement
theorem intersection_in_square :
  ∃ (x y : ℝ), 
    intersection_function x = 0 ∧ 
    4 ≤ x ∧ x < 5 ∧
    2017 ≤ y ∧ y < 2018 ∧
    y = radar_line x :=
sorry

end NUMINAMATH_CALUDE_intersection_in_square_l62_6269


namespace NUMINAMATH_CALUDE_movie_theater_revenue_l62_6264

theorem movie_theater_revenue
  (adult_price : ℕ)
  (child_price : ℕ)
  (total_tickets : ℕ)
  (adult_tickets : ℕ)
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_tickets = 900)
  (h4 : adult_tickets = 500)
  : adult_price * adult_tickets + child_price * (total_tickets - adult_tickets) = 5100 := by
  sorry

end NUMINAMATH_CALUDE_movie_theater_revenue_l62_6264


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l62_6240

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (4 : ℝ)^x + (2 : ℝ)^x - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l62_6240


namespace NUMINAMATH_CALUDE_third_row_sum_is_401_l62_6262

/-- Represents a position in the grid -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents the grid -/
def Grid := ℕ → ℕ → ℕ

/-- The size of the grid -/
def gridSize : ℕ := 16

/-- The starting position (centermost) -/
def startPos : Position :=
  { row := 9, col := 9 }

/-- Fills the grid in a clockwise spiral pattern -/
def fillGrid : Grid :=
  sorry

/-- Gets the numbers in a specific row -/
def getRowNumbers (g : Grid) (row : ℕ) : List ℕ :=
  sorry

/-- Theorem: The sum of the greatest and least number in the third row from the top is 401 -/
theorem third_row_sum_is_401 :
  let g := fillGrid
  let thirdRow := getRowNumbers g 3
  (List.maximum thirdRow).getD 0 + (List.minimum thirdRow).getD 0 = 401 := by
  sorry

end NUMINAMATH_CALUDE_third_row_sum_is_401_l62_6262


namespace NUMINAMATH_CALUDE_gilbert_basil_bushes_l62_6265

/-- The number of basil bushes Gilbert planted initially -/
def initial_basil_bushes : ℕ := 1

/-- The total number of herb plants at the end of spring -/
def total_plants : ℕ := 5

/-- The number of mint types (which were eaten) -/
def mint_types : ℕ := 2

/-- The number of parsley plants -/
def parsley_plants : ℕ := 1

/-- The number of extra basil plants that grew during spring -/
def extra_basil : ℕ := 1

theorem gilbert_basil_bushes :
  initial_basil_bushes = total_plants - mint_types - parsley_plants - extra_basil :=
by sorry

end NUMINAMATH_CALUDE_gilbert_basil_bushes_l62_6265


namespace NUMINAMATH_CALUDE_solution_set_range_l62_6285

-- Define the function f(x) for a given a
def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - 1) * x^2 - (a - 1) * x - 1

-- Define the property that f(x) < 0 for all real x
def always_negative (a : ℝ) : Prop := ∀ x : ℝ, f a x < 0

-- Define the set of a for which f(x) < 0 for all real x
def solution_set : Set ℝ := {a : ℝ | always_negative a}

-- State the theorem
theorem solution_set_range : solution_set = Set.Ioc (-3/5) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_range_l62_6285


namespace NUMINAMATH_CALUDE_square_side_length_average_l62_6253

theorem square_side_length_average (a b c : ℝ) 
  (ha : a = 36) (hb : b = 64) (hc : c = 144) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 26 / 3 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_average_l62_6253


namespace NUMINAMATH_CALUDE_work_completion_time_l62_6201

/-- The number of days x needs to finish the work alone -/
def x_days : ℝ := 18

/-- The number of days y worked before leaving -/
def y_worked : ℝ := 5

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℝ := 12

/-- The number of days y needs to finish the work alone -/
def y_days : ℝ := 15

theorem work_completion_time : 
  (y_worked / y_days) + (x_remaining / x_days) = 1 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l62_6201


namespace NUMINAMATH_CALUDE_find_A_l62_6223

theorem find_A : ∀ A : ℕ, (A / 9 = 2 ∧ A % 9 = 6) → A = 24 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l62_6223


namespace NUMINAMATH_CALUDE_translation_theorem_l62_6268

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def applyTranslation (t : Translation) (p : Point) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_theorem (A B C D : Point) (t : Translation) :
  A.x = -1 ∧ A.y = 4 ∧
  C.x = 4 ∧ C.y = 7 ∧
  B.x = -4 ∧ B.y = -1 ∧
  C = applyTranslation t A ∧
  D = applyTranslation t B →
  D.x = 1 ∧ D.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l62_6268


namespace NUMINAMATH_CALUDE_cultivation_equation_correct_l62_6203

/-- Represents the cultivation problem of a farmer --/
structure CultivationProblem where
  paddy_area : ℝ
  dry_area : ℝ
  dry_rate_difference : ℝ
  time_ratio : ℝ

/-- The equation representing the cultivation problem --/
def cultivation_equation (p : CultivationProblem) (x : ℝ) : Prop :=
  p.paddy_area / x = 2 * (p.dry_area / (x + p.dry_rate_difference))

/-- Theorem stating that the given equation correctly represents the cultivation problem --/
theorem cultivation_equation_correct (p : CultivationProblem) :
  p.paddy_area = 36 ∧ 
  p.dry_area = 30 ∧ 
  p.dry_rate_difference = 4 ∧ 
  p.time_ratio = 2 →
  ∃ x : ℝ, cultivation_equation p x :=
by sorry

end NUMINAMATH_CALUDE_cultivation_equation_correct_l62_6203


namespace NUMINAMATH_CALUDE_journey_time_calculation_l62_6220

/-- Proves that if walking twice the distance of running takes 30 minutes,
    then walking one-third and running two-thirds of the same distance takes 24 minutes,
    given that running speed is twice the walking speed. -/
theorem journey_time_calculation (v : ℝ) (S : ℝ) (h1 : v > 0) (h2 : S > 0) :
  (2 * S / v + S / (2 * v) = 30) →
  (S / v + 2 * S / (2 * v) = 24) :=
by sorry

end NUMINAMATH_CALUDE_journey_time_calculation_l62_6220


namespace NUMINAMATH_CALUDE_binomial_square_constant_l62_6279

theorem binomial_square_constant (x : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 150*x + 5625 = (x + a)^2) := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l62_6279


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l62_6208

/-- A decreasing function on (0, +∞) -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ 0 < y ∧ x < y → f y < f x

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : DecreasingFunction f)
  (h_inequality : f (2 * a^2 + a + 1) < f (3 * a^2 - 4 * a + 1)) :
  (0 < a ∧ a < 1/3) ∨ (1 < a ∧ a < 5) :=
by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l62_6208


namespace NUMINAMATH_CALUDE_three_parallel_lines_planes_l62_6218

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary fields to represent a line in 3D space
  -- This is a placeholder and may need to be adjusted based on Lean's geometry libraries

-- Define a predicate for parallel lines
def parallel (l1 l2 : Line3D) : Prop :=
  sorry -- Definition of parallel lines

-- Define a predicate for coplanar lines
def coplanar (l1 l2 l3 : Line3D) : Prop :=
  sorry -- Definition of coplanar lines

-- Define a function to count planes through two lines
def count_planes_through_two_lines (l1 l2 : Line3D) : ℕ :=
  sorry -- Definition to count planes through two lines

-- Theorem statement
theorem three_parallel_lines_planes (a b c : Line3D) :
  parallel a b ∧ parallel b c ∧ parallel a c ∧ ¬coplanar a b c →
  (count_planes_through_two_lines a b +
   count_planes_through_two_lines b c +
   count_planes_through_two_lines a c) = 3 :=
by sorry

end NUMINAMATH_CALUDE_three_parallel_lines_planes_l62_6218


namespace NUMINAMATH_CALUDE_permutations_of_six_distinct_objects_l62_6237

theorem permutations_of_six_distinct_objects : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_six_distinct_objects_l62_6237


namespace NUMINAMATH_CALUDE_largest_increase_2007_2008_l62_6210

/-- Represents the number of students taking AMC 10 for each year from 2002 to 2008 -/
def students : Fin 7 → ℕ
  | 0 => 50  -- 2002
  | 1 => 55  -- 2003
  | 2 => 60  -- 2004
  | 3 => 65  -- 2005
  | 4 => 72  -- 2006
  | 5 => 80  -- 2007
  | 6 => 90  -- 2008

/-- Calculates the percentage increase between two consecutive years -/
def percentageIncrease (year : Fin 6) : ℚ :=
  (students (year.succ) - students year) / students year * 100

/-- Theorem stating that the percentage increase between 2007 and 2008 is the largest -/
theorem largest_increase_2007_2008 :
  ∀ year : Fin 6, percentageIncrease 5 ≥ percentageIncrease year :=
by sorry

end NUMINAMATH_CALUDE_largest_increase_2007_2008_l62_6210


namespace NUMINAMATH_CALUDE_best_fit_model_l62_6235

/-- Represents a regression model with a correlation coefficient -/
structure RegressionModel where
  R : ℝ
  h_R_range : R ≥ 0 ∧ R ≤ 1

/-- Defines when one model has a better fit than another -/
def better_fit (m1 m2 : RegressionModel) : Prop := m1.R > m2.R

theorem best_fit_model (model1 model2 model3 model4 : RegressionModel)
  (h1 : model1.R = 0.98)
  (h2 : model2.R = 0.80)
  (h3 : model3.R = 0.50)
  (h4 : model4.R = 0.25) :
  better_fit model1 model2 ∧ better_fit model1 model3 ∧ better_fit model1 model4 := by
  sorry


end NUMINAMATH_CALUDE_best_fit_model_l62_6235


namespace NUMINAMATH_CALUDE_hit_rate_calculation_l62_6226

theorem hit_rate_calculation (p₁ p₂ : ℚ) : 
  (p₁ * (1 - p₂) * (1/3 : ℚ) = 1/18) →
  (p₂ * (2/3 : ℚ) = 4/9) →
  p₁ = 1/2 ∧ p₂ = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_hit_rate_calculation_l62_6226


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l62_6259

/-- Represents a square quilt block -/
structure QuiltBlock where
  size : Nat
  total_squares : Nat
  fully_shaded : Nat
  half_shaded : Nat

/-- Calculates the fraction of shaded area in a quilt block -/
def shaded_fraction (q : QuiltBlock) : Rat :=
  let total_area : Rat := q.total_squares
  let shaded_area : Rat := q.fully_shaded + q.half_shaded / 2
  shaded_area / total_area

/-- Theorem stating the shaded fraction of the specific quilt block -/
theorem quilt_shaded_fraction :
  let q : QuiltBlock := ⟨4, 16, 2, 4⟩
  shaded_fraction q = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l62_6259


namespace NUMINAMATH_CALUDE_max_diagonal_path_l62_6280

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a diagonal path in the rectangle -/
structure DiagonalPath where
  rectangle : Rectangle
  num_diagonals : ℕ

/-- Checks if a path is valid according to the problem constraints -/
def is_valid_path (path : DiagonalPath) : Prop :=
  path.num_diagonals > 0 ∧
  path.num_diagonals ≤ path.rectangle.width * path.rectangle.height / 2

/-- The main theorem stating the maximum number of diagonals in the path -/
theorem max_diagonal_path (rect : Rectangle) 
    (h1 : rect.width = 5) 
    (h2 : rect.height = 8) : 
  ∃ (path : DiagonalPath), 
    path.rectangle = rect ∧ 
    is_valid_path path ∧ 
    path.num_diagonals = 24 ∧
    ∀ (other_path : DiagonalPath), 
      other_path.rectangle = rect → 
      is_valid_path other_path → 
      other_path.num_diagonals ≤ path.num_diagonals :=
sorry

end NUMINAMATH_CALUDE_max_diagonal_path_l62_6280


namespace NUMINAMATH_CALUDE_set_A_equivalent_range_of_a_l62_6238

-- Define set A
def A : Set ℝ := {x | (3*x - 5)/(x + 1) ≤ 1}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}

-- Theorem for part 1
theorem set_A_equivalent : A = {x : ℝ | -1 < x ∧ x ≤ 3} := by sorry

-- Theorem for part 2
theorem range_of_a (a : ℝ) : B a ∩ (Set.univ \ A) = B a → a ≤ -2 ∨ a > 4 := by sorry

end NUMINAMATH_CALUDE_set_A_equivalent_range_of_a_l62_6238


namespace NUMINAMATH_CALUDE_remainder_eight_n_mod_seven_l62_6281

theorem remainder_eight_n_mod_seven (n : ℤ) (h : n % 4 = 3) : (8 * n) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_eight_n_mod_seven_l62_6281


namespace NUMINAMATH_CALUDE_watch_cost_price_l62_6286

/-- The cost price of a watch satisfying certain conditions -/
theorem watch_cost_price : ∃ (cp : ℚ), 
  cp > 0 ∧ 
  (0.9 * cp = cp - 0.1 * cp) ∧ 
  (1.04 * cp = cp + 0.04 * cp) ∧ 
  (1.04 * cp - 0.9 * cp = 168) ∧ 
  cp = 1200 := by
sorry

end NUMINAMATH_CALUDE_watch_cost_price_l62_6286


namespace NUMINAMATH_CALUDE_percentage_problem_l62_6239

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 400) : 1.2 * x = 2400 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l62_6239


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l62_6257

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- The common ratio of a geometric sequence -/
def CommonRatio (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h1 : GeometricSequence a)
    (h2 : a 2 + a 4 = 3)
    (h3 : a 3 * a 5 = 1) :
    ∃ q : ℝ, CommonRatio a q ∧ q = Real.sqrt 2 / 2 ∧
    ∀ n : ℕ, a n = 2 ^ ((n + 2 : ℝ) / 2) :=
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l62_6257


namespace NUMINAMATH_CALUDE_inequality_proof_l62_6267

theorem inequality_proof (n : ℕ+) (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  (1 - x + x^2 / 2)^(n : ℝ) - (1 - x)^(n : ℝ) ≤ x / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l62_6267


namespace NUMINAMATH_CALUDE_binomial_150_150_equals_1_l62_6200

theorem binomial_150_150_equals_1 : Nat.choose 150 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_150_equals_1_l62_6200


namespace NUMINAMATH_CALUDE_complex_division_negative_l62_6293

theorem complex_division_negative (m : ℝ) : 
  let z₁ : ℂ := 2 + 3*I
  let z₂ : ℂ := m - I
  (z₁ / z₂).re < 0 ∧ (z₁ / z₂).im = 0 → m = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_complex_division_negative_l62_6293


namespace NUMINAMATH_CALUDE_perpendicular_unit_vector_l62_6229

def a : ℝ × ℝ := (2, 1)

theorem perpendicular_unit_vector :
  let v : ℝ × ℝ := (Real.sqrt 5 / 5, -2 * Real.sqrt 5 / 5)
  (v.1 * v.1 + v.2 * v.2 = 1) ∧ (a.1 * v.1 + a.2 * v.2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_unit_vector_l62_6229


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l62_6219

/-- Given a circle intersected by three equally spaced parallel lines creating chords of lengths 36, 36, and 40, the distance between two adjacent parallel lines is 4√19/3 -/
theorem parallel_lines_distance (r : ℝ) (d : ℝ) : 
  (36 * r^2 = 648 + (9/4) * d^2) ∧ 
  (40 * r^2 = 800 + (45/4) * d^2) →
  d = (4 * Real.sqrt 19) / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l62_6219


namespace NUMINAMATH_CALUDE_passenger_trips_scientific_notation_l62_6290

/-- The number of operating passenger trips in millions -/
def passenger_trips : ℝ := 56.99

/-- The scientific notation representation of the passenger trips -/
def scientific_notation : ℝ := 5.699 * (10^7)

/-- Theorem stating that the number of passenger trips in millions 
    is equal to its scientific notation representation -/
theorem passenger_trips_scientific_notation : 
  passenger_trips * 10^6 = scientific_notation := by sorry

end NUMINAMATH_CALUDE_passenger_trips_scientific_notation_l62_6290


namespace NUMINAMATH_CALUDE_f_composition_of_three_l62_6250

def f (x : ℤ) : ℤ :=
  if x % 3 = 0 then x / 3 else 5 * x + 2

theorem f_composition_of_three : f (f (f (f 3))) = 187 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_three_l62_6250


namespace NUMINAMATH_CALUDE_circle_ratio_new_circumference_to_area_increase_l62_6222

/-- The ratio of new circumference to increase in area when a circle's radius is increased -/
theorem circle_ratio_new_circumference_to_area_increase 
  (r k : ℝ) (h : k > 0) : 
  (2 * Real.pi * (r + k)) / (Real.pi * ((r + k)^2 - r^2)) = 2 * (r + k) / (2 * r * k + k^2) :=
sorry

end NUMINAMATH_CALUDE_circle_ratio_new_circumference_to_area_increase_l62_6222


namespace NUMINAMATH_CALUDE_middle_part_of_proportional_division_l62_6252

theorem middle_part_of_proportional_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 120)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_prop : a = 2 * b ∧ c = (1/2) * b) : 
  b = 240/7 := by
  sorry

end NUMINAMATH_CALUDE_middle_part_of_proportional_division_l62_6252


namespace NUMINAMATH_CALUDE_rent_increase_new_mean_l62_6289

theorem rent_increase_new_mean 
  (num_friends : ℕ) 
  (initial_average : ℝ) 
  (increased_rent : ℝ) 
  (increase_percentage : ℝ) : 
  num_friends = 4 → 
  initial_average = 800 → 
  increased_rent = 800 → 
  increase_percentage = 0.25 → 
  (num_friends * initial_average + increased_rent * increase_percentage) / num_friends = 850 := by
  sorry

end NUMINAMATH_CALUDE_rent_increase_new_mean_l62_6289


namespace NUMINAMATH_CALUDE_smallest_triangle_perimeter_smallest_triangle_perimeter_proof_l62_6258

/-- The smallest possible perimeter of a triangle with consecutive integer side lengths,
    where the smallest side is at least 4. -/
theorem smallest_triangle_perimeter : ℕ → Prop :=
  fun p => (∃ n : ℕ, n ≥ 4 ∧ p = n + (n + 1) + (n + 2)) ∧
           (∀ m : ℕ, m ≥ 4 → m + (m + 1) + (m + 2) ≥ p) →
           p = 15

/-- Proof of the smallest_triangle_perimeter theorem -/
theorem smallest_triangle_perimeter_proof : smallest_triangle_perimeter 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_triangle_perimeter_smallest_triangle_perimeter_proof_l62_6258


namespace NUMINAMATH_CALUDE_cube_edge_length_l62_6297

theorem cube_edge_length 
  (paint_cost : ℝ) 
  (coverage_per_quart : ℝ) 
  (total_cost : ℝ) 
  (h1 : paint_cost = 3.20)
  (h2 : coverage_per_quart = 120)
  (h3 : total_cost = 16) : 
  ∃ (edge_length : ℝ), edge_length = 10 ∧ 
  6 * edge_length^2 = (total_cost / paint_cost) * coverage_per_quart :=
by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l62_6297


namespace NUMINAMATH_CALUDE_binomial_coefficient_21_15_l62_6273

theorem binomial_coefficient_21_15 :
  (Nat.choose 20 13 = 77520) →
  (Nat.choose 20 14 = 38760) →
  (Nat.choose 22 15 = 170544) →
  (Nat.choose 21 15 = 54264) :=
by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_21_15_l62_6273


namespace NUMINAMATH_CALUDE_probability_same_activity_l62_6243

/-- The probability that two specific students participate in the same activity
    when four students are divided into two groups. -/
theorem probability_same_activity (n : ℕ) (m : ℕ) : 
  n = 4 → m = 2 → (m : ℚ) / (Nat.choose n 2) = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_same_activity_l62_6243


namespace NUMINAMATH_CALUDE_f_properties_l62_6214

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (2*a + 1)*x + a * log x

theorem f_properties (a : ℝ) :
  (∀ x > 0, HasDerivAt (f a) ((2*x - (2*a + 1) + a/x) : ℝ) x) ∧
  (HasDerivAt (f a) 0 1 ↔ a = 1) ∧
  (∀ x > 1, f a x > 0 ↔ a ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l62_6214


namespace NUMINAMATH_CALUDE_sum_of_roots_times_two_l62_6261

theorem sum_of_roots_times_two (a b : ℝ) : 
  (a^2 + a - 6 = 0) → (b^2 + b - 6 = 0) → (2*a + 2*b = -2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_times_two_l62_6261


namespace NUMINAMATH_CALUDE_minimal_sum_of_squares_l62_6266

theorem minimal_sum_of_squares (a b c : ℕ+) : 
  a ≠ b → b ≠ c → a ≠ c →
  ∃ p q r : ℕ+, (a + b : ℕ) = p^2 ∧ (b + c : ℕ) = q^2 ∧ (a + c : ℕ) = r^2 →
  (a : ℕ) + b + c ≥ 55 :=
sorry

end NUMINAMATH_CALUDE_minimal_sum_of_squares_l62_6266


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l62_6215

-- Define the function f(x) = -x³
def f (x : ℝ) : ℝ := -x^3

-- Theorem stating that f is both odd and decreasing
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l62_6215


namespace NUMINAMATH_CALUDE_continued_proportionality_and_linear_combination_l62_6282

theorem continued_proportionality_and_linear_combination :
  -- Part (1)
  (∀ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 →
    x / (2*y + z) = y / (2*z + x) ∧ y / (2*z + x) = z / (2*x + y) →
    x / (2*y + z) = 1/3) ∧
  -- Part (2)
  (∀ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ a →
    (a + b) / (a - b) = (b + c) / (2*(b - c)) ∧
    (b + c) / (2*(b - c)) = (c + a) / (3*(c - a)) →
    8*a + 9*b + 5*c = 0) := by
  sorry

end NUMINAMATH_CALUDE_continued_proportionality_and_linear_combination_l62_6282


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l62_6248

/-- Given that -l, a, b, c, and -9 form a geometric sequence, prove that b = -3 and ac = 9 -/
theorem geometric_sequence_problem (l a b c : ℝ) 
  (h1 : ∃ (r : ℝ), a / (-l) = r ∧ b / a = r ∧ c / b = r ∧ (-9) / c = r) : 
  b = -3 ∧ a * c = 9 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l62_6248


namespace NUMINAMATH_CALUDE_a_plus_b_value_l62_6205

theorem a_plus_b_value (a b : ℝ) : 
  ((a + 2)^2 = 1 ∧ 3^3 = b - 3) → (a + b = 29 ∨ a + b = 27) := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l62_6205


namespace NUMINAMATH_CALUDE_janes_number_l62_6207

theorem janes_number : ∃ x : ℚ, 5 * (3 * x + 16) = 250 ∧ x = 34/3 := by
  sorry

end NUMINAMATH_CALUDE_janes_number_l62_6207


namespace NUMINAMATH_CALUDE_modulus_of_z_l62_6202

/-- Given a complex number z satisfying (1-i)z = 2i, prove that its modulus is √2 -/
theorem modulus_of_z (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l62_6202


namespace NUMINAMATH_CALUDE_marquita_garden_width_marquita_garden_width_proof_l62_6241

/-- The width of Marquita's gardens given the conditions of the problem -/
theorem marquita_garden_width : ℝ :=
  let mancino_garden_count : ℕ := 3
  let mancino_garden_length : ℝ := 16
  let mancino_garden_width : ℝ := 5
  let marquita_garden_count : ℕ := 2
  let marquita_garden_length : ℝ := 8
  let total_area : ℝ := 304

  let mancino_total_area := mancino_garden_count * mancino_garden_length * mancino_garden_width
  let marquita_total_area := total_area - mancino_total_area
  let marquita_garden_area := marquita_total_area / marquita_garden_count
  let marquita_garden_width := marquita_garden_area / marquita_garden_length

  4

theorem marquita_garden_width_proof : marquita_garden_width = 4 := by
  sorry

end NUMINAMATH_CALUDE_marquita_garden_width_marquita_garden_width_proof_l62_6241


namespace NUMINAMATH_CALUDE_monochromatic_state_reachable_final_color_independent_l62_6206

/-- Represents the three possible colors of glass pieces -/
inductive Color
  | Red
  | Yellow
  | Blue

/-- Represents the state of glass pieces -/
structure GlassState :=
  (red : Nat)
  (yellow : Nat)
  (blue : Nat)
  (total : Nat)
  (total_eq : red + yellow + blue = total)

/-- Represents an operation on glass pieces -/
def perform_operation (state : GlassState) : GlassState :=
  sorry

/-- Theorem stating that it's always possible to reach a monochromatic state -/
theorem monochromatic_state_reachable (initial_state : GlassState) 
  (h : initial_state.total = 1987) :
  ∃ (final_state : GlassState) (c : Color), 
    (final_state.red = initial_state.total ∧ c = Color.Red) ∨
    (final_state.yellow = initial_state.total ∧ c = Color.Yellow) ∨
    (final_state.blue = initial_state.total ∧ c = Color.Blue) :=
  sorry

/-- Theorem stating that the final color is independent of operation order -/
theorem final_color_independent (initial_state : GlassState) 
  (h : initial_state.total = 1987) :
  ∀ (final_state1 final_state2 : GlassState) (c1 c2 : Color),
    ((final_state1.red = initial_state.total ∧ c1 = Color.Red) ∨
     (final_state1.yellow = initial_state.total ∧ c1 = Color.Yellow) ∨
     (final_state1.blue = initial_state.total ∧ c1 = Color.Blue)) →
    ((final_state2.red = initial_state.total ∧ c2 = Color.Red) ∨
     (final_state2.yellow = initial_state.total ∧ c2 = Color.Yellow) ∨
     (final_state2.blue = initial_state.total ∧ c2 = Color.Blue)) →
    c1 = c2 :=
  sorry

end NUMINAMATH_CALUDE_monochromatic_state_reachable_final_color_independent_l62_6206


namespace NUMINAMATH_CALUDE_special_triangle_unique_values_l62_6231

/-- An isosceles triangle with a specific internal point -/
structure SpecialTriangle where
  -- The side length of the two equal sides
  s : ℝ
  -- The base length
  t : ℝ
  -- Coordinates of the internal point P
  px : ℝ
  py : ℝ
  -- Assertion that the triangle is isosceles
  h_isosceles : s > 0
  -- Assertion that P is inside the triangle
  h_inside : 0 < px ∧ px < t ∧ 0 < py ∧ py < s
  -- Distance from A to P is 2
  h_ap : px^2 + py^2 = 4
  -- Distance from B to P is 2√2
  h_bp : (t - px)^2 + py^2 = 8
  -- Distance from C to P is 3
  h_cp : px^2 + (s - py)^2 = 9

/-- The theorem stating the unique values of s and t -/
theorem special_triangle_unique_values (tri : SpecialTriangle) : 
  tri.s = 2 * Real.sqrt 3 ∧ tri.t = 6 := by sorry

end NUMINAMATH_CALUDE_special_triangle_unique_values_l62_6231


namespace NUMINAMATH_CALUDE_length_EC_l62_6256

-- Define the points
variable (A B C D E : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (h1 : ∃ t : ℝ, C = A + t • (C - A) ∧ D = B + t • (D - B))
variable (h2 : ‖A - E‖ = ‖A - B‖ - 1)
variable (h3 : ‖A - E‖ = ‖D - C‖)
variable (h4 : ‖A - D‖ = ‖B - E‖)
variable (h5 : angle A D C = angle D E C)

-- The theorem to prove
theorem length_EC : ‖E - C‖ = 1 := by
  sorry

end NUMINAMATH_CALUDE_length_EC_l62_6256


namespace NUMINAMATH_CALUDE_james_beef_pork_ratio_l62_6228

/-- Proves that the ratio of beef to pork James bought is 2:1 given the problem conditions --/
theorem james_beef_pork_ratio :
  ∀ (beef pork : ℝ) (meals : ℕ),
    beef = 20 →
    meals * 20 = 400 →
    meals * 1.5 = beef + pork →
    beef / pork = 2 := by
  sorry

end NUMINAMATH_CALUDE_james_beef_pork_ratio_l62_6228


namespace NUMINAMATH_CALUDE_root_product_equals_eight_l62_6251

theorem root_product_equals_eight :
  (32 : ℝ) ^ (1/5 : ℝ) * (8 : ℝ) ^ (1/3 : ℝ) * (4 : ℝ) ^ (1/2 : ℝ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_root_product_equals_eight_l62_6251


namespace NUMINAMATH_CALUDE_sid_shopping_l62_6263

def shopping_problem (initial_amount : ℕ) (snack_cost : ℕ) (remaining_amount_extra : ℕ) : Prop :=
  let computer_accessories_cost := initial_amount - snack_cost - (initial_amount / 2 + remaining_amount_extra)
  computer_accessories_cost = 12

theorem sid_shopping :
  shopping_problem 48 8 4 :=
sorry

end NUMINAMATH_CALUDE_sid_shopping_l62_6263


namespace NUMINAMATH_CALUDE_least_phrases_to_learn_l62_6292

theorem least_phrases_to_learn (total_phrases : ℕ) (min_grade : ℚ) : 
  total_phrases = 600 → min_grade = 90 / 100 → 
  ∃ (least_phrases : ℕ), 
    (least_phrases : ℚ) / total_phrases ≥ min_grade ∧
    ∀ (n : ℕ), (n : ℚ) / total_phrases ≥ min_grade → n ≥ least_phrases ∧
    least_phrases = 540 :=
by sorry

end NUMINAMATH_CALUDE_least_phrases_to_learn_l62_6292


namespace NUMINAMATH_CALUDE_father_total_spending_l62_6233

def heaven_spending : ℕ := 2 * 5 + 4 * 5
def brother_eraser_spending : ℕ := 10 * 4
def brother_highlighter_spending : ℕ := 30

theorem father_total_spending :
  heaven_spending + brother_eraser_spending + brother_highlighter_spending = 100 := by
  sorry

end NUMINAMATH_CALUDE_father_total_spending_l62_6233


namespace NUMINAMATH_CALUDE_partner_investment_time_l62_6255

/-- Given two partners p and q with investment and profit ratios, prove q's investment time -/
theorem partner_investment_time
  (investment_ratio_p investment_ratio_q : ℚ)
  (profit_ratio_p profit_ratio_q : ℚ)
  (investment_time_p : ℚ) :
  investment_ratio_p = 7 →
  investment_ratio_q = 5 →
  profit_ratio_p = 7 →
  profit_ratio_q = 10 →
  investment_time_p = 2 →
  ∃ (investment_time_q : ℚ),
    investment_time_q = 4 ∧
    (profit_ratio_p / profit_ratio_q) =
    ((investment_ratio_p * investment_time_p) /
     (investment_ratio_q * investment_time_q)) :=
by sorry


end NUMINAMATH_CALUDE_partner_investment_time_l62_6255


namespace NUMINAMATH_CALUDE_particle_prob_origin_prob_form_l62_6254

/-- A particle starts at (4,4) and moves randomly until it hits a coordinate axis. 
    At each step, it moves to one of (a-1, b), (a, b-1), or (a-1, b-1) with equal probability. -/
def particle_movement (a b : ℕ) : Fin 3 → ℕ × ℕ
| 0 => (a - 1, b)
| 1 => (a, b - 1)
| 2 => (a - 1, b - 1)

/-- The probability of the particle reaching (0,0) when starting from (4,4) -/
def prob_reach_origin : ℚ :=
  63 / 3^8

/-- Theorem stating that the probability of reaching (0,0) is 63/3^8 -/
theorem particle_prob_origin : 
  prob_reach_origin = 63 / 3^8 := by sorry

/-- The probability can be expressed as m/3^n where m is not divisible by 3 -/
theorem prob_form (m n : ℕ) (h : m % 3 ≠ 0) : 
  prob_reach_origin = m / 3^n := by sorry

end NUMINAMATH_CALUDE_particle_prob_origin_prob_form_l62_6254


namespace NUMINAMATH_CALUDE_proportion_theorem_l62_6270

theorem proportion_theorem (A B C p q r : ℝ) 
  (h1 : A / B = p) 
  (h2 : B / C = q) 
  (h3 : C / A = r) : 
  ∃ k : ℝ, k > 0 ∧ 
    A = k * (p^2 * q / r)^(1/3) ∧ 
    B = k * (q^2 * r / p)^(1/3) ∧ 
    C = k * (r^2 * p / q)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_proportion_theorem_l62_6270


namespace NUMINAMATH_CALUDE_intersection_M_N_l62_6232

-- Define the sets M and N
def M : Set ℝ := {y | y ≥ 0}
def N : Set ℝ := {y | -Real.sqrt 2 ≤ y ∧ y ≤ Real.sqrt 2}

-- State the theorem
theorem intersection_M_N : M ∩ N = {y | 0 ≤ y ∧ y ≤ Real.sqrt 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l62_6232


namespace NUMINAMATH_CALUDE_max_value_problem_l62_6283

theorem max_value_problem (a₁ a₂ a₃ a₄ : ℝ) 
  (h_pos₁ : 0 < a₁) (h_pos₂ : 0 < a₂) (h_pos₃ : 0 < a₃) (h_pos₄ : 0 < a₄)
  (h₁ : a₁ ≥ a₂ * a₃^2) (h₂ : a₂ ≥ a₃ * a₄^2) 
  (h₃ : a₃ ≥ a₄ * a₁^2) (h₄ : a₄ ≥ a₁ * a₂^2) : 
  a₁ * a₂ * a₃ * a₄ * (a₁ - a₂ * a₃^2) * (a₂ - a₃ * a₄^2) * 
  (a₃ - a₄ * a₁^2) * (a₄ - a₁ * a₂^2) ≤ 1 / 256 := by
  sorry

end NUMINAMATH_CALUDE_max_value_problem_l62_6283
