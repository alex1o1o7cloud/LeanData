import Mathlib

namespace NUMINAMATH_CALUDE_perimeter_ratio_of_similar_squares_l2491_249190

theorem perimeter_ratio_of_similar_squares (s : ℝ) (h : s > 0) : 
  let s1 := s * ((Real.sqrt 5 + 1) / 2)
  let p1 := 4 * s1
  let p2 := 4 * s
  let diagonal_first := Real.sqrt (2 * s1 ^ 2)
  diagonal_first = s → p1 / p2 = (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_of_similar_squares_l2491_249190


namespace NUMINAMATH_CALUDE_greg_bike_ride_l2491_249125

/-- Proves that Greg wants to ride 8 blocks given the conditions of the problem -/
theorem greg_bike_ride (rotations_per_block : ℕ) (rotations_so_far : ℕ) (rotations_needed : ℕ) :
  rotations_per_block = 200 →
  rotations_so_far = 600 →
  rotations_needed = 1000 →
  (rotations_so_far + rotations_needed) / rotations_per_block = 8 := by
  sorry

end NUMINAMATH_CALUDE_greg_bike_ride_l2491_249125


namespace NUMINAMATH_CALUDE_removed_number_value_l2491_249174

theorem removed_number_value (S : ℝ) (X : ℝ) : 
  S / 50 = 56 →
  (S - X - 55) / 48 = 56.25 →
  X = 45 := by
sorry

end NUMINAMATH_CALUDE_removed_number_value_l2491_249174


namespace NUMINAMATH_CALUDE_f_value_at_one_l2491_249168

def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 20

def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 50*x + c

theorem f_value_at_one (a b c : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    g a r₁ = 0 ∧ g a r₂ = 0 ∧ g a r₃ = 0) →
  (∀ x : ℝ, g a x = 0 → f b c x = 0) →
  f b c 1 = -217 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_one_l2491_249168


namespace NUMINAMATH_CALUDE_increments_theorem_l2491_249196

-- Define the function z(x, y)
def z (x y : ℝ) : ℝ := x^2 * y

-- Define the initial values and increments
def x₀ : ℝ := 1
def y₀ : ℝ := 2
def Δx : ℝ := 0.1
def Δy : ℝ := -0.2

-- Theorem statement
theorem increments_theorem :
  let Δ_x_z := z (x₀ + Δx) y₀ - z x₀ y₀
  let Δ_y_z := z x₀ (y₀ + Δy) - z x₀ y₀
  let Δz := z (x₀ + Δx) (y₀ + Δy) - z x₀ y₀
  (Δ_x_z = 0.42) ∧ (Δ_y_z = -0.2) ∧ (Δz = 0.178) := by
  sorry

end NUMINAMATH_CALUDE_increments_theorem_l2491_249196


namespace NUMINAMATH_CALUDE_ellipse_equation_l2491_249130

/-- The equation of an ellipse passing through points (1, √3/2) and (2, 0) -/
theorem ellipse_equation : ∃ (a b : ℝ), 
  (a > 0 ∧ b > 0) ∧ 
  (1 / a^2 + (Real.sqrt 3 / 2)^2 / b^2 = 1) ∧ 
  (4 / a^2 = 1) ∧
  ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2491_249130


namespace NUMINAMATH_CALUDE_initial_amount_calculation_l2491_249198

/-- Given a person receives additional money and the difference between their
initial amount and the received amount is known, calculate their initial amount. -/
theorem initial_amount_calculation (received : ℕ) (difference : ℕ) : 
  received = 13 → difference = 11 → received + difference = 24 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_calculation_l2491_249198


namespace NUMINAMATH_CALUDE_CD_length_l2491_249124

-- Define the points
variable (A B C D : ℝ × ℝ)

-- Define the conditions
def AD_perp_AB : (A.1 - D.1) * (B.1 - A.1) + (A.2 - D.2) * (B.2 - A.2) = 0 := sorry
def BC_perp_AB : (B.1 - C.1) * (B.1 - A.1) + (B.2 - C.2) * (B.2 - A.2) = 0 := sorry
def CD_perp_AC : (C.1 - D.1) * (C.1 - A.1) + (C.2 - D.2) * (C.2 - A.2) = 0 := sorry

def AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 4 := sorry
def BC_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 3 := sorry

-- Theorem to prove
theorem CD_length : 
  Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) = 20/3 :=
by sorry

end NUMINAMATH_CALUDE_CD_length_l2491_249124


namespace NUMINAMATH_CALUDE_age_height_not_function_l2491_249137

-- Define the relationships
def angle_sine_relation : Set (ℝ × ℝ) := sorry
def square_side_area_relation : Set (ℝ × ℝ) := sorry
def polygon_sides_angles_relation : Set (ℕ × ℝ) := sorry
def age_height_relation : Set (ℕ × ℝ) := sorry

-- Define the property of being a function
def is_function (r : Set (α × β)) : Prop := 
  ∀ x y z, (x, y) ∈ r → (x, z) ∈ r → y = z

-- State the theorem
theorem age_height_not_function :
  is_function angle_sine_relation ∧ 
  is_function square_side_area_relation ∧ 
  is_function polygon_sides_angles_relation → 
  ¬ is_function age_height_relation := by
sorry

end NUMINAMATH_CALUDE_age_height_not_function_l2491_249137


namespace NUMINAMATH_CALUDE_clothes_batch_size_l2491_249149

/-- Proves that the number of sets of clothes in a batch is 30, given the production rates of two workers and their time difference. -/
theorem clothes_batch_size :
  let wang_rate : ℚ := 3  -- Wang's production rate (sets per day)
  let li_rate : ℚ := 5    -- Li's production rate (sets per day)
  let time_diff : ℚ := 4  -- Time difference in days
  let batch_size : ℚ := (wang_rate * li_rate * time_diff) / (li_rate - wang_rate)
  batch_size = 30 := by
  sorry


end NUMINAMATH_CALUDE_clothes_batch_size_l2491_249149


namespace NUMINAMATH_CALUDE_blackboard_sum_divisibility_l2491_249146

theorem blackboard_sum_divisibility (n : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ∃ x ∈ Finset.range n, ¬ ∃ y ∈ Finset.range n,
    (((n * (3 * n - 1)) / 2 - (n + x)) % (n + y) = 0) := by
  sorry

end NUMINAMATH_CALUDE_blackboard_sum_divisibility_l2491_249146


namespace NUMINAMATH_CALUDE_intersection_sum_is_eight_l2491_249144

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x + 2)^2
def parabola2 (x y : ℝ) : Prop := x + 5 = (y - 4)^2

-- Define the set of intersection points
def intersectionPoints : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

-- Theorem statement
theorem intersection_sum_is_eight :
  ∃ (points : Finset (ℝ × ℝ)), points.toSet = intersectionPoints ∧
  points.card = 4 ∧
  (points.sum (λ p => p.1) + points.sum (λ p => p.2) = 8) := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_is_eight_l2491_249144


namespace NUMINAMATH_CALUDE_complex_magnitude_l2491_249122

theorem complex_magnitude (z : ℂ) (h : 3 * z + Complex.I = 1 - 4 * Complex.I * z) :
  Complex.abs z = Real.sqrt 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2491_249122


namespace NUMINAMATH_CALUDE_power_of_seven_mod_four_l2491_249132

theorem power_of_seven_mod_four : 7^150 % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_four_l2491_249132


namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_expressions_equivalent_l2491_249162

/-- The polynomial expression to be simplified -/
def original_expression (x : ℝ) : ℝ :=
  5 * (2 * x^3 - 3 * x^2 + 4) - 6 * (x^4 - 2 * x^3 + 3 * x - 2)

/-- The simplified form of the polynomial -/
def simplified_expression (x : ℝ) : ℝ :=
  -6 * x^4 + 22 * x^3 - 15 * x^2 - 18 * x + 32

/-- Coefficients of the simplified polynomial -/
def coefficients : List ℝ := [-6, 22, -15, -18, 32]

/-- Theorem stating that the sum of squares of coefficients is 2093 -/
theorem sum_of_squares_of_coefficients :
  (coefficients.map (λ c => c^2)).sum = 2093 := by sorry

/-- Theorem stating that the original and simplified expressions are equivalent -/
theorem expressions_equivalent :
  ∀ x : ℝ, original_expression x = simplified_expression x := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_expressions_equivalent_l2491_249162


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2491_249143

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 - a*b + b^2 = 0) :
  (a^8 + b^8) / (a^2 + b^2)^4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2491_249143


namespace NUMINAMATH_CALUDE_angel_food_cake_egg_whites_angel_food_cake_proof_l2491_249186

theorem angel_food_cake_egg_whites (aquafaba_per_egg_white : ℕ) 
  (num_cakes : ℕ) (total_aquafaba : ℕ) : ℕ :=
  let egg_whites_per_cake := (total_aquafaba / aquafaba_per_egg_white) / num_cakes
  egg_whites_per_cake

theorem angel_food_cake_proof : 
  angel_food_cake_egg_whites 2 2 32 = 8 := by
  sorry

end NUMINAMATH_CALUDE_angel_food_cake_egg_whites_angel_food_cake_proof_l2491_249186


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2491_249171

theorem absolute_value_inequality (x : ℝ) :
  |((3 * x + 2) / (x - 2))| > 3 ↔ 2/3 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2491_249171


namespace NUMINAMATH_CALUDE_matrix_transformation_and_eigenvalues_l2491_249151

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 3; 2, 2]

theorem matrix_transformation_and_eigenvalues :
  -- 1) A transforms (1, 2) to (7, 6)
  A.mulVec ![1, 2] = ![7, 6] ∧
  -- 2) The eigenvalues of A are -1 and 4
  (A.charpoly.roots.toFinset = {-1, 4}) ∧
  -- 3) [3, -2] is an eigenvector for λ = -1
  (A.mulVec ![3, -2] = (-1 : ℝ) • ![3, -2]) ∧
  -- 4) [1, 1] is an eigenvector for λ = 4
  (A.mulVec ![1, 1] = (4 : ℝ) • ![1, 1]) := by
sorry


end NUMINAMATH_CALUDE_matrix_transformation_and_eigenvalues_l2491_249151


namespace NUMINAMATH_CALUDE_unique_solution_cube_root_plus_square_root_l2491_249111

theorem unique_solution_cube_root_plus_square_root (x : ℝ) :
  (((x - 3) ^ (1/3 : ℝ)) + ((5 - x) ^ (1/2 : ℝ)) = 2) ↔ (x = 4) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_cube_root_plus_square_root_l2491_249111


namespace NUMINAMATH_CALUDE_chinese_math_problem_l2491_249179

-- Define the system of equations
def equation_system (x y : ℝ) : Prop :=
  5 * x + 2 * y = 19 ∧ 2 * x + 5 * y = 16

-- Define the profit function
def profit_function (m : ℝ) : ℝ := 0.5 * m + 5

-- Theorem statement
theorem chinese_math_problem :
  (∃ (x y : ℝ), equation_system x y ∧ x = 3 ∧ y = 2) ∧
  (∀ m : ℝ, 0 ≤ m ∧ m ≤ 5 → profit_function m ≤ profit_function 5) :=
by sorry

end NUMINAMATH_CALUDE_chinese_math_problem_l2491_249179


namespace NUMINAMATH_CALUDE_radio_selling_price_l2491_249170

/-- Calculates the selling price of an item given its cost price and loss percentage. -/
def selling_price (cost_price : ℚ) (loss_percentage : ℚ) : ℚ :=
  cost_price * (1 - loss_percentage / 100)

/-- Theorem: The selling price of a radio with a cost price of 1500 and a loss percentage of 17 is 1245. -/
theorem radio_selling_price :
  selling_price 1500 17 = 1245 := by
  sorry

end NUMINAMATH_CALUDE_radio_selling_price_l2491_249170


namespace NUMINAMATH_CALUDE_zoo_animal_count_l2491_249103

/-- The number of animals Brinley counted at the San Diego Zoo --/
theorem zoo_animal_count :
  let snakes : ℕ := 100
  let arctic_foxes : ℕ := 80
  let leopards : ℕ := 20
  let bee_eaters : ℕ := 10 * leopards
  let cheetahs : ℕ := snakes / 2
  let alligators : ℕ := 2 * (arctic_foxes + leopards)
  snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators = 650 :=
by sorry


end NUMINAMATH_CALUDE_zoo_animal_count_l2491_249103


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l2491_249141

/-- Represents a person at the table -/
structure Person :=
  (id : Nat)

/-- Represents the circular table with 40 people -/
def Table := Fin 40 → Person

/-- Defines the relationship of acquaintance between two people -/
def IsAcquainted (table : Table) (p1 p2 : Person) : Prop := sorry

/-- Counts the number of people between two given positions on the table -/
def PeopleBetween (pos1 pos2 : Fin 40) : Nat := sorry

/-- States that for any two people with an even number between them, there's a common acquaintance -/
def EvenHaveCommonAcquaintance (table : Table) : Prop :=
  ∀ (pos1 pos2 : Fin 40), Even (PeopleBetween pos1 pos2) →
    ∃ (p : Person), IsAcquainted table (table pos1) p ∧ IsAcquainted table (table pos2) p

/-- States that for any two people with an odd number between them, there's no common acquaintance -/
def OddNoCommonAcquaintance (table : Table) : Prop :=
  ∀ (pos1 pos2 : Fin 40), Odd (PeopleBetween pos1 pos2) →
    ∀ (p : Person), ¬(IsAcquainted table (table pos1) p ∧ IsAcquainted table (table pos2) p)

/-- The main theorem stating that no arrangement satisfies both conditions -/
theorem no_valid_arrangement :
  ¬∃ (table : Table), EvenHaveCommonAcquaintance table ∧ OddNoCommonAcquaintance table :=
sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l2491_249141


namespace NUMINAMATH_CALUDE_least_number_divisible_by_smallest_primes_l2491_249142

def smallest_primes : List Nat := [2, 3, 5, 7, 11]

def product_of_smallest_primes : Nat := smallest_primes.prod

theorem least_number_divisible_by_smallest_primes :
  ∀ n : Nat, n > 0 ∧ (∀ p ∈ smallest_primes, n % p = 0) → n ≥ product_of_smallest_primes :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_smallest_primes_l2491_249142


namespace NUMINAMATH_CALUDE_odd_function_sum_property_l2491_249108

def is_odd_function (v : ℝ → ℝ) : Prop := ∀ x, v (-x) = -v x

theorem odd_function_sum_property (v : ℝ → ℝ) (a b : ℝ) 
  (h : is_odd_function v) : 
  v (-a) + v (-b) + v b + v a = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_property_l2491_249108


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2491_249185

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (-1, x) (-2, 4) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2491_249185


namespace NUMINAMATH_CALUDE_island_puzzle_l2491_249123

/-- Represents a person who is either a knight or a liar -/
inductive Person
| Knight
| Liar

/-- The statement made by a person about the number of liars -/
def Statement := Fin 5 → ℕ

/-- Checks if a person's statement is truthful given the actual number of liars -/
def isStatementTruthful (p : Person) (s : ℕ) (actualLiars : ℕ) : Prop :=
  match p with
  | Person.Knight => s = actualLiars
  | Person.Liar => s ≠ actualLiars

/-- The main theorem to prove -/
theorem island_puzzle :
  ∀ (people : Fin 5 → Person) (statements : Statement),
  (∀ i j : Fin 5, i ≠ j → statements i ≠ statements j) →
  (∀ i : Fin 5, statements i = i.val + 1) →
  (∃! i : Fin 5, people i = Person.Knight) →
  (∀ i : Fin 5, isStatementTruthful (people i) (statements i) 4) :=
sorry

end NUMINAMATH_CALUDE_island_puzzle_l2491_249123


namespace NUMINAMATH_CALUDE_smallest_number_with_same_prime_factors_l2491_249129

def alice_number : ℕ := 72

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ n → p ∣ m)

theorem smallest_number_with_same_prime_factors :
  ∃ k : ℕ, k > 0 ∧ has_all_prime_factors alice_number k ∧
  ∀ m : ℕ, m > 0 → has_all_prime_factors alice_number m → k ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_same_prime_factors_l2491_249129


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l2491_249172

theorem inequality_solution_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ((a + 1) * x > 2 * a + 2) ↔ (x < 2)) →
  a < -1 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l2491_249172


namespace NUMINAMATH_CALUDE_oranges_taken_l2491_249181

theorem oranges_taken (initial_oranges final_oranges : ℕ) 
  (h1 : initial_oranges = 60)
  (h2 : final_oranges = 25) :
  initial_oranges - final_oranges = 35 := by
  sorry

end NUMINAMATH_CALUDE_oranges_taken_l2491_249181


namespace NUMINAMATH_CALUDE_product_of_fractions_equals_one_l2491_249101

theorem product_of_fractions_equals_one :
  (7 / 4 : ℚ) * (8 / 14 : ℚ) * (21 / 12 : ℚ) * (16 / 28 : ℚ) *
  (49 / 28 : ℚ) * (24 / 42 : ℚ) * (63 / 36 : ℚ) * (32 / 56 : ℚ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_equals_one_l2491_249101


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2491_249106

def U : Finset ℕ := {1,2,3,4,5,6}
def A : Finset ℕ := {2,4,5}
def B : Finset ℕ := {1,3}

theorem intersection_complement_equality : A ∩ (U \ B) = {2,4,5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2491_249106


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_area_l2491_249117

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line in the form y = mx + c -/
structure Line where
  m : ℝ
  c : ℝ

/-- Calculate the area of triangle AOB formed by the intersection of an ellipse and a line -/
def area_triangle_AOB (e : Ellipse) (l : Line) : ℝ :=
  sorry

theorem ellipse_line_intersection_area :
  ∀ (e : Ellipse) (l : Line),
    e.b = 1 →
    e.a^2 = 2 →
    l.m = 1 ∧ l.c = Real.sqrt 2 →
    area_triangle_AOB e l = 2/3 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_area_l2491_249117


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2491_249135

theorem purely_imaginary_complex_number (m : ℝ) : 
  (m^2 - 1 : ℂ) + (m + 1 : ℂ) * Complex.I = Complex.I * y → m = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2491_249135


namespace NUMINAMATH_CALUDE_new_average_production_l2491_249173

theorem new_average_production (n : ℕ) (past_average : ℝ) (today_production : ℝ) 
  (h1 : n = 11)
  (h2 : past_average = 50)
  (h3 : today_production = 110) : 
  (n * past_average + today_production) / (n + 1) = 55 :=
by sorry

end NUMINAMATH_CALUDE_new_average_production_l2491_249173


namespace NUMINAMATH_CALUDE_present_ages_of_deepak_and_rajat_l2491_249193

-- Define the present ages as variables
variable (R D Ra : ℕ)

-- Define the conditions
def present_age_ratio : Prop := R / D = 4 / 3 ∧ Ra / D = 5 / 3
def rahul_future_age : Prop := R + 4 = 32
def rajat_future_age : Prop := Ra + 7 = 50

-- State the theorem
theorem present_ages_of_deepak_and_rajat 
  (h1 : present_age_ratio R D Ra)
  (h2 : rahul_future_age R)
  (h3 : rajat_future_age Ra) :
  D = 21 ∧ Ra = 43 := by
  sorry

end NUMINAMATH_CALUDE_present_ages_of_deepak_and_rajat_l2491_249193


namespace NUMINAMATH_CALUDE_birds_meeting_point_l2491_249116

/-- The distance between West-town and East-town in kilometers -/
def total_distance : ℝ := 20

/-- The speed of the first bird in kilometers per minute -/
def speed_bird1 : ℝ := 4

/-- The speed of the second bird in kilometers per minute -/
def speed_bird2 : ℝ := 1

/-- The distance traveled by the first bird before meeting -/
def distance_bird1 : ℝ := 16

/-- The distance traveled by the second bird before meeting -/
def distance_bird2 : ℝ := 4

theorem birds_meeting_point :
  distance_bird1 + distance_bird2 = total_distance ∧
  distance_bird1 / speed_bird1 = distance_bird2 / speed_bird2 ∧
  distance_bird1 = 16 := by sorry

end NUMINAMATH_CALUDE_birds_meeting_point_l2491_249116


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l2491_249102

theorem complex_subtraction_simplification :
  (-5 - 3 * Complex.I) - (2 - 5 * Complex.I) = -7 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l2491_249102


namespace NUMINAMATH_CALUDE_fuchsia_survey_l2491_249131

theorem fuchsia_survey (total : ℕ) (kinda_pink : ℕ) (both : ℕ) (neither : ℕ)
  (h_total : total = 100)
  (h_kinda_pink : kinda_pink = 60)
  (h_both : both = 27)
  (h_neither : neither = 17) :
  ∃ (purply : ℕ), purply = 50 ∧ purply = total - (kinda_pink - both + neither) :=
by sorry

end NUMINAMATH_CALUDE_fuchsia_survey_l2491_249131


namespace NUMINAMATH_CALUDE_sequence_is_decreasing_l2491_249140

def is_decreasing (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n ≥ a (n + 1)

theorem sequence_is_decreasing (a : ℕ+ → ℝ) 
  (h : ∀ n : ℕ+, a n - a (n + 1) = 10) : 
  is_decreasing a :=
sorry

end NUMINAMATH_CALUDE_sequence_is_decreasing_l2491_249140


namespace NUMINAMATH_CALUDE_complementary_angle_adjustment_l2491_249183

/-- Proves that when two complementary angles with a ratio of 3:7 have the smaller angle
    increased by 20%, the larger angle must decrease by 8.571% to maintain complementary angles. -/
theorem complementary_angle_adjustment (smaller larger : ℝ) : 
  smaller + larger = 90 →  -- angles are complementary
  smaller / larger = 3 / 7 →  -- ratio of angles is 3:7
  let new_smaller := smaller * 1.20  -- smaller angle increased by 20%
  let new_larger := 90 - new_smaller  -- new larger angle to maintain complementary
  (larger - new_larger) / larger * 100 = 8.571 :=  -- percentage decrease of larger angle
by sorry

end NUMINAMATH_CALUDE_complementary_angle_adjustment_l2491_249183


namespace NUMINAMATH_CALUDE_five_teachers_three_classes_l2491_249139

/-- The number of ways to assign n teachers to k distinct classes, 
    with at least one teacher per class -/
def teacher_assignments (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 150 ways to assign 5 teachers to 3 classes -/
theorem five_teachers_three_classes : 
  teacher_assignments 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_five_teachers_three_classes_l2491_249139


namespace NUMINAMATH_CALUDE_smallest_number_with_divisible_digit_sums_l2491_249159

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number satisfies the divisibility condition -/
def satisfiesCondition (n : ℕ) : Prop :=
  17 ∣ sumOfDigits n ∧ 17 ∣ sumOfDigits (n + 1)

theorem smallest_number_with_divisible_digit_sums :
  satisfiesCondition 8899 ∧ ∀ m < 8899, ¬satisfiesCondition m := by sorry

end NUMINAMATH_CALUDE_smallest_number_with_divisible_digit_sums_l2491_249159


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2491_249156

theorem arithmetic_mean_of_fractions :
  let a := 3 / 8
  let b := 5 / 9
  (a + b) / 2 = 67 / 144 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2491_249156


namespace NUMINAMATH_CALUDE_decimal_division_subtraction_l2491_249118

theorem decimal_division_subtraction : (0.24 / 0.004) - 0.1 = 59.9 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_subtraction_l2491_249118


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l2491_249138

theorem smallest_solution_of_equation :
  let f : ℝ → ℝ := λ x => (3*x)/(x-3) + (3*x^2 - 27*x)/x
  ∃ x : ℝ, f x = 14 ∧ x = (-41 - Real.sqrt 4633) / 12 ∧
  ∀ y : ℝ, f y = 14 → y ≥ (-41 - Real.sqrt 4633) / 12 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l2491_249138


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l2491_249119

theorem cos_2alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin α - Real.cos α = Real.sqrt 5 / 5) : 
  Real.cos (2 * α) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l2491_249119


namespace NUMINAMATH_CALUDE_valid_placement_exists_l2491_249184

/-- Represents the configuration of the circles --/
inductive Position
| TopLeft | TopMiddle | TopRight
| MiddleLeft | MiddleRight
| BottomLeft | BottomMiddle | BottomRight
| Center

/-- A function type that maps positions to numbers --/
def Placement := Position → Fin 9

/-- Checks if two numbers are adjacent in the configuration --/
def are_adjacent (p1 p2 : Position) : Bool :=
  match p1, p2 with
  | Position.TopLeft, Position.TopMiddle => true
  | Position.TopLeft, Position.MiddleLeft => true
  | Position.TopMiddle, Position.TopRight => true
  | Position.TopMiddle, Position.Center => true
  | Position.TopRight, Position.MiddleRight => true
  | Position.MiddleLeft, Position.BottomLeft => true
  | Position.MiddleLeft, Position.Center => true
  | Position.MiddleRight, Position.BottomRight => true
  | Position.MiddleRight, Position.Center => true
  | Position.BottomLeft, Position.BottomMiddle => true
  | Position.BottomMiddle, Position.BottomRight => true
  | Position.BottomMiddle, Position.Center => true
  | _, _ => false

/-- The main theorem stating the existence of a valid placement --/
theorem valid_placement_exists : ∃ (p : Placement),
  (∀ pos1 pos2, pos1 ≠ pos2 → p pos1 ≠ p pos2) ∧
  (∀ pos1 pos2, are_adjacent pos1 pos2 → Nat.gcd (p pos1).val.succ (p pos2).val.succ = 1) :=
sorry

end NUMINAMATH_CALUDE_valid_placement_exists_l2491_249184


namespace NUMINAMATH_CALUDE_sum_of_fraction_parts_l2491_249120

def repeating_decimal : ℚ := 123 / 999

theorem sum_of_fraction_parts : ∃ (n d : ℕ), 
  repeating_decimal = n / d ∧ 
  Nat.gcd n d = 1 ∧ 
  n + d = 374 := by sorry

end NUMINAMATH_CALUDE_sum_of_fraction_parts_l2491_249120


namespace NUMINAMATH_CALUDE_point_on_line_segment_l2491_249113

def A (m : ℝ) : ℝ × ℝ := (m^2, 2)
def B (m : ℝ) : ℝ × ℝ := (2*m^2 + 2, 2)
def M (m : ℝ) : ℝ × ℝ := (-m^2, 2)
def N (m : ℝ) : ℝ × ℝ := (m^2, m^2 + 2)
def P (m : ℝ) : ℝ × ℝ := (m^2 + 1, 2)
def Q (m : ℝ) : ℝ × ℝ := (3*m^2, 2)

theorem point_on_line_segment (m : ℝ) :
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P m = ((1 - t) • (A m) + t • (B m))) ∧
  (¬ ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → M m = ((1 - t) • (A m) + t • (B m))) ∧
  (¬ ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → N m = ((1 - t) • (A m) + t • (B m))) ∧
  (¬ ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → Q m = ((1 - t) • (A m) + t • (B m))) :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_segment_l2491_249113


namespace NUMINAMATH_CALUDE_unique_solution_mn_l2491_249115

theorem unique_solution_mn : 
  ∃! (m n : ℕ+), 18 * (m : ℕ) * (n : ℕ) = 72 - 9 * (m : ℕ) - 4 * (n : ℕ) ∧ m = 8 ∧ n = 36 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_mn_l2491_249115


namespace NUMINAMATH_CALUDE_linear_equation_solution_l2491_249199

theorem linear_equation_solution (k : ℝ) (x : ℝ) :
  k - 2 = 0 →        -- Condition for linearity
  4 * k ≠ 0 →        -- Ensure non-trivial equation
  (k - 2) * x^2 + 4 * k * x - 5 = 0 →
  x = 5 / 8 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l2491_249199


namespace NUMINAMATH_CALUDE_f_has_unique_root_in_interval_l2491_249165

/-- The polynomial function we're analyzing -/
def f (x : ℝ) : ℝ := x^11 + 9*x^10 + 20*x^9 + 2000*x^8 - 1500*x^7

/-- Theorem stating that f has exactly one root in (0,2) -/
theorem f_has_unique_root_in_interval :
  ∃! x : ℝ, 0 < x ∧ x < 2 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_f_has_unique_root_in_interval_l2491_249165


namespace NUMINAMATH_CALUDE_probability_ratio_l2491_249107

def total_slips : ℕ := 50
def numbers_range : ℕ := 10
def slips_per_number : ℕ := 5
def drawn_slips : ℕ := 5

def probability_same_number (total : ℕ) (range : ℕ) (per_number : ℕ) (drawn : ℕ) : ℚ :=
  (range : ℚ) / Nat.choose total drawn

def probability_three_two (total : ℕ) (range : ℕ) (per_number : ℕ) (drawn : ℕ) : ℚ :=
  (Nat.choose range 2 * Nat.choose per_number 3 * Nat.choose per_number 2 : ℚ) / Nat.choose total drawn

theorem probability_ratio :
  (probability_three_two total_slips numbers_range slips_per_number drawn_slips) /
  (probability_same_number total_slips numbers_range slips_per_number drawn_slips) = 450 := by
  sorry

end NUMINAMATH_CALUDE_probability_ratio_l2491_249107


namespace NUMINAMATH_CALUDE_balls_in_boxes_l2491_249167

/-- The number of ways to place balls into boxes -/
def place_balls (num_balls : ℕ) (num_boxes : ℕ) (max_per_box : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of ways to place the balls -/
theorem balls_in_boxes : place_balls 3 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l2491_249167


namespace NUMINAMATH_CALUDE_vector_expression_l2491_249128

theorem vector_expression (a b c : ℝ × ℝ) :
  a = (1, 2) →
  a + b = (0, 3) →
  c = (1, 5) →
  c = 2 • a + b := by
sorry

end NUMINAMATH_CALUDE_vector_expression_l2491_249128


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2491_249100

theorem complex_modulus_problem (x y : ℝ) (i : ℂ) (h : i^2 = -1) 
  (eq : (1 + i) * (x + y*i) = 2) : 
  Complex.abs (2*x + y*i) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2491_249100


namespace NUMINAMATH_CALUDE_yellow_peaches_count_l2491_249145

/-- The number of yellow peaches in a basket -/
def yellow_peaches (red green yellow total_green_yellow : ℕ) : Prop :=
  red = 5 ∧ green = 6 ∧ total_green_yellow = 20 → yellow = 14

theorem yellow_peaches_count : ∀ (red green yellow total_green_yellow : ℕ),
  yellow_peaches red green yellow total_green_yellow :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_peaches_count_l2491_249145


namespace NUMINAMATH_CALUDE_johns_age_ratio_l2491_249192

/-- The ratio of John's age 5 years ago to his age in 8 years -/
def age_ratio (current_age : ℕ) : ℚ :=
  (current_age - 5 : ℚ) / (current_age + 8)

/-- Theorem stating that the ratio of John's age 5 years ago to his age in 8 years is 1:2 -/
theorem johns_age_ratio :
  age_ratio 18 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_ratio_l2491_249192


namespace NUMINAMATH_CALUDE_greatest_prime_divisor_digit_sum_l2491_249161

def n : ℕ := 4095

def is_greatest_prime_divisor (p : ℕ) : Prop :=
  Nat.Prime p ∧ p ∣ n ∧ ∀ q, Nat.Prime q → q ∣ n → q ≤ p

def sum_of_digits (m : ℕ) : ℕ :=
  if m < 10 then m else m % 10 + sum_of_digits (m / 10)

theorem greatest_prime_divisor_digit_sum :
  ∃ p, is_greatest_prime_divisor p ∧ sum_of_digits p = 4 := by sorry

end NUMINAMATH_CALUDE_greatest_prime_divisor_digit_sum_l2491_249161


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l2491_249155

theorem polynomial_root_sum (p q r s : ℝ) : 
  let g : ℂ → ℂ := λ x => x^4 + p*x^3 + q*x^2 + r*x + s
  (g (-3*I) = 0 ∧ g (1 + I) = 0) → p + q + r + s = 9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l2491_249155


namespace NUMINAMATH_CALUDE_coin_flip_probability_difference_l2491_249180

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def prob_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ n

/-- The theorem statement -/
theorem coin_flip_probability_difference : 
  |prob_k_heads 6 4 - prob_k_heads 6 6| = 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_difference_l2491_249180


namespace NUMINAMATH_CALUDE_g_of_six_equals_eleven_l2491_249158

theorem g_of_six_equals_eleven (g : ℝ → ℝ) 
  (h : ∀ x : ℝ, g (4 * x - 2) = x^2 + 2 * x + 3) : 
  g 6 = 11 := by
sorry

end NUMINAMATH_CALUDE_g_of_six_equals_eleven_l2491_249158


namespace NUMINAMATH_CALUDE_unique_solution_value_l2491_249104

theorem unique_solution_value (k : ℝ) : 
  (∃! x : ℝ, (x + 3) / (k * x + 2) = x) ↔ k = -1/12 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_value_l2491_249104


namespace NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l2491_249163

theorem unique_solution_for_exponential_equation :
  ∀ x y : ℕ, x ≥ 1 → y ≥ 1 → (7^x = 3^y + 4) → (x = 1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l2491_249163


namespace NUMINAMATH_CALUDE_zeros_of_g_l2491_249136

/-- Given a linear function f(x) = ax + b with a zero at x = 1,
    prove that the zeros of g(x) = bx^2 - ax are 0 and -1 -/
theorem zeros_of_g (a b : ℝ) (h : a + b = 0) (ha : a ≠ 0) :
  let f := λ x : ℝ => a * x + b
  let g := λ x : ℝ => b * x^2 - a * x
  (∀ x : ℝ, g x = 0 ↔ x = 0 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_zeros_of_g_l2491_249136


namespace NUMINAMATH_CALUDE_matthew_crackers_l2491_249160

theorem matthew_crackers (initial_crackers remaining_crackers crackers_per_friend : ℕ) 
  (h1 : initial_crackers = 23)
  (h2 : remaining_crackers = 11)
  (h3 : crackers_per_friend = 6) :
  (initial_crackers - remaining_crackers) / crackers_per_friend = 2 :=
by sorry

end NUMINAMATH_CALUDE_matthew_crackers_l2491_249160


namespace NUMINAMATH_CALUDE_fraction_simplification_l2491_249164

theorem fraction_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  (a^3 - a^2*b) / (a^2*b) - (a^2*b - b^3) / (a*b - b^2) - (a*b) / (a^2 - b^2) = -3*a / (a^2 - b^2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2491_249164


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2491_249147

/-- A hyperbola with foci on the y-axis and asymptotes y = ±4x has eccentricity √17/4 -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a = 4*b) : 
  let e := (Real.sqrt (a^2 + b^2)) / a
  e = Real.sqrt 17 / 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2491_249147


namespace NUMINAMATH_CALUDE_cubic_roots_sum_squares_l2491_249148

theorem cubic_roots_sum_squares (a b c : ℝ) : 
  a^3 - 3*a - 2 = 0 ∧ 
  b^3 - 3*b - 2 = 0 ∧ 
  c^3 - 3*c - 2 = 0 → 
  a^2*(b - c)^2 + b^2*(c - a)^2 + c^2*(a - b)^2 = 9 := by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_squares_l2491_249148


namespace NUMINAMATH_CALUDE_range_of_m_l2491_249191

theorem range_of_m (m : ℝ) : 
  (∃ α : ℝ, 0 < α ∧ α < π / 2 ∧ Real.sqrt 3 * Real.sin α + Real.cos α = m) → 
  1 < m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2491_249191


namespace NUMINAMATH_CALUDE_scalper_discount_l2491_249157

def discount_problem (normal_price : ℝ) (scalper_markup : ℝ) (friend_discount : ℝ) (total_payment : ℝ) : Prop :=
  let website_tickets := 2 * normal_price
  let scalper_tickets := 2 * (normal_price * scalper_markup)
  let friend_ticket := normal_price * friend_discount
  let total_before_discount := website_tickets + scalper_tickets + friend_ticket
  total_before_discount - total_payment = 10

theorem scalper_discount :
  discount_problem 50 2.4 0.6 360 := by
  sorry

end NUMINAMATH_CALUDE_scalper_discount_l2491_249157


namespace NUMINAMATH_CALUDE_exists_card_with_1024_l2491_249169

/-- The number of cards for each natural number up to 1968 -/
def num_cards (n : ℕ) : ℕ := n

/-- The condition that each card has divisors of its number written on it -/
def has_divisors (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → d ≤ 1968 → (num_cards d) ≥ 1

/-- The main theorem to prove -/
theorem exists_card_with_1024 (h : ∀ n ≤ 1968, has_divisors n) :
  (num_cards 1024) > 0 :=
sorry

end NUMINAMATH_CALUDE_exists_card_with_1024_l2491_249169


namespace NUMINAMATH_CALUDE_claire_photos_l2491_249126

theorem claire_photos (lisa robert claire : ℕ) 
  (h1 : lisa = robert)
  (h2 : lisa = 3 * claire)
  (h3 : robert = claire + 10) :
  claire = 5 := by
sorry

end NUMINAMATH_CALUDE_claire_photos_l2491_249126


namespace NUMINAMATH_CALUDE_opposite_numbers_fraction_equals_one_l2491_249134

theorem opposite_numbers_fraction_equals_one (a b : ℝ) 
  (h1 : a + b = 0) 
  (h2 : |a - b| = 2) : 
  (a^2 + 2*a*b + 2*b^2 + 2*a + 2*b + 1) / (a^2 + 3*a*b + b^2 + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_fraction_equals_one_l2491_249134


namespace NUMINAMATH_CALUDE_bicycle_parking_income_l2491_249175

/-- Represents the total income from bicycle parking --/
def total_income (x : ℝ) : ℝ := -0.3 * x + 1600

/-- Theorem stating the relationship between the number of ordinary bicycles parked and the total income --/
theorem bicycle_parking_income (x : ℝ) 
  (h1 : 0 ≤ x) (h2 : x ≤ 2000) : 
  total_income x = 0.5 * x + 0.8 * (2000 - x) := by
  sorry

#check bicycle_parking_income

end NUMINAMATH_CALUDE_bicycle_parking_income_l2491_249175


namespace NUMINAMATH_CALUDE_robert_gets_two_more_than_kate_l2491_249114

/-- The number of candy pieces each child receives. -/
structure CandyDistribution where
  robert : ℕ
  kate : ℕ
  bill : ℕ
  mary : ℕ

/-- The conditions of the candy distribution problem. -/
def ValidDistribution (d : CandyDistribution) : Prop :=
  d.robert + d.kate + d.bill + d.mary = 20 ∧
  d.robert > d.kate ∧
  d.bill = d.mary - 6 ∧
  d.mary = d.robert + 2 ∧
  d.kate = d.bill + 2 ∧
  d.kate = 4

theorem robert_gets_two_more_than_kate (d : CandyDistribution) 
  (h : ValidDistribution d) : d.robert - d.kate = 2 := by
  sorry

end NUMINAMATH_CALUDE_robert_gets_two_more_than_kate_l2491_249114


namespace NUMINAMATH_CALUDE_edwards_remaining_money_l2491_249105

/-- 
Given that Edward had $18 initially and spent $16, 
prove that his remaining money is $2.
-/
theorem edwards_remaining_money :
  let initial_amount : ℕ := 18
  let spent_amount : ℕ := 16
  let remaining_amount : ℕ := initial_amount - spent_amount
  remaining_amount = 2 := by sorry

end NUMINAMATH_CALUDE_edwards_remaining_money_l2491_249105


namespace NUMINAMATH_CALUDE_clock_right_angles_in_day_l2491_249182

/-- Represents a clock with an hour hand and a minute hand. -/
structure Clock :=
  (hour_hand : ℕ)
  (minute_hand : ℕ)

/-- Represents a day consisting of 24 hours. -/
def Day := 24

/-- Checks if the hands of a clock are at right angles. -/
def is_right_angle (c : Clock) : Prop :=
  (c.hour_hand * 5 - c.minute_hand) % 60 = 15 ∨ (c.minute_hand - c.hour_hand * 5) % 60 = 15

/-- Counts the number of times the clock hands are at right angles in a day. -/
def count_right_angles (d : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that the hands of a clock are at right angles 44 times in a day. -/
theorem clock_right_angles_in_day :
  count_right_angles Day = 44 :=
sorry

end NUMINAMATH_CALUDE_clock_right_angles_in_day_l2491_249182


namespace NUMINAMATH_CALUDE_cooper_fence_length_l2491_249178

/-- The length of each wall in Cooper's fence --/
def wall_length : ℕ := 20

/-- The number of walls in Cooper's fence --/
def num_walls : ℕ := 4

/-- The height of each wall in bricks --/
def wall_height : ℕ := 5

/-- The depth of each wall in bricks --/
def wall_depth : ℕ := 2

/-- The total number of bricks needed for the fence --/
def total_bricks : ℕ := 800

theorem cooper_fence_length :
  wall_length * num_walls * wall_height * wall_depth = total_bricks :=
by sorry

end NUMINAMATH_CALUDE_cooper_fence_length_l2491_249178


namespace NUMINAMATH_CALUDE_equation_solutions_l2491_249127

theorem equation_solutions :
  (∃ x : ℚ, 1 - 1 / (x - 5) = x / (x + 5) ∧ x = 15 / 2) ∧
  (∃ x : ℚ, 3 / (x - 1) - 2 / (x + 1) = 1 / (x^2 - 1) ∧ x = -4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2491_249127


namespace NUMINAMATH_CALUDE_no_cyclic_prime_divisibility_l2491_249153

theorem no_cyclic_prime_divisibility : ¬∃ (p : Fin 2007 → ℕ), 
  (∀ i, Nat.Prime (p i)) ∧ 
  (∀ i : Fin 2006, (p i)^2 - 1 ∣ p (i + 1)) ∧
  ((p 2006)^2 - 1 ∣ p 0) := by
  sorry

end NUMINAMATH_CALUDE_no_cyclic_prime_divisibility_l2491_249153


namespace NUMINAMATH_CALUDE_primality_test_upper_bound_l2491_249154

theorem primality_test_upper_bound :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 1100 →
  (∀ p : ℕ, p.Prime ∧ p ≤ 31 → ¬(p ∣ n)) →
  n.Prime ∨ n = 1 :=
sorry

end NUMINAMATH_CALUDE_primality_test_upper_bound_l2491_249154


namespace NUMINAMATH_CALUDE_appended_number_divisible_by_seven_l2491_249177

theorem appended_number_divisible_by_seven (a b : ℕ) (ha : 100 ≤ a ∧ a < 1000) 
  (hb : 100 ≤ b ∧ b < 1000) (h_rem : a % 7 = b % 7) :
  ∃ k : ℕ, 1000 * a + b = 7 * k :=
by sorry

end NUMINAMATH_CALUDE_appended_number_divisible_by_seven_l2491_249177


namespace NUMINAMATH_CALUDE_smaller_integer_problem_l2491_249197

theorem smaller_integer_problem (x y : ℤ) (h1 : y = 5 * x + 2) (h2 : y - x = 26) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_smaller_integer_problem_l2491_249197


namespace NUMINAMATH_CALUDE_probability_one_shirt_two_pants_one_sock_l2491_249166

def num_shirts : ℕ := 3
def num_pants : ℕ := 6
def num_socks : ℕ := 9
def total_items : ℕ := num_shirts + num_pants + num_socks
def num_items_to_remove : ℕ := 4

def probability_specific_combination : ℚ :=
  (Nat.choose num_shirts 1 * Nat.choose num_pants 2 * Nat.choose num_socks 1) /
  Nat.choose total_items num_items_to_remove

theorem probability_one_shirt_two_pants_one_sock :
  probability_specific_combination = 15 / 114 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_shirt_two_pants_one_sock_l2491_249166


namespace NUMINAMATH_CALUDE_exactly_three_correct_deliveries_l2491_249150

def n : ℕ := 5

-- The probability of exactly k successes in n trials
def probability (k : ℕ) : ℚ :=
  (n.choose k * (n - k).factorial) / n.factorial

-- The main theorem
theorem exactly_three_correct_deliveries : probability 3 = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_exactly_three_correct_deliveries_l2491_249150


namespace NUMINAMATH_CALUDE_sin_sum_75_15_degrees_l2491_249187

theorem sin_sum_75_15_degrees :
  Real.sin (75 * π / 180) ^ 2 + Real.sin (15 * π / 180) ^ 2 + 
  Real.sin (75 * π / 180) * Real.sin (15 * π / 180) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_75_15_degrees_l2491_249187


namespace NUMINAMATH_CALUDE_sam_march_aug_earnings_l2491_249195

/-- Represents Sam's work and financial situation --/
structure SamFinances where
  hourly_rate : ℝ
  march_aug_hours : ℕ := 23
  sept_feb_hours : ℕ := 8
  additional_hours : ℕ := 16
  console_cost : ℕ := 600
  car_repair_cost : ℕ := 340

/-- Theorem stating Sam's earnings from March to August --/
theorem sam_march_aug_earnings (sam : SamFinances) :
  sam.hourly_rate * sam.march_aug_hours = 460 :=
by
  have total_needed : ℝ := sam.console_cost + sam.car_repair_cost
  have total_hours : ℕ := sam.march_aug_hours + sam.sept_feb_hours + sam.additional_hours
  have : sam.hourly_rate * total_hours = total_needed :=
    sorry
  sorry

#check sam_march_aug_earnings

end NUMINAMATH_CALUDE_sam_march_aug_earnings_l2491_249195


namespace NUMINAMATH_CALUDE_triangle_division_l2491_249189

/-- The number of triangles formed by n points inside a triangle -/
def numTriangles (n : ℕ) : ℕ := 2 * n + 1

/-- Theorem stating that 1997 points inside a triangle creates 3995 smaller triangles -/
theorem triangle_division (n : ℕ) (h : n = 1997) : numTriangles n = 3995 := by
  sorry

end NUMINAMATH_CALUDE_triangle_division_l2491_249189


namespace NUMINAMATH_CALUDE_race_speed_ratio_l2491_249176

/-- Represents the speeds and distances in a race between two runners A and B -/
structure RaceParameters where
  speedA : ℝ
  speedB : ℝ
  totalDistance : ℝ
  headStart : ℝ

/-- Theorem stating that if A and B finish at the same time in a race with given parameters,
    then A's speed is 4 times B's speed -/
theorem race_speed_ratio 
  (race : RaceParameters) 
  (h1 : race.totalDistance = 100)
  (h2 : race.headStart = 75)
  (h3 : race.totalDistance / race.speedA = (race.totalDistance - race.headStart) / race.speedB) :
  race.speedA = 4 * race.speedB := by
  sorry


end NUMINAMATH_CALUDE_race_speed_ratio_l2491_249176


namespace NUMINAMATH_CALUDE_solution_set_inequality_proof_l2491_249121

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 3|
def g (x : ℝ) : ℝ := |x - 2|

-- Theorem for the solution set of f(x) + g(x) < 2
theorem solution_set : 
  {x : ℝ | f x + g x < 2} = {x : ℝ | 3/2 < x ∧ x < 7/2} := by sorry

-- Theorem for the inequality proof
theorem inequality_proof (x y : ℝ) (hx : f x ≤ 1) (hy : g y ≤ 1) : 
  |x - 2*y + 1| ≤ 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_proof_l2491_249121


namespace NUMINAMATH_CALUDE_inequality_holds_infinitely_often_l2491_249109

theorem inequality_holds_infinitely_often (a : ℕ → ℝ) 
  (h : ∀ n, a n > 0) : 
  ∀ m : ℕ, ∃ n : ℕ, n > m ∧ 1 + a n > a (n - 1) * (2 ^ (1 / n : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_inequality_holds_infinitely_often_l2491_249109


namespace NUMINAMATH_CALUDE_rectangle_width_l2491_249110

theorem rectangle_width (perimeter : ℝ) (length_difference : ℝ) (width : ℝ) : 
  perimeter = 48 →
  length_difference = 2 →
  perimeter = 2 * (width + length_difference) + 2 * width →
  width = 11 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l2491_249110


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2491_249188

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 5 * a 14 = 5 →
  a 8 * a 9 * a 10 * a 11 = 25 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2491_249188


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2491_249194

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | |x - 1| < 1}
def Q : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2491_249194


namespace NUMINAMATH_CALUDE_credit_rating_equation_l2491_249112

theorem credit_rating_equation (x : ℝ) : 
  (96 : ℝ) = x * (1 + 0.2) ↔ 
  (96 : ℝ) = x + x * 0.2 := by sorry

end NUMINAMATH_CALUDE_credit_rating_equation_l2491_249112


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l2491_249133

-- Define the triangle ABC
variable (A B C : ℝ)

-- Define the dot product of vectors AB and BC
def dot_product_AB_BC : ℝ := sorry

-- Define the area of triangle ABC
def area_ABC : ℝ := sorry

-- State the theorem
theorem triangle_ABC_properties 
  (h1 : dot_product_AB_BC = (3/2) * area_ABC)
  (h2 : A - C = π/4) : 
  Real.sin B = 4/5 ∧ Real.cos A = (Real.sqrt (50 + 5 * Real.sqrt 2)) / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l2491_249133


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l2491_249152

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l2491_249152
