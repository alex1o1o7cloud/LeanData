import Mathlib

namespace inscribed_box_radius_l1659_165960

/-- A rectangular box inscribed in a sphere --/
structure InscribedBox where
  x : ℝ
  y : ℝ
  z : ℝ
  r : ℝ
  h_surface_area : 2 * (x*y + x*z + y*z) = 432
  h_edge_sum : 4 * (x + y + z) = 104
  h_inscribed : (2*r)^2 = x^2 + y^2 + z^2

/-- Theorem: If a rectangular box Q is inscribed in a sphere, with surface area 432,
    sum of edge lengths 104, and one dimension 8, then the radius of the sphere is 7 --/
theorem inscribed_box_radius (Q : InscribedBox) (h_x : Q.x = 8) : Q.r = 7 := by
  sorry

end inscribed_box_radius_l1659_165960


namespace circle_diameter_and_circumference_l1659_165925

theorem circle_diameter_and_circumference (A : ℝ) (h : A = 16 * Real.pi) :
  ∃ (d c : ℝ), d = 8 ∧ c = 8 * Real.pi ∧ A = Real.pi * (d / 2)^2 ∧ c = Real.pi * d := by
  sorry

end circle_diameter_and_circumference_l1659_165925


namespace real_roots_of_polynomial_l1659_165964

def p (x : ℝ) : ℝ := x^5 - 3*x^4 + 3*x^3 - x^2 - 4*x + 4

theorem real_roots_of_polynomial :
  ∀ x : ℝ, p x = 0 ↔ x = -1 ∨ x = 1 ∨ x = 2 :=
by sorry

end real_roots_of_polynomial_l1659_165964


namespace total_walking_hours_l1659_165944

/-- Represents the types of dogs Charlotte walks -/
inductive DogType
  | Poodle
  | Chihuahua
  | Labrador

/-- Represents a day of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday

/-- Returns the number of hours it takes to walk a dog of a given type -/
def walkingTime (d : DogType) : Nat :=
  match d with
  | DogType.Poodle => 2
  | DogType.Chihuahua => 1
  | DogType.Labrador => 3

/-- Returns the number of dogs of a given type walked on a specific day -/
def dogsWalked (day : Day) (dogType : DogType) : Nat :=
  match day, dogType with
  | Day.Monday, DogType.Poodle => 4
  | Day.Monday, DogType.Chihuahua => 2
  | Day.Monday, DogType.Labrador => 0
  | Day.Tuesday, DogType.Poodle => 4
  | Day.Tuesday, DogType.Chihuahua => 2
  | Day.Tuesday, DogType.Labrador => 0
  | Day.Wednesday, DogType.Poodle => 0
  | Day.Wednesday, DogType.Chihuahua => 0
  | Day.Wednesday, DogType.Labrador => 4

/-- Calculates the total hours spent walking dogs on a given day -/
def hoursPerDay (day : Day) : Nat :=
  (dogsWalked day DogType.Poodle * walkingTime DogType.Poodle) +
  (dogsWalked day DogType.Chihuahua * walkingTime DogType.Chihuahua) +
  (dogsWalked day DogType.Labrador * walkingTime DogType.Labrador)

/-- Theorem stating that the total hours for dog-walking this week is 32 -/
theorem total_walking_hours :
  hoursPerDay Day.Monday + hoursPerDay Day.Tuesday + hoursPerDay Day.Wednesday = 32 := by
  sorry


end total_walking_hours_l1659_165944


namespace taxi_charge_correct_l1659_165983

/-- Calculates the total charge for a taxi trip given the initial fee, per-increment charge, increment distance, and total trip distance. -/
def total_charge (initial_fee : ℚ) (per_increment_charge : ℚ) (increment_distance : ℚ) (trip_distance : ℚ) : ℚ :=
  initial_fee + (trip_distance / increment_distance).floor * per_increment_charge

/-- Proves that the total charge for a specific taxi trip is correct. -/
theorem taxi_charge_correct :
  let initial_fee : ℚ := 41/20  -- $2.05
  let per_increment_charge : ℚ := 7/20  -- $0.35
  let increment_distance : ℚ := 2/5  -- 2/5 mile
  let trip_distance : ℚ := 18/5  -- 3.6 miles
  total_charge initial_fee per_increment_charge increment_distance trip_distance = 26/5  -- $5.20
  := by sorry

end taxi_charge_correct_l1659_165983


namespace vector_at_minus_2_l1659_165927

/-- A line in a plane parameterized by t -/
def line (t : ℝ) : ℝ × ℝ := sorry

/-- The vector at t = 5 is (0, 5) -/
axiom vector_at_5 : line 5 = (0, 5)

/-- The vector at t = 8 is (9, 1) -/
axiom vector_at_8 : line 8 = (9, 1)

/-- The theorem to prove -/
theorem vector_at_minus_2 : line (-2) = (21, -23/3) := by sorry

end vector_at_minus_2_l1659_165927


namespace phi_value_l1659_165975

open Real

noncomputable def f (x φ : ℝ) : ℝ := sin (Real.sqrt 3 * x + φ)

noncomputable def f_deriv (x φ : ℝ) : ℝ := Real.sqrt 3 * cos (Real.sqrt 3 * x + φ)

theorem phi_value (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) 
  (h3 : ∀ x, f x φ + f_deriv x φ = -(f (-x) φ + f_deriv (-x) φ)) : 
  φ = 2 * π / 3 := by
sorry

end phi_value_l1659_165975


namespace third_chest_silver_excess_l1659_165943

/-- Represents the number of coins in each chest -/
structure ChestContents where
  gold : ℕ
  silver : ℕ

/-- Problem setup -/
def coin_problem (chest1 chest2 chest3 : ChestContents) : Prop :=
  let total_gold := chest1.gold + chest2.gold + chest3.gold
  let total_silver := chest1.silver + chest2.silver + chest3.silver
  total_gold = 40 ∧
  total_silver = 40 ∧
  chest1.gold = chest1.silver + 7 ∧
  chest2.gold = chest2.silver + 15

/-- Theorem statement -/
theorem third_chest_silver_excess 
  (chest1 chest2 chest3 : ChestContents) 
  (h : coin_problem chest1 chest2 chest3) : 
  chest3.silver = chest3.gold + 22 := by
  sorry

#check third_chest_silver_excess

end third_chest_silver_excess_l1659_165943


namespace factor_tree_product_l1659_165972

theorem factor_tree_product : ∀ (X F G H : ℕ),
  X = F * G →
  F = 11 * 7 →
  G = 7 * H →
  H = 17 * 2 →
  X = 57556 := by
sorry

end factor_tree_product_l1659_165972


namespace valid_sampling_interval_l1659_165919

def total_population : ℕ := 102
def removed_individuals : ℕ := 2
def sampling_interval : ℕ := 10

theorem valid_sampling_interval :
  (total_population - removed_individuals) % sampling_interval = 0 := by
  sorry

end valid_sampling_interval_l1659_165919


namespace brand_preference_survey_l1659_165988

theorem brand_preference_survey (total : ℕ) (ratio : ℚ) (brand_x : ℕ) : 
  total = 250 → 
  ratio = 4/1 → 
  brand_x = total * (ratio / (1 + ratio)) → 
  brand_x = 200 := by
sorry

end brand_preference_survey_l1659_165988


namespace solutions_eq1_solutions_eq2_l1659_165977

-- Equation 1
theorem solutions_eq1 : 
  ∀ x : ℝ, x^2 - 6*x - 7 = 0 ↔ x = 7 ∨ x = -1 := by sorry

-- Equation 2
theorem solutions_eq2 : 
  ∀ x : ℝ, 3*x^2 - 1 = 2*x ↔ x = 1 ∨ x = -1/3 := by sorry

end solutions_eq1_solutions_eq2_l1659_165977


namespace perpendicular_vectors_k_value_l1659_165967

/-- Given points A and B, and vector a, proves that if AB is perpendicular to a, then k = 1 -/
theorem perpendicular_vectors_k_value (k : ℝ) :
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, k)
  let a : ℝ × ℝ := (-1, 2)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  (AB.1 * a.1 + AB.2 * a.2 = 0) → k = 1 := by
sorry

end perpendicular_vectors_k_value_l1659_165967


namespace bert_stamps_correct_l1659_165917

/-- The number of stamps Bert bought -/
def stamps_bought : ℕ := 300

/-- The number of stamps Bert had before the purchase -/
def stamps_before : ℕ := stamps_bought / 2

/-- The total number of stamps Bert has after the purchase -/
def total_stamps : ℕ := 450

/-- Theorem stating that the number of stamps Bert bought is correct -/
theorem bert_stamps_correct :
  stamps_bought = 300 ∧
  stamps_before = stamps_bought / 2 ∧
  total_stamps = stamps_before + stamps_bought :=
by sorry

end bert_stamps_correct_l1659_165917


namespace number_problem_l1659_165978

theorem number_problem (x : ℝ) : 
  (0.3 * x = 0.6 * 150 + 120) → x = 700 := by
sorry

end number_problem_l1659_165978


namespace arithmetic_sequence_20th_term_l1659_165920

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_20th_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 3 + a 5 = 105)
  (h_sum2 : a 2 + a 4 + a 6 = 99) :
  a 20 = 1 := by
  sorry

end arithmetic_sequence_20th_term_l1659_165920


namespace geometric_sum_first_10_terms_l1659_165989

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₁ * r^(n - 1)

def geometric_sum (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₁ * (1 - r^n) / (1 - r)

theorem geometric_sum_first_10_terms :
  let a₁ : ℚ := 12
  let r : ℚ := 1/3
  let n : ℕ := 10
  geometric_sum a₁ r n = 1062864/59049 := by sorry

end geometric_sum_first_10_terms_l1659_165989


namespace opposite_roots_imply_k_value_l1659_165986

theorem opposite_roots_imply_k_value (k : ℝ) : 
  (∃ x : ℝ, x^2 + (k^2 - 4)*x + (k - 1) = 0 ∧ 
             ∃ y : ℝ, y^2 + (k^2 - 4)*y + (k - 1) = 0 ∧ 
             x = -y ∧ x ≠ y) → 
  k = -2 :=
by sorry

end opposite_roots_imply_k_value_l1659_165986


namespace remaining_volume_cube_with_cylindrical_hole_l1659_165970

/-- The remaining volume of a cube after drilling a cylindrical hole -/
theorem remaining_volume_cube_with_cylindrical_hole :
  let cube_side : ℝ := 6
  let hole_radius : ℝ := 3
  let hole_height : ℝ := 6
  let cube_volume : ℝ := cube_side ^ 3
  let cylinder_volume : ℝ := π * hole_radius ^ 2 * hole_height
  let remaining_volume : ℝ := cube_volume - cylinder_volume
  remaining_volume = 216 - 54 * π := by
  sorry


end remaining_volume_cube_with_cylindrical_hole_l1659_165970


namespace solution_difference_l1659_165912

theorem solution_difference (x₀ y₀ : ℝ) : 
  (x₀^3 - 2023*x₀ = y₀^3 - 2023*y₀ + 2020) →
  (x₀^2 + x₀*y₀ + y₀^2 = 2022) →
  (x₀ - y₀ = -2020) := by
sorry

end solution_difference_l1659_165912


namespace total_hats_bought_l1659_165993

theorem total_hats_bought (blue_cost green_cost total_price green_hats : ℕ) 
  (h1 : blue_cost = 6)
  (h2 : green_cost = 7)
  (h3 : total_price = 550)
  (h4 : green_hats = 40) :
  ∃ (blue_hats : ℕ), blue_cost * blue_hats + green_cost * green_hats = total_price ∧
                     blue_hats + green_hats = 85 := by
  sorry

end total_hats_bought_l1659_165993


namespace negative_fraction_comparison_l1659_165923

theorem negative_fraction_comparison : -3/5 < -4/7 := by
  sorry

end negative_fraction_comparison_l1659_165923


namespace quadratic_equations_solutions_l1659_165956

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁^2 + 2*x₁ = 0 ∧ x₂^2 + 2*x₂ = 0 ∧ x₁ = 0 ∧ x₂ = -2) ∧
  (∃ x₁ x₂ : ℝ, (x₁+1)^2 - 144 = 0 ∧ (x₂+1)^2 - 144 = 0 ∧ x₁ = 11 ∧ x₂ = -13) ∧
  (∃ x₁ x₂ : ℝ, 3*(x₁-2)^2 = x₁*(x₁-2) ∧ 3*(x₂-2)^2 = x₂*(x₂-2) ∧ x₁ = 2 ∧ x₂ = 3) ∧
  (∃ x₁ x₂ : ℝ, x₁^2 + 5*x₁ - 1 = 0 ∧ x₂^2 + 5*x₂ - 1 = 0 ∧ 
    x₁ = (-5 + Real.sqrt 29) / 2 ∧ x₂ = (-5 - Real.sqrt 29) / 2) :=
by
  sorry


end quadratic_equations_solutions_l1659_165956


namespace time_per_toy_l1659_165962

/-- Given a worker who makes 50 toys in 150 hours, prove that the time taken to make one toy is 3 hours. -/
theorem time_per_toy (total_hours : ℝ) (total_toys : ℝ) (h1 : total_hours = 150) (h2 : total_toys = 50) :
  total_hours / total_toys = 3 := by
  sorry

end time_per_toy_l1659_165962


namespace sequence_sum_l1659_165974

theorem sequence_sum (a : ℕ → ℕ) (h : ∀ k : ℕ, k > 0 → a k + a (k + 1) = 2 * k + 1) :
  a 1 + a 100 = 101 := by
  sorry

end sequence_sum_l1659_165974


namespace trajectory_is_hyperbola_l1659_165959

-- Define the complex plane
def ComplexPlane := ℂ

-- Define the condition for the trajectory
def TrajectoryCondition (z : ℂ) : Prop :=
  Complex.abs (Complex.abs (z - 1) - Complex.abs (z + Complex.I)) = 1

-- Define a hyperbola in the complex plane
def IsHyperbola (S : Set ℂ) : Prop :=
  ∃ (F₁ F₂ : ℂ) (a : ℝ), a > 0 ∧ Complex.abs (F₁ - F₂) > 2 * a ∧
    S = {z : ℂ | Complex.abs (Complex.abs (z - F₁) - Complex.abs (z - F₂)) = 2 * a}

-- Theorem statement
theorem trajectory_is_hyperbola :
  IsHyperbola {z : ℂ | TrajectoryCondition z} :=
sorry

end trajectory_is_hyperbola_l1659_165959


namespace sqrt_of_negative_nine_l1659_165945

theorem sqrt_of_negative_nine :
  (3 * Complex.I)^2 = -9 ∧ (-3 * Complex.I)^2 = -9 := by
  sorry

end sqrt_of_negative_nine_l1659_165945


namespace custom_op_solution_l1659_165908

-- Define the custom operation
def customOp (a b : ℤ) : ℤ := (a - 1) * (b - 1)

-- State the theorem
theorem custom_op_solution :
  ∀ y : ℤ, customOp y 10 = 90 → y = 11 := by
  sorry

end custom_op_solution_l1659_165908


namespace arithmetic_sequence_ratio_l1659_165966

/-- An arithmetic sequence with first four terms a, y, b, 3y has a/b = 0 -/
theorem arithmetic_sequence_ratio (a y b : ℝ) : 
  (∃ d : ℝ, y = a + d ∧ b = y + d ∧ 3*y = b + d) → a / b = 0 :=
by sorry

end arithmetic_sequence_ratio_l1659_165966


namespace train_distance_l1659_165950

/-- The distance covered by a train traveling at a constant speed for a given time. -/
theorem train_distance (speed : ℝ) (time : ℝ) (h1 : speed = 150) (h2 : time = 8) :
  speed * time = 1200 := by
  sorry

end train_distance_l1659_165950


namespace sqrt_360000_equals_600_l1659_165947

theorem sqrt_360000_equals_600 : Real.sqrt 360000 = 600 := by
  sorry

end sqrt_360000_equals_600_l1659_165947


namespace price_restoration_l1659_165980

theorem price_restoration (original_price : ℝ) (reduced_price : ℝ) (increase_percentage : ℝ) : 
  reduced_price = original_price * (1 - 0.2) →
  reduced_price * (1 + increase_percentage) = original_price →
  increase_percentage = 0.25 := by
  sorry

#check price_restoration

end price_restoration_l1659_165980


namespace jet_bar_sales_difference_l1659_165934

def weekly_target : ℕ := 90
def monday_sales : ℕ := 45
def remaining_sales : ℕ := 16

theorem jet_bar_sales_difference : 
  monday_sales - (weekly_target - remaining_sales - monday_sales) = 16 := by
  sorry

end jet_bar_sales_difference_l1659_165934


namespace median_list_i_is_eight_l1659_165968

def list_i : List ℕ := [9, 2, 4, 7, 10, 11]
def list_ii : List ℕ := [3, 3, 4, 6, 7, 10]

def median (l : List ℕ) : ℚ := sorry
def mode (l : List ℕ) : ℕ := sorry

theorem median_list_i_is_eight :
  median list_i = 8 :=
by
  have h1 : median list_ii + mode list_ii = 8 := by sorry
  have h2 : median list_i = median list_ii + mode list_ii := by sorry
  sorry

end median_list_i_is_eight_l1659_165968


namespace cos_alpha_plus_pi_12_l1659_165914

theorem cos_alpha_plus_pi_12 (α : Real) (h : Real.tan (α + π/3) = -2) :
  Real.cos (α + π/12) = Real.sqrt 10 / 10 ∨ Real.cos (α + π/12) = -Real.sqrt 10 / 10 := by
  sorry

end cos_alpha_plus_pi_12_l1659_165914


namespace mixed_fraction_power_product_l1659_165942

theorem mixed_fraction_power_product (n : ℕ) (m : ℕ) :
  (-(3 : ℚ) / 2) ^ (2021 : ℕ) * (2 : ℚ) / 3 ^ (2023 : ℕ) = -(4 : ℚ) / 9 := by
  sorry

end mixed_fraction_power_product_l1659_165942


namespace chessboard_inner_square_probability_l1659_165953

/-- Represents a square chessboard -/
structure Chessboard where
  size : ℕ

/-- Calculates the number of squares on the perimeter of the chessboard -/
def perimeterSquares (board : Chessboard) : ℕ :=
  4 * board.size - 4

/-- Calculates the number of squares not on the perimeter of the chessboard -/
def innerSquares (board : Chessboard) : ℕ :=
  board.size * board.size - perimeterSquares board

/-- The probability of choosing an inner square on the chessboard -/
def innerSquareProbability (board : Chessboard) : ℚ :=
  innerSquares board / (board.size * board.size)

theorem chessboard_inner_square_probability :
  let board := Chessboard.mk 10
  innerSquareProbability board = 16 / 25 := by
  sorry

end chessboard_inner_square_probability_l1659_165953


namespace compute_expression_l1659_165957

theorem compute_expression : 10 + 4 * (5 + 3)^3 = 2058 := by
  sorry

end compute_expression_l1659_165957


namespace plains_total_area_l1659_165906

def plain_problem (region_B region_A total : ℕ) : Prop :=
  (region_B = 200) ∧
  (region_A = region_B - 50) ∧
  (total = region_A + region_B)

theorem plains_total_area : 
  ∃ (region_B region_A total : ℕ), 
    plain_problem region_B region_A total ∧ total = 350 :=
by
  sorry

end plains_total_area_l1659_165906


namespace desiree_age_proof_l1659_165915

/-- Desiree's current age -/
def desiree_age : ℕ := 6

/-- Desiree's cousin's current age -/
def cousin_age : ℕ := 3

/-- Proves that Desiree's current age is 6 years old -/
theorem desiree_age_proof :
  (desiree_age = 2 * cousin_age) ∧
  (desiree_age + 30 = (2/3 : ℚ) * (cousin_age + 30) + 14) →
  desiree_age = 6 := by
sorry

end desiree_age_proof_l1659_165915


namespace negation_of_universal_proposition_l1659_165940

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2^x - 1 < 0) ↔ (∃ x : ℝ, x^2 + 2^x - 1 ≥ 0) := by
  sorry

end negation_of_universal_proposition_l1659_165940


namespace reflection_line_sum_l1659_165963

/-- Given a line y = mx + b, if the reflection of point (2, 3) across this line is (10, -1), then m + b = -9 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    -- The point (x, y) is on the line y = mx + b
    y = m * x + b ∧ 
    -- The point (x, y) is equidistant from (2, 3) and (10, -1)
    (x - 2)^2 + (y - 3)^2 = (x - 10)^2 + (y + 1)^2 ∧
    -- The line through (2, 3) and (10, -1) is perpendicular to y = mx + b
    m * ((10 - 2) / ((-1) - 3)) = -1) →
  m + b = -9 := by sorry


end reflection_line_sum_l1659_165963


namespace subtracted_number_l1659_165992

theorem subtracted_number (x n : ℚ) : 
  x / 4 - (x - n) / 6 = 1 → x = 6 → n = 3 := by
  sorry

end subtracted_number_l1659_165992


namespace apple_bags_problem_l1659_165971

theorem apple_bags_problem (A B C : ℕ) 
  (h1 : A + B + C = 24)
  (h2 : B + C = 18)
  (h3 : A + C = 19) :
  A + B = 11 := by
sorry

end apple_bags_problem_l1659_165971


namespace remainder_97_103_mod_9_l1659_165985

theorem remainder_97_103_mod_9 : (97 * 103) % 9 = 1 := by
  sorry

end remainder_97_103_mod_9_l1659_165985


namespace geometric_sequence_property_l1659_165930

def is_geometric_sequence (α : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, α (n + 1) = α n * r

theorem geometric_sequence_property 
  (α : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence α) 
  (h_product : α 4 * α 5 * α 6 = 27) : 
  α 5 = 3 := by
sorry

end geometric_sequence_property_l1659_165930


namespace seating_arrangements_l1659_165936

/-- The number of boys -/
def num_boys : Nat := 5

/-- The number of girls -/
def num_girls : Nat := 4

/-- The total number of chairs -/
def total_chairs : Nat := 9

/-- The number of odd-numbered chairs -/
def odd_chairs : Nat := (total_chairs + 1) / 2

/-- The number of even-numbered chairs -/
def even_chairs : Nat := total_chairs / 2

/-- Factorial function -/
def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem seating_arrangements :
  factorial num_boys * factorial num_girls = 2880 :=
by sorry

end seating_arrangements_l1659_165936


namespace quadratic_maximum_quadratic_maximum_achieved_l1659_165981

theorem quadratic_maximum (s : ℝ) : -7 * s^2 + 56 * s - 18 ≤ 94 := by sorry

theorem quadratic_maximum_achieved : ∃ s : ℝ, -7 * s^2 + 56 * s - 18 = 94 := by sorry

end quadratic_maximum_quadratic_maximum_achieved_l1659_165981


namespace expression_evaluation_l1659_165913

theorem expression_evaluation (x : ℝ) (h : x = 4) :
  (x - 1 - 3 / (x + 1)) / ((x^2 - 2*x) / (x + 1)) = 3/2 := by
  sorry

end expression_evaluation_l1659_165913


namespace logarithm_expression_equals_two_l1659_165916

theorem logarithm_expression_equals_two :
  (Real.log 243 / Real.log 3) / (Real.log 3 / Real.log 81) -
  (Real.log 729 / Real.log 3) / (Real.log 3 / Real.log 27) = 2 := by
  sorry

end logarithm_expression_equals_two_l1659_165916


namespace beidou_satellite_altitude_scientific_notation_l1659_165973

theorem beidou_satellite_altitude_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 18500000 = a * (10 : ℝ) ^ n ∧ a = 1.85 ∧ n = 7 := by
  sorry

end beidou_satellite_altitude_scientific_notation_l1659_165973


namespace range_of_a_for_non_negative_x_l1659_165928

theorem range_of_a_for_non_negative_x (a x : ℝ) : 
  (x - a = 1 - 2*x ∧ x ≥ 0) → a ≥ -1 := by
sorry

end range_of_a_for_non_negative_x_l1659_165928


namespace inverse_proportion_problem_l1659_165931

/-- Two numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ) 
  (h1 : InverselyProportional x y)
  (h2 : ∃ x₀ y₀ : ℝ, x₀ + y₀ = 60 ∧ x₀ = 3 * y₀ ∧ InverselyProportional x₀ y₀) :
  x = 12 → y = 56.25 := by
  sorry

end inverse_proportion_problem_l1659_165931


namespace bens_old_car_cost_l1659_165946

/-- Proves that Ben's old car cost $1900 given the problem conditions -/
theorem bens_old_car_cost :
  ∀ (old_car_cost new_car_cost : ℕ),
  new_car_cost = 2 * old_car_cost →
  old_car_cost = 1800 →
  new_car_cost = 1800 + 2000 →
  old_car_cost = 1900 := by
  sorry

#check bens_old_car_cost

end bens_old_car_cost_l1659_165946


namespace binomial_expansion_constant_term_l1659_165941

theorem binomial_expansion_constant_term (x : ℝ) (n : ℕ) :
  (∀ k : ℕ, k ≤ n → (n.choose k) ≤ (n.choose 4)) →
  (∃ k : ℕ, (8 : ℝ) - (4 * k) / 3 = 0) →
  (∃ c : ℝ, c = (n.choose 6) * (1/2)^2 * (-1)^6) →
  c = 7 := by
sorry

end binomial_expansion_constant_term_l1659_165941


namespace stock_rise_amount_l1659_165949

/-- Represents the daily change in stock value -/
structure StockChange where
  morning_rise : ℝ
  afternoon_fall : ℝ

/-- Calculates the stock value after n days given initial value and daily change -/
def stock_value_after_days (initial_value : ℝ) (daily_change : StockChange) (n : ℕ) : ℝ :=
  initial_value + n * (daily_change.morning_rise - daily_change.afternoon_fall)

theorem stock_rise_amount (initial_value : ℝ) (daily_change : StockChange) :
  initial_value = 100 →
  daily_change.afternoon_fall = 1 →
  stock_value_after_days initial_value daily_change 100 = 200 →
  daily_change.morning_rise = 2 := by
  sorry

#eval stock_value_after_days 100 ⟨2, 1⟩ 100

end stock_rise_amount_l1659_165949


namespace circular_sector_properties_l1659_165982

/-- A circular sector with given area and perimeter -/
structure CircularSector where
  area : ℝ
  perimeter : ℝ

/-- The central angle of a circular sector -/
def central_angle (s : CircularSector) : ℝ := sorry

/-- The chord length of a circular sector -/
def chord_length (s : CircularSector) : ℝ := sorry

/-- Theorem stating the properties of a specific circular sector -/
theorem circular_sector_properties :
  let s : CircularSector := { area := 1, perimeter := 4 }
  (central_angle s = 2) ∧ (chord_length s = 2 * Real.sin 1) := by sorry

end circular_sector_properties_l1659_165982


namespace parallel_planes_from_perpendicular_lines_perpendicular_line_plane_from_intersection_l1659_165998

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (intersection : Plane → Plane → Line)
variable (contained_in : Line → Plane → Prop)

-- Theorem for statement ②
theorem parallel_planes_from_perpendicular_lines 
  (l m : Line) (α β : Plane) :
  parallel l m →
  perpendicular_line_plane m α →
  perpendicular_line_plane l β →
  parallel_plane α β :=
sorry

-- Theorem for statement ④
theorem perpendicular_line_plane_from_intersection 
  (l : Line) (α β : Plane) (m : Line) :
  perpendicular_plane α β →
  intersection α β = m →
  contained_in l β →
  perpendicular l m →
  perpendicular_line_plane l α :=
sorry

end parallel_planes_from_perpendicular_lines_perpendicular_line_plane_from_intersection_l1659_165998


namespace line_passes_through_fixed_point_l1659_165951

/-- The line equation passing through a fixed point for any real k -/
def line_equation (k x y : ℝ) : ℝ := (2*k - 1)*x - (k + 3)*y - (k - 11)

/-- The fixed point that the line always passes through -/
def fixed_point : ℝ × ℝ := (2, 3)

/-- Theorem stating that the line always passes through the fixed point -/
theorem line_passes_through_fixed_point :
  ∀ k : ℝ, line_equation k (fixed_point.1) (fixed_point.2) = 0 := by
sorry

end line_passes_through_fixed_point_l1659_165951


namespace range_of_a_range_of_m_l1659_165976

-- Define the sets
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | (x - 5) / x ≤ 0}
def C (a : ℝ) : Set ℝ := {x | 3 * a ≤ x ∧ x ≤ 2 * a + 1}
def D (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1/2}

-- Part 1
theorem range_of_a : 
  ∀ a : ℝ, (C a ⊆ (A ∩ B)) ↔ (a ∈ Set.Ioo 0 (1/2) ∪ Set.Ioi 1) :=
sorry

-- Part 2
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ D m → x ∈ (A ∪ B)) ∧ 
           (∃ y : ℝ, y ∈ (A ∪ B) ∧ y ∉ D m) ↔
           m ∈ Set.Icc (-2) (9/2) :=
sorry

end range_of_a_range_of_m_l1659_165976


namespace candy_boxes_problem_l1659_165924

/-- Given that Paul bought 6 boxes of chocolate candy and 4 boxes of caramel candy,
    with a total of 90 candies, and each box contains the same number of pieces,
    prove that there are 9 pieces of candy in each box. -/
theorem candy_boxes_problem (pieces_per_box : ℕ) : 
  (6 * pieces_per_box + 4 * pieces_per_box = 90) → pieces_per_box = 9 := by
sorry

end candy_boxes_problem_l1659_165924


namespace prob_select_one_from_2_7_l1659_165938

/-- The decimal representation of 2/7 -/
def decimal_rep_2_7 : List Nat := [2, 8, 5, 7, 1, 4]

/-- The probability of selecting a specific digit from the decimal representation of 2/7 -/
def prob_select_digit (d : Nat) : Rat :=
  (decimal_rep_2_7.count d) / (decimal_rep_2_7.length)

theorem prob_select_one_from_2_7 :
  prob_select_digit 1 = 1 / 6 := by sorry

end prob_select_one_from_2_7_l1659_165938


namespace no_integer_solution_exists_l1659_165999

theorem no_integer_solution_exists (a b : ℤ) : 
  ∃ c : ℤ, ∀ m n : ℤ, m^2 + a*m + b ≠ 2*n^2 + 2*n + c :=
by sorry

end no_integer_solution_exists_l1659_165999


namespace equivalent_operations_l1659_165954

theorem equivalent_operations (x : ℝ) : 
  (x * (4/5)) / (2/7) = x * (7/4) := by
  sorry

end equivalent_operations_l1659_165954


namespace two_in_A_l1659_165932

def A : Set ℝ := {x | x^2 - 4 = 0}

theorem two_in_A : 2 ∈ A := by sorry

end two_in_A_l1659_165932


namespace sport_popularity_order_l1659_165903

/-- Represents a sport with its popularity fraction -/
structure Sport where
  name : String
  popularity : Rat

/-- Determines if one sport is more popular than another -/
def morePopularThan (s1 s2 : Sport) : Prop :=
  s1.popularity > s2.popularity

theorem sport_popularity_order (basketball tennis volleyball : Sport)
  (h_basketball : basketball.name = "Basketball" ∧ basketball.popularity = 9/24)
  (h_tennis : tennis.name = "Tennis" ∧ tennis.popularity = 8/24)
  (h_volleyball : volleyball.name = "Volleyball" ∧ volleyball.popularity = 7/24) :
  morePopularThan basketball tennis ∧ 
  morePopularThan tennis volleyball ∧
  [basketball.name, tennis.name, volleyball.name] = ["Basketball", "Tennis", "Volleyball"] :=
by sorry

end sport_popularity_order_l1659_165903


namespace expression_simplification_appropriate_integer_value_l1659_165961

theorem expression_simplification (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) (h3 : x ≠ -2) :
  (x / (x - 2) - 4 / (x^2 - 2*x)) / ((x + 2) / x^2) = x :=
by sorry

theorem appropriate_integer_value :
  ∃ (x : ℤ), -2 ≤ x ∧ x < Real.sqrt 7 ∧ x ≠ 0 ∧ x ≠ 2 ∧ x = 1 :=
by sorry

end expression_simplification_appropriate_integer_value_l1659_165961


namespace coefficient_x_cubed_is_73_l1659_165952

def p₁ (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 + x + 1
def p₂ (x : ℝ) : ℝ := 2 * x^2 + x + 4
def p₃ (x : ℝ) : ℝ := x^2 + 2 * x + 3

def product (x : ℝ) : ℝ := p₁ x * p₂ x * p₃ x

theorem coefficient_x_cubed_is_73 :
  ∃ (a b c d : ℝ), product = fun x ↦ 73 * x^3 + a * x^4 + b * x^2 + c * x + d :=
by sorry

end coefficient_x_cubed_is_73_l1659_165952


namespace raghu_investment_l1659_165996

/-- Represents the investment amounts of Raghu, Trishul, and Vishal -/
structure Investments where
  raghu : ℝ
  trishul : ℝ
  vishal : ℝ

/-- The conditions of the investment problem -/
def investment_conditions (inv : Investments) : Prop :=
  inv.trishul = 0.9 * inv.raghu ∧
  inv.vishal = 1.1 * inv.trishul ∧
  inv.raghu + inv.trishul + inv.vishal = 6069

/-- Theorem stating that under the given conditions, Raghu's investment is 2100 -/
theorem raghu_investment (inv : Investments) 
  (h : investment_conditions inv) : inv.raghu = 2100 := by
  sorry

end raghu_investment_l1659_165996


namespace no_real_roots_quadratic_l1659_165901

theorem no_real_roots_quadratic : 
  {x : ℝ | x^2 - x + 1 = 0} = ∅ := by sorry

end no_real_roots_quadratic_l1659_165901


namespace sector_area_l1659_165921

/-- Given a sector with arc length and radius both equal to 2, its area is 2. -/
theorem sector_area (arc_length : ℝ) (radius : ℝ) (h1 : arc_length = 2) (h2 : radius = 2) :
  (1 / 2) * arc_length * radius = 2 := by
  sorry

end sector_area_l1659_165921


namespace range_m_when_not_p_false_range_m_when_p_or_q_true_and_p_and_q_false_l1659_165907

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, x^2 - m ≤ 0

def q (m : ℝ) : Prop := ∃ a b : ℝ, a > b ∧ b > 0 ∧
  ∀ x y : ℝ, x^2 / m^2 + y^2 / 4 = 1 ↔ (x/a)^2 + (y/b)^2 = 1

-- Theorem 1
theorem range_m_when_not_p_false (m : ℝ) :
  ¬(¬(p m)) → m ≥ 1 := by sorry

-- Theorem 2
theorem range_m_when_p_or_q_true_and_p_and_q_false (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) →
  m ∈ Set.Ioi (-2) ∪ Set.Icc 1 2 := by sorry

end range_m_when_not_p_false_range_m_when_p_or_q_true_and_p_and_q_false_l1659_165907


namespace square_of_101_l1659_165933

theorem square_of_101 : 101 * 101 = 10201 := by
  sorry

end square_of_101_l1659_165933


namespace geometric_sequence_property_l1659_165991

theorem geometric_sequence_property (a : ℕ → ℝ) (h_geom : ∀ n m : ℕ, a (n + m) = a n * a m) :
  a 6 = 6 → a 9 = 9 → a 3 = 4 := by
  sorry

end geometric_sequence_property_l1659_165991


namespace arithmetic_sequence_general_term_l1659_165937

/-- An arithmetic sequence with given second and fifth terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m
  second_term : a 2 = 2
  fifth_term : a 5 = 5

/-- The general term of the arithmetic sequence is n -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.a n = n := by
  sorry

end arithmetic_sequence_general_term_l1659_165937


namespace exam_percentage_l1659_165995

theorem exam_percentage (total_items : ℕ) (correct_A correct_B incorrect_B : ℕ) :
  total_items = 60 →
  correct_B = correct_A + 2 →
  incorrect_B = 4 →
  correct_B + incorrect_B = total_items →
  (correct_A : ℚ) / total_items * 100 = 90 := by
  sorry

end exam_percentage_l1659_165995


namespace brian_commission_l1659_165922

/-- Calculates the commission for a given sale price and commission rate -/
def calculate_commission (sale_price : ℝ) (commission_rate : ℝ) : ℝ :=
  sale_price * commission_rate

/-- Calculates the total commission for multiple sales -/
def total_commission (sale_prices : List ℝ) (commission_rate : ℝ) : ℝ :=
  sale_prices.map (λ price => calculate_commission price commission_rate) |>.sum

theorem brian_commission :
  let commission_rate : ℝ := 0.02
  let sale_prices : List ℝ := [157000, 499000, 125000]
  total_commission sale_prices commission_rate = 15620 := by
  sorry

end brian_commission_l1659_165922


namespace room_length_proof_l1659_165984

/-- Given a rectangular room with known width, total paving cost, and paving rate per square meter,
    prove that the length of the room is 5.5 meters. -/
theorem room_length_proof (width : ℝ) (total_cost : ℝ) (paving_rate : ℝ) :
  width = 3.75 ∧ total_cost = 20625 ∧ paving_rate = 1000 →
  total_cost / paving_rate / width = 5.5 :=
by sorry

end room_length_proof_l1659_165984


namespace temperature_conversion_l1659_165918

theorem temperature_conversion (C F : ℝ) : 
  C = (4/7) * (F - 40) → C = 28 → F = 89 := by
  sorry

end temperature_conversion_l1659_165918


namespace closest_to_standard_weight_l1659_165911

def quality_errors : List ℝ := [-0.02, 0.1, -0.23, -0.3, 0.2]

theorem closest_to_standard_weight :
  ∀ x ∈ quality_errors, |(-0.02)| ≤ |x| :=
by sorry

end closest_to_standard_weight_l1659_165911


namespace monotonic_increasing_condition_l1659_165948

open Real

theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 (π / 2), StrictMono (fun x => (sin x + a) / cos x)) →
  a ≥ -1 := by
  sorry

end monotonic_increasing_condition_l1659_165948


namespace vector_properties_l1659_165987

def a : ℝ × ℝ := (2, 0)
def b : ℝ × ℝ := (-1, -1)

theorem vector_properties :
  (a.1 * b.1 + a.2 * b.2 ≠ 2) ∧
  (∃ (k : ℝ), a ≠ k • b) ∧
  (b.1 * (a.1 + b.1) + b.2 * (a.2 + b.2) = 0) ∧
  (a.1^2 + a.2^2 ≠ b.1^2 + b.2^2) :=
by sorry

end vector_properties_l1659_165987


namespace cards_distribution_l1659_165955

/-- 
Given 52 cards dealt to 8 people as evenly as possible, 
this theorem proves that 4 people will have fewer than 7 cards.
-/
theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 52)
  (h2 : num_people = 8) :
  (num_people - (total_cards % num_people)) = 4 := by
  sorry

end cards_distribution_l1659_165955


namespace min_value_exponential_sum_l1659_165990

theorem min_value_exponential_sum (x y : ℝ) (h : x + y = 5) :
  3^x + 3^y ≥ 18 * Real.sqrt 3 := by
  sorry

end min_value_exponential_sum_l1659_165990


namespace museum_wings_l1659_165965

/-- Represents a museum with paintings and artifacts -/
structure Museum where
  painting_wings : ℕ
  artifact_wings : ℕ
  large_paintings : ℕ
  small_paintings : ℕ
  artifacts_per_wing : ℕ

/-- Calculates the total number of paintings in the museum -/
def total_paintings (m : Museum) : ℕ :=
  m.large_paintings + m.small_paintings

/-- Calculates the total number of artifacts in the museum -/
def total_artifacts (m : Museum) : ℕ :=
  m.artifact_wings * m.artifacts_per_wing

/-- Theorem stating the total number of wings in the museum -/
theorem museum_wings (m : Museum) 
  (h1 : m.painting_wings = 3)
  (h2 : m.large_paintings = 1)
  (h3 : m.small_paintings = 24)
  (h4 : m.artifacts_per_wing = 20)
  (h5 : total_artifacts m = 4 * total_paintings m) :
  m.painting_wings + m.artifact_wings = 8 := by
  sorry

#check museum_wings

end museum_wings_l1659_165965


namespace max_value_sqrt_sum_l1659_165904

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (x_nonneg : x ≥ 0)
  (y_geq : y ≥ -3/2)
  (z_geq : z ≥ -1) :
  ∃ (max : ℝ), max = 3 * Real.sqrt 2 ∧ 
    ∀ a b c : ℝ, a + b + c = 3 → a ≥ 0 → b ≥ -3/2 → c ≥ -1 →
      Real.sqrt (2 * a) + Real.sqrt (2 * b + 3) + Real.sqrt (2 * c + 2) ≤ max :=
by sorry

end max_value_sqrt_sum_l1659_165904


namespace panda_survival_probability_l1659_165958

theorem panda_survival_probability (p_10 p_15 : ℝ) 
  (h1 : p_10 = 0.8) 
  (h2 : p_15 = 0.6) : 
  p_15 / p_10 = 0.75 := by
  sorry

end panda_survival_probability_l1659_165958


namespace x_minus_y_equals_three_l1659_165994

theorem x_minus_y_equals_three (x y : ℝ) (h : |x - 2| + (y + 1)^2 = 0) : x - y = 3 := by
  sorry

end x_minus_y_equals_three_l1659_165994


namespace product_repeating_decimal_and_seven_l1659_165926

theorem product_repeating_decimal_and_seven (x : ℚ) : 
  (x = 1/3) → (x * 7 = 7/3) := by
  sorry

end product_repeating_decimal_and_seven_l1659_165926


namespace degree_of_polynomial_l1659_165979

/-- The degree of (5x^3 + 7)^10 is 30 -/
theorem degree_of_polynomial (x : ℝ) : 
  Polynomial.degree ((5 * X + 7 : Polynomial ℝ) ^ 10) = 30 := by
  sorry

end degree_of_polynomial_l1659_165979


namespace fraction_equality_l1659_165939

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) : (2 * a) / (2 * b) = a / b := by
  sorry

end fraction_equality_l1659_165939


namespace trigonometric_expressions_l1659_165900

theorem trigonometric_expressions (θ : ℝ) 
  (h : (4 * Real.sin θ - 2 * Real.cos θ) / (3 * Real.sin θ + 5 * Real.cos θ) = 6 / 11) : 
  (5 * (Real.cos θ)^2) / ((Real.sin θ)^2 + 2 * Real.sin θ * Real.cos θ - 3 * (Real.cos θ)^2) = 1 ∧ 
  1 - 4 * Real.sin θ * Real.cos θ + 2 * (Real.cos θ)^2 = -1/5 := by
  sorry

end trigonometric_expressions_l1659_165900


namespace exists_non_negative_sums_l1659_165905

/-- Represents a sign change operation on a matrix -/
inductive SignChange
| Row (i : Nat)
| Col (j : Nat)

/-- Apply a sequence of sign changes to a matrix -/
def applySignChanges (A : Matrix (Fin m) (Fin n) ℝ) (changes : List SignChange) : Matrix (Fin m) (Fin n) ℝ :=
  sorry

/-- Check if all row and column sums are non-negative -/
def allSumsNonNegative (A : Matrix (Fin m) (Fin n) ℝ) : Prop :=
  sorry

/-- Main theorem: For any real matrix, there exists a sequence of sign changes
    that results in all row and column sums being non-negative -/
theorem exists_non_negative_sums (A : Matrix (Fin m) (Fin n) ℝ) :
  ∃ (changes : List SignChange), allSumsNonNegative (applySignChanges A changes) :=
  sorry

end exists_non_negative_sums_l1659_165905


namespace b_value_l1659_165902

theorem b_value (a b : ℚ) : 
  (let x := 2 + Real.sqrt 3
   x^3 + a*x^2 + b*x - 20 = 0) →
  b = 81 := by
sorry

end b_value_l1659_165902


namespace spool_length_problem_l1659_165910

/-- Calculates the length of each spool of wire -/
def spool_length (total_spools : ℕ) (wire_per_necklace : ℕ) (total_necklaces : ℕ) : ℕ :=
  (wire_per_necklace * total_necklaces) / total_spools

theorem spool_length_problem :
  let total_spools : ℕ := 3
  let wire_per_necklace : ℕ := 4
  let total_necklaces : ℕ := 15
  spool_length total_spools wire_per_necklace total_necklaces = 20 := by
  sorry

end spool_length_problem_l1659_165910


namespace planted_fraction_for_specific_field_l1659_165935

/-- Represents a right triangle field with an unplanted square at the right angle -/
structure TriangleField where
  /-- Length of the first leg of the triangle -/
  leg1 : ℝ
  /-- Length of the second leg of the triangle -/
  leg2 : ℝ
  /-- Side length of the unplanted square -/
  square_side : ℝ
  /-- Shortest distance from the square to the hypotenuse -/
  distance_to_hypotenuse : ℝ

/-- The fraction of the field that is planted -/
def planted_fraction (field : TriangleField) : ℝ :=
  sorry

/-- Theorem stating the planted fraction for the specific field described in the problem -/
theorem planted_fraction_for_specific_field :
  let field : TriangleField := {
    leg1 := 5,
    leg2 := 12,
    square_side := 60 / 49,
    distance_to_hypotenuse := 3
  }
  planted_fraction field = 11405 / 12005 := by
  sorry

end planted_fraction_for_specific_field_l1659_165935


namespace total_purchase_options_l1659_165929

/-- The number of oreo flavors --/
def num_oreo_flavors : ℕ := 6

/-- The number of milk flavors --/
def num_milk_flavors : ℕ := 4

/-- The total number of product types --/
def total_product_types : ℕ := num_oreo_flavors + num_milk_flavors

/-- The total number of products they purchase --/
def total_purchases : ℕ := 4

/-- Charlie's purchase options --/
def charlie_options (k : ℕ) : ℕ := Nat.choose total_product_types k

/-- Delta's oreo purchase options when buying k oreos --/
def delta_options (k : ℕ) : ℕ :=
  Nat.choose num_oreo_flavors k +
  if k ≥ 2 then num_oreo_flavors * Nat.choose (num_oreo_flavors - 1) (k - 2) else 0 +
  if k = 3 then num_oreo_flavors else 0

/-- The main theorem stating the total number of ways to purchase --/
theorem total_purchase_options : 
  (charlie_options 3 * num_oreo_flavors) +
  (charlie_options 2 * delta_options 2) +
  (charlie_options 1 * delta_options 3) = 2225 := by
  sorry

end total_purchase_options_l1659_165929


namespace unique_solution_l1659_165969

theorem unique_solution (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (sum_squares_eq : x^2 + y^2 + z^2 = 3)
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end unique_solution_l1659_165969


namespace dave_book_cost_l1659_165997

/-- Calculates the total cost of Dave's books including discounts and taxes -/
def total_cost (animal_books_count : ℕ) (animal_book_price : ℚ)
                (space_books_count : ℕ) (space_book_price : ℚ)
                (train_books_count : ℕ) (train_book_price : ℚ)
                (history_books_count : ℕ) (history_book_price : ℚ)
                (science_books_count : ℕ) (science_book_price : ℚ)
                (animal_discount_rate : ℚ) (science_tax_rate : ℚ) : ℚ :=
  let animal_cost := animal_books_count * animal_book_price * (1 - animal_discount_rate)
  let space_cost := space_books_count * space_book_price
  let train_cost := train_books_count * train_book_price
  let history_cost := history_books_count * history_book_price
  let science_cost := science_books_count * science_book_price * (1 + science_tax_rate)
  animal_cost + space_cost + train_cost + history_cost + science_cost

/-- Theorem stating that the total cost of Dave's books is $379.5 -/
theorem dave_book_cost :
  total_cost 8 10 6 12 9 8 4 15 5 18 (1/10) (15/100) = 379.5 := by
  sorry

end dave_book_cost_l1659_165997


namespace polynomial_division_l1659_165909

-- Define the theorem
theorem polynomial_division (a b : ℝ) (h : b ≠ 2*a) : 
  (4*a^2 - b^2) / (b - 2*a) = -2*a - b :=
by sorry

end polynomial_division_l1659_165909
