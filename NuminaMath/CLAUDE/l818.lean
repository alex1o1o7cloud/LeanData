import Mathlib

namespace min_value_of_f_l818_81803

/-- The function f(x, y) as defined in the problem -/
def f (x y : ℝ) : ℝ := 6 * (x^2 + y^2) * (x + y) - 4 * (x^2 + x*y + y^2) - 3 * (x + y) + 5

/-- The theorem statement -/
theorem min_value_of_f :
  (∀ x y : ℝ, x > 0 → y > 0 → f x y ≥ 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ f x y = 2) :=
sorry

end min_value_of_f_l818_81803


namespace sqrt_two_minus_one_power_l818_81802

theorem sqrt_two_minus_one_power (n : ℤ) :
  ∃ k : ℤ, (Real.sqrt 2 - 1) ^ n = Real.sqrt (k + 1) - Real.sqrt k := by
  sorry

end sqrt_two_minus_one_power_l818_81802


namespace pens_to_classmates_l818_81844

/-- Represents the problem of calculating the fraction of remaining pens given to classmates. -/
theorem pens_to_classmates 
  (boxes : ℕ) 
  (pens_per_box : ℕ) 
  (friend_percentage : ℚ) 
  (pens_left : ℕ) 
  (h1 : boxes = 20) 
  (h2 : pens_per_box = 5) 
  (h3 : friend_percentage = 2/5) 
  (h4 : pens_left = 45) : 
  (boxes * pens_per_box - pens_left - (friend_percentage * (boxes * pens_per_box))) / 
  ((1 - friend_percentage) * (boxes * pens_per_box)) = 1/4 := by
sorry

end pens_to_classmates_l818_81844


namespace initial_maple_trees_l818_81807

theorem initial_maple_trees (cut_trees : ℝ) (remaining_trees : ℕ) 
  (h1 : cut_trees = 2.0)
  (h2 : remaining_trees = 7) :
  cut_trees + remaining_trees = 9.0 := by
  sorry

end initial_maple_trees_l818_81807


namespace imaginary_part_of_complex_fraction_l818_81833

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i ^ 2 = -1 →
  Complex.im (i / (2 + i)) = 2 / 5 := by
  sorry

end imaginary_part_of_complex_fraction_l818_81833


namespace simplify_expression_l818_81881

theorem simplify_expression (x y : ℝ) : 3*y + 5*y + 6*y + 2*x + 4*x = 14*y + 6*x := by
  sorry

end simplify_expression_l818_81881


namespace length_of_BC_l818_81805

-- Define the parabola
def parabola (x : ℝ) : ℝ := 4 * x^2

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.A = (2, 16) ∧
  parabola 2 = 16 ∧
  t.B.1 = -t.C.1 ∧
  t.B.2 = t.C.2 ∧
  (1/2 : ℝ) * |t.B.1 - t.C.1| * |t.A.2 - t.B.2| = 128

-- Theorem statement
theorem length_of_BC (t : Triangle) (h : satisfies_conditions t) : 
  |t.B.1 - t.C.1| = 8 := by sorry

end length_of_BC_l818_81805


namespace terminal_side_point_theorem_l818_81836

theorem terminal_side_point_theorem (m : ℝ) (hm : m ≠ 0) :
  let α := Real.arctan (3 * m / (-4 * m))
  (2 * Real.sin α + Real.cos α = 2/5) ∨ (2 * Real.sin α + Real.cos α = -2/5) := by
  sorry

end terminal_side_point_theorem_l818_81836


namespace value_of_a_l818_81890

theorem value_of_a (a b c : ℝ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 6) 
  (eq3 : c = 4) : 
  a = 2 := by sorry

end value_of_a_l818_81890


namespace normal_distribution_probability_l818_81886

/-- A random variable following a normal distribution with mean 1 and standard deviation σ > 0 -/
def normal_rv (σ : ℝ) : Type := ℝ

/-- The probability density function of the normal distribution -/
noncomputable def pdf (σ : ℝ) (x : ℝ) : ℝ := sorry

/-- The probability that the random variable takes a value in the interval (a, b) -/
noncomputable def prob (σ : ℝ) (a b : ℝ) : ℝ := sorry

/-- Theorem: If P(0 < ξ < 1) = 0.4, then P(0 < ξ < 2) = 0.8 for a normal distribution with mean 1 -/
theorem normal_distribution_probability (σ : ℝ) (hσ : σ > 0) :
  prob σ 0 1 = 0.4 → prob σ 0 2 = 0.8 := by sorry

end normal_distribution_probability_l818_81886


namespace total_space_after_compaction_l818_81832

/-- Represents the types of cans -/
inductive CanType
  | Small
  | Large

/-- Represents the properties of a can type -/
structure CanProperties where
  originalSize : ℕ
  compactionRate : ℚ

/-- Calculates the space taken by a type of can after compaction -/
def spaceAfterCompaction (props : CanProperties) (quantity : ℕ) : ℚ :=
  ↑(props.originalSize * quantity) * props.compactionRate

theorem total_space_after_compaction :
  let smallCanProps : CanProperties := ⟨20, 3/10⟩
  let largeCanProps : CanProperties := ⟨40, 4/10⟩
  let smallCanQuantity : ℕ := 50
  let largeCanQuantity : ℕ := 50
  let totalSpaceAfterCompaction :=
    spaceAfterCompaction smallCanProps smallCanQuantity +
    spaceAfterCompaction largeCanProps largeCanQuantity
  totalSpaceAfterCompaction = 1100 := by
  sorry


end total_space_after_compaction_l818_81832


namespace harmonic_mean_three_fourths_five_sixths_l818_81894

def harmonic_mean (a b : ℚ) : ℚ := 2 / (1/a + 1/b)

theorem harmonic_mean_three_fourths_five_sixths :
  harmonic_mean (3/4) (5/6) = 15/19 := by
  sorry

end harmonic_mean_three_fourths_five_sixths_l818_81894


namespace ted_losses_l818_81883

/-- Represents a player in the game --/
inductive Player
| Carl
| James
| Saif
| Ted

/-- Records the number of wins and losses for a player --/
structure PlayerRecord where
  wins : Nat
  losses : Nat

/-- Represents the game results for all players --/
def GameResults := Player → PlayerRecord

theorem ted_losses (results : GameResults) :
  (results Player.Carl).wins = 5 ∧
  (results Player.Carl).losses = 0 ∧
  (results Player.James).wins = 4 ∧
  (results Player.James).losses = 2 ∧
  (results Player.Saif).wins = 1 ∧
  (results Player.Saif).losses = 6 ∧
  (results Player.Ted).wins = 4 ∧
  (∀ p : Player, (results p).wins + (results p).losses = 
    (results Player.Carl).wins + (results Player.James).wins + 
    (results Player.Saif).wins + (results Player.Ted).wins) →
  (results Player.Ted).losses = 6 := by
  sorry

end ted_losses_l818_81883


namespace right_triangle_from_special_case_l818_81866

/-- 
Given a triangle with sides a, 2a, and c, where the angle between sides a and 2a is 60°,
prove that the angle opposite side 2a is 90°.
-/
theorem right_triangle_from_special_case (a : ℝ) (h : a > 0) :
  let c := a * Real.sqrt 3
  let cos_alpha := (a^2 + c^2 - (2*a)^2) / (2 * a * c)
  cos_alpha = 0 := by sorry

end right_triangle_from_special_case_l818_81866


namespace starting_lineup_combinations_l818_81821

def team_size : ℕ := 12
def center_players : ℕ := 2
def lineup_size : ℕ := 4

theorem starting_lineup_combinations :
  (center_players) * (team_size - 1) * (team_size - 2) * (team_size - 3) = 1980 :=
by sorry

end starting_lineup_combinations_l818_81821


namespace bolzano_weierstrass_l818_81827

-- Define a bounded sequence
def BoundedSequence (a : ℕ → ℝ) : Prop :=
  ∃ (M : ℝ), ∀ (n : ℕ), |a n| ≤ M

-- Define a limit point
def LimitPoint (a : ℕ → ℝ) (x : ℝ) : Prop :=
  ∀ (ε : ℝ), ε > 0 → ∀ (N : ℕ), ∃ (n : ℕ), n ≥ N ∧ |a n - x| < ε

-- Bolzano-Weierstrass theorem
theorem bolzano_weierstrass (a : ℕ → ℝ) :
  BoundedSequence a → ∃ (x : ℝ), LimitPoint a x :=
sorry

end bolzano_weierstrass_l818_81827


namespace square_of_binomial_l818_81810

theorem square_of_binomial (a : ℚ) :
  (∃ b : ℚ, ∀ x : ℚ, 9 * x^2 + 15 * x + a = (3 * x + b)^2) → a = 25 / 4 := by
  sorry

end square_of_binomial_l818_81810


namespace smallest_multiple_of_9_l818_81893

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

theorem smallest_multiple_of_9 :
  ∀ d : ℕ, d < 10 →
    (is_multiple_of_9 456786 ∧
     is_six_digit 456786 ∧
     456786 = 45678 * 10 + 6) ∧
    (∀ n : ℕ, is_six_digit n ∧ n < 456786 ∧ ∃ d' : ℕ, d' < 10 ∧ n = 45678 * 10 + d' →
      ¬is_multiple_of_9 n) :=
by sorry

end smallest_multiple_of_9_l818_81893


namespace inequality_lower_bound_l818_81860

theorem inequality_lower_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 2*y) * (2/x + 1/y) ≥ 8 ∧
  ∀ ε > 0, ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ (x₀ + 2*y₀) * (2/x₀ + 1/y₀) < 8 + ε :=
sorry

end inequality_lower_bound_l818_81860


namespace sqrt_seven_to_sixth_l818_81856

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end sqrt_seven_to_sixth_l818_81856


namespace ellipse_equation_proof_l818_81892

theorem ellipse_equation_proof (a b : ℝ) : 
  (∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ∧ x = 2 ∧ y = 0) → -- ellipse passes through (2, 0)
  (a^2 - b^2 = 2) → -- ellipse shares focus with hyperbola x² - y² = 1
  (a^2 = 4 ∧ b^2 = 2) → -- derived from the conditions
  (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 / 2 = 1) :=
by sorry


end ellipse_equation_proof_l818_81892


namespace number_of_schnauzers_l818_81801

/-- Given the number of Doberman puppies and an equation relating it to the number of Schnauzers,
    this theorem proves the number of Schnauzers. -/
theorem number_of_schnauzers (D S : ℤ) (h1 : 3*D - 5 + (D - S) = 90) (h2 : D = 20) : S = 45 := by
  sorry

end number_of_schnauzers_l818_81801


namespace journey_distance_ratio_l818_81826

/-- Proves that the ratio of North distance to East distance is 2:1 given the problem conditions --/
theorem journey_distance_ratio :
  let south_distance : ℕ := 40
  let east_distance : ℕ := south_distance + 20
  let total_distance : ℕ := 220
  let north_distance : ℕ := total_distance - south_distance - east_distance
  (north_distance : ℚ) / east_distance = 2 := by
  sorry

end journey_distance_ratio_l818_81826


namespace floor_sqrt_27_squared_l818_81897

theorem floor_sqrt_27_squared : ⌊Real.sqrt 27⌋^2 = 25 := by
  sorry

end floor_sqrt_27_squared_l818_81897


namespace scientific_notation_393000_l818_81806

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_393000 :
  toScientificNotation 393000 = ScientificNotation.mk 3.93 5 (by sorry) :=
sorry

end scientific_notation_393000_l818_81806


namespace pyramid_edges_l818_81812

/-- Represents a pyramid with a polygonal base -/
structure Pyramid where
  base_sides : ℕ

/-- The number of vertices in a pyramid -/
def num_vertices (p : Pyramid) : ℕ := p.base_sides + 1

/-- The number of faces in a pyramid -/
def num_faces (p : Pyramid) : ℕ := p.base_sides + 1

/-- The number of edges in a pyramid -/
def num_edges (p : Pyramid) : ℕ := p.base_sides + p.base_sides

theorem pyramid_edges (p : Pyramid) :
  num_vertices p + num_faces p = 16 → num_edges p = 14 := by
  sorry

end pyramid_edges_l818_81812


namespace age_when_dog_born_is_15_l818_81889

/-- The age of the person when their dog was born -/
def age_when_dog_born (current_age : ℕ) (dog_future_age : ℕ) (years_until_future : ℕ) : ℕ :=
  current_age - (dog_future_age - years_until_future)

/-- Theorem stating the age when the dog was born -/
theorem age_when_dog_born_is_15 :
  age_when_dog_born 17 4 2 = 15 := by
  sorry

end age_when_dog_born_is_15_l818_81889


namespace special_rectangle_perimeter_l818_81850

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  small_perimeter : ℝ
  width_half_length : width = length / 2
  divides_into_three : length = 3 * width
  small_rect_perimeter : small_perimeter = 2 * (width + length / 3)
  small_perimeter_value : small_perimeter = 40

/-- The perimeter of a SpecialRectangle is 72 -/
theorem special_rectangle_perimeter (rect : SpecialRectangle) : 
  2 * (rect.length + rect.width) = 72 := by
  sorry

end special_rectangle_perimeter_l818_81850


namespace absolute_value_equality_implies_product_zero_l818_81864

theorem absolute_value_equality_implies_product_zero (x y : ℝ) :
  |x - Real.log y| = x + Real.log y → x * (y - 1) = 0 := by
  sorry

end absolute_value_equality_implies_product_zero_l818_81864


namespace simplified_expression_equals_result_l818_81841

theorem simplified_expression_equals_result (a b : ℝ) 
  (ha : a = 4) (hb : b = 3) : 
  (a * Real.sqrt (1 / a) + Real.sqrt (4 * b)) - (Real.sqrt a / 2 - b * Real.sqrt (1 / b)) = 1 + 3 * Real.sqrt 3 := by
  sorry

end simplified_expression_equals_result_l818_81841


namespace point_translation_l818_81847

def initial_point : ℝ × ℝ := (-2, 3)

def translate_down (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 - units)

def translate_right (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1 + units, p.2)

theorem point_translation :
  (translate_right (translate_down initial_point 3) 1) = (-1, 0) := by
  sorry

end point_translation_l818_81847


namespace terms_are_like_when_k_is_two_l818_81820

/-- Two monomials are like terms if they have the same variables raised to the same powers -/
def like_terms (term1 term2 : ℕ → ℕ) : Prop :=
  ∀ var, term1 var = term2 var

/-- The first term: -3x²y³ᵏ -/
def term1 (k : ℕ) : ℕ → ℕ
| 0 => 2  -- x has power 2
| 1 => 3 * k  -- y has power 3k
| _ => 0  -- other variables have power 0

/-- The second term: 4x²y⁶ -/
def term2 : ℕ → ℕ
| 0 => 2  -- x has power 2
| 1 => 6  -- y has power 6
| _ => 0  -- other variables have power 0

/-- Theorem: When k = 2, -3x²y³ᵏ and 4x²y⁶ are like terms -/
theorem terms_are_like_when_k_is_two : like_terms (term1 2) term2 := by
  sorry

end terms_are_like_when_k_is_two_l818_81820


namespace car_speed_problem_l818_81823

/-- Proves that given the conditions of the car problem, the speed of Car B is 50 km/h -/
theorem car_speed_problem (speed_b : ℝ) : 
  let speed_a := 3 * speed_b
  let time_a := 6
  let time_b := 2
  let total_distance := 1000
  speed_a * time_a + speed_b * time_b = total_distance →
  speed_b = 50 := by
sorry

end car_speed_problem_l818_81823


namespace actual_car_body_mass_l818_81868

/-- Represents the scale factor between the model and the actual car body -/
def scaleFactor : ℝ := 10

/-- Represents the mass of the model car body in kilograms -/
def modelMass : ℝ := 1

/-- Calculates the volume ratio between the actual car body and the model -/
def volumeRatio : ℝ := scaleFactor ^ 3

/-- Calculates the mass of the actual car body in kilograms -/
def actualMass : ℝ := modelMass * volumeRatio

/-- Theorem stating that the mass of the actual car body is 1000 kg -/
theorem actual_car_body_mass : actualMass = 1000 := by
  sorry

end actual_car_body_mass_l818_81868


namespace expected_pairs_in_both_arrangements_l818_81878

/-- Represents a 7x7 grid arrangement of numbers 1 through 49 -/
def Arrangement := Fin 49 → Fin 7 × Fin 7

/-- The number of rows in the grid -/
def num_rows : Nat := 7

/-- The number of columns in the grid -/
def num_cols : Nat := 7

/-- The total number of numbers in the grid -/
def total_numbers : Nat := num_rows * num_cols

/-- Calculates the expected number of pairs that occur in the same row or column in both arrangements -/
noncomputable def expected_pairs (a1 a2 : Arrangement) : ℝ :=
  (total_numbers.choose 2 : ℝ) * (1 / 16)

/-- The main theorem stating the expected number of pairs -/
theorem expected_pairs_in_both_arrangements :
  ∀ a1 a2 : Arrangement, expected_pairs a1 a2 = 73.5 := by
  sorry

end expected_pairs_in_both_arrangements_l818_81878


namespace adult_ticket_cost_l818_81871

theorem adult_ticket_cost (child_cost : ℝ) : 
  (child_cost + 6 = 19) ∧ 
  (2 * (child_cost + 6) + 3 * child_cost = 77) := by
  sorry

end adult_ticket_cost_l818_81871


namespace sixteen_team_tournament_games_l818_81848

/-- Calculates the number of games in a single-elimination tournament. -/
def num_games_in_tournament (num_teams : ℕ) : ℕ :=
  num_teams - 1

/-- Theorem: In a single-elimination tournament with 16 teams, 15 games are played to determine the winner. -/
theorem sixteen_team_tournament_games :
  num_games_in_tournament 16 = 15 := by
  sorry

#eval num_games_in_tournament 16  -- Should output 15

end sixteen_team_tournament_games_l818_81848


namespace coefficient_x_squared_in_expansion_l818_81853

theorem coefficient_x_squared_in_expansion : 
  let expansion := (fun x : ℝ => (x - x⁻¹)^6)
  ∃ (a b c : ℝ), ∀ x : ℝ, x ≠ 0 → 
    expansion x = a*x^3 + 15*x^2 + b*x + c + (x⁻¹ * (1 + x⁻¹ * (1 + x⁻¹ * (1)))) := by
  sorry

end coefficient_x_squared_in_expansion_l818_81853


namespace grandma_olga_grandchildren_l818_81825

/-- Calculates the total number of grandchildren for a grandmother with the given family structure -/
def total_grandchildren (num_daughters num_sons sons_per_daughter daughters_per_son : ℕ) : ℕ :=
  (num_daughters * sons_per_daughter) + (num_sons * daughters_per_son)

/-- Theorem: Grandma Olga's total number of grandchildren is 33 -/
theorem grandma_olga_grandchildren :
  total_grandchildren 3 3 6 5 = 33 := by
  sorry

#eval total_grandchildren 3 3 6 5

end grandma_olga_grandchildren_l818_81825


namespace greatest_multiple_5_and_6_less_than_800_l818_81813

theorem greatest_multiple_5_and_6_less_than_800 : 
  ∃ n : ℕ, n = 780 ∧ 
  (∀ m : ℕ, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 800 → m ≤ n) ∧
  n % 5 = 0 ∧ n % 6 = 0 ∧ n < 800 :=
sorry

end greatest_multiple_5_and_6_less_than_800_l818_81813


namespace spencer_walk_distance_l818_81887

/-- Represents the distances walked by Spencer -/
structure WalkDistances where
  total : ℝ
  libraryToPostOffice : ℝ
  postOfficeToHome : ℝ

/-- Calculates the distance from house to library -/
def distanceHouseToLibrary (w : WalkDistances) : ℝ :=
  w.total - w.libraryToPostOffice - w.postOfficeToHome

/-- Theorem stating that the distance from house to library is 0.3 miles -/
theorem spencer_walk_distance (w : WalkDistances) 
  (h_total : w.total = 0.8)
  (h_lib_post : w.libraryToPostOffice = 0.1)
  (h_post_home : w.postOfficeToHome = 0.4) : 
  distanceHouseToLibrary w = 0.3 := by
  sorry

end spencer_walk_distance_l818_81887


namespace geometric_sequence_product_l818_81862

theorem geometric_sequence_product (a b c : ℝ) : 
  (8/3 < a) ∧ (a < b) ∧ (b < c) ∧ (c < 27/2) ∧ 
  (∃ q : ℝ, q ≠ 0 ∧ a = 8/3 * q ∧ b = 8/3 * q^2 ∧ c = 8/3 * q^3 ∧ 27/2 = 8/3 * q^4) →
  a * b * c = 216 := by
sorry


end geometric_sequence_product_l818_81862


namespace uncoolParentsOnlyChildCount_l818_81840

/-- Represents a class of students -/
structure PhysicsClass where
  total : ℕ
  coolDads : ℕ
  coolMoms : ℕ
  coolBothAndSiblings : ℕ

/-- Calculates the number of students with uncool parents and no siblings -/
def uncoolParentsOnlyChild (c : PhysicsClass) : ℕ :=
  c.total - (c.coolDads + c.coolMoms - c.coolBothAndSiblings)

/-- The theorem to be proved -/
theorem uncoolParentsOnlyChildCount (c : PhysicsClass) 
  (h1 : c.total = 40)
  (h2 : c.coolDads = 20)
  (h3 : c.coolMoms = 22)
  (h4 : c.coolBothAndSiblings = 10) :
  uncoolParentsOnlyChild c = 8 := by
  sorry

#eval uncoolParentsOnlyChild { total := 40, coolDads := 20, coolMoms := 22, coolBothAndSiblings := 10 }

end uncoolParentsOnlyChildCount_l818_81840


namespace find_number_l818_81843

theorem find_number (N : ℚ) : 
  (N / (4/5) = (4/5) * N + 36) → N = 80 := by
  sorry

end find_number_l818_81843


namespace matrix_not_invertible_iff_y_eq_one_seventh_l818_81830

theorem matrix_not_invertible_iff_y_eq_one_seventh :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2 + y, 5; 4 - y, 9]
  ¬(IsUnit (Matrix.det A)) ↔ y = (1 : ℝ) / 7 :=
by sorry

end matrix_not_invertible_iff_y_eq_one_seventh_l818_81830


namespace order_of_roots_l818_81863

theorem order_of_roots (a b c : ℝ) (ha : a = 2^(4/3)) (hb : b = 3^(2/3)) (hc : c = 25^(1/3)) :
  b < a ∧ a < c := by
  sorry

end order_of_roots_l818_81863


namespace triangle_area_from_squares_l818_81824

theorem triangle_area_from_squares (a b c : ℝ) 
  (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100)
  (right_triangle : a^2 + b^2 = c^2) : 
  (1/2) * a * b = 24 := by
sorry

end triangle_area_from_squares_l818_81824


namespace parabola_equation_l818_81872

/-- A parabola with the origin as vertex, coordinate axes as axes of symmetry, 
    and passing through the point (6, 4) has the equation y² = 8/3 * x or x² = 9 * y -/
theorem parabola_equation : ∃ (f : ℝ → ℝ),
  (∀ x y : ℝ, f x = y ↔ (y^2 = 8/3 * x ∨ x^2 = 9 * y)) ∧
  f 0 = 0 ∧
  (∀ x : ℝ, f (-x) = f x) ∧
  f 6 = 4 := by
  sorry

end parabola_equation_l818_81872


namespace cone_radius_l818_81849

/-- Given a cone with angle π/3 between generatrix and base, and volume 3π, its base radius is √3 -/
theorem cone_radius (angle : Real) (volume : Real) (radius : Real) : 
  angle = π / 3 → volume = 3 * π → 
  (1 / 3) * π * radius^2 * (radius * Real.sqrt 3) = volume → 
  radius = Real.sqrt 3 := by
sorry

end cone_radius_l818_81849


namespace robot_purchase_strategy_l818_81808

/-- The problem of finding optimal robot purchase strategy -/
theorem robot_purchase_strategy 
  (price_difference : ℕ) 
  (cost_A cost_B : ℕ) 
  (total_units : ℕ) 
  (discount_rate : ℚ) : 
  price_difference = 200 →
  cost_A = 2000 →
  cost_B = 1200 →
  total_units = 40 →
  discount_rate = 1/5 →
  ∃ (price_A price_B units_A units_B min_cost : ℕ),
    -- Unit prices
    price_A = 500 ∧ 
    price_B = 300 ∧ 
    price_A = price_B + price_difference ∧
    cost_A * price_B = cost_B * price_A ∧
    -- Optimal purchase strategy
    units_A = 10 ∧
    units_B = 30 ∧
    units_A + units_B = total_units ∧
    units_B ≤ 3 * units_A ∧
    min_cost = 11200 ∧
    min_cost = (price_A * units_A + price_B * units_B) * (1 - discount_rate) ∧
    ∀ (other_A other_B : ℕ), 
      other_A + other_B = total_units →
      other_B ≤ 3 * other_A →
      min_cost ≤ (price_A * other_A + price_B * other_B) * (1 - discount_rate) :=
by sorry

end robot_purchase_strategy_l818_81808


namespace fraction_simplification_l818_81809

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hyz : y^3 - 1/x ≠ 0) : 
  (x^3 - 1/y) / (y^3 - 1/x) = x / y := by
  sorry

end fraction_simplification_l818_81809


namespace telephone_number_D_is_9_l818_81885

def TelephoneNumber (A B C D E F G H I J : ℕ) : Prop :=
  A > B ∧ B > C ∧ D > E ∧ E > F ∧ G > H ∧ H > I ∧ I > J ∧
  A % 2 = 0 ∧ B = A - 2 ∧ C = B - 2 ∧
  D % 2 = 1 ∧ E = D - 2 ∧ F = E - 2 ∧
  H + I + J = 9 ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ I ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ I ∧ G ≠ J ∧
  H ≠ I ∧ H ≠ J ∧
  I ≠ J

theorem telephone_number_D_is_9 :
  ∀ A B C D E F G H I J, TelephoneNumber A B C D E F G H I J → D = 9 :=
by sorry

end telephone_number_D_is_9_l818_81885


namespace veronica_cherry_pitting_time_l818_81815

/-- Represents the time needed to pit cherries for a cherry pie --/
def cherry_pitting_time (pounds_needed : ℕ) 
                        (cherries_per_pound : ℕ) 
                        (first_pound_rate : ℚ) 
                        (second_pound_rate : ℚ) 
                        (third_pound_rate : ℚ) 
                        (interruptions : ℕ) 
                        (interruption_duration : ℚ) : ℚ :=
  sorry

theorem veronica_cherry_pitting_time :
  cherry_pitting_time 3 80 (10/20) (8/20) (12/20) 2 15 = 5/2 := by
  sorry

end veronica_cherry_pitting_time_l818_81815


namespace compound_simple_interest_principal_l818_81831

theorem compound_simple_interest_principal (P r : ℝ) : 
  P * (1 + r)^2 - P = 11730 → P * r * 2 = 10200 → P = 17000 := by
  sorry

end compound_simple_interest_principal_l818_81831


namespace parallelogram_angle_measure_l818_81865

theorem parallelogram_angle_measure (a b : ℝ) : 
  a = 70 → b = a + 40 → b = 110 := by sorry

end parallelogram_angle_measure_l818_81865


namespace cos_pi_4_plus_alpha_l818_81879

theorem cos_pi_4_plus_alpha (α : Real) 
  (h : Real.sin (π / 4 - α) = Real.sqrt 2 / 2) : 
  Real.cos (π / 4 + α) = Real.sqrt 2 / 2 := by
  sorry

end cos_pi_4_plus_alpha_l818_81879


namespace g_composition_equals_1200_l818_81842

def g (x : ℝ) : ℝ := 7 * x + 3

theorem g_composition_equals_1200 : g (g (g 3)) = 1200 := by
  sorry

end g_composition_equals_1200_l818_81842


namespace max_product_of_prime_factors_l818_81829

def primes : List Nat := [2, 3, 5, 7, 11, 13, 17]

theorem max_product_of_prime_factors :
  ∃ (a b c d e f g : Nat),
    a ∈ primes ∧ b ∈ primes ∧ c ∈ primes ∧ d ∈ primes ∧
    e ∈ primes ∧ f ∈ primes ∧ g ∈ primes ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
    e ≠ f ∧ e ≠ g ∧
    f ≠ g ∧
    (a + b + c + d) * (e + f + g) = 841 ∧
    ∀ (x y z w u v t : Nat),
      x ∈ primes → y ∈ primes → z ∈ primes → w ∈ primes →
      u ∈ primes → v ∈ primes → t ∈ primes →
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ u ∧ x ≠ v ∧ x ≠ t ∧
      y ≠ z ∧ y ≠ w ∧ y ≠ u ∧ y ≠ v ∧ y ≠ t ∧
      z ≠ w ∧ z ≠ u ∧ z ≠ v ∧ z ≠ t ∧
      w ≠ u ∧ w ≠ v ∧ w ≠ t ∧
      u ≠ v ∧ u ≠ t ∧
      v ≠ t →
      (x + y + z + w) * (u + v + t) ≤ 841 :=
by sorry

end max_product_of_prime_factors_l818_81829


namespace extremum_implies_f_two_l818_81859

/-- A cubic function with integer coefficients -/
def f (a b : ℤ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

/-- Theorem stating that if f has an extremum of 10 at x = 1, then f(2) = 2 -/
theorem extremum_implies_f_two (a b : ℤ) :
  (f a b 1 = 10) →  -- f(1) = 10
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≤ f a b 1) →  -- local maximum at x = 1
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≥ f a b 1) →  -- local minimum at x = 1
  f a b 2 = 2 := by
  sorry

end extremum_implies_f_two_l818_81859


namespace solution1_satisfies_system1_solution2_satisfies_system2_l818_81846

-- Part 1
def system1 (y z : ℝ → ℝ) : Prop :=
  ∀ x, (deriv y x + 2 * y x - 4 * z x = 0) ∧
       (deriv z x + y x - 3 * z x = 3 * x^2)

def solution1 (C₁ C₂ : ℝ) (y z : ℝ → ℝ) : Prop :=
  ∀ x, y x = C₁ * Real.exp (-x) + C₂ * Real.exp (2*x) - 6*x^2 + 6*x - 9 ∧
       z x = (1/4) * C₁ * Real.exp (-x) + C₂ * Real.exp (2*x) - 3*x^2 - 3

theorem solution1_satisfies_system1 (C₁ C₂ : ℝ) :
  ∀ y z, solution1 C₁ C₂ y z → system1 y z := by sorry

-- Part 2
def system2 (u v w : ℝ → ℝ) : Prop :=
  ∀ x, (6 * deriv u x - u x - 7 * v x + 5 * w x = 10 * Real.exp x) ∧
       (2 * deriv v x + u x + v x - w x = 0) ∧
       (3 * deriv w x - u x + 2 * v x - w x = Real.exp x)

def solution2 (C₁ C₂ C₃ : ℝ) (u v w : ℝ → ℝ) : Prop :=
  ∀ x, u x = C₁ + C₂ * Real.cos x + C₃ * Real.sin x + Real.exp x ∧
       v x = 2*C₁ + (1/2)*(C₃ - C₂)*Real.cos x - (1/2)*(C₃ + C₂)*Real.sin x ∧
       w x = 3*C₁ - (1/2)*(C₂ + C₃)*Real.cos x + (1/2)*(C₂ - C₃)*Real.sin x + Real.exp x

theorem solution2_satisfies_system2 (C₁ C₂ C₃ : ℝ) :
  ∀ u v w, solution2 C₁ C₂ C₃ u v w → system2 u v w := by sorry

end solution1_satisfies_system1_solution2_satisfies_system2_l818_81846


namespace rational_quadratic_integer_solutions_l818_81857

theorem rational_quadratic_integer_solutions (r : ℚ) :
  (∃ x : ℤ, r * x^2 + (r + 1) * x + r = 1) ↔ (r = 1 ∨ r = -1/7) := by
  sorry

end rational_quadratic_integer_solutions_l818_81857


namespace largest_three_digit_divisible_by_digits_and_twelve_l818_81839

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def divisible_by_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ≠ 0 → d ≠ 10 → d < 10 → (n % 10 = d ∨ (n / 10) % 10 = d ∨ n / 100 = d) → n % d = 0

theorem largest_three_digit_divisible_by_digits_and_twelve :
  ∀ n : ℕ, is_three_digit n → divisible_by_digits n → n % 12 = 0 → n ≤ 864 :=
sorry

end largest_three_digit_divisible_by_digits_and_twelve_l818_81839


namespace tim_weekly_earnings_l818_81858

/-- Tim's daily tasks -/
def daily_tasks : ℕ := 100

/-- Pay per task in dollars -/
def pay_per_task : ℚ := 6/5

/-- Number of working days per week -/
def working_days_per_week : ℕ := 6

/-- Tim's weekly earnings in dollars -/
def weekly_earnings : ℚ := daily_tasks * pay_per_task * working_days_per_week

theorem tim_weekly_earnings : weekly_earnings = 720 := by sorry

end tim_weekly_earnings_l818_81858


namespace consecutive_integers_product_l818_81817

theorem consecutive_integers_product (x : ℕ) :
  x > 0 ∧ x * (x + 1) = 812 → (x + 1)^2 - x = 813 := by
  sorry

end consecutive_integers_product_l818_81817


namespace abc_system_solution_l818_81874

theorem abc_system_solution (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a * b = 3 * (a + b))
  (hbc : b * c = 4 * (b + c))
  (hac : a * c = 5 * (a + c)) :
  a = 120 / 17 ∧ b = 120 / 23 ∧ c = 120 / 7 :=
by sorry

end abc_system_solution_l818_81874


namespace meeting_time_prove_meeting_time_l818_81851

/-- The time it takes for a motorcyclist and a cyclist to meet under specific conditions -/
theorem meeting_time : ℝ → Prop := fun t =>
  ∀ (D vm vb : ℝ),
  D > 0 →  -- Total distance between A and B is positive
  vm > 0 →  -- Motorcyclist's speed is positive
  vb > 0 →  -- Cyclist's speed is positive
  (1/3) * vm = D/2 + 2 →  -- Motorcyclist's position after 20 minutes
  (1/2) * vb = D/2 - 3 →  -- Cyclist's position after 30 minutes
  t * (vm + vb) = D →  -- They meet when they cover the total distance
  t = 24/60  -- The meeting time is 24 minutes (converted to hours)

/-- Proof of the meeting time theorem -/
theorem prove_meeting_time : meeting_time (24/60) := by
  sorry

end meeting_time_prove_meeting_time_l818_81851


namespace remainder_calculation_l818_81822

-- Define the remainder function
def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

-- Theorem statement
theorem remainder_calculation :
  rem (5/9 : ℚ) (-3/4 : ℚ) = -7/36 := by
  sorry

end remainder_calculation_l818_81822


namespace parallel_chords_central_angles_l818_81834

/-- Given a circle with parallel chords of lengths 5, 12, and 13 determining
    central angles α, β, and α + β radians respectively, where α + β < π,
    prove that α + β = π/2 -/
theorem parallel_chords_central_angles
  (α β : Real)
  (h1 : 0 < α) (h2 : 0 < β)
  (h3 : α + β < π)
  (h4 : 2 * Real.sin (α / 2) = 5 / (2 * R))
  (h5 : 2 * Real.sin (β / 2) = 12 / (2 * R))
  (h6 : 2 * Real.sin ((α + β) / 2) = 13 / (2 * R))
  (R : Real) (h7 : R > 0) :
  α + β = π / 2 := by
sorry

end parallel_chords_central_angles_l818_81834


namespace paul_gave_35_books_l818_81818

/-- The number of books Paul gave to his friend -/
def books_given_to_friend (initial_books sold_books remaining_books : ℕ) : ℕ :=
  initial_books - sold_books - remaining_books

/-- Theorem stating that Paul gave 35 books to his friend -/
theorem paul_gave_35_books : books_given_to_friend 108 11 62 = 35 := by
  sorry

end paul_gave_35_books_l818_81818


namespace triangle_properties_l818_81870

/-- Given two 2D vectors a and b, proves statements about the triangle formed by 0, a, and b -/
theorem triangle_properties (a b : Fin 2 → ℝ) 
  (ha : a = ![4, -1]) 
  (hb : b = ![2, 6]) : 
  (1/2 * abs (a 0 * b 1 - a 1 * b 0) = 13) ∧ 
  ((((a 0 + b 0)/2)^2 + ((a 1 + b 1)/2)^2) = 15.25) := by
  sorry

end triangle_properties_l818_81870


namespace equation_solutions_l818_81888

theorem equation_solutions : 
  let solutions : List ℂ := [
    4 + Complex.I * Real.sqrt 6,
    4 - Complex.I * Real.sqrt 6,
    4 + Complex.I * Real.sqrt (21 + Real.sqrt 433),
    4 - Complex.I * Real.sqrt (21 + Real.sqrt 433),
    4 + Complex.I * Real.sqrt (21 - Real.sqrt 433),
    4 - Complex.I * Real.sqrt (21 - Real.sqrt 433)
  ]
  ∀ x ∈ solutions, (x - 2)^6 + (x - 6)^6 = 32 ∧
  ∀ x : ℂ, (x - 2)^6 + (x - 6)^6 = 32 → x ∈ solutions :=
by sorry

end equation_solutions_l818_81888


namespace harry_snails_collection_l818_81884

/-- Represents the number of sea stars Harry collected initially -/
def sea_stars : ℕ := 34

/-- Represents the number of seashells Harry collected initially -/
def seashells : ℕ := 21

/-- Represents the total number of items Harry had at the end of his walk -/
def total_items_left : ℕ := 59

/-- Represents the number of sea creatures Harry lost during his walk -/
def lost_sea_creatures : ℕ := 25

/-- Represents the number of snails Harry collected initially -/
def snails_collected : ℕ := total_items_left - (sea_stars + seashells - lost_sea_creatures)

theorem harry_snails_collection :
  snails_collected = 29 :=
sorry

end harry_snails_collection_l818_81884


namespace puppy_feeding_theorem_l818_81816

/-- Given the number of puppies, portions of formula, and days, 
    calculates the number of times each puppy should be fed per day. -/
def feeding_frequency (puppies : ℕ) (portions : ℕ) (days : ℕ) : ℕ :=
  (portions / days) / puppies

/-- Proves that for 7 puppies, 105 portions, and 5 days, 
    the feeding frequency is 3 times per day. -/
theorem puppy_feeding_theorem :
  feeding_frequency 7 105 5 = 3 := by
  sorry

end puppy_feeding_theorem_l818_81816


namespace amy_remaining_money_l818_81869

/-- Calculates the remaining money after a purchase -/
def remaining_money (initial_amount : ℕ) (num_items : ℕ) (item_cost : ℕ) : ℕ :=
  initial_amount - (num_items * item_cost)

/-- Proves that Amy has $97 left after her purchase -/
theorem amy_remaining_money :
  remaining_money 100 3 1 = 97 := by
  sorry

end amy_remaining_money_l818_81869


namespace series_sum_equals_four_implies_x_equals_half_l818_81804

/-- The sum of the infinite series 1 + 2x + 3x^2 + ... -/
noncomputable def S (x : ℝ) : ℝ := ∑' n, (n + 1) * x^n

/-- The theorem stating that if S(x) = 4, then x = 1/2 -/
theorem series_sum_equals_four_implies_x_equals_half :
  ∀ x : ℝ, x < 1 → S x = 4 → x = 1/2 := by sorry

end series_sum_equals_four_implies_x_equals_half_l818_81804


namespace train_passing_bridge_l818_81861

/-- Time taken for a train to pass a bridge -/
theorem train_passing_bridge
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 870)
  (h2 : train_speed_kmh = 90)
  (h3 : bridge_length = 370) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 49.6 :=
by sorry

end train_passing_bridge_l818_81861


namespace arithmetic_calculation_l818_81898

theorem arithmetic_calculation : 12 / 4 - 3 - 16 + 4 * 6 = 8 := by
  sorry

end arithmetic_calculation_l818_81898


namespace square_extension_theorem_l818_81811

/-- A configuration of points derived from a unit square. -/
structure SquareExtension where
  /-- The unit square ABCD -/
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  /-- Extension point E on AB extended -/
  E : ℝ × ℝ
  /-- Extension point F on DA extended -/
  F : ℝ × ℝ
  /-- Point G on ray FC such that FG = FE -/
  G : ℝ × ℝ
  /-- Point H on ray FC such that FH = 1 -/
  H : ℝ × ℝ
  /-- Intersection of FE and line through G parallel to CE -/
  J : ℝ × ℝ
  /-- Intersection of FE and line through H parallel to CJ -/
  K : ℝ × ℝ

  /-- ABCD forms a unit square -/
  h_unit_square : A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1)
  /-- BE = 1 -/
  h_BE : E = (2, 0)
  /-- AF = 5/9 -/
  h_AF : F = (0, 5/9)
  /-- FG = FE -/
  h_FG_eq_FE : dist F G = dist F E
  /-- FH = 1 -/
  h_FH : dist F H = 1
  /-- G is on ray FC -/
  h_G_on_FC : ∃ t : ℝ, t > 0 ∧ G = F + t • (C - F)
  /-- H is on ray FC -/
  h_H_on_FC : ∃ t : ℝ, t > 0 ∧ H = F + t • (C - F)
  /-- Line through G is parallel to CE -/
  h_G_parallel_CE : (G.2 - J.2) / (G.1 - J.1) = (C.2 - E.2) / (C.1 - E.1)
  /-- Line through H is parallel to CJ -/
  h_H_parallel_CJ : (H.2 - K.2) / (H.1 - K.1) = (C.2 - J.2) / (C.1 - J.1)

/-- The main theorem stating that FK = 349/97 in the given configuration. -/
theorem square_extension_theorem (se : SquareExtension) : dist se.F se.K = 349/97 := by
  sorry

end square_extension_theorem_l818_81811


namespace max_profit_at_eight_days_max_profit_value_l818_81835

/-- Profit function for fruit wholesaler --/
def profit (x : ℕ) : ℝ :=
  let initial_amount := 500
  let purchase_price := 40
  let base_selling_price := 60
  let daily_price_increase := 2
  let daily_loss := 10
  let daily_storage_cost := 40
  let selling_price := base_selling_price + daily_price_increase * x
  let remaining_amount := initial_amount - daily_loss * x
  (selling_price * remaining_amount) - (daily_storage_cost * x) - (initial_amount * purchase_price)

/-- Maximum storage time in days --/
def max_storage_time : ℕ := 8

/-- Theorem: Maximum profit is achieved at 8 days of storage --/
theorem max_profit_at_eight_days :
  ∀ x : ℕ, x ≤ max_storage_time → profit x ≤ profit max_storage_time :=
sorry

/-- Theorem: Maximum profit is 11600 yuan --/
theorem max_profit_value :
  profit max_storage_time = 11600 :=
sorry

end max_profit_at_eight_days_max_profit_value_l818_81835


namespace platform_length_l818_81845

/-- Given a train of length 300 meters, which takes 39 seconds to cross a platform
    and 9 seconds to cross a signal pole, the length of the platform is 1000 meters. -/
theorem platform_length
  (train_length : ℝ)
  (platform_crossing_time : ℝ)
  (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 39)
  (h3 : pole_crossing_time = 9) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 1000 := by
sorry

end platform_length_l818_81845


namespace three_A_students_l818_81854

-- Define the students
inductive Student : Type
| Edward : Student
| Fiona : Student
| George : Student
| Hannah : Student
| Ian : Student

-- Define a predicate for getting an A
def got_A : Student → Prop := sorry

-- Define the statements
axiom Edward_statement : got_A Student.Edward → got_A Student.Fiona
axiom Fiona_statement : got_A Student.Fiona → got_A Student.George
axiom George_statement : got_A Student.George → got_A Student.Hannah
axiom Hannah_statement : got_A Student.Hannah → got_A Student.Ian

-- Define the condition that exactly three students got an A
axiom three_A : ∃ (s1 s2 s3 : Student), 
  (s1 ≠ s2 ∧ s1 ≠ s3 ∧ s2 ≠ s3) ∧
  got_A s1 ∧ got_A s2 ∧ got_A s3 ∧
  (∀ (s : Student), got_A s → (s = s1 ∨ s = s2 ∨ s = s3))

-- The theorem to prove
theorem three_A_students : 
  got_A Student.George ∧ got_A Student.Hannah ∧ got_A Student.Ian ∧
  ¬got_A Student.Edward ∧ ¬got_A Student.Fiona :=
sorry

end three_A_students_l818_81854


namespace circle_diameter_from_area_l818_81873

theorem circle_diameter_from_area (A : ℝ) (d : ℝ) : 
  A = 400 * Real.pi → d = 40 → A = Real.pi * (d / 2)^2 :=
by sorry

end circle_diameter_from_area_l818_81873


namespace steak_entree_cost_l818_81895

theorem steak_entree_cost 
  (total_guests : ℕ) 
  (chicken_cost : ℕ) 
  (total_budget : ℕ) 
  (h1 : total_guests = 80)
  (h2 : chicken_cost = 18)
  (h3 : total_budget = 1860)
  : (total_budget - (total_guests / 4 * chicken_cost)) / (3 * total_guests / 4) = 25 := by
  sorry

end steak_entree_cost_l818_81895


namespace circle_max_sum_of_abs_l818_81891

theorem circle_max_sum_of_abs (x y : ℝ) :
  x^2 + y^2 = 4 → |x| + |y| ≤ 2 * Real.sqrt 2 :=
by sorry

end circle_max_sum_of_abs_l818_81891


namespace simplify_trig_expression_l818_81819

theorem simplify_trig_expression : 
  Real.sqrt (1 - Real.sin (160 * π / 180) ^ 2) = Real.cos (20 * π / 180) := by
  sorry

end simplify_trig_expression_l818_81819


namespace gum_pack_size_l818_81852

-- Define the number of cherry and grape gum pieces
def cherry_gum : ℚ := 25
def grape_gum : ℚ := 35

-- Define the number of packs of grape gum found
def grape_packs_found : ℚ := 6

-- Define the variable x as the number of pieces in a complete pack
variable (x : ℚ)

-- Define the equality condition
def equality_condition (x : ℚ) : Prop :=
  (cherry_gum - x) / grape_gum = cherry_gum / (grape_gum + grape_packs_found * x)

-- Theorem statement
theorem gum_pack_size :
  equality_condition x → x = 115 / 6 :=
by
  sorry

end gum_pack_size_l818_81852


namespace planes_perpendicular_from_line_l818_81867

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular_from_line (a : Line) (M N : Plane) :
  perpendicular a M → parallel a N → perpendicularPlanes N M :=
sorry

end planes_perpendicular_from_line_l818_81867


namespace tan_150_degrees_l818_81814

theorem tan_150_degrees : Real.tan (150 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l818_81814


namespace nathan_ate_four_boxes_l818_81896

def gumballs_per_box : ℕ := 5
def gumballs_eaten : ℕ := 20

theorem nathan_ate_four_boxes : 
  gumballs_eaten / gumballs_per_box = 4 := by
  sorry

end nathan_ate_four_boxes_l818_81896


namespace total_pencils_l818_81838

theorem total_pencils (boxes : ℕ) (pencils_per_box : ℕ) (h1 : boxes = 162) (h2 : pencils_per_box = 4) : 
  boxes * pencils_per_box = 648 := by
  sorry

end total_pencils_l818_81838


namespace f_evaluation_l818_81837

/-- The function f(x) = 3x^2 - 5x + 8 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 8

/-- Theorem stating that 3f(4) + 2f(-4) = 260 -/
theorem f_evaluation : 3 * f 4 + 2 * f (-4) = 260 := by
  sorry

end f_evaluation_l818_81837


namespace percentage_equality_l818_81855

theorem percentage_equality (x : ℝ) : (80 / 100 * 600 = 50 / 100 * x) → x = 960 := by
  sorry

end percentage_equality_l818_81855


namespace two_car_garage_count_l818_81800

theorem two_car_garage_count (total_houses : ℕ) (pool_houses : ℕ) (garage_and_pool : ℕ) (neither : ℕ) :
  total_houses = 65 →
  pool_houses = 40 →
  garage_and_pool = 35 →
  neither = 10 →
  ∃ (garage_houses : ℕ), garage_houses = 50 ∧ 
    total_houses = garage_houses + pool_houses - garage_and_pool + neither :=
by sorry

end two_car_garage_count_l818_81800


namespace bills_omelet_preparation_time_l818_81876

/-- Represents the time in minutes for various tasks in omelet preparation --/
structure OmeletPreparationTime where
  chop_pepper : ℕ
  chop_onion : ℕ
  grate_cheese : ℕ
  assemble_and_cook : ℕ

/-- Represents the quantities of ingredients and omelets --/
structure OmeletQuantities where
  peppers : ℕ
  onions : ℕ
  omelets : ℕ

/-- Calculates the total time for omelet preparation given preparation times and quantities --/
def total_preparation_time (prep_time : OmeletPreparationTime) (quantities : OmeletQuantities) : ℕ :=
  prep_time.chop_pepper * quantities.peppers +
  prep_time.chop_onion * quantities.onions +
  prep_time.grate_cheese * quantities.omelets +
  prep_time.assemble_and_cook * quantities.omelets

/-- Theorem stating that Bill's total preparation time for five omelets is 50 minutes --/
theorem bills_omelet_preparation_time :
  let prep_time : OmeletPreparationTime := {
    chop_pepper := 3,
    chop_onion := 4,
    grate_cheese := 1,
    assemble_and_cook := 5
  }
  let quantities : OmeletQuantities := {
    peppers := 4,
    onions := 2,
    omelets := 5
  }
  total_preparation_time prep_time quantities = 50 := by
  sorry

end bills_omelet_preparation_time_l818_81876


namespace calculation_proof_l818_81899

theorem calculation_proof : -1^4 + 16 / (-2)^3 * |(-3) - 1| = -9 := by
  sorry

end calculation_proof_l818_81899


namespace joeys_route_length_l818_81875

theorem joeys_route_length 
  (time_one_way : ℝ) 
  (avg_speed : ℝ) 
  (return_speed : ℝ) 
  (h1 : time_one_way = 1)
  (h2 : avg_speed = 8)
  (h3 : return_speed = 12) : 
  ∃ (route_length : ℝ), route_length = 6 ∧ 
    route_length / return_speed + time_one_way = 2 * route_length / avg_speed :=
by sorry

end joeys_route_length_l818_81875


namespace sequence_realignment_l818_81828

def letter_cycle_length : ℕ := 6
def digit_cycle_length : ℕ := 4

theorem sequence_realignment :
  ∃ n : ℕ, n > 0 ∧ n % letter_cycle_length = 0 ∧ n % digit_cycle_length = 0 ∧
  ∀ m : ℕ, (m > 0 ∧ m % letter_cycle_length = 0 ∧ m % digit_cycle_length = 0) → m ≥ n :=
by
  sorry

end sequence_realignment_l818_81828


namespace pyramid_section_volume_l818_81882

/-- Given a pyramid with base area 3 and volume 3, and two parallel cross-sections with areas 1 and 2,
    the volume of the part of the pyramid between these cross-sections is (2√6 - √3) / 3. -/
theorem pyramid_section_volume 
  (base_area : ℝ) 
  (pyramid_volume : ℝ) 
  (section_area_1 : ℝ) 
  (section_area_2 : ℝ) 
  (h_base_area : base_area = 3) 
  (h_pyramid_volume : pyramid_volume = 3) 
  (h_section_area_1 : section_area_1 = 1) 
  (h_section_area_2 : section_area_2 = 2) : 
  ∃ (section_volume : ℝ), section_volume = (2 * Real.sqrt 6 - Real.sqrt 3) / 3 :=
by sorry

end pyramid_section_volume_l818_81882


namespace inequality_proof_l818_81877

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ((2*a + b + c)^2 / (2*a^2 + (b + c)^2)) + 
  ((2*b + c + a)^2 / (2*b^2 + (c + a)^2)) + 
  ((2*c + a + b)^2 / (2*c^2 + (a + b)^2)) ≤ 8 := by
sorry

end inequality_proof_l818_81877


namespace cricket_game_initial_overs_l818_81880

/-- Proves that the number of overs played initially is 10 in a cricket game scenario -/
theorem cricket_game_initial_overs (target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) 
  (remaining_overs : ℝ) (h1 : target = 282) (h2 : initial_rate = 3.2) 
  (h3 : required_rate = 6.25) (h4 : remaining_overs = 40) : 
  ∃ (initial_overs : ℝ), initial_overs = 10 ∧ 
  initial_rate * initial_overs + required_rate * remaining_overs = target :=
by sorry

end cricket_game_initial_overs_l818_81880
