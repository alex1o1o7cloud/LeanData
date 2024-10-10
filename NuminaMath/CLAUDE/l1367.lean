import Mathlib

namespace complement_of_A_l1367_136784

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x ≥ 2}

-- State the theorem
theorem complement_of_A : Set.compl A = {x : ℝ | x < 2} := by
  sorry

end complement_of_A_l1367_136784


namespace max_handshakes_equals_combinations_l1367_136717

/-- The number of men in the group -/
def n : ℕ := 20

/-- The number of men involved in each handshake -/
def k : ℕ := 2

/-- Calculates the number of combinations of k items from n items -/
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: The maximum number of unique pairwise handshakes among n men is equal to the number of combinations of k=2 men from n men -/
theorem max_handshakes_equals_combinations :
  combinations n k = 190 := by sorry

end max_handshakes_equals_combinations_l1367_136717


namespace vector_magnitude_sum_l1367_136767

/-- Given two vectors a and b in ℝ², prove that if |a| = 3, |b| = 4, 
    and a - b = (√2, √7), then |a + b| = √41 -/
theorem vector_magnitude_sum (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 3)
  (h2 : ‖b‖ = 4)
  (h3 : a - b = (Real.sqrt 2, Real.sqrt 7)) :
  ‖a + b‖ = Real.sqrt 41 := by
  sorry


end vector_magnitude_sum_l1367_136767


namespace integer_solution_squared_sum_eq_product_l1367_136729

theorem integer_solution_squared_sum_eq_product (a b c : ℤ) :
  a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end integer_solution_squared_sum_eq_product_l1367_136729


namespace product_of_numbers_l1367_136707

theorem product_of_numbers (x y : ℝ) (sum_eq : x + y = 18) (sum_squares_eq : x^2 + y^2 = 180) : x * y = 72 := by
  sorry

end product_of_numbers_l1367_136707


namespace inspector_meter_count_l1367_136713

/-- Given an inspector who rejects 10% of meters as defective and finds 20 meters to be defective,
    the total number of meters examined is 200. -/
theorem inspector_meter_count (reject_rate : ℝ) (defective_count : ℕ) (total_count : ℕ) : 
  reject_rate = 0.1 →
  defective_count = 20 →
  (reject_rate : ℝ) * total_count = defective_count →
  total_count = 200 := by
  sorry

end inspector_meter_count_l1367_136713


namespace paco_initial_salty_cookies_l1367_136708

/-- The number of salty cookies Paco had initially -/
def initial_salty_cookies : ℕ := sorry

/-- The number of sweet cookies Paco had initially -/
def initial_sweet_cookies : ℕ := 40

/-- The number of salty cookies Paco ate -/
def eaten_salty_cookies : ℕ := 28

/-- The number of sweet cookies Paco ate -/
def eaten_sweet_cookies : ℕ := 15

/-- The difference between salty and sweet cookies eaten -/
def salty_sweet_difference : ℕ := 13

theorem paco_initial_salty_cookies :
  initial_salty_cookies = 56 :=
by sorry

end paco_initial_salty_cookies_l1367_136708


namespace prob_different_suits_l1367_136758

/-- Represents a standard 52-card deck -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards in the combined deck -/
def CombinedDeck : ℕ := 2 * StandardDeck

/-- Represents the number of cards of the same suit in the combined deck -/
def SameSuitCards : ℕ := 26

/-- The probability of drawing two cards of different suits from a pile of two shuffled standard 52-card decks -/
theorem prob_different_suits : 
  (CombinedDeck - 1 - SameSuitCards) / (CombinedDeck - 1 : ℚ) = 78 / 103 := by
sorry

end prob_different_suits_l1367_136758


namespace calculation_result_solution_set_l1367_136773

-- Problem 1
theorem calculation_result : (Real.pi - 2023) ^ 0 + |-Real.sqrt 3| - 2 * Real.sin (π / 3) = 1 := by sorry

-- Problem 2
def system_of_inequalities (x : ℝ) : Prop :=
  2 * (x + 3) ≥ 8 ∧ x < (x + 4) / 2

theorem solution_set :
  ∀ x : ℝ, system_of_inequalities x ↔ 1 ≤ x ∧ x < 4 := by sorry

end calculation_result_solution_set_l1367_136773


namespace largest_binomial_coefficient_fifth_term_l1367_136749

/-- 
Theorem: There exists a natural number n such that the binomial coefficient 
of the 5th term in the expansion of (x - 2/x)^n is the largest, and n = 7 
is one such value.
-/
theorem largest_binomial_coefficient_fifth_term : 
  ∃ n : ℕ, (
    -- The binomial coefficient of the 5th term is the largest
    ∀ k : ℕ, k ≤ n → (n.choose 4) ≥ (n.choose k)
  ) ∧ 
  -- n = 7 is a valid solution
  (7 : ℕ) ∈ { m : ℕ | ∀ k : ℕ, k ≤ m → (m.choose 4) ≥ (m.choose k) } :=
by sorry


end largest_binomial_coefficient_fifth_term_l1367_136749


namespace martha_cakes_l1367_136704

theorem martha_cakes (num_children : ℕ) (cakes_per_child : ℕ) (h1 : num_children = 3) (h2 : cakes_per_child = 6) :
  num_children * cakes_per_child = 18 :=
by sorry

end martha_cakes_l1367_136704


namespace inequality_chain_l1367_136753

theorem inequality_chain (a b : ℝ) (h : a < b ∧ b < 0) : a^2 > a*b ∧ a*b > b^2 := by
  sorry

end inequality_chain_l1367_136753


namespace hyperbola_a_value_l1367_136756

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the focal length
def focal_length : ℝ := 12

-- Define the condition for point M
def point_M_condition (a b c : ℝ) : Prop :=
  b^2 / a = 2 * (a + c)

-- Theorem statement
theorem hyperbola_a_value (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧
  c = focal_length / 2 ∧
  c^2 = a^2 + b^2 ∧
  point_M_condition a b c →
  a = 2 := by
  sorry

end hyperbola_a_value_l1367_136756


namespace clothing_division_l1367_136780

theorem clothing_division (total : ℕ) (first_load : ℕ) (h1 : total = 36) (h2 : first_load = 18) :
  (total - first_load) / 2 = 9 := by
sorry

end clothing_division_l1367_136780


namespace AB_BA_parallel_l1367_136774

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = k • w ∨ w = k • v

/-- Vector AB is defined as the difference between points B and A -/
def vector_AB (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

/-- Vector BA is defined as the difference between points A and B -/
def vector_BA (A B : ℝ × ℝ) : ℝ × ℝ :=
  (A.1 - B.1, A.2 - B.2)

/-- Theorem: Vectors AB and BA are parallel -/
theorem AB_BA_parallel (A B : ℝ × ℝ) :
  are_parallel (vector_AB A B) (vector_BA A B) := by
  sorry

end AB_BA_parallel_l1367_136774


namespace right_triangle_arctan_sum_l1367_136739

/-- Given a right-angled triangle ABC with ∠A = π/2, D is the foot of the altitude from A to BC,
    BD = m, and DC = n. This theorem proves that arctan(b/(m+c)) + arctan(c/(n+b)) = π/4. -/
theorem right_triangle_arctan_sum (a b c m n : ℝ) (h_right : a^2 = b^2 + c^2)
  (h_altitude : m * n = a^2) (h_sum : m * b + c * n = b * n + c * m) :
  Real.arctan (b / (m + c)) + Real.arctan (c / (n + b)) = π / 4 := by
  sorry

end right_triangle_arctan_sum_l1367_136739


namespace b_visited_city_b_l1367_136746

-- Define the types for students and cities
inductive Student : Type
| A : Student
| B : Student
| C : Student

inductive City : Type
| A : City
| B : City
| C : City

-- Define a function to represent whether a student has visited a city
def hasVisited : Student → City → Prop := sorry

-- State the theorem
theorem b_visited_city_b 
  (h1 : ∀ c : City, hasVisited Student.A c → hasVisited Student.B c)
  (h2 : ¬ hasVisited Student.A City.C)
  (h3 : ¬ hasVisited Student.B City.A)
  (h4 : ∃ c : City, hasVisited Student.A c ∧ hasVisited Student.B c ∧ hasVisited Student.C c)
  : hasVisited Student.B City.B :=
sorry

end b_visited_city_b_l1367_136746


namespace function_divisibility_property_l1367_136702

theorem function_divisibility_property (f : ℕ+ → ℕ+) : 
  (∀ a b : ℕ+, ∃ k : ℕ+, a^2 + f a * f b = k * (f a + b)) →
  (∀ n : ℕ+, f n = n) :=
by sorry

end function_divisibility_property_l1367_136702


namespace smallest_gcd_multiple_l1367_136764

theorem smallest_gcd_multiple (a b : ℕ+) (h : Nat.gcd a b = 10) :
  (∃ (a' b' : ℕ+), Nat.gcd a' b' = 10 ∧ Nat.gcd (12 * a') (20 * b') = 40) ∧
  (∀ (a'' b'' : ℕ+), Nat.gcd a'' b'' = 10 → Nat.gcd (12 * a'') (20 * b'') ≥ 40) :=
by sorry

end smallest_gcd_multiple_l1367_136764


namespace scientific_notation_of_ten_million_two_hundred_thousand_l1367_136775

theorem scientific_notation_of_ten_million_two_hundred_thousand :
  ∃ (a : ℝ) (n : ℤ), 10200000 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 10.2 ∧ n = 7 :=
sorry

end scientific_notation_of_ten_million_two_hundred_thousand_l1367_136775


namespace number_subtracted_from_x_l1367_136791

-- Define the problem conditions
def problem_conditions (x y z a : ℤ) : Prop :=
  ((x - a) * (y - 5) * (z - 2) = 1000) ∧
  (∀ (x' y' z' : ℤ), ((x' - a) * (y' - 5) * (z' - 2) = 1000) → (x + y + z ≤ x' + y' + z')) ∧
  (x + y + z = 7)

-- State the theorem
theorem number_subtracted_from_x :
  ∃ (x y z a : ℤ), problem_conditions x y z a ∧ a = -30 :=
sorry

end number_subtracted_from_x_l1367_136791


namespace pentagon_rods_l1367_136730

theorem pentagon_rods (rods : Finset ℕ) : 
  rods = {4, 9, 18, 25} →
  (∀ e ∈ Finset.range 41 \ rods, 
    (e ≠ 0 ∧ 
     e < 4 + 9 + 18 + 25 ∧
     4 + 9 + 18 + e > 25 ∧
     4 + 9 + 25 + e > 18 ∧
     4 + 18 + 25 + e > 9 ∧
     9 + 18 + 25 + e > 4)) →
  (Finset.range 41 \ rods).card = 51 :=
sorry

end pentagon_rods_l1367_136730


namespace unique_valid_number_l1367_136786

def shares_one_digit (n m : Nat) : Bool :=
  let n_digits := n.digits 10
  let m_digits := m.digits 10
  (n_digits.filter (fun d => m_digits.contains d)).length = 1

def is_valid_number (n : Nat) : Bool :=
  n ≥ 100 ∧ n < 1000 ∧
  shares_one_digit n 543 ∧
  shares_one_digit n 142 ∧
  shares_one_digit n 562

theorem unique_valid_number : 
  ∀ n : Nat, is_valid_number n ↔ n = 163 :=
sorry

end unique_valid_number_l1367_136786


namespace prob_two_black_is_25_102_l1367_136711

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total_cards : Nat)
  (black_cards : Nat)
  (h_total : total_cards = 52)
  (h_black : black_cards = 26)

/-- The probability of drawing two black cards from a standard deck -/
def prob_two_black (d : Deck) : Rat :=
  (d.black_cards * (d.black_cards - 1)) / (d.total_cards * (d.total_cards - 1))

/-- Theorem stating the probability of drawing two black cards is 25/102 -/
theorem prob_two_black_is_25_102 (d : Deck) : prob_two_black d = 25 / 102 := by
  sorry

end prob_two_black_is_25_102_l1367_136711


namespace bacteria_growth_proof_l1367_136734

/-- The number of quadrupling cycles in two minutes -/
def quadrupling_cycles : ℕ := 8

/-- The number of bacteria after two minutes -/
def final_bacteria_count : ℕ := 4194304

/-- The growth factor for each cycle -/
def growth_factor : ℕ := 4

/-- The initial number of bacteria -/
def initial_bacteria : ℕ := 64

theorem bacteria_growth_proof :
  initial_bacteria * growth_factor ^ quadrupling_cycles = final_bacteria_count :=
by sorry

end bacteria_growth_proof_l1367_136734


namespace all_positives_can_be_written_l1367_136743

/-- The predicate that determines if a number can be written on the board -/
def CanBeWritten (n : ℕ) : Prop :=
  ∃ (sequence : ℕ → ℕ), sequence 0 = 1 ∧
  (∀ k, ∃ b, (sequence k + b + 1) ∣ (sequence k^2 + b^2 + 1) ∧
            sequence (k + 1) = b)

/-- The main theorem stating that any positive integer can be written on the board -/
theorem all_positives_can_be_written :
  ∀ n : ℕ, n > 0 → CanBeWritten n :=
sorry

end all_positives_can_be_written_l1367_136743


namespace line_tangent_to_curve_l1367_136795

/-- Tangency condition for a line to a curve -/
theorem line_tangent_to_curve
  {m n u v : ℝ}
  (hm : m > 1)
  (hn : m⁻¹ + n⁻¹ = 1) :
  (∀ x y : ℝ, u * x + v * y = 1 →
    (∃ a : ℝ, x^m + y^m = a) →
    (∀ δ ε : ℝ, δ ≠ 0 ∨ ε ≠ 0 →
      (x + δ)^m + (y + ε)^m > a)) ↔
  u^n + v^n = 1 :=
sorry

end line_tangent_to_curve_l1367_136795


namespace bazylev_inequality_l1367_136762

theorem bazylev_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^y + y^z + z^x > 1 := by
  sorry

end bazylev_inequality_l1367_136762


namespace intersection_A_B_l1367_136769

def A : Set ℝ := {-2, -1, 1, 2, 4}

def B : Set ℝ := {x : ℝ | (x + 2) * (x - 3) < 0}

theorem intersection_A_B : A ∩ B = {-1, 1, 2} := by sorry

end intersection_A_B_l1367_136769


namespace johns_spending_l1367_136751

/-- Given John's allowance B, prove that he spends 4/13 of B on movie ticket and soda combined -/
theorem johns_spending (B : ℝ) (t d : ℝ) 
  (ht : t = 0.25 * (B - d)) 
  (hd : d = 0.1 * (B - t)) : 
  t + d = (4 / 13) * B := by
sorry

end johns_spending_l1367_136751


namespace quadratic_roots_max_value_l1367_136772

/-- Given a quadratic x^2 - tx + q with roots α and β, where 
    α + β = α^2 + β^2 = α^3 + β^3 = ⋯ = α^2010 + β^2010,
    the maximum value of 1/α^2012 + 1/β^2012 is 2. -/
theorem quadratic_roots_max_value (t q α β : ℝ) : 
  α^2 - t*α + q = 0 →
  β^2 - t*β + q = 0 →
  (∀ n : ℕ, n ≤ 2010 → α^n + β^n = α + β) →
  (∃ M : ℝ, M = 2 ∧ 
    ∀ t' q' α' β' : ℝ, 
      α'^2 - t'*α' + q' = 0 →
      β'^2 - t'*β' + q' = 0 →
      (∀ n : ℕ, n ≤ 2010 → α'^n + β'^n = α' + β') →
      1/α'^2012 + 1/β'^2012 ≤ M) :=
by sorry

end quadratic_roots_max_value_l1367_136772


namespace division_problem_l1367_136752

theorem division_problem : (72 : ℝ) / (6 / (3 / 2)) = 18 := by
  sorry

end division_problem_l1367_136752


namespace percentage_comparison_l1367_136794

theorem percentage_comparison : 
  (0.85 * 250 - 0.75 * 180) < 0.90 * 320 := by
  sorry

end percentage_comparison_l1367_136794


namespace smallest_sum_of_reciprocals_l1367_136757

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 20) :
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 20 → (x : ℕ) + (y : ℕ) ≤ (a : ℕ) + (b : ℕ) :=
by sorry

end smallest_sum_of_reciprocals_l1367_136757


namespace visible_painted_cubes_12_l1367_136736

/-- The number of visible painted unit cubes from a corner of a cube -/
def visiblePaintedCubes (n : ℕ) : ℕ :=
  3 * n^2 - 3 * (n - 1) + 1

/-- Theorem: The number of visible painted unit cubes from a corner of a 12×12×12 cube is 400 -/
theorem visible_painted_cubes_12 :
  visiblePaintedCubes 12 = 400 := by
  sorry

#eval visiblePaintedCubes 12  -- This will evaluate to 400

end visible_painted_cubes_12_l1367_136736


namespace sqrt_expressions_equality_l1367_136715

theorem sqrt_expressions_equality : 
  (2 * Real.sqrt 3 - 3 * Real.sqrt 12 + 5 * Real.sqrt 27 = 11 * Real.sqrt 3) ∧
  ((1 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 6) - (2 * Real.sqrt 3 - 1)^2 = 
   -2 * Real.sqrt 2 + 4 * Real.sqrt 3 - 13) := by
  sorry

end sqrt_expressions_equality_l1367_136715


namespace nested_fraction_equality_l1367_136745

theorem nested_fraction_equality : 
  1 + 2 / (3 + 6 / (7 + 8 / 9)) = 409 / 267 := by sorry

end nested_fraction_equality_l1367_136745


namespace geometric_sequence_sum_4_l1367_136761

/-- A geometric sequence with its sum -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1  -- Geometric sequence property

/-- Theorem: For a geometric sequence satisfying given conditions, S_4 = 75 -/
theorem geometric_sequence_sum_4 (seq : GeometricSequence)
  (h1 : seq.a 3 - seq.a 1 = 15)
  (h2 : seq.a 2 - seq.a 1 = 5) :
  seq.S 4 = 75 := by
  sorry

end geometric_sequence_sum_4_l1367_136761


namespace may_scarf_count_l1367_136740

/-- Represents the number of scarves that can be made from one yarn of a given color -/
def scarvesPerYarn (color : String) : ℕ :=
  match color with
  | "red" => 3
  | "blue" => 2
  | "yellow" => 4
  | "green" => 5
  | "purple" => 6
  | _ => 0

/-- Represents the number of yarns May has for each color -/
def yarnCount (color : String) : ℕ :=
  match color with
  | "red" => 1
  | "blue" => 1
  | "yellow" => 1
  | "green" => 3
  | "purple" => 2
  | _ => 0

/-- The list of colors May has yarn for -/
def colors : List String := ["red", "blue", "yellow", "green", "purple"]

/-- The total number of scarves May can make -/
def totalScarves : ℕ := (colors.map (fun c => scarvesPerYarn c * yarnCount c)).sum

theorem may_scarf_count : totalScarves = 36 := by
  sorry

end may_scarf_count_l1367_136740


namespace angle_A_measure_l1367_136777

/-- A triangle with an internal point creating three smaller triangles -/
structure TriangleWithInternalPoint where
  /-- Angle B of the large triangle -/
  angle_B : ℝ
  /-- Angle C of the large triangle -/
  angle_C : ℝ
  /-- Angle D at the internal point -/
  angle_D : ℝ
  /-- Angle A of one of the smaller triangles -/
  angle_A : ℝ
  /-- The sum of angles in a triangle is 180° -/
  triangle_sum : angle_B + angle_C + angle_D + (180 - angle_A) = 180

/-- Theorem: If m∠B = 50°, m∠C = 40°, and m∠D = 30°, then m∠A = 120° -/
theorem angle_A_measure (t : TriangleWithInternalPoint)
    (hB : t.angle_B = 50)
    (hC : t.angle_C = 40)
    (hD : t.angle_D = 30) :
    t.angle_A = 120 := by
  sorry


end angle_A_measure_l1367_136777


namespace shyam_weight_increase_l1367_136719

-- Define the original weight ratio
def weight_ratio : ℚ := 7 / 9

-- Define Ram's weight increase percentage
def ram_increase : ℚ := 12 / 100

-- Define the total new weight
def total_new_weight : ℚ := 165.6

-- Define the total weight increase percentage
def total_increase : ℚ := 20 / 100

-- Theorem to prove
theorem shyam_weight_increase : ∃ (original_ram : ℚ) (original_shyam : ℚ),
  original_shyam = original_ram / weight_ratio ∧
  (original_ram * (1 + ram_increase) + original_shyam * (1 + x)) = total_new_weight ∧
  (original_ram + original_shyam) * (1 + total_increase) = total_new_weight ∧
  abs (x - 26.29 / 100) < 0.0001 := by
  sorry

end shyam_weight_increase_l1367_136719


namespace quadratic_positive_combination_l1367_136793

/-- A quadratic polynomial -/
def QuadraticPolynomial := ℝ → ℝ

/-- Predicate to check if a function is negative on an interval -/
def NegativeOnInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a < x ∧ x < b → f x < 0

/-- Predicate to check if two intervals are non-overlapping -/
def NonOverlappingIntervals (a b c d : ℝ) : Prop :=
  b < c ∨ d < a

/-- Main theorem statement -/
theorem quadratic_positive_combination
  (f g : QuadraticPolynomial)
  (a b c d : ℝ)
  (hf : NegativeOnInterval f a b)
  (hg : NegativeOnInterval g c d)
  (h_non_overlap : NonOverlappingIntervals a b c d) :
  ∃ (α β : ℝ), α > 0 ∧ β > 0 ∧ ∀ x, α * f x + β * g x > 0 :=
sorry

end quadratic_positive_combination_l1367_136793


namespace parallel_vectors_k_value_l1367_136771

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t ≠ 0 ∧ a.1 = t * b.1 ∧ a.2 = t * b.2

/-- Given vectors a and b, if they are parallel, then k = 1/2 -/
theorem parallel_vectors_k_value (k : ℝ) :
  let a : ℝ × ℝ := (1, k)
  let b : ℝ × ℝ := (2, 1)
  are_parallel a b → k = 1/2 := by
  sorry

end parallel_vectors_k_value_l1367_136771


namespace interns_escape_probability_l1367_136710

/-- A permutation on n elements -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- The probability that a random permutation on n elements has no cycle longer than k -/
noncomputable def prob_no_long_cycle (n k : ℕ) : ℝ := sorry

/-- The number of interns/drawers -/
def num_interns : ℕ := 44

/-- The maximum allowed cycle length for survival -/
def max_cycle_length : ℕ := 21

/-- The minimum required survival probability -/
def min_survival_prob : ℝ := 0.30

theorem interns_escape_probability :
  prob_no_long_cycle num_interns max_cycle_length > min_survival_prob := by sorry

end interns_escape_probability_l1367_136710


namespace inequality_proof_l1367_136796

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0) (non_neg_d : d ≥ 0)
  (sum_eq_four : a + b + c + d = 4) :
  a * Real.sqrt (3*a + b + c) + b * Real.sqrt (3*b + c + d) + 
  c * Real.sqrt (3*c + d + a) + d * Real.sqrt (3*d + a + b) ≥ 4 * Real.sqrt 5 := by
  sorry

end inequality_proof_l1367_136796


namespace nora_watch_cost_l1367_136727

/-- The cost of a watch in dollars, given the number of dimes paid and the value of a dime in dollars. -/
def watch_cost (dimes_paid : ℕ) (dime_value : ℚ) : ℚ :=
  (dimes_paid : ℚ) * dime_value

/-- Theorem stating that if Nora paid 90 dimes for a watch, and 1 dime is worth $0.10, the cost of the watch is $9.00. -/
theorem nora_watch_cost :
  let dimes_paid : ℕ := 90
  let dime_value : ℚ := 1/10
  watch_cost dimes_paid dime_value = 9 := by
sorry

end nora_watch_cost_l1367_136727


namespace tyler_meal_choices_l1367_136789

def meat_options : ℕ := 3
def vegetable_options : ℕ := 5
def dessert_options : ℕ := 4
def drink_options : ℕ := 4
def vegetables_to_choose : ℕ := 3

def number_of_meals : ℕ := meat_options * Nat.choose vegetable_options vegetables_to_choose * dessert_options * drink_options

theorem tyler_meal_choices : number_of_meals = 480 := by
  sorry

end tyler_meal_choices_l1367_136789


namespace daily_toy_production_l1367_136741

/-- Given a factory that produces toys, this theorem proves the daily production
    when the weekly production and number of working days are known. -/
theorem daily_toy_production
  (weekly_production : ℕ)
  (working_days : ℕ)
  (h_weekly : weekly_production = 6500)
  (h_days : working_days = 5)
  (h_equal_daily : weekly_production % working_days = 0) :
  weekly_production / working_days = 1300 := by
  sorry

#check daily_toy_production

end daily_toy_production_l1367_136741


namespace garage_door_properties_l1367_136748

/-- Represents a garage door mechanism -/
structure GarageDoor where
  AC : ℝ
  BC : ℝ
  CY : ℝ
  AX : ℝ
  BD : ℝ

/-- Properties of the garage door mechanism -/
def isValidGarageDoor (door : GarageDoor) : Prop :=
  door.AC = 0.5 ∧ door.BC = 0.5 ∧ door.CY = 0.5 ∧ door.AX = 1 ∧ door.BD = 2

/-- Calculate CR given XS -/
def calculateCR (door : GarageDoor) (XS : ℝ) : ℝ := sorry

/-- Check if Y's height remains constant -/
def isYHeightConstant (door : GarageDoor) : Prop := sorry

/-- Calculate DT given XT -/
def calculateDT (door : GarageDoor) (XT : ℝ) : ℝ := sorry

/-- Main theorem about the garage door mechanism -/
theorem garage_door_properties (door : GarageDoor) 
  (h : isValidGarageDoor door) : 
  calculateCR door 0.2 = 0.1 ∧ 
  isYHeightConstant door ∧ 
  calculateDT door 0.4 = 0.6 := by sorry

end garage_door_properties_l1367_136748


namespace sourball_candies_distribution_l1367_136766

def nellie_limit : ℕ := 12
def jacob_limit : ℕ := nellie_limit / 2
def lana_limit : ℕ := jacob_limit - 3
def total_candies : ℕ := 30
def num_people : ℕ := 3

theorem sourball_candies_distribution :
  (total_candies - (nellie_limit + jacob_limit + lana_limit)) / num_people = 3 := by
  sorry

end sourball_candies_distribution_l1367_136766


namespace cookie_cost_is_16_l1367_136755

/-- The cost of each cookie Josiah bought in March --/
def cookie_cost (total_spent : ℕ) (days_in_march : ℕ) (cookies_per_day : ℕ) : ℚ :=
  total_spent / (days_in_march * cookies_per_day)

/-- Theorem stating that each cookie costs 16 dollars --/
theorem cookie_cost_is_16 :
  cookie_cost 992 31 2 = 16 := by
  sorry

end cookie_cost_is_16_l1367_136755


namespace simultaneous_truth_probability_l1367_136735

/-- The probability of A telling the truth -/
def prob_A_truth : ℝ := 0.8

/-- The probability of B telling the truth -/
def prob_B_truth : ℝ := 0.6

/-- The probability of A and B telling the truth simultaneously -/
def prob_both_truth : ℝ := prob_A_truth * prob_B_truth

theorem simultaneous_truth_probability :
  prob_both_truth = 0.48 :=
by sorry

end simultaneous_truth_probability_l1367_136735


namespace fraction_product_simplification_l1367_136768

theorem fraction_product_simplification :
  (3 : ℚ) / 4 * (4 : ℚ) / 5 * (5 : ℚ) / 6 * (6 : ℚ) / 7 * (7 : ℚ) / 8 = (3 : ℚ) / 8 :=
by sorry

end fraction_product_simplification_l1367_136768


namespace concert_attendance_l1367_136797

/-- The number of buses used for the concert -/
def num_buses : ℕ := 12

/-- The number of students each bus can carry -/
def students_per_bus : ℕ := 57

/-- The total number of students who went to the concert -/
def total_students : ℕ := num_buses * students_per_bus

theorem concert_attendance : total_students = 684 := by
  sorry

end concert_attendance_l1367_136797


namespace makenna_larger_garden_l1367_136706

-- Define the dimensions of Karl's garden
def karl_length : ℕ := 30
def karl_width : ℕ := 50

-- Define the dimensions of Makenna's garden
def makenna_length : ℕ := 35
def makenna_width : ℕ := 45

-- Define the area Karl allocates for trees
def karl_tree_area : ℕ := 300

-- Calculate the areas of both gardens
def karl_total_area : ℕ := karl_length * karl_width
def makenna_total_area : ℕ := makenna_length * makenna_width

-- Calculate Karl's vegetable area
def karl_veg_area : ℕ := karl_total_area - karl_tree_area

-- Define the difference between vegetable areas
def veg_area_difference : ℕ := makenna_total_area - karl_veg_area

-- Theorem statement
theorem makenna_larger_garden : veg_area_difference = 375 := by
  sorry

end makenna_larger_garden_l1367_136706


namespace arithmetic_sequence_fifth_term_l1367_136799

/-- Given an arithmetic sequence with the first four terms as specified,
    prove that the fifth term is x^2 - (21x)/5 -/
theorem arithmetic_sequence_fifth_term (x y : ℝ) :
  let a₁ := x^2 + 3*y
  let a₂ := (x - 2) * y
  let a₃ := x^2 - y
  let a₄ := x / (y + 1)
  -- The sequence is arithmetic
  (a₂ - a₁ = a₃ - a₂) ∧ (a₃ - a₂ = a₄ - a₃) →
  -- The fifth term
  ∃ (a₅ : ℝ), a₅ = x^2 - (21 * x) / 5 :=
by sorry

end arithmetic_sequence_fifth_term_l1367_136799


namespace fraction_not_on_time_l1367_136738

/-- Represents the attendees at the monthly meeting -/
structure Attendees where
  total : ℕ
  males : ℕ
  females : ℕ
  malesOnTime : ℕ
  femalesOnTime : ℕ

/-- The conditions of the problem -/
def meetingConditions (a : Attendees) : Prop :=
  a.males = (2 * a.total) / 3 ∧
  a.females = a.total - a.males ∧
  a.malesOnTime = (3 * a.males) / 4 ∧
  a.femalesOnTime = (5 * a.females) / 6

/-- The theorem to be proved -/
theorem fraction_not_on_time (a : Attendees) 
  (h : meetingConditions a) : 
  (a.total - (a.malesOnTime + a.femalesOnTime)) / a.total = 1 / 4 := by
  sorry


end fraction_not_on_time_l1367_136738


namespace angle_ABH_in_regular_octagon_l1367_136790

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The measure of an angle in degrees -/
def angle_measure (a b c : ℝ × ℝ) : ℝ := sorry

theorem angle_ABH_in_regular_octagon (ABCDEFGH : RegularOctagon) :
  let vertices := ABCDEFGH.vertices
  angle_measure (vertices 0) (vertices 1) (vertices 7) = 22.5 := by
  sorry

end angle_ABH_in_regular_octagon_l1367_136790


namespace nine_sided_polygon_diagonals_l1367_136723

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex 9-sided polygon has 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by
  sorry

end nine_sided_polygon_diagonals_l1367_136723


namespace paths_with_consecutive_right_moves_l1367_136765

/-- The number of paths on a grid with specified conditions -/
def num_paths (horizontal_steps vertical_steps : ℕ) : ℕ :=
  Nat.choose (horizontal_steps + vertical_steps - 1) vertical_steps

/-- The main theorem stating the number of paths under given conditions -/
theorem paths_with_consecutive_right_moves :
  num_paths 7 6 = 924 :=
by
  sorry

end paths_with_consecutive_right_moves_l1367_136765


namespace min_value_x_plus_y_l1367_136731

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 16*y = x*y) :
  ∀ z w : ℝ, z > 0 → w > 0 → z + 16*w = z*w → x + y ≤ z + w ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 16*b = a*b ∧ a + b = 25 :=
by sorry

end min_value_x_plus_y_l1367_136731


namespace namek_clock_overlap_time_l1367_136732

/-- Represents the clock on Namek --/
structure NamekClock where
  minutes_per_hour : ℕ
  hour_hand_rate : ℚ
  minute_hand_rate : ℚ

/-- The time when the hour and minute hands overlap on Namek's clock --/
def overlap_time (clock : NamekClock) : ℚ :=
  360 / (clock.minute_hand_rate - clock.hour_hand_rate)

/-- Theorem stating that the overlap time for Namek's clock is 20/19 hours --/
theorem namek_clock_overlap_time :
  let clock : NamekClock := {
    minutes_per_hour := 100,
    hour_hand_rate := 360 / 20,
    minute_hand_rate := 360 / (100 / 60)
  }
  overlap_time clock = 20 / 19 := by sorry

end namek_clock_overlap_time_l1367_136732


namespace computer_store_optimal_solution_l1367_136712

/-- Represents the profit optimization problem for a computer store. -/
def ComputerStoreProblem (total_computers : ℕ) (profit_A profit_B : ℕ) : Prop :=
  ∃ (x : ℕ) (y : ℤ),
    -- Total number of computers is fixed
    x + (total_computers - x) = total_computers ∧
    -- Profit calculation
    y = -100 * x + 50000 ∧
    -- Constraint on type B computers
    (total_computers - x) ≤ 3 * x ∧
    -- x is the optimal number of type A computers
    ∀ (x' : ℕ), x' ≠ x →
      (-100 * x' + 50000 : ℤ) ≤ (-100 * x + 50000 : ℤ) ∧
    -- Maximum profit is achieved
    y = 47500

/-- Theorem stating the existence of an optimal solution for the computer store problem. -/
theorem computer_store_optimal_solution :
  ComputerStoreProblem 100 400 500 :=
sorry

end computer_store_optimal_solution_l1367_136712


namespace cube_root_125_times_fourth_root_256_times_fifth_root_32_l1367_136742

theorem cube_root_125_times_fourth_root_256_times_fifth_root_32 :
  (125 : ℝ) ^ (1/3) * (256 : ℝ) ^ (1/4) * (32 : ℝ) ^ (1/5) = 40 := by
  sorry

end cube_root_125_times_fourth_root_256_times_fifth_root_32_l1367_136742


namespace point_in_second_quadrant_l1367_136725

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the point corresponding to i + i^2
def point : ℂ := i + i^2

-- Theorem stating that the point is in the second quadrant
theorem point_in_second_quadrant : 
  Complex.re point < 0 ∧ Complex.im point > 0 := by
  sorry

end point_in_second_quadrant_l1367_136725


namespace intersection_of_A_and_B_l1367_136782

def A : Set ℕ := {2, 3, 5, 7}
def B : Set ℕ := {1, 2, 3, 5, 8}

theorem intersection_of_A_and_B : A ∩ B = {2, 3, 5} := by sorry

end intersection_of_A_and_B_l1367_136782


namespace total_books_read_is_72cs_l1367_136701

/-- The total number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  let books_per_month : ℕ := 6
  let months_per_year : ℕ := 12
  let books_per_student_per_year : ℕ := books_per_month * months_per_year
  let total_students : ℕ := c * s
  books_per_student_per_year * total_students

/-- Theorem stating that the total number of books read is 72cs -/
theorem total_books_read_is_72cs (c s : ℕ) :
  total_books_read c s = 72 * c * s := by
  sorry

end total_books_read_is_72cs_l1367_136701


namespace snow_probability_first_week_l1367_136785

def probability_of_snow (days : ℕ) (daily_prob : ℚ) : ℚ :=
  1 - (1 - daily_prob) ^ days

theorem snow_probability_first_week :
  let prob_first_four := probability_of_snow 4 (1/4)
  let prob_next_three := probability_of_snow 3 (1/3)
  let total_prob := 1 - (1 - prob_first_four) * (1 - prob_next_three)
  total_prob = 29/32 := by
  sorry

end snow_probability_first_week_l1367_136785


namespace arithmetic_sequence_problem_l1367_136770

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₃ + a₈ = 6, prove that 3a₂ + a₁₆ = 12 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arithmetic : arithmetic_sequence a) 
    (h_sum : a 3 + a 8 = 6) : 
  3 * a 2 + a 16 = 12 := by
sorry

end arithmetic_sequence_problem_l1367_136770


namespace second_round_votes_l1367_136787

/-- Represents the total number of votes in the second round of an election. -/
def total_votes : ℕ := sorry

/-- Represents the percentage of votes received by Candidate A in the second round. -/
def candidate_a_percentage : ℚ := 50 / 100

/-- Represents the percentage of votes received by Candidate B in the second round. -/
def candidate_b_percentage : ℚ := 30 / 100

/-- Represents the percentage of votes received by Candidate C in the second round. -/
def candidate_c_percentage : ℚ := 20 / 100

/-- Represents the majority of votes by which Candidate A won over Candidate B. -/
def majority : ℕ := 1350

theorem second_round_votes : 
  (candidate_a_percentage - candidate_b_percentage) * total_votes = majority ∧
  total_votes = 6750 := by sorry

end second_round_votes_l1367_136787


namespace integer_roots_quadratic_l1367_136716

theorem integer_roots_quadratic (n : ℕ+) : 
  (∃ x : ℤ, x^2 - 4*x + n.val = 0) ↔ (n.val = 3 ∨ n.val = 4) := by
  sorry

end integer_roots_quadratic_l1367_136716


namespace hotel_rooms_rented_l1367_136705

theorem hotel_rooms_rented (total_rooms : ℝ) (h1 : total_rooms > 0) : 
  let air_conditioned := (3/5) * total_rooms
  let rented_air_conditioned := (2/3) * air_conditioned
  let not_rented := total_rooms - (rented_air_conditioned + (1/5) * air_conditioned)
  (total_rooms - not_rented) / total_rooms = 3/4 := by
  sorry

end hotel_rooms_rented_l1367_136705


namespace smallest_sum_of_three_factors_of_125_l1367_136792

theorem smallest_sum_of_three_factors_of_125 :
  ∀ a b c : ℕ+,
  a * b * c = 125 →
  ∀ x y z : ℕ+,
  x * y * z = 125 →
  a + b + c ≤ x + y + z →
  a + b + c = 15 :=
by sorry

end smallest_sum_of_three_factors_of_125_l1367_136792


namespace root_minus_one_l1367_136763

theorem root_minus_one (p : ℝ) (hp : p ≠ 1 ∧ p ≠ -1) : 
  (2 * (1 - p + p^2) / (1 - p^2)) * (-1)^2 + 
  ((2 - p) / (1 + p)) * (-1) - 
  (p / (1 - p)) = 0 := by
sorry

end root_minus_one_l1367_136763


namespace no_solution_for_2023_l1367_136750

theorem no_solution_for_2023 : ¬ ∃ (a b : ℤ), a^2 + b^2 = 2023 := by
  sorry

end no_solution_for_2023_l1367_136750


namespace coefficient_of_monomial_l1367_136781

/-- The coefficient of a monomial is the numerical factor multiplied by the variables. -/
def coefficient (m : ℝ) (x y : ℝ) : ℝ := m

/-- For the monomial -2π * x^2 * y, prove that its coefficient is -2π. -/
theorem coefficient_of_monomial :
  coefficient (-2 * Real.pi) (x ^ 2) y = -2 * Real.pi := by
  sorry

end coefficient_of_monomial_l1367_136781


namespace fraction_subtraction_l1367_136779

theorem fraction_subtraction : (16 : ℚ) / 40 - (3 : ℚ) / 9 = (1 : ℚ) / 15 := by
  sorry

end fraction_subtraction_l1367_136779


namespace cheapest_lamp_cost_l1367_136718

theorem cheapest_lamp_cost (frank_money : ℕ) (remaining : ℕ) (price_ratio : ℕ) : 
  frank_money = 90 →
  remaining = 30 →
  price_ratio = 3 →
  (frank_money - remaining) / price_ratio = 20 :=
by sorry

end cheapest_lamp_cost_l1367_136718


namespace custom_op_theorem_l1367_136721

/-- Custom operation ⊗ defined as x ⊗ y = x^3 + y^3 -/
def custom_op (x y : ℝ) : ℝ := x^3 + y^3

/-- Theorem stating that h ⊗ (h ⊗ h) = h^3 + 8h^9 -/
theorem custom_op_theorem (h : ℝ) : custom_op h (custom_op h h) = h^3 + 8*h^9 := by
  sorry

end custom_op_theorem_l1367_136721


namespace regular_polygon_144_degrees_has_10_sides_l1367_136700

/-- A regular polygon with interior angles of 144 degrees has 10 sides -/
theorem regular_polygon_144_degrees_has_10_sides :
  ∀ (n : ℕ), n > 2 →
  (180 * (n - 2) : ℚ) / n = 144 →
  n = 10 :=
by
  sorry

end regular_polygon_144_degrees_has_10_sides_l1367_136700


namespace polynomial_divisibility_theorem_l1367_136709

def is_prime (p : ℕ) : Prop := Nat.Prime p

def divides (p n : ℕ) : Prop := n % p = 0

def polynomial_with_int_coeffs (P : ℕ → ℤ) : Prop :=
  ∃ (coeffs : List ℤ), ∀ x, P x = (coeffs.enum.map (λ (i, a) => a * (x ^ i))).sum

def constant_polynomial (P : ℕ → ℤ) : Prop :=
  ∃ c : ℤ, c ≠ 0 ∧ ∀ x, P x = c

def S (P : ℕ → ℤ) : Set ℕ :=
  {p | is_prime p ∧ ∃ n, divides p (P n).natAbs}

theorem polynomial_divisibility_theorem (P : ℕ → ℤ) 
  (h_poly : polynomial_with_int_coeffs P) :
  (Set.Finite (S P)) ↔ (constant_polynomial P) :=
sorry

end polynomial_divisibility_theorem_l1367_136709


namespace smallest_candy_count_l1367_136778

theorem smallest_candy_count : ∃ (n : ℕ), 
  (n ≥ 100 ∧ n ≤ 999) ∧ 
  (n + 6) % 9 = 0 ∧ 
  (n - 9) % 6 = 0 ∧ 
  (∀ m : ℕ, m ≥ 100 ∧ m ≤ 999 ∧ (m + 6) % 9 = 0 ∧ (m - 9) % 6 = 0 → m ≥ n) ∧
  n = 111 := by
sorry

end smallest_candy_count_l1367_136778


namespace sandy_spending_percentage_l1367_136703

def total_amount : ℝ := 320
def amount_left : ℝ := 224

theorem sandy_spending_percentage :
  (total_amount - amount_left) / total_amount * 100 = 30 := by
  sorry

end sandy_spending_percentage_l1367_136703


namespace hundred_thirteen_in_sequence_l1367_136759

/-- Ewan's sequence starting at 3 and increasing by 11 each time -/
def ewans_sequence (n : ℕ) : ℤ := 11 * n - 8

/-- Theorem stating that 113 is in Ewan's sequence -/
theorem hundred_thirteen_in_sequence : ∃ n : ℕ, ewans_sequence n = 113 := by
  sorry

end hundred_thirteen_in_sequence_l1367_136759


namespace consecutive_sum_fifteen_l1367_136724

theorem consecutive_sum_fifteen (n : ℤ) : n + (n + 1) + (n + 2) = 15 → n = 4 := by
  sorry

end consecutive_sum_fifteen_l1367_136724


namespace sequence_periodicity_l1367_136760

def M (m : ℕ) : Set ℕ :=
  {x | x ∈ Finset.range m ∨ (x > m ∧ x ≤ 2*m ∧ x % 2 = 1)}

def next_term (m : ℕ) (a : ℕ) : ℕ :=
  if a % 2 = 0 then a / 2 else a + m

def is_periodic (m : ℕ) (a : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, 
    (Nat.iterate (next_term m) k a) = (Nat.iterate (next_term m) n a)

theorem sequence_periodicity (m : ℕ) (h : m > 0) :
  ∀ a : ℕ, is_periodic m a ↔ a ∈ M m :=
sorry

end sequence_periodicity_l1367_136760


namespace logical_implications_l1367_136744

theorem logical_implications (p q : Prop) : 
  (((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q))) ∧
  (((p ∧ q) → ¬(¬p)) ∧ ¬(¬(¬p) → (p ∧ q))) ∧
  ((¬p → ¬(p ∧ q)) ∧ ¬(¬(p ∧ q) → ¬p)) := by
  sorry

end logical_implications_l1367_136744


namespace area_pda_equals_sqrt_vw_l1367_136754

-- Define the rectangular pyramid
structure RectangularPyramid where
  -- Lengths of edges
  a : ℝ
  b : ℝ
  h : ℝ
  -- Areas of triangles
  u : ℝ
  v : ℝ
  w : ℝ
  -- Conditions
  pos_a : 0 < a
  pos_b : 0 < b
  pos_h : 0 < h
  area_pab : u = (1/2) * a * h
  area_pbc : v = (1/2) * a * b
  area_pcd : w = (1/2) * b * h

-- Theorem statement
theorem area_pda_equals_sqrt_vw (pyramid : RectangularPyramid) :
  (1/2) * pyramid.b * pyramid.h = Real.sqrt (pyramid.v * pyramid.w) :=
sorry

end area_pda_equals_sqrt_vw_l1367_136754


namespace point_always_on_line_l1367_136726

theorem point_always_on_line (m b : ℝ) (h : m * b < 0) :
  0 = m * 2003 + b := by sorry

end point_always_on_line_l1367_136726


namespace x_plus_y_value_l1367_136747

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| + x + y = 10) 
  (h2 : x + |y| - y = 12) : 
  x + y = 3.6 := by
sorry

end x_plus_y_value_l1367_136747


namespace class_size_proof_l1367_136788

/-- Represents the number of pupils in a class. -/
def num_pupils : ℕ := 56

/-- Represents the wrongly entered mark. -/
def wrong_mark : ℕ := 73

/-- Represents the correct mark. -/
def correct_mark : ℕ := 45

/-- Represents the increase in average marks due to the error. -/
def avg_increase : ℚ := 1/2

theorem class_size_proof :
  (wrong_mark - correct_mark : ℚ) / num_pupils = avg_increase :=
sorry

end class_size_proof_l1367_136788


namespace chord_intersection_probability_l1367_136737

/-- Given 1988 points evenly distributed on a circle, this function represents
    the probability that chord PQ intersects chord RS when selecting four distinct points
    P, Q, R, and S with all quadruples being equally likely. -/
def probability_chords_intersect (n : ℕ) : ℚ :=
  if n = 1988 then 1/3 else 0

/-- Theorem stating that the probability of chord PQ intersecting chord RS
    is 1/3 when selecting 4 points from 1988 evenly distributed points on a circle. -/
theorem chord_intersection_probability :
  probability_chords_intersect 1988 = 1/3 := by sorry

end chord_intersection_probability_l1367_136737


namespace distance_after_two_hours_l1367_136776

/-- Anna's walking speed in miles per minute -/
def anna_speed : ℚ := 1 / 20

/-- Mark's jogging speed in miles per minute -/
def mark_speed : ℚ := 3 / 40

/-- Duration of walking in minutes -/
def duration : ℕ := 120

/-- The distance between Anna and Mark after walking for the given duration -/
def distance_apart : ℚ := anna_speed * duration + mark_speed * duration

theorem distance_after_two_hours :
  distance_apart = 15 := by sorry

end distance_after_two_hours_l1367_136776


namespace f_properties_l1367_136720

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

theorem f_properties :
  (∃ (x : ℝ), -2 < x ∧ x < 2 ∧ f x = 5 ∧ ∀ (y : ℝ), -2 < y ∧ y < 2 → f y ≤ 5) ∧
  (∀ (m : ℝ), ∃ (x : ℝ), -2 < x ∧ x < 2 ∧ f x < m) :=
by sorry

end f_properties_l1367_136720


namespace muffin_banana_cost_ratio_l1367_136714

theorem muffin_banana_cost_ratio :
  ∀ (m b : ℝ),
  m > 0 → b > 0 →
  4 * m + 3 * b > 0 →
  2 * (4 * m + 3 * b) = 2 * m + 16 * b →
  m / b = 5 / 3 :=
by sorry

end muffin_banana_cost_ratio_l1367_136714


namespace algebraic_expression_value_l1367_136722

/-- Given that x³ + x + m = 7 when x = 1, prove that x³ + x + m = 3 when x = -1. -/
theorem algebraic_expression_value (m : ℝ) 
  (h : 1^3 + 1 + m = 7) : 
  (-1)^3 + (-1) + m = 3 := by
  sorry

end algebraic_expression_value_l1367_136722


namespace cannon_probability_l1367_136783

theorem cannon_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 0.5) (h2 : p2 = 0.8) (h3 : p3 = 0.7) : 
  p1 * p2 * p3 = 0.28 := by
  sorry

end cannon_probability_l1367_136783


namespace moles_of_Na2SO4_formed_l1367_136733

-- Define the reactants and products
structure Compound where
  name : String
  coefficient : ℚ

-- Define the reaction
def reaction : List Compound → List Compound → Prop :=
  λ reactants products => reactants.length = 2 ∧ products.length = 2

-- Define the balanced equation
def balancedEquation : Prop :=
  reaction
    [⟨"H2SO4", 1⟩, ⟨"NaOH", 2⟩]
    [⟨"Na2SO4", 1⟩, ⟨"H2O", 2⟩]

-- Define the given amounts of reactants
def givenReactants : List Compound :=
  [⟨"H2SO4", 1⟩, ⟨"NaOH", 2⟩]

-- Theorem to prove
theorem moles_of_Na2SO4_formed
  (h1 : balancedEquation)
  (h2 : givenReactants = [⟨"H2SO4", 1⟩, ⟨"NaOH", 2⟩]) :
  ∃ (product : Compound),
    product.name = "Na2SO4" ∧ product.coefficient = 1 :=
  sorry

end moles_of_Na2SO4_formed_l1367_136733


namespace value_of_q_l1367_136798

theorem value_of_q (p q : ℝ) 
  (h1 : 1 < p) 
  (h2 : p < q) 
  (h3 : (1 / p) + (1 / q) = 1) 
  (h4 : p * q = 16 / 3) : 
  q = 4 := by
sorry

end value_of_q_l1367_136798


namespace cyclic_sum_inequality_l1367_136728

-- Define the variables as positive real numbers
variable (x y z : ℝ) 

-- Define the hypothesis that x, y, and z are positive
variable (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)

-- Define the main theorem
theorem cyclic_sum_inequality :
  let f (a b c : ℝ) := Real.sqrt (a / (b + c)) * Real.sqrt ((a * b + a * c + b^2 + c^2) / (b^2 + c^2))
  f x y z + f y z x + f z x y ≥ 2 * Real.sqrt 2 := by
  sorry

end cyclic_sum_inequality_l1367_136728
