import Mathlib

namespace lease_problem_l139_13978

theorem lease_problem (elapsed_time : ℝ) : 
  elapsed_time > 0 ∧ 
  elapsed_time < 99 ∧
  (2 / 3) * elapsed_time = (4 / 5) * (99 - elapsed_time) →
  elapsed_time = 54 := by
sorry

end lease_problem_l139_13978


namespace problem_solution_l139_13904

open Real

noncomputable def f (a k x : ℝ) : ℝ := a^x - (k-1)*a^(-x)

theorem problem_solution (a k : ℝ) (h_a : a > 0) (h_a_neq_1 : a ≠ 1)
  (h_odd : ∀ x, f a k (-x) = -f a k x) :
  (k = 2) ∧
  (f a k 1 < 0 →
    ∀ t, (∀ x, f a k (x^2 + t*x) + f a k (4 - x) < 0) ↔ -3 < t ∧ t < 5) ∧
  (f a k 1 = 3/2 →
    ∃ m, (∀ x ≥ 1, a^(2*x) + a^(-2*x) - m * f a k x ≥ 5/4) ∧
         (∃ x ≥ 1, a^(2*x) + a^(-2*x) - m * f a k x = 5/4) ∧
         m = 2) :=
by sorry

end problem_solution_l139_13904


namespace apple_transport_trucks_l139_13933

theorem apple_transport_trucks (total_apples : ℕ) (transported_apples : ℕ) (truck_capacity : ℕ) 
  (h1 : total_apples = 80)
  (h2 : transported_apples = 56)
  (h3 : truck_capacity = 4)
  : (total_apples - transported_apples) / truck_capacity = 6 := by
  sorry

end apple_transport_trucks_l139_13933


namespace digits_of_3_15_times_5_10_l139_13975

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The number of digits in 3^15 * 5^10 is 14 -/
theorem digits_of_3_15_times_5_10 : num_digits (3^15 * 5^10) = 14 := by sorry

end digits_of_3_15_times_5_10_l139_13975


namespace nested_bracket_equals_two_l139_13912

-- Define the bracket operation
def bracket (x y z : ℚ) : ℚ := (x + y) / z

-- State the theorem
theorem nested_bracket_equals_two :
  bracket (bracket 45 15 60) (bracket 3 3 6) (bracket 20 10 30) = 2 := by
  sorry

end nested_bracket_equals_two_l139_13912


namespace profit_percent_calculation_l139_13957

/-- Proves that the profit percent is 26% when selling an article at a certain price,
    given that selling it at 2/3 of that price results in a 16% loss. -/
theorem profit_percent_calculation (P C : ℝ) 
  (h : (2/3) * P = 0.84 * C) : 
  (P - C) / C * 100 = 26 := by
  sorry

end profit_percent_calculation_l139_13957


namespace projectile_max_height_l139_13991

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

/-- Theorem stating that the maximum height of the projectile is 161 meters -/
theorem projectile_max_height :
  ∃ t : ℝ, h t = 161 ∧ ∀ s : ℝ, h s ≤ h t :=
sorry

end projectile_max_height_l139_13991


namespace resulting_polygon_has_24_sides_l139_13941

/-- Calculates the number of sides in the resulting polygon formed by sequentially 
    attaching regular polygons from triangle to octagon. -/
def resulting_polygon_sides : ℕ :=
  let initial_triangle := 3
  let square_addition := 4 - 2
  let pentagon_addition := 5 - 2
  let hexagon_addition := 6 - 2
  let heptagon_addition := 7 - 2
  let octagon_addition := 8 - 1
  initial_triangle + square_addition + pentagon_addition + 
  hexagon_addition + heptagon_addition + octagon_addition

/-- The resulting polygon has 24 sides. -/
theorem resulting_polygon_has_24_sides : resulting_polygon_sides = 24 := by
  sorry

end resulting_polygon_has_24_sides_l139_13941


namespace gala_arrangement_count_l139_13907

/-- The number of programs in the New Year's gala. -/
def total_programs : ℕ := 8

/-- The number of non-singing programs in the New Year's gala. -/
def non_singing_programs : ℕ := 3

/-- The number of singing programs in the New Year's gala. -/
def singing_programs : ℕ := total_programs - non_singing_programs

/-- A function that calculates the number of ways to arrange the programs
    such that non-singing programs are not adjacent and the first and last
    programs are singing programs. -/
def arrangement_count : ℕ :=
  Nat.choose (total_programs - 2) non_singing_programs *
  Nat.factorial non_singing_programs *
  Nat.factorial (singing_programs - 2)

/-- Theorem stating that the number of ways to arrange the programs
    under the given conditions is 720. -/
theorem gala_arrangement_count :
  arrangement_count = 720 :=
by sorry

end gala_arrangement_count_l139_13907


namespace contrapositive_at_least_one_even_l139_13984

theorem contrapositive_at_least_one_even (a b c : ℕ) :
  (¬ (Even a ∨ Even b ∨ Even c)) ↔ (Odd a ∧ Odd b ∧ Odd c) := by
  sorry

end contrapositive_at_least_one_even_l139_13984


namespace sequences_satisfy_conditions_l139_13906

-- Define the sequences A and B
def A (n : ℕ) : ℝ × ℝ := (n, n^3)
def B (n : ℕ) : ℝ × ℝ := (-n, -n^3)

-- Define a function to check if a point is on a line through two other points
def is_on_line (p q r : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := p
  let (x₂, y₂) := q
  let (x₃, y₃) := r
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

-- State the theorem
theorem sequences_satisfy_conditions :
  ∀ (i j k : ℕ), 1 ≤ i → i < j → j < k →
    (is_on_line (A i) (A j) (B k) ↔ k = i + j) ∧
    (is_on_line (B i) (B j) (A k) ↔ k = i + j) :=
by sorry

end sequences_satisfy_conditions_l139_13906


namespace min_value_theorem_l139_13946

theorem min_value_theorem (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2) ≥ 3 * Real.sqrt 3 / 2 := by
sorry

end min_value_theorem_l139_13946


namespace max_tiles_on_floor_l139_13909

/-- Calculates the maximum number of tiles that can fit on a rectangular floor --/
def max_tiles (floor_length floor_width tile_length tile_width : ℕ) : ℕ :=
  let orientation1 := (floor_length / tile_length) * (floor_width / tile_width)
  let orientation2 := (floor_length / tile_width) * (floor_width / tile_length)
  max orientation1 orientation2

/-- Theorem stating the maximum number of tiles that can be accommodated on the given floor --/
theorem max_tiles_on_floor :
  max_tiles 180 120 25 16 = 49 := by
  sorry

end max_tiles_on_floor_l139_13909


namespace max_parts_three_planes_correct_l139_13927

/-- The maximum number of parts into which three non-overlapping planes can divide space -/
def max_parts_three_planes : ℕ := 8

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the specifics of a plane for this statement

/-- A configuration of three non-overlapping planes in 3D space -/
structure ThreePlaneConfiguration where
  plane1 : Plane3D
  plane2 : Plane3D
  plane3 : Plane3D
  non_overlapping : plane1 ≠ plane2 ∧ plane1 ≠ plane3 ∧ plane2 ≠ plane3

/-- The number of parts into which a configuration of three planes divides space -/
def num_parts (config : ThreePlaneConfiguration) : ℕ :=
  sorry -- The actual calculation would go here

theorem max_parts_three_planes_correct :
  ∀ (config : ThreePlaneConfiguration), num_parts config ≤ max_parts_three_planes ∧
  ∃ (config : ThreePlaneConfiguration), num_parts config = max_parts_three_planes :=
sorry

end max_parts_three_planes_correct_l139_13927


namespace parallel_implies_magnitude_perpendicular_implies_k_obtuse_angle_implies_k_range_l139_13988

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b (k : ℝ) : ℝ × ℝ := (-3, k)

-- Theorem 1
theorem parallel_implies_magnitude (k : ℝ) :
  (∃ (t : ℝ), a = t • (b k)) → ‖b k‖ = 3 * Real.sqrt 5 := by
  sorry

-- Theorem 2
theorem perpendicular_implies_k :
  (a • (a + 2 • (b (1/4))) = 0) → (1/4 : ℝ) = 1/4 := by
  sorry

-- Theorem 3
theorem obtuse_angle_implies_k_range (k : ℝ) :
  (a • (b k) < 0) → k < 3/2 ∧ k ≠ -6 := by
  sorry

end parallel_implies_magnitude_perpendicular_implies_k_obtuse_angle_implies_k_range_l139_13988


namespace series_evaluation_l139_13925

open Real

noncomputable def series_sum : ℝ := ∑' k, (k : ℝ)^2 / 3^k

theorem series_evaluation : series_sum = 7 := by sorry

end series_evaluation_l139_13925


namespace hexagon_enclosure_octagon_enclosure_l139_13944

-- Define the shapes
def Square (sideLength : ℝ) : Type := Unit
def RegularHexagon (sideLength : ℝ) : Type := Unit
def Circle (diameter : ℝ) : Type := Unit

-- Define the derived shapes
def Hexagon (s : Square 1) : Type := Unit
def Octagon (h : RegularHexagon (Real.sqrt 3 / 3)) : Type := Unit

-- Define the enclosure property
def CanEnclose (shape : Type) (figure : Circle 1) : Prop := sorry

-- State the theorems
theorem hexagon_enclosure (s : Square 1) (f : Circle 1) :
  CanEnclose (Hexagon s) f := sorry

theorem octagon_enclosure (h : RegularHexagon (Real.sqrt 3 / 3)) (f : Circle 1) :
  CanEnclose (Octagon h) f := sorry

end hexagon_enclosure_octagon_enclosure_l139_13944


namespace angle_B_is_30_degrees_l139_13924

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.c * Real.cos t.B + t.b * Real.cos t.C = t.a * Real.sin t.A ∧
  (Real.sqrt 3 / 4) * (t.b^2 + t.a^2 - t.c^2) = (1/2) * t.a * t.b * Real.sin t.C

-- Theorem statement
theorem angle_B_is_30_degrees (t : Triangle) 
  (h : satisfies_conditions t) : t.B = 30 * (Real.pi / 180) := by
  sorry

end angle_B_is_30_degrees_l139_13924


namespace greatest_q_minus_r_l139_13970

theorem greatest_q_minus_r : ∃ (q r : ℕ), 
  1259 = 23 * q + r ∧ 
  q > 0 ∧ 
  r > 0 ∧
  ∀ (q' r' : ℕ), (1259 = 23 * q' + r' ∧ q' > 0 ∧ r' > 0) → q' - r' ≤ q - r ∧ 
  q - r = 37 := by
  sorry

end greatest_q_minus_r_l139_13970


namespace total_cost_with_tip_l139_13901

def hair_cost : ℝ := 50
def nail_cost : ℝ := 30
def tip_percentage : ℝ := 0.20

theorem total_cost_with_tip : 
  (hair_cost + nail_cost) * (1 + tip_percentage) = 96 := by
  sorry

end total_cost_with_tip_l139_13901


namespace special_function_property_l139_13962

/-- A real-valued function on rational numbers satisfying specific properties -/
def special_function (f : ℚ → ℝ) : Prop :=
  (f 0 = 0) ∧
  (∀ α, α ≠ 0 → f α > 0) ∧
  (∀ α β, f (α * β) = f α * f β) ∧
  (∀ α β, f (α + β) ≤ f α + f β) ∧
  (∀ m : ℤ, f m ≤ 1989)

/-- Theorem stating that f(α + β) = max{f(α), f(β)} when f(α) ≠ f(β) -/
theorem special_function_property (f : ℚ → ℝ) (h : special_function f) :
  ∀ α β : ℚ, f α ≠ f β → f (α + β) = max (f α) (f β) :=
sorry

end special_function_property_l139_13962


namespace acme_vowel_soup_sequences_l139_13973

/-- The number of distinct elements in the set -/
def n : ℕ := 5

/-- The number of times each element appears -/
def k : ℕ := 6

/-- The length of the sequences to be formed -/
def seq_length : ℕ := 6

/-- The number of possible sequences -/
def num_sequences : ℕ := n ^ seq_length

theorem acme_vowel_soup_sequences :
  num_sequences = 15625 :=
sorry

end acme_vowel_soup_sequences_l139_13973


namespace sequence_sum_l139_13966

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) : ℕ → ℕ
  | n => a₁ + (n - 1) * d

def geometric_sequence (b₁ : ℕ) (r : ℕ) : ℕ → ℕ
  | n => b₁ * r^(n - 1)

theorem sequence_sum (a₁ : ℕ) :
  let a := arithmetic_sequence a₁ 2
  let b := geometric_sequence 1 2
  a (b 2) + a (b 3) + a (b 4) = 25 := by
  sorry

end sequence_sum_l139_13966


namespace nuts_per_bag_l139_13965

theorem nuts_per_bag (bags : ℕ) (students : ℕ) (nuts_per_student : ℕ) 
  (h1 : bags = 65)
  (h2 : students = 13)
  (h3 : nuts_per_student = 75) :
  (students * nuts_per_student) / bags = 15 := by
sorry

end nuts_per_bag_l139_13965


namespace prisoners_puzzle_solution_l139_13968

-- Define the hair colors
inductive HairColor
| Blonde
| Red
| Brunette

-- Define the prisoners
inductive Prisoner
| P1
| P2
| P3
| P4
| P5

-- Define the ladies
structure Lady where
  name : String
  hairColor : HairColor

-- Define the statement of a prisoner
structure Statement where
  prisoner : Prisoner
  ownLady : Lady
  neighborLadies : List HairColor

-- Define the truthfulness of a prisoner
inductive Truthfulness
| AlwaysTruth
| AlwaysLie
| Variable

-- Define the problem setup
def prisonerSetup : List (Prisoner × Truthfulness) := 
  [(Prisoner.P1, Truthfulness.AlwaysTruth),
   (Prisoner.P2, Truthfulness.AlwaysLie),
   (Prisoner.P3, Truthfulness.AlwaysTruth),
   (Prisoner.P4, Truthfulness.AlwaysLie),
   (Prisoner.P5, Truthfulness.Variable)]

-- Define the statements of the prisoners
def prisonerStatements : List Statement := 
  [{ prisoner := Prisoner.P1, 
     ownLady := { name := "Anna", hairColor := HairColor.Blonde },
     neighborLadies := [HairColor.Blonde] },
   { prisoner := Prisoner.P2,
     ownLady := { name := "Brynhild", hairColor := HairColor.Red },
     neighborLadies := [HairColor.Brunette, HairColor.Brunette] },
   { prisoner := Prisoner.P3,
     ownLady := { name := "Clotilde", hairColor := HairColor.Red },
     neighborLadies := [HairColor.Red, HairColor.Red] },
   { prisoner := Prisoner.P4,
     ownLady := { name := "Gudrun", hairColor := HairColor.Red },
     neighborLadies := [HairColor.Brunette, HairColor.Brunette] },
   { prisoner := Prisoner.P5,
     ownLady := { name := "Johanna", hairColor := HairColor.Brunette },
     neighborLadies := [HairColor.Brunette, HairColor.Blonde] }]

-- Define the correct solution
def correctSolution : List Lady := 
  [{ name := "Anna", hairColor := HairColor.Blonde },
   { name := "Brynhild", hairColor := HairColor.Red },
   { name := "Clotilde", hairColor := HairColor.Red },
   { name := "Gudrun", hairColor := HairColor.Red },
   { name := "Johanna", hairColor := HairColor.Brunette }]

-- Theorem statement
theorem prisoners_puzzle_solution :
  ∀ (solution : List Lady),
  (∀ p ∈ prisonerSetup, 
   ∀ s ∈ prisonerStatements,
   p.1 = s.prisoner →
   (p.2 = Truthfulness.AlwaysTruth → 
    (s.ownLady ∈ solution ∧ 
     ∀ c ∈ s.neighborLadies, ∃ l ∈ solution, l.hairColor = c)) ∧
   (p.2 = Truthfulness.AlwaysLie → 
    (s.ownLady ∉ solution ∨ 
     ∃ c ∈ s.neighborLadies, ∀ l ∈ solution, l.hairColor ≠ c))) →
  solution = correctSolution :=
sorry

end prisoners_puzzle_solution_l139_13968


namespace investment_average_interest_rate_l139_13953

/-- Prove that given a total investment split between two interest rates with equal annual returns, the average rate of interest is as calculated. -/
theorem investment_average_interest_rate 
  (total_investment : ℝ)
  (rate1 rate2 : ℝ)
  (h1 : total_investment = 6000)
  (h2 : rate1 = 0.03)
  (h3 : rate2 = 0.07)
  (h4 : ∃ (x : ℝ), x * rate2 = (total_investment - x) * rate1) :
  (rate1 * (total_investment - (180 / 0.1)) + rate2 * (180 / 0.1)) / total_investment = 0.042 := by
  sorry

end investment_average_interest_rate_l139_13953


namespace probability_two_even_toys_l139_13903

def total_toys : ℕ := 21
def even_toys : ℕ := 10

theorem probability_two_even_toys :
  let p1 := even_toys / total_toys
  let p2 := (even_toys - 1) / (total_toys - 1)
  p1 * p2 = 3 / 14 := by sorry

end probability_two_even_toys_l139_13903


namespace tangent_line_at_x_1_increase_decrease_intervals_l139_13986

noncomputable section

-- Define the function f(x) = ln x - ax
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

-- Theorem for the tangent line equation when a = -2
theorem tangent_line_at_x_1 (a : ℝ) (h : a = -2) :
  ∃ m b : ℝ, ∀ x y : ℝ, y = f a x → (x = 1 → m * x + b = y) ∧ 
  (m * x - y + b = 0 ↔ 3 * x - y - 1 = 0) :=
sorry

-- Theorem for the intervals of increase and decrease
theorem increase_decrease_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (a > 0 → (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1/a → f a x₁ < f a x₂) ∧
           (∀ x₁ x₂ : ℝ, 1/a < x₁ → x₁ < x₂ → f a x₂ < f a x₁)) :=
sorry

end tangent_line_at_x_1_increase_decrease_intervals_l139_13986


namespace triangle_similarity_problem_l139_13915

theorem triangle_similarity_problem (DC CB : ℝ) (AB ED AD : ℝ) (FC : ℝ) :
  DC = 9 →
  CB = 9 →
  AB = (1 / 3) * AD →
  ED = (2 / 3) * AD →
  -- Assuming triangle similarity
  (∃ (k₁ k₂ : ℝ), k₁ > 0 ∧ k₂ > 0 ∧
    AB / AD = k₁ ∧
    FC / (CB + AB) = k₁ ∧
    ED / AD = k₂ ∧
    FC / (CB + AB) = k₂) →
  FC = 12 :=
by sorry

end triangle_similarity_problem_l139_13915


namespace lcm_and_sum_of_numbers_l139_13960

def numbers : List Nat := [14, 21, 35]

theorem lcm_and_sum_of_numbers :
  (Nat.lcm (Nat.lcm 14 21) 35 = 210) ∧ (numbers.sum = 70) := by
  sorry

end lcm_and_sum_of_numbers_l139_13960


namespace polygon_interior_angles_l139_13985

theorem polygon_interior_angles (n : ℕ) (h : n = 14) : 
  (n - 2) * 180 - 180 = 2000 :=
sorry

end polygon_interior_angles_l139_13985


namespace hyperbola_standard_equation_l139_13932

/-- A hyperbola with one asymptote defined by x±y=0 and passing through (-1,-2) -/
structure Hyperbola where
  /-- One asymptote of the hyperbola is defined by x±y=0 -/
  asymptote : ∀ (x y : ℝ), x = y ∨ x = -y
  /-- The hyperbola passes through the point (-1,-2) -/
  passes_through : ∃ (f : ℝ → ℝ → ℝ), f (-1) (-2) = 0

/-- The standard equation of the hyperbola is y²/3 - x²/3 = 1 -/
theorem hyperbola_standard_equation (h : Hyperbola) :
  ∃ (f : ℝ → ℝ → ℝ), (∀ x y, f x y = y^2/3 - x^2/3 - 1) ∧ (∀ x y, f x y = 0 ↔ h.passes_through.choose x y = 0) :=
sorry

end hyperbola_standard_equation_l139_13932


namespace fraction_sum_simplification_l139_13996

theorem fraction_sum_simplification : (1 : ℚ) / 462 + 17 / 42 = 94 / 231 := by
  sorry

end fraction_sum_simplification_l139_13996


namespace meeting_speed_l139_13972

theorem meeting_speed
  (total_distance : ℝ)
  (time : ℝ)
  (speed_diff : ℝ)
  (h1 : total_distance = 45)
  (h2 : time = 5)
  (h3 : speed_diff = 1)
  (h4 : ∀ (v_a v_b : ℝ), v_a = v_b + speed_diff → v_a * time + v_b * time = total_distance)
  : ∃ (v_a : ℝ), v_a = 5 ∧ ∃ (v_b : ℝ), v_a = v_b + speed_diff ∧ v_a * time + v_b * time = total_distance :=
by sorry

end meeting_speed_l139_13972


namespace lawn_care_supplies_cost_l139_13948

/-- The total cost of supplies for a lawn care company -/
theorem lawn_care_supplies_cost 
  (num_blades : ℕ) 
  (blade_cost : ℕ) 
  (string_cost : ℕ) :
  num_blades = 4 →
  blade_cost = 8 →
  string_cost = 7 →
  num_blades * blade_cost + string_cost = 39 :=
by
  sorry

end lawn_care_supplies_cost_l139_13948


namespace ryosuke_trip_cost_l139_13971

/-- Calculates the cost of gas for a trip given the odometer readings, fuel efficiency, and gas price -/
def gas_cost (initial_reading final_reading : ℕ) (fuel_efficiency : ℚ) (gas_price : ℚ) : ℚ :=
  let distance := final_reading - initial_reading
  let gas_used := (distance : ℚ) / fuel_efficiency
  gas_used * gas_price

/-- Proves that the cost of gas for Ryosuke's trip is $5.04 -/
theorem ryosuke_trip_cost :
  let initial_reading : ℕ := 74580
  let final_reading : ℕ := 74610
  let fuel_efficiency : ℚ := 25
  let gas_price : ℚ := 21/5
  gas_cost initial_reading final_reading fuel_efficiency gas_price = 504/100 := by
  sorry

end ryosuke_trip_cost_l139_13971


namespace sequence_relation_l139_13920

-- Define the sequence u
def u (n : ℕ) : ℝ := 17^n * (n + 2)

-- State the theorem
theorem sequence_relation (a b : ℝ) :
  (∀ n : ℕ, u (n + 2) = a * u (n + 1) + b * u n) →
  a^2 - b = 144.5 :=
by sorry

end sequence_relation_l139_13920


namespace parabola_kite_sum_l139_13961

/-- Given two parabolas that intersect the coordinate axes in four points forming a kite -/
theorem parabola_kite_sum (a b : ℝ) : 
  (∀ x y : ℝ, (y = a * x^2 + 4 ∨ y = 6 - b * x^2) → 
    (x = 0 ∨ y = 0)) →  -- intersect coordinate axes
  (∃! p q r s : ℝ × ℝ, 
    (p.2 = a * p.1^2 + 4 ∨ p.2 = 6 - b * p.1^2) ∧
    (q.2 = a * q.1^2 + 4 ∨ q.2 = 6 - b * q.1^2) ∧
    (r.2 = a * r.1^2 + 4 ∨ r.2 = 6 - b * r.1^2) ∧
    (s.2 = a * s.1^2 + 4 ∨ s.2 = 6 - b * s.1^2) ∧
    (p.1 = 0 ∨ p.2 = 0) ∧ (q.1 = 0 ∨ q.2 = 0) ∧
    (r.1 = 0 ∨ r.2 = 0) ∧ (s.1 = 0 ∨ s.2 = 0)) →  -- exactly four intersection points
  (∃ d₁ d₂ : ℝ, d₁ * d₂ / 2 = 18) →  -- kite area is 18
  a + b = 2/81 :=
by sorry

end parabola_kite_sum_l139_13961


namespace negation_of_forall_leq_zero_l139_13959

theorem negation_of_forall_leq_zero :
  (¬ ∀ x : ℝ, x^2 - x ≤ 0) ↔ (∃ x₀ : ℝ, x₀^2 - x₀ > 0) := by
  sorry

end negation_of_forall_leq_zero_l139_13959


namespace rope_cutting_problem_l139_13918

theorem rope_cutting_problem :
  let rope1 : ℕ := 44
  let rope2 : ℕ := 54
  let rope3 : ℕ := 74
  Nat.gcd rope1 (Nat.gcd rope2 rope3) = 2 := by
sorry

end rope_cutting_problem_l139_13918


namespace initial_workers_count_l139_13939

/-- Represents the productivity of workers in digging holes -/
structure DiggingProductivity where
  initialWorkers : ℕ
  initialDepth : ℝ
  initialTime : ℝ
  newDepth : ℝ
  newTime : ℝ
  extraWorkers : ℕ

/-- Proves that the initial number of workers is 45 given the conditions -/
theorem initial_workers_count (p : DiggingProductivity) 
  (h1 : p.initialDepth = 30)
  (h2 : p.initialTime = 8)
  (h3 : p.newDepth = 45)
  (h4 : p.newTime = 6)
  (h5 : p.extraWorkers = 45)
  (h6 : p.initialWorkers > 0)
  (h7 : p.initialDepth > 0)
  (h8 : p.initialTime > 0)
  (h9 : p.newDepth > 0)
  (h10 : p.newTime > 0) :
  p.initialWorkers = 45 := by
  sorry


end initial_workers_count_l139_13939


namespace complement_A_intersect_B_l139_13902

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set Nat := {2, 5, 8}
def B : Set Nat := {1, 3, 5, 7}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {1, 3, 7} := by sorry

end complement_A_intersect_B_l139_13902


namespace prop_1_prop_3_prop_4_l139_13949

open Real

-- Define the second quadrant
def second_quadrant (θ : ℝ) : Prop := π/2 < θ ∧ θ < π

-- Proposition 1
theorem prop_1 (θ : ℝ) (h : second_quadrant θ) : sin θ * tan θ < 0 := by
  sorry

-- Proposition 3
theorem prop_3 : sin 1 * cos 2 * tan 3 > 0 := by
  sorry

-- Proposition 4
theorem prop_4 (θ : ℝ) (h : 3*π/2 < θ ∧ θ < 2*π) : sin (π + θ) > 0 := by
  sorry

end prop_1_prop_3_prop_4_l139_13949


namespace cube_with_holes_properties_l139_13997

/-- Represents a cube with square holes on each face -/
structure CubeWithHoles where
  edge_length : ℝ
  hole_side_length : ℝ
  hole_depth : ℝ

/-- Calculate the total surface area of a cube with holes, including inside surfaces -/
def total_surface_area (c : CubeWithHoles) : ℝ :=
  6 * c.edge_length^2 + 6 * (c.hole_side_length^2 + 4 * c.hole_side_length * c.hole_depth)

/-- Calculate the total volume of material removed from a cube due to holes -/
def total_volume_removed (c : CubeWithHoles) : ℝ :=
  6 * c.hole_side_length^2 * c.hole_depth

/-- The main theorem stating the properties of the specific cube with holes -/
theorem cube_with_holes_properties :
  let c := CubeWithHoles.mk 4 2 1
  total_surface_area c = 144 ∧ total_volume_removed c = 24 := by
  sorry


end cube_with_holes_properties_l139_13997


namespace D_96_l139_13992

/-- D(n) is the number of ways of writing n as a product of integers greater than 1, where the order matters -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem: D(96) = 112 -/
theorem D_96 : D 96 = 112 := by sorry

end D_96_l139_13992


namespace quadratic_integer_roots_count_l139_13974

theorem quadratic_integer_roots_count :
  ∃! (S : Finset ℝ), 
    (∀ a ∈ S, ∃ r s : ℤ, r^2 + a*r + 9*a = 0 ∧ s^2 + a*s + 9*a = 0) ∧
    (∀ a : ℝ, (∃ r s : ℤ, r^2 + a*r + 9*a = 0 ∧ s^2 + a*s + 9*a = 0) → a ∈ S) ∧
    S.card = 6 :=
by sorry

end quadratic_integer_roots_count_l139_13974


namespace special_op_is_addition_l139_13989

/-- A binary operation on real numbers satisfying (a * b) * c = a + b + c -/
def special_op (a b : ℝ) : ℝ := sorry

/-- The property of the special operation -/
axiom special_op_property (a b c : ℝ) : special_op (special_op a b) c = a + b + c

/-- Theorem: The special operation is equivalent to addition -/
theorem special_op_is_addition (a b : ℝ) : special_op a b = a + b := by sorry

end special_op_is_addition_l139_13989


namespace arithmetic_calculation_l139_13983

theorem arithmetic_calculation : 2 + 3 * 4 - 5 / 5 + 7 = 20 := by
  sorry

end arithmetic_calculation_l139_13983


namespace correct_answer_l139_13926

theorem correct_answer (x : ℝ) (h : x / 3 = 27) : x * 3 = 243 := by
  sorry

end correct_answer_l139_13926


namespace club_equation_solution_l139_13938

-- Define the operation ♣
def club (A B : ℝ) : ℝ := 3 * A^2 + 2 * B + 5

-- Theorem statement
theorem club_equation_solution :
  ∃ B : ℝ, club 4 B = 101 ∧ B = 24 := by
  sorry

end club_equation_solution_l139_13938


namespace probability_less_than_one_third_l139_13917

/-- The probability of selecting a number less than 1/3 from the interval (0, 1/2) is 2/3 -/
theorem probability_less_than_one_third : 
  let total_interval : ℝ := 1/2 - 0
  let desired_interval : ℝ := 1/3 - 0
  desired_interval / total_interval = 2/3 := by
sorry

end probability_less_than_one_third_l139_13917


namespace max_a_value_exists_max_a_l139_13928

theorem max_a_value (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 12, x^2 + 25 + |x^3 - 5*x^2| ≥ a*x) → 
  a ≤ 10 :=
by sorry

theorem exists_max_a : 
  ∃ a : ℝ, a = 10 ∧ 
  (∀ x ∈ Set.Icc 1 12, x^2 + 25 + |x^3 - 5*x^2| ≥ a*x) ∧
  ∀ b > a, ∃ x ∈ Set.Icc 1 12, x^2 + 25 + |x^3 - 5*x^2| < b*x :=
by sorry

end max_a_value_exists_max_a_l139_13928


namespace solution_set_inequality_l139_13911

theorem solution_set_inequality (x : ℝ) :
  (((1 - 2*x) / (3*x^2 - 4*x + 7)) ≥ 0) ↔ (x ≤ 1/2) :=
by sorry

end solution_set_inequality_l139_13911


namespace max_value_x2_y2_z3_max_value_achieved_l139_13943

theorem max_value_x2_y2_z3 (x y z : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_constraint : x + 2*y + 3*z = 1) : 
  x^2 + y^2 + z^3 ≤ 1 :=
by sorry

theorem max_value_achieved (x y z : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_constraint : x + 2*y + 3*z = 1) : 
  ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + 2*b + 3*c = 1 ∧ a^2 + b^2 + c^3 = 1 :=
by sorry

end max_value_x2_y2_z3_max_value_achieved_l139_13943


namespace stamps_problem_l139_13940

theorem stamps_problem (A B C D : ℕ) : 
  A + B + C + D = 251 →
  A = 2 * B + 2 →
  A = 3 * C + 6 →
  A = 4 * D - 16 →
  D = 32 := by
sorry

end stamps_problem_l139_13940


namespace dave_total_rides_l139_13919

/-- The number of rides Dave went on the first day -/
def first_day_rides : ℕ := 4

/-- The number of rides Dave went on the second day -/
def second_day_rides : ℕ := 3

/-- The total number of rides Dave went on over two days -/
def total_rides : ℕ := first_day_rides + second_day_rides

theorem dave_total_rides :
  total_rides = 7 := by sorry

end dave_total_rides_l139_13919


namespace math_competition_proof_l139_13908

def math_competition (sammy_score : ℕ) (opponent_score : ℕ) : Prop :=
  let gab_score : ℕ := 2 * sammy_score
  let cher_score : ℕ := 2 * gab_score
  let alex_score : ℕ := cher_score + (cher_score / 10)
  let combined_score : ℕ := sammy_score + gab_score + cher_score + alex_score
  combined_score - opponent_score = 143

theorem math_competition_proof :
  math_competition 20 85 := by sorry

end math_competition_proof_l139_13908


namespace correct_calculation_l139_13947

theorem correct_calculation : (-36 : ℚ) / (-1/2 + 1/6 - 1/3) = 54 := by
  sorry

end correct_calculation_l139_13947


namespace room_height_calculation_l139_13955

/-- Calculates the height of a room given its dimensions, door and window sizes, and whitewashing cost. -/
theorem room_height_calculation (room_length room_width : ℝ)
  (door_length door_width : ℝ)
  (window_length window_width : ℝ)
  (num_windows : ℕ)
  (whitewash_cost_per_sqft : ℝ)
  (total_cost : ℝ)
  (h : room_length = 25 ∧ room_width = 15 ∧ 
       door_length = 6 ∧ door_width = 3 ∧
       window_length = 4 ∧ window_width = 3 ∧
       num_windows = 3 ∧
       whitewash_cost_per_sqft = 5 ∧
       total_cost = 4530) :
  ∃ (room_height : ℝ),
    room_height = 12 ∧
    total_cost = whitewash_cost_per_sqft * 
      (2 * (room_length + room_width) * room_height - 
       (door_length * door_width + num_windows * window_length * window_width)) :=
by sorry

end room_height_calculation_l139_13955


namespace total_clients_l139_13995

/-- Represents the number of clients needing vegan meals -/
def vegan : ℕ := 7

/-- Represents the number of clients needing kosher meals -/
def kosher : ℕ := 8

/-- Represents the number of clients needing both vegan and kosher meals -/
def both : ℕ := 3

/-- Represents the number of clients needing neither vegan nor kosher meals -/
def neither : ℕ := 18

/-- Theorem stating that the total number of clients is 30 -/
theorem total_clients : vegan + kosher - both + neither = 30 := by
  sorry

end total_clients_l139_13995


namespace printer_X_time_l139_13952

/-- The time (in hours) it takes for printer Y to complete the job alone -/
def time_Y : ℝ := 12

/-- The time (in hours) it takes for printer Z to complete the job alone -/
def time_Z : ℝ := 8

/-- The ratio of the time it takes printer X alone to the time it takes printers Y and Z together -/
def ratio : ℝ := 3.333333333333333

theorem printer_X_time : ∃ (time_X : ℝ), time_X = 16 ∧
  ratio = time_X / (1 / (1 / time_Y + 1 / time_Z)) :=
sorry

end printer_X_time_l139_13952


namespace planes_perpendicular_l139_13916

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (intersects : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular
  (α β : Plane) (a b c : Line)
  (h1 : contains α a)
  (h2 : contains α b)
  (h3 : intersects a b)
  (h4 : perpendicular c a)
  (h5 : perpendicular c b)
  (h6 : parallel c β) :
  plane_perpendicular α β :=
sorry

end planes_perpendicular_l139_13916


namespace lcm_of_8_24_36_54_l139_13981

theorem lcm_of_8_24_36_54 : Nat.lcm 8 (Nat.lcm 24 (Nat.lcm 36 54)) = 216 := by sorry

end lcm_of_8_24_36_54_l139_13981


namespace hex_A08_equals_2568_l139_13950

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  if c.isDigit then c.toNat - '0'.toNat
  else if 'A' ≤ c ∧ c ≤ 'F' then c.toNat - 'A'.toNat + 10
  else 0

/-- Converts a hexadecimal string to its decimal value -/
def hex_string_to_dec (s : String) : ℕ :=
  s.foldr (fun c acc => hex_to_dec c + 16 * acc) 0

/-- The hexadecimal representation of the number -/
def hex_number : String := "A08"

/-- Theorem stating that the hexadecimal number A08 is equal to 2568 in decimal -/
theorem hex_A08_equals_2568 : hex_string_to_dec hex_number = 2568 := by
  sorry


end hex_A08_equals_2568_l139_13950


namespace solution_characterization_l139_13935

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(4/3, 4/3, -5/3), (4/3, -5/3, 4/3), (-5/3, 4/3, 4/3),
   (-4/3, -4/3, 5/3), (-4/3, 5/3, -4/3), (5/3, -4/3, -4/3)}

def satisfies_equations (x y z : ℝ) : Prop :=
  x^2 - y*z = |y - z| + 1 ∧
  y^2 - z*x = |z - x| + 1 ∧
  z^2 - x*y = |x - y| + 1

theorem solution_characterization :
  {p : ℝ × ℝ × ℝ | satisfies_equations p.1 p.2.1 p.2.2} = solution_set :=
by sorry

end solution_characterization_l139_13935


namespace lamp_height_difference_example_l139_13990

/-- The height difference between two lamps -/
def lamp_height_difference (new_height old_height : ℝ) : ℝ :=
  new_height - old_height

/-- Theorem: The height difference between a new lamp of 2.33 feet and an old lamp of 1 foot is 1.33 feet -/
theorem lamp_height_difference_example :
  lamp_height_difference 2.33 1 = 1.33 := by
  sorry

end lamp_height_difference_example_l139_13990


namespace five_number_average_l139_13934

theorem five_number_average (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 20 →
  a + b + c = 48 →
  a = 2 * b →
  (d + e) / 2 = 26 := by
sorry

end five_number_average_l139_13934


namespace distinct_roots_sum_bound_l139_13942

theorem distinct_roots_sum_bound (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ → 
  r₁^2 + p*r₁ + 8 = 0 → 
  r₂^2 + p*r₂ + 8 = 0 → 
  |r₁ + r₂| > 4 * Real.sqrt 2 :=
sorry

end distinct_roots_sum_bound_l139_13942


namespace quarter_percent_of_120_l139_13958

theorem quarter_percent_of_120 : (1 / 4 : ℚ) / 100 * 120 = 0.3 := by
  sorry

end quarter_percent_of_120_l139_13958


namespace cylinder_radius_determination_l139_13936

theorem cylinder_radius_determination (z : ℝ) : 
  let original_height : ℝ := 3
  let volume_increase (r : ℝ) : ℝ → ℝ := λ h => π * (r^2 * h - r^2 * original_height)
  ∀ r : ℝ, 
    (volume_increase r (original_height + 4) = z ∧ 
     volume_increase (r + 4) original_height = z) → 
    r = 8 :=
by sorry

end cylinder_radius_determination_l139_13936


namespace path_of_vertex_A_l139_13922

/-- Represents a rectangle in a 2D plane -/
structure Rectangle where
  ab : ℝ
  bc : ℝ

/-- Calculates the path traveled by vertex A of a rectangle when rotated 90° around D and translated -/
def pathTraveledByA (rect : Rectangle) (rotationAngle : ℝ) (translation : ℝ) : ℝ :=
  sorry

/-- Theorem stating the path traveled by vertex A of the specific rectangle -/
theorem path_of_vertex_A :
  let rect : Rectangle := { ab := 3, bc := 5 }
  let rotationAngle : ℝ := π / 2  -- 90° in radians
  let translation : ℝ := 3
  pathTraveledByA rect rotationAngle translation = 2.5 * π + 3 := by
  sorry

end path_of_vertex_A_l139_13922


namespace range_of_m_l139_13931

-- Define the sets M and N
def M (m : ℝ) : Set ℝ := {x | x + m ≥ 0}
def N : Set ℝ := {x | x^2 - 2*x - 8 < 0}

-- Define the universe U as the real numbers
def U : Set ℝ := Set.univ

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (∃ x, x ∈ (U \ M m) ∩ N) → m ≤ 2 := by
  sorry

end range_of_m_l139_13931


namespace parabola_point_focus_distance_l139_13956

/-- A point on a parabola and its distance to the focus -/
theorem parabola_point_focus_distance :
  ∀ (x y : ℝ),
  y^2 = 8*x →  -- Point (x, y) is on the parabola y^2 = 8x
  x = 4 →      -- The x-coordinate of the point is 4
  Real.sqrt ((x - 2)^2 + y^2) = 6 :=  -- The distance to the focus (2, 0) is 6
by sorry

end parabola_point_focus_distance_l139_13956


namespace inequality_solution_l139_13998

-- Define the inequality
def inequality (x a : ℝ) : Prop := x^2 - 2*a*x - 3*a^2 < 0

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  {x | inequality x a}

-- Theorem statement
theorem inequality_solution :
  ∀ a : ℝ,
    (a = 0 → solution_set a = ∅) ∧
    (a > 0 → solution_set a = {x | -a < x ∧ x < 3*a}) ∧
    (a < 0 → solution_set a = {x | 3*a < x ∧ x < -a}) :=
by sorry

end inequality_solution_l139_13998


namespace sum_of_squares_l139_13923

theorem sum_of_squares (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 1) (h5 : a^3 + b^3 + c^3 = a^5 + b^5 + c^5 + 1) :
  a^2 + b^2 + c^2 = 7/5 := by
  sorry

end sum_of_squares_l139_13923


namespace a_range_l139_13951

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - 2*a)^x < (3 - 2*a)^y

def range_of_a (a : ℝ) : Prop := (1 ≤ a ∧ a < 2) ∨ a ≤ -2

theorem a_range (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) : range_of_a a :=
sorry

end a_range_l139_13951


namespace min_boxes_for_load_l139_13980

theorem min_boxes_for_load (total_load : ℝ) (max_box_weight : ℝ) : 
  total_load = 13.5 * 1000 → 
  max_box_weight = 350 → 
  ⌈total_load / max_box_weight⌉ ≥ 39 := by
sorry

end min_boxes_for_load_l139_13980


namespace vector_relations_l139_13977

/-- Two-dimensional vector -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Parallel vectors -/
def parallel (v w : Vec2D) : Prop :=
  v.x * w.y = v.y * w.x

/-- Perpendicular vectors -/
def perpendicular (v w : Vec2D) : Prop :=
  v.x * w.x + v.y * w.y = 0

theorem vector_relations (m : ℝ) :
  let a : Vec2D := ⟨1, 2⟩
  let b : Vec2D := ⟨-2, m⟩
  (parallel a b → m = -4) ∧
  (perpendicular a b → m = 1) := by
  sorry

end vector_relations_l139_13977


namespace ellipse_foci_l139_13954

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1

-- Define the foci coordinates
def foci : Set (ℝ × ℝ) := {(Real.sqrt 7, 0), (-Real.sqrt 7, 0)}

-- Theorem statement
theorem ellipse_foci :
  ∀ (x y : ℝ), ellipse_equation x y →
  ∃ (f : ℝ × ℝ), f ∈ foci ∧
  (x - f.1)^2 + y^2 = (4 + Real.sqrt 7)^2 ∨
  (x - f.1)^2 + y^2 = (4 - Real.sqrt 7)^2 :=
sorry

end ellipse_foci_l139_13954


namespace product_of_roots_l139_13921

theorem product_of_roots (x : ℝ) : 
  (2 * x^3 - 24 * x^2 + 96 * x + 56 = 0) → 
  (∃ r₁ r₂ r₃ : ℝ, (x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ r₁ * r₂ * r₃ = -28) := by
sorry

end product_of_roots_l139_13921


namespace oscar_swag_bag_scarves_l139_13993

/-- Represents the contents and value of an Oscar swag bag -/
structure SwagBag where
  totalValue : ℕ
  earringCost : ℕ
  iphoneCost : ℕ
  scarfCost : ℕ
  numScarves : ℕ

/-- Theorem stating that given the specific costs and total value, 
    the number of scarves in the swag bag is 4 -/
theorem oscar_swag_bag_scarves (bag : SwagBag) 
    (h1 : bag.totalValue = 20000)
    (h2 : bag.earringCost = 6000)
    (h3 : bag.iphoneCost = 2000)
    (h4 : bag.scarfCost = 1500)
    (h5 : bag.totalValue = 2 * bag.earringCost + bag.iphoneCost + bag.numScarves * bag.scarfCost) :
  bag.numScarves = 4 := by
  sorry

#check oscar_swag_bag_scarves

end oscar_swag_bag_scarves_l139_13993


namespace isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l139_13967

/-- An isosceles triangle with side lengths 4 and 9 has a perimeter of 22 -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∀ a b c : ℝ,
      a = 4 ∧ b = 9 ∧ c = 9 →  -- Two sides are 9, one side is 4
      a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
      (a = b ∨ b = c ∨ c = a) →  -- Isosceles condition
      perimeter = a + b + c →  -- Perimeter definition
      perimeter = 22

-- The proof is omitted
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 22 := by sorry

end isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l139_13967


namespace num_distinct_paths_l139_13929

/-- The number of rows in the grid -/
def rows : ℕ := 6

/-- The number of columns in the grid -/
def cols : ℕ := 5

/-- The number of dominoes used -/
def num_dominoes : ℕ := 5

/-- The number of moves to the right required to reach the bottom right corner -/
def moves_right : ℕ := cols - 1

/-- The number of moves down required to reach the bottom right corner -/
def moves_down : ℕ := rows - 1

/-- The total number of moves required to reach the bottom right corner -/
def total_moves : ℕ := moves_right + moves_down

/-- Theorem stating the number of distinct paths from top-left to bottom-right corner -/
theorem num_distinct_paths : (total_moves.choose moves_right) = 126 := by
  sorry

end num_distinct_paths_l139_13929


namespace counterfeit_coin_identification_l139_13963

/-- Represents the result of a weighing -/
inductive WeighingResult
| Equal : WeighingResult
| LeftHeavier : WeighingResult
| RightHeavier : WeighingResult

/-- Represents a coin -/
inductive Coin
| Real : Coin
| Counterfeit : Coin

/-- Represents a set of coins -/
def CoinSet := List Coin

/-- Represents a weighing action -/
def Weighing := CoinSet → CoinSet → WeighingResult

/-- The maximum number of weighings allowed -/
def MaxWeighings : Nat := 4

/-- The number of unknown coins -/
def UnknownCoins : Nat := 12

/-- The number of known real coins -/
def KnownRealCoins : Nat := 5

/-- The number of known counterfeit coins -/
def KnownCounterfeitCoins : Nat := 5

/-- A strategy is a function that takes the current state and returns the next weighing to perform -/
def Strategy := List WeighingResult → Weighing

/-- Determines if a strategy is successful in identifying the number of counterfeit coins -/
def IsSuccessfulStrategy (s : Strategy) : Prop := sorry

/-- The main theorem: There exists a successful strategy to determine the number of counterfeit coins -/
theorem counterfeit_coin_identification :
  ∃ (s : Strategy), IsSuccessfulStrategy s := by sorry

end counterfeit_coin_identification_l139_13963


namespace parabola_vertex_coordinates_l139_13910

/-- The vertex coordinates of the parabola y = x^2 - 4x + 3 are (2, -1) -/
theorem parabola_vertex_coordinates :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x + 3
  ∃ x y : ℝ, (x = 2 ∧ y = -1) ∧
    (∀ t : ℝ, f t ≥ f x) ∧
    (y = f x) :=
by sorry

end parabola_vertex_coordinates_l139_13910


namespace blue_tetrahedron_volume_l139_13900

/-- Represents a cube with alternately colored vertices -/
structure ColoredCube where
  sideLength : ℝ
  vertexColors : Fin 8 → Bool  -- True for blue, False for red

/-- The volume of a tetrahedron formed by four vertices of a cube -/
def tetrahedronVolume (c : ColoredCube) (v1 v2 v3 v4 : Fin 8) : ℝ :=
  sorry

/-- Theorem: The volume of the tetrahedron formed by blue vertices in a cube with side length 8 is 170⅔ -/
theorem blue_tetrahedron_volume (c : ColoredCube) 
  (h1 : c.sideLength = 8)
  (h2 : ∀ i j : Fin 8, i ≠ j → c.vertexColors i ≠ c.vertexColors j) :
  ∃ v1 v2 v3 v4 : Fin 8, 
    (c.vertexColors v1 ∧ c.vertexColors v2 ∧ c.vertexColors v3 ∧ c.vertexColors v4) ∧
    tetrahedronVolume c v1 v2 v3 v4 = 170 + 2/3 :=
  sorry

end blue_tetrahedron_volume_l139_13900


namespace tangent_lines_to_parabola_l139_13930

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 + 4*x + 9

-- Define the point B
def B : ℝ × ℝ := (-1, 2)

-- Define the two lines
def line1 (x : ℝ) : ℝ := -2*x
def line2 (x : ℝ) : ℝ := 6*x + 8

-- Theorem statement
theorem tangent_lines_to_parabola :
  (∃ x₀ : ℝ, line1 x₀ = parabola x₀ ∧ 
    (∀ x : ℝ, x ≠ x₀ → line1 x < parabola x) ∧
    line1 (B.1) = B.2) ∧
  (∃ x₀ : ℝ, line2 x₀ = parabola x₀ ∧ 
    (∀ x : ℝ, x ≠ x₀ → line2 x < parabola x) ∧
    line2 (B.1) = B.2) :=
sorry

end tangent_lines_to_parabola_l139_13930


namespace total_participants_is_260_l139_13987

/-- Represents the voting scenario for a school disco date --/
structure VotingScenario where
  initial_oct22_percent : ℝ
  initial_oct29_percent : ℝ
  additional_oct22_votes : ℕ
  final_oct29_percent : ℝ

/-- Calculates the total number of participants in the voting --/
def total_participants (scenario : VotingScenario) : ℕ :=
  sorry

/-- Theorem stating that the total number of participants is 260 --/
theorem total_participants_is_260 (scenario : VotingScenario) 
  (h1 : scenario.initial_oct22_percent = 0.35)
  (h2 : scenario.initial_oct29_percent = 0.65)
  (h3 : scenario.additional_oct22_votes = 80)
  (h4 : scenario.final_oct29_percent = 0.45) :
  total_participants scenario = 260 := by
  sorry

end total_participants_is_260_l139_13987


namespace polynomial_division_remainder_l139_13905

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (X^5 - 2*X^3 + X - 1) * (X^3 - X + 1) = (X^2 + X + 1) * q + (2*X) :=
by sorry

end polynomial_division_remainder_l139_13905


namespace chickens_and_rabbits_l139_13969

theorem chickens_and_rabbits (x y : ℕ) : 
  (x + y = 35 ∧ 2*x + 4*y = 94) ↔ 
  (x + y = 35 ∧ x * 2 + y * 4 = 94) := by sorry

end chickens_and_rabbits_l139_13969


namespace no_solution_exists_l139_13999

theorem no_solution_exists : ¬ ∃ (a b : ℤ), a^2 = b^15 + 1004 := by
  sorry

end no_solution_exists_l139_13999


namespace sector_properties_l139_13982

/-- Represents a circular sector --/
structure Sector where
  α : Real  -- Central angle in radians
  r : Real  -- Radius
  h_r_pos : r > 0

/-- Calculates the arc length of a sector --/
def arcLength (s : Sector) : Real :=
  s.α * s.r

/-- Calculates the perimeter of a sector --/
def perimeter (s : Sector) : Real :=
  s.r * (s.α + 2)

/-- Calculates the area of a sector --/
def area (s : Sector) : Real :=
  0.5 * s.α * s.r^2

theorem sector_properties :
  ∃ (s1 s2 : Sector),
    s1.α = 2 * Real.pi / 3 ∧
    s1.r = 6 ∧
    arcLength s1 = 4 * Real.pi ∧
    perimeter s2 = 24 ∧
    s2.α = 2 ∧
    area s2 = 36 ∧
    ∀ (s : Sector), perimeter s = 24 → area s ≤ area s2 := by
  sorry

end sector_properties_l139_13982


namespace circle_radius_l139_13964

theorem circle_radius (A : ℝ) (h : A = 196 * Real.pi) : 
  ∃ r : ℝ, r > 0 ∧ A = Real.pi * r^2 ∧ r = 14 := by
  sorry

end circle_radius_l139_13964


namespace product_segment_doubles_when_unit_halved_l139_13976

/-- Theorem: Product segment length doubles when unit segment is halved -/
theorem product_segment_doubles_when_unit_halved 
  (a b e d : ℝ) 
  (h1 : d = a * b / e) 
  (e' : ℝ) 
  (h2 : e' = e / 2) 
  (d' : ℝ) 
  (h3 : d' = a * b / e') : 
  d' = 2 * d := by
sorry

end product_segment_doubles_when_unit_halved_l139_13976


namespace even_quadratic_iff_b_eq_zero_l139_13914

/-- A quadratic function f(x) = ax^2 + bx + c is even if and only if b = 0, given a ≠ 0 -/
theorem even_quadratic_iff_b_eq_zero (a b c : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c = a * (-x)^2 + b * (-x) + c) ↔ b = 0 := by
  sorry

end even_quadratic_iff_b_eq_zero_l139_13914


namespace exists_constant_function_l139_13937

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- State the theorem
theorem exists_constant_function (x : ℝ) : ∃ k : ℝ, 2 * f 3 - 10 = f k ∧ k = 1 := by
  sorry

end exists_constant_function_l139_13937


namespace factorization_equality_l139_13945

theorem factorization_equality (x : ℝ) : 
  75 * x^19 + 165 * x^38 = 15 * x^19 * (5 + 11 * x^19) := by
  sorry

end factorization_equality_l139_13945


namespace total_checks_purchased_l139_13994

/-- Represents the number of travelers checks purchased -/
structure TravelersChecks where
  fifty : ℕ    -- number of $50 checks
  hundred : ℕ  -- number of $100 checks

/-- The total value of all travelers checks -/
def total_value (tc : TravelersChecks) : ℕ :=
  50 * tc.fifty + 100 * tc.hundred

/-- The average value of remaining checks after spending 6 $50 checks -/
def average_remaining (tc : TravelersChecks) : ℚ :=
  (total_value tc - 300) / (tc.fifty + tc.hundred - 6 : ℚ)

/-- Theorem stating the total number of travelers checks purchased -/
theorem total_checks_purchased :
  ∃ (tc : TravelersChecks),
    total_value tc = 1800 ∧
    average_remaining tc = 62.5 ∧
    tc.fifty + tc.hundred = 33 :=
  sorry

end total_checks_purchased_l139_13994


namespace local_min_implies_b_in_open_unit_interval_l139_13979

/-- If f(x) = x^3 - 3bx + b has a local minimum in (0, 1), then b ∈ (0, 1) -/
theorem local_min_implies_b_in_open_unit_interval (b : ℝ) : 
  (∃ c ∈ Set.Ioo 0 1, IsLocalMin (fun x => x^3 - 3*b*x + b) c) → 
  b ∈ Set.Ioo 0 1 := by
sorry

end local_min_implies_b_in_open_unit_interval_l139_13979


namespace intersection_A_B_complement_B_union_A_complementB_l139_13913

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 2}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- Define the complement of B
def complementB : Set ℝ := {x | ¬ (x ∈ B)}

-- Theorem statements
theorem intersection_A_B : A ∩ B = {x | -2 < x ∧ x < -1} := by sorry

theorem complement_B : complementB = {x | -1 ≤ x ∧ x ≤ 4} := by sorry

theorem union_A_complementB : A ∪ complementB = {x | -2 < x ∧ x ≤ 4} := by sorry

end intersection_A_B_complement_B_union_A_complementB_l139_13913
