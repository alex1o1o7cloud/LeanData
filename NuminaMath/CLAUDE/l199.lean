import Mathlib

namespace pancake_theorem_l199_19995

/-- The fraction of pancakes that could be flipped -/
def flipped_fraction : ℚ := 4 / 5

/-- The fraction of flipped pancakes that didn't burn -/
def not_burnt_fraction : ℚ := 51 / 100

/-- The fraction of edible pancakes that weren't dropped -/
def not_dropped_fraction : ℚ := 5 / 6

/-- The percentage of pancakes Anya could offer her family -/
def offered_percentage : ℚ := flipped_fraction * not_burnt_fraction * not_dropped_fraction * 100

theorem pancake_theorem : 
  ∃ (ε : ℚ), abs (offered_percentage - 34) < ε ∧ ε > 0 := by
  sorry

end pancake_theorem_l199_19995


namespace five_digit_divisible_by_6_l199_19966

def is_divisible_by_6 (n : Nat) : Prop :=
  ∃ k : Nat, 7123 * 10 + n = 6 * k

theorem five_digit_divisible_by_6 :
  ∀ n : Nat, n < 10 →
    (is_divisible_by_6 n ↔ (n = 2 ∨ n = 8)) :=
by sorry

end five_digit_divisible_by_6_l199_19966


namespace pulley_centers_distance_l199_19940

/-- The distance between the centers of two circular pulleys with an uncrossed belt -/
theorem pulley_centers_distance (r₁ r₂ d : ℝ) (hr₁ : r₁ = 10) (hr₂ : r₂ = 6) (hd : d = 30) :
  Real.sqrt ((r₁ - r₂)^2 + d^2) = 2 * Real.sqrt 229 := by
  sorry

end pulley_centers_distance_l199_19940


namespace notebook_length_l199_19943

/-- Given a rectangular notebook with area 1.77 cm² and width 3 cm, prove its length is 0.59 cm -/
theorem notebook_length (area : ℝ) (width : ℝ) (length : ℝ) :
  area = 1.77 ∧ width = 3 ∧ area = length * width → length = 0.59 := by
  sorry

end notebook_length_l199_19943


namespace no_consecutive_beeches_probability_l199_19951

/-- The number of oaks to be planted -/
def num_oaks : ℕ := 3

/-- The number of holm oaks to be planted -/
def num_holm_oaks : ℕ := 4

/-- The number of beeches to be planted -/
def num_beeches : ℕ := 5

/-- The total number of trees to be planted -/
def total_trees : ℕ := num_oaks + num_holm_oaks + num_beeches

/-- The probability of no two beeches being consecutive when planted randomly -/
def prob_no_consecutive_beeches : ℚ := 7 / 99

theorem no_consecutive_beeches_probability :
  let total_arrangements := (total_trees.factorial) / (num_oaks.factorial * num_holm_oaks.factorial * num_beeches.factorial)
  let favorable_arrangements := (Nat.choose 8 5) * ((num_oaks + num_holm_oaks).factorial / (num_oaks.factorial * num_holm_oaks.factorial))
  (favorable_arrangements : ℚ) / total_arrangements = prob_no_consecutive_beeches := by
  sorry

end no_consecutive_beeches_probability_l199_19951


namespace intersection_line_slope_is_one_third_l199_19904

/-- The equation of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 5 = 0

/-- The equation of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10*x + 16*y + 24 = 0

/-- The slope of the line passing through the intersection points of two circles -/
def intersectionLineSlope (c1 c2 : (ℝ → ℝ → Prop)) : ℝ := sorry

theorem intersection_line_slope_is_one_third :
  intersectionLineSlope circle1 circle2 = 1/3 := by sorry

end intersection_line_slope_is_one_third_l199_19904


namespace committee_selection_l199_19935

theorem committee_selection (n : ℕ) (h : Nat.choose n 3 = 15) : Nat.choose n 4 = 15 := by
  sorry

end committee_selection_l199_19935


namespace quadratic_root_relation_l199_19924

theorem quadratic_root_relation (a c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ y = 3 * x ∧ a * x^2 + 6 * x + c = 0 ∧ a * y^2 + 6 * y + c = 0) →
  c = 27 / (4 * a) := by
sorry

end quadratic_root_relation_l199_19924


namespace number_of_players_is_16_l199_19914

def jersey_cost : ℚ := 25
def shorts_cost : ℚ := 15.20
def socks_cost : ℚ := 6.80
def total_cost : ℚ := 752

def equipment_cost_per_player : ℚ := jersey_cost + shorts_cost + socks_cost

theorem number_of_players_is_16 :
  (total_cost / equipment_cost_per_player : ℚ) = 16 := by sorry

end number_of_players_is_16_l199_19914


namespace D_72_l199_19950

/-- D(n) represents the number of ways of expressing the positive integer n 
    as a product of integers greater than 1, where the order matters. -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem stating that D(72) is equal to 103 -/
theorem D_72 : D 72 = 103 := by sorry

end D_72_l199_19950


namespace divisible_by_five_l199_19953

theorem divisible_by_five (B : ℕ) : 
  B < 10 → (5270 + B) % 5 = 0 ↔ B = 0 ∨ B = 5 := by sorry

end divisible_by_five_l199_19953


namespace odd_integer_not_divides_power_plus_one_l199_19910

theorem odd_integer_not_divides_power_plus_one (n m : ℕ) : 
  n > 1 → Odd n → m ≥ 1 → ¬(n ∣ m^(n-1) + 1) := by
  sorry

end odd_integer_not_divides_power_plus_one_l199_19910


namespace student_age_l199_19934

theorem student_age (student_age man_age : ℕ) : 
  man_age = student_age + 26 →
  man_age + 2 = 2 * (student_age + 2) →
  student_age = 24 := by
sorry

end student_age_l199_19934


namespace smallest_integer_with_remainders_l199_19988

theorem smallest_integer_with_remainders : ∃! x : ℕ+, 
  (x : ℕ) % 4 = 3 ∧ 
  (x : ℕ) % 3 = 2 ∧ 
  ∀ y : ℕ+, y < x → (y : ℕ) % 4 ≠ 3 ∨ (y : ℕ) % 3 ≠ 2 :=
by sorry

end smallest_integer_with_remainders_l199_19988


namespace g_of_neg_three_eq_eight_l199_19922

/-- Given functions f and g, prove that g(-3) = 8 -/
theorem g_of_neg_three_eq_eight
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x, f x = 4 * x - 7)
  (hg : ∀ x, g (f x) = 3 * x^2 + 4 * x + 1) :
  g (-3) = 8 := by
sorry

end g_of_neg_three_eq_eight_l199_19922


namespace min_cosine_sine_fraction_l199_19905

open Real

theorem min_cosine_sine_fraction (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  (cos x)^3 / sin x + (sin x)^3 / cos x ≥ 1 ∧
  ∃ y, 0 < y ∧ y < π / 2 ∧ (cos y)^3 / sin y + (sin y)^3 / cos y = 1 :=
by sorry

end min_cosine_sine_fraction_l199_19905


namespace employed_males_percentage_l199_19907

/-- The percentage of employed people in the population -/
def employed_percentage : ℝ := 64

/-- The percentage of employed people who are female -/
def female_employed_percentage : ℝ := 25

/-- The theorem stating the percentage of the population that are employed males -/
theorem employed_males_percentage :
  (employed_percentage / 100) * (1 - female_employed_percentage / 100) * 100 = 48 := by
  sorry

end employed_males_percentage_l199_19907


namespace crayon_selection_l199_19997

theorem crayon_selection (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 5) :
  (Nat.choose (n - 1) (k - 1)) = 1001 := by
  sorry

end crayon_selection_l199_19997


namespace stating_two_thousandth_hit_on_second_string_l199_19969

/-- Represents the number of strings on the guitar. -/
def num_strings : ℕ := 6

/-- Represents the total number of hits we're interested in. -/
def total_hits : ℕ := 2000

/-- 
Represents the string number for a given hit in the sequence.
n: The hit number
-/
def string_number (n : ℕ) : ℕ :=
  let cycle_length := 2 * num_strings - 2
  let position_in_cycle := n % cycle_length
  if position_in_cycle ≤ num_strings
  then position_in_cycle
  else 2 * num_strings - position_in_cycle

/-- 
Theorem stating that the 2000th hit lands on string number 2.
-/
theorem two_thousandth_hit_on_second_string : 
  string_number total_hits = 2 := by
  sorry

end stating_two_thousandth_hit_on_second_string_l199_19969


namespace smallest_debate_club_size_l199_19974

/-- Represents the number of students in each grade --/
structure GradeCount where
  eighth : ℕ
  sixth : ℕ
  seventh : ℕ
  ninth : ℕ

/-- Checks if the given counts satisfy the ratio conditions --/
def satisfiesRatios (counts : GradeCount) : Prop :=
  7 * counts.sixth = 4 * counts.eighth ∧
  6 * counts.seventh = 5 * counts.eighth ∧
  9 * counts.ninth = 2 * counts.eighth

/-- Calculates the total number of students --/
def totalStudents (counts : GradeCount) : ℕ :=
  counts.eighth + counts.sixth + counts.seventh + counts.ninth

/-- Theorem stating that the smallest number of students satisfying the ratios is 331 --/
theorem smallest_debate_club_size :
  ∀ counts : GradeCount,
    satisfiesRatios counts →
    totalStudents counts ≥ 331 ∧
    ∃ counts' : GradeCount, satisfiesRatios counts' ∧ totalStudents counts' = 331 :=
by sorry

end smallest_debate_club_size_l199_19974


namespace tan_value_from_trig_equation_l199_19965

theorem tan_value_from_trig_equation (α : ℝ) 
  (h : (Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α) = 5/16) :
  Real.tan α = -1/3 := by sorry

end tan_value_from_trig_equation_l199_19965


namespace no_solution_equation_l199_19925

theorem no_solution_equation :
  ∀ (x : ℝ), x^2 + x ≠ 0 ∧ x + 1 ≠ 0 →
  (5*x + 2) / (x^2 + x) ≠ 3 / (x + 1) :=
by
  sorry

end no_solution_equation_l199_19925


namespace product_increase_theorem_l199_19939

theorem product_increase_theorem :
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ),
    (a₁ - 3) * (a₂ - 3) * (a₃ - 3) * (a₄ - 3) * (a₅ - 3) * (a₆ - 3) * (a₇ - 3) =
    13 * (a₁ * a₂ * a₃ * a₄ * a₅ * a₆ * a₇) :=
by
  sorry

end product_increase_theorem_l199_19939


namespace triangle_right_angled_l199_19978

theorem triangle_right_angled (A B C : ℝ) (h : A - C = B) : A = 90 := by
  sorry

end triangle_right_angled_l199_19978


namespace locus_of_midpoints_l199_19906

-- Define the circles and their properties
def circle1_radius : ℝ := 1
def circle2_radius : ℝ := 3
def distance_between_centers : ℝ := 10

-- Define the locus
def locus_inner_radius : ℝ := 1
def locus_outer_radius : ℝ := 2

-- Theorem statement
theorem locus_of_midpoints (p : ℝ × ℝ) : 
  (∃ (p1 p2 : ℝ × ℝ), 
    (p1.1 - 0)^2 + (p1.2 - 0)^2 = circle1_radius^2 ∧ 
    (p2.1 - distance_between_centers)^2 + p2.2^2 = circle2_radius^2 ∧
    p = ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)) ↔ 
  (locus_inner_radius^2 ≤ (p.1 - distance_between_centers / 2)^2 + p.2^2 ∧ 
   (p.1 - distance_between_centers / 2)^2 + p.2^2 ≤ locus_outer_radius^2) :=
sorry

end locus_of_midpoints_l199_19906


namespace opposite_numbers_iff_differ_in_sign_l199_19942

/-- Two real numbers are opposite if and only if they differ only in their sign -/
theorem opposite_numbers_iff_differ_in_sign (a b : ℝ) : 
  (a = -b) ↔ (abs a = abs b) := by sorry

end opposite_numbers_iff_differ_in_sign_l199_19942


namespace smallest_n_satisfying_property_l199_19912

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The property given in the problem -/
def property (n : ℕ) : Prop :=
  binomial (n + 1) 7 - binomial n 7 = binomial n 8

/-- The theorem to be proved -/
theorem smallest_n_satisfying_property : 
  ∀ n : ℕ, n > 0 → (property n ↔ n ≥ 14) :=
sorry

end smallest_n_satisfying_property_l199_19912


namespace cubic_inequality_solution_l199_19923

theorem cubic_inequality_solution (x : ℝ) :
  x^3 - 10*x^2 + 15*x > 0 ↔ x ∈ Set.Ioo 0 (5 - Real.sqrt 10) ∪ Set.Ioi (5 + Real.sqrt 10) :=
by sorry

end cubic_inequality_solution_l199_19923


namespace last_digit_2008_2005_l199_19917

theorem last_digit_2008_2005 : (2008^2005) % 10 = 8 := by
  sorry

end last_digit_2008_2005_l199_19917


namespace seventh_power_sum_l199_19992

theorem seventh_power_sum (α₁ α₂ α₃ : ℂ) 
  (h1 : α₁ + α₂ + α₃ = 2)
  (h2 : α₁^2 + α₂^2 + α₃^2 = 6)
  (h3 : α₁^3 + α₂^3 + α₃^3 = 14) :
  α₁^7 + α₂^7 + α₃^7 = 478 := by
  sorry

end seventh_power_sum_l199_19992


namespace vector_expression_in_quadrilateral_l199_19990

/-- Given a quadrilateral OABC in space, prove that MN = -1/2 * a + 1/2 * b + 1/2 * c -/
theorem vector_expression_in_quadrilateral
  (O A B C M N : EuclideanSpace ℝ (Fin 3))
  (a b c : EuclideanSpace ℝ (Fin 3))
  (h1 : A - O = a)
  (h2 : B - O = b)
  (h3 : C - O = c)
  (h4 : M - O = (1/2) • (A - O))
  (h5 : N - B = (1/2) • (C - B)) :
  N - M = (-1/2) • a + (1/2) • b + (1/2) • c := by
  sorry

end vector_expression_in_quadrilateral_l199_19990


namespace correct_proposition_l199_19977

theorem correct_proposition :
  (∀ a b : ℝ, a > |b| → a^2 > b^2) ∧
  (∃ a b c : ℝ, a > b ∧ ¬(a*c^2 > b*c^2)) ∧
  (∃ a b : ℝ, a > b ∧ ¬(a^2 > b^2)) ∧
  (∃ a b : ℝ, |a| > b ∧ ¬(a^2 > b^2)) :=
by sorry

end correct_proposition_l199_19977


namespace cone_sphere_ratio_angle_and_k_permissible_k_values_l199_19967

-- Define the cone and sphere
structure ConeWithInscribedSphere where
  R : ℝ  -- radius of the cone's base
  α : ℝ  -- angle between slant height and base plane
  k : ℝ  -- ratio of cone volume to sphere volume

-- Define the theorem
theorem cone_sphere_ratio_angle_and_k (c : ConeWithInscribedSphere) :
  c.k ≥ 2 →
  c.α = 2 * Real.arctan (Real.sqrt ((c.k + Real.sqrt (c.k^2 - 2*c.k)) / (2*c.k))) ∨
  c.α = 2 * Real.arctan (Real.sqrt ((c.k - Real.sqrt (c.k^2 - 2*c.k)) / (2*c.k))) :=
by sorry

-- Define the permissible values of k
theorem permissible_k_values (c : ConeWithInscribedSphere) :
  c.k ≥ 2 :=
by sorry

end cone_sphere_ratio_angle_and_k_permissible_k_values_l199_19967


namespace alcohol_solution_volume_l199_19980

/-- Given an initial solution with volume V and 5% alcohol concentration,
    adding 5.5 liters of alcohol and 4.5 liters of water results in
    a new solution with 15% alcohol concentration if and only if
    the initial volume V is 40 liters. -/
theorem alcohol_solution_volume (V : ℝ) : 
  (0.15 * (V + 10) = 0.05 * V + 5.5) ↔ V = 40 := by sorry

end alcohol_solution_volume_l199_19980


namespace geometric_sequence_sum_l199_19941

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  (∀ n, a (n + 1) = a n * q) →
  a 3 + a 4 + a 5 = 84 := by
sorry

end geometric_sequence_sum_l199_19941


namespace buddy_fraction_l199_19926

theorem buddy_fraction (s₆ : ℕ) (n₉ : ℕ) : 
  s₆ > 0 ∧ n₉ > 0 →  -- Ensure positive numbers of students
  (n₉ : ℚ) / 4 = (s₆ : ℚ) / 3 →  -- 1/4 of ninth graders paired with 1/3 of sixth graders
  (s₆ : ℚ) / 3 / ((4 * s₆ : ℚ) / 3 + s₆) = 1 / 7 :=
by sorry

#check buddy_fraction

end buddy_fraction_l199_19926


namespace race_outcomes_five_participants_l199_19911

/-- The number of different 1st-2nd-3rd place outcomes in a race -/
def raceOutcomes (n : ℕ) : ℕ :=
  if n < 3 then 0 else (n - 1) * (n - 2)

/-- Theorem: In a race with 5 participants where one must finish first and there are no ties,
    the number of different 1st-2nd-3rd place outcomes is 12. -/
theorem race_outcomes_five_participants :
  raceOutcomes 5 = 12 := by
  sorry

end race_outcomes_five_participants_l199_19911


namespace planes_parallel_if_perpendicular_to_same_line_l199_19972

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (a : Line) (α β : Plane) :
  perpendicular a α → perpendicular a β → parallel α β :=
sorry

end planes_parallel_if_perpendicular_to_same_line_l199_19972


namespace jellybeans_left_in_jar_l199_19948

theorem jellybeans_left_in_jar
  (total_jellybeans : ℕ)
  (total_kids : ℕ)
  (absent_kids : ℕ)
  (jellybeans_per_kid : ℕ)
  (h1 : total_jellybeans = 100)
  (h2 : total_kids = 24)
  (h3 : absent_kids = 2)
  (h4 : jellybeans_per_kid = 3) :
  total_jellybeans - (total_kids - absent_kids) * jellybeans_per_kid = 34 :=
by
  sorry


end jellybeans_left_in_jar_l199_19948


namespace max_sum_cyclic_fraction_l199_19932

open Real BigOperators

/-- The maximum value of the sum for positive real numbers with sum 1 -/
theorem max_sum_cyclic_fraction (n : ℕ) (a : ℕ → ℝ) 
  (hn : n ≥ 4)
  (ha_pos : ∀ k, a k > 0)
  (ha_sum : ∑ k in Finset.range n, a k = 1) :
  (∑ k in Finset.range n, (a k)^2 / (a k + a ((k + 1) % n) + a ((k + 2) % n))) ≤ 1/3 :=
sorry


end max_sum_cyclic_fraction_l199_19932


namespace derivative_f_at_3_l199_19985

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem derivative_f_at_3 : 
  deriv f 3 = 6 := by sorry

end derivative_f_at_3_l199_19985


namespace on_time_speed_l199_19913

-- Define the variables
def distance : ℝ → ℝ → ℝ := λ speed time => speed * time

-- Define the conditions
def early_arrival (d : ℝ) (T : ℝ) : Prop := distance 20 (T - 0.5) = d
def late_arrival (d : ℝ) (T : ℝ) : Prop := distance 12 (T + 0.5) = d

-- Define the theorem
theorem on_time_speed (d : ℝ) (T : ℝ) :
  early_arrival d T → late_arrival d T → distance 15 T = d :=
by sorry

end on_time_speed_l199_19913


namespace coursework_materials_theorem_l199_19979

def total_budget : ℝ := 1000

def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.25

def coursework_materials_spending : ℝ := 
  total_budget * (1 - (food_percentage + accommodation_percentage + entertainment_percentage))

theorem coursework_materials_theorem : 
  coursework_materials_spending = 300 := by sorry

end coursework_materials_theorem_l199_19979


namespace no_real_roots_quadratic_l199_19938

theorem no_real_roots_quadratic (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + (a - 5) * x + 2 ≠ 0) → 1 < a ∧ a < 9 := by
  sorry

end no_real_roots_quadratic_l199_19938


namespace parallelogram_existence_l199_19945

/-- Represents a cell in the table -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents a table with marked cells -/
structure Table where
  size : Nat
  markedCells : Finset Cell

/-- Represents a parallelogram in the table -/
structure Parallelogram where
  v1 : Cell
  v2 : Cell
  v3 : Cell
  v4 : Cell

/-- Checks if a cell is within the table bounds -/
def Cell.isValid (c : Cell) (n : Nat) : Prop :=
  c.row < n ∧ c.col < n

/-- Checks if a parallelogram is valid (all vertices are marked and form a parallelogram) -/
def Parallelogram.isValid (p : Parallelogram) (t : Table) : Prop :=
  p.v1 ∈ t.markedCells ∧ p.v2 ∈ t.markedCells ∧ p.v3 ∈ t.markedCells ∧ p.v4 ∈ t.markedCells ∧
  (p.v1.row - p.v2.row = p.v4.row - p.v3.row) ∧
  (p.v1.col - p.v2.col = p.v4.col - p.v3.col)

/-- Main theorem: In an n × n table with 2n marked cells, there exists a valid parallelogram -/
theorem parallelogram_existence (t : Table) (h1 : t.markedCells.card = 2 * t.size) :
  ∃ p : Parallelogram, p.isValid t :=
sorry

end parallelogram_existence_l199_19945


namespace intersection_point_k_value_l199_19962

/-- Given two lines that intersect at x = -12, prove that k = 65 -/
theorem intersection_point_k_value :
  ∀ (y : ℝ),
  -3 * (-12) + y = k →
  0.75 * (-12) + y = 20 →
  k = 65 := by
sorry

end intersection_point_k_value_l199_19962


namespace octagon_diagonals_l199_19996

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_sides = 20 := by
  sorry

end octagon_diagonals_l199_19996


namespace three_lines_plane_count_l199_19946

/-- Represents a line in 3D space -/
structure Line3D where
  -- We don't need to define the internal structure of a line for this problem

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- We don't need to define the internal structure of a plane for this problem

/-- Predicate to check if a line intersects another line -/
def intersects (l1 l2 : Line3D) : Prop :=
  sorry

/-- Function to determine the number of planes formed by three lines -/
def num_planes_formed (l1 l2 l3 : Line3D) : ℕ :=
  sorry

/-- Theorem stating that three lines, where one intersects the other two,
    can form either 1, 2, or 3 planes -/
theorem three_lines_plane_count 
  (l1 l2 l3 : Line3D) 
  (h1 : intersects l1 l2) 
  (h2 : intersects l1 l3) : 
  let n := num_planes_formed l1 l2 l3
  n = 1 ∨ n = 2 ∨ n = 3 :=
by sorry

end three_lines_plane_count_l199_19946


namespace volume_maximized_at_10cm_l199_19991

/-- The volume of a lidless container made from a rectangular sheet -/
def containerVolume (sheetLength sheetWidth height : ℝ) : ℝ :=
  (sheetLength - 2 * height) * (sheetWidth - 2 * height) * height

/-- The statement that the volume is maximized at a specific height -/
theorem volume_maximized_at_10cm (sheetLength sheetWidth : ℝ) 
  (hLength : sheetLength = 90) 
  (hWidth : sheetWidth = 48) :
  ∃ (maxHeight : ℝ), maxHeight = 10 ∧ 
  ∀ (h : ℝ), 0 < h → h < 24 → 
  containerVolume sheetLength sheetWidth h ≤ containerVolume sheetLength sheetWidth maxHeight :=
sorry

end volume_maximized_at_10cm_l199_19991


namespace multiple_condition_l199_19975

theorem multiple_condition (n : ℕ+) : 
  (∃ k : ℕ, 3^n.val + 5^n.val = k * (3^(n.val - 1) + 5^(n.val - 1))) ↔ n = 1 := by
  sorry

end multiple_condition_l199_19975


namespace absolute_value_equation_l199_19956

theorem absolute_value_equation (x : ℝ) : 
  |2*x - 1| = Real.sqrt 2 - 1 → x = Real.sqrt 2 / 2 ∨ x = (2 - Real.sqrt 2) / 2 := by
  sorry

end absolute_value_equation_l199_19956


namespace f_has_root_in_interval_l199_19976

-- Define the function f(x) = x^3 - 3x - 3
def f (x : ℝ) : ℝ := x^3 - 3*x - 3

-- State the theorem
theorem f_has_root_in_interval : 
  ∃ x ∈ Set.Ioo 2 3, f x = 0 := by
  sorry

end f_has_root_in_interval_l199_19976


namespace jose_investment_is_45000_l199_19987

/-- Represents the investment problem with Tom and Jose --/
structure InvestmentProblem where
  tom_investment : ℕ
  jose_join_delay : ℕ
  total_profit : ℕ
  jose_profit_share : ℕ

/-- Calculates Jose's investment amount based on the given parameters --/
def calculate_jose_investment (problem : InvestmentProblem) : ℕ :=
  let tom_investment_months : ℕ := problem.tom_investment * 12
  let jose_investment_months : ℕ := (12 - problem.jose_join_delay) * (problem.jose_profit_share * tom_investment_months) / (problem.total_profit - problem.jose_profit_share)
  jose_investment_months / (12 - problem.jose_join_delay)

/-- Theorem stating that Jose's investment is 45000 given the problem conditions --/
theorem jose_investment_is_45000 (problem : InvestmentProblem) 
  (h1 : problem.tom_investment = 30000)
  (h2 : problem.jose_join_delay = 2)
  (h3 : problem.total_profit = 54000)
  (h4 : problem.jose_profit_share = 30000) :
  calculate_jose_investment problem = 45000 := by
  sorry

end jose_investment_is_45000_l199_19987


namespace function_transformation_l199_19984

theorem function_transformation (f : ℝ → ℝ) : 
  (∀ x, f (x - 1) = 19 * x^2 + 55 * x - 44) → 
  (∀ x, f x = 19 * x^2 + 93 * x + 30) :=
by sorry

end function_transformation_l199_19984


namespace base_conversion_theorem_l199_19919

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun digit acc => digit + base * acc) 0

theorem base_conversion_theorem :
  let base := 5
  let T := 0
  let P := 1
  let Q := 2
  let R := 3
  let S := 4
  let dividend_base5 := [P, Q, R, S, R, Q, P]
  let divisor_base5 := [Q, R, Q]
  (to_decimal dividend_base5 base = 24336) ∧
  (to_decimal divisor_base5 base = 67) := by
  sorry

end base_conversion_theorem_l199_19919


namespace farm_tax_calculation_l199_19999

/-- Represents the farm tax calculation for a village and an individual landowner. -/
theorem farm_tax_calculation 
  (total_tax : ℝ) 
  (individual_land_ratio : ℝ) 
  (h1 : total_tax = 3840) 
  (h2 : individual_land_ratio = 0.5) : 
  individual_land_ratio * total_tax = 1920 := by
  sorry


end farm_tax_calculation_l199_19999


namespace running_time_ratio_l199_19954

theorem running_time_ratio :
  ∀ (danny_time steve_time : ℝ),
    danny_time = 27 →
    steve_time / 2 = danny_time / 2 + 13.5 →
    danny_time / steve_time = 1 / 2 :=
by
  sorry

end running_time_ratio_l199_19954


namespace sum_and_count_theorem_l199_19947

def sum_of_range (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_in_range (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_theorem :
  sum_of_range 30 50 + count_even_in_range 30 50 = 851 := by
  sorry

end sum_and_count_theorem_l199_19947


namespace seventh_twenty_ninth_725th_digit_l199_19900

def decimal_representation (n d : ℕ) : List ℕ := sorry

theorem seventh_twenty_ninth_725th_digit :
  let rep := decimal_representation 7 29
  -- The decimal representation has a period of 28 digits
  ∀ i, rep.get? i = rep.get? (i + 28)
  -- The 725th digit is 6
  → rep.get? 724 = some 6 := by
  sorry

end seventh_twenty_ninth_725th_digit_l199_19900


namespace hill_distance_l199_19982

theorem hill_distance (speed_up speed_down : ℝ) (total_time : ℝ) 
  (h1 : speed_up = 1.5)
  (h2 : speed_down = 4.5)
  (h3 : total_time = 6) :
  ∃ d : ℝ, d = 6.75 ∧ d / speed_up + d / speed_down = total_time :=
sorry

end hill_distance_l199_19982


namespace minimal_discs_to_separate_l199_19901

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a disc in a plane -/
structure Disc where
  center : Point
  radius : ℝ

/-- A function that checks if a disc separates two points -/
def separates (d : Disc) (p1 p2 : Point) : Prop :=
  (((p1.x - d.center.x)^2 + (p1.y - d.center.y)^2 < d.radius^2) ∧
   ((p2.x - d.center.x)^2 + (p2.y - d.center.y)^2 > d.radius^2)) ∨
  (((p1.x - d.center.x)^2 + (p1.y - d.center.y)^2 > d.radius^2) ∧
   ((p2.x - d.center.x)^2 + (p2.y - d.center.y)^2 < d.radius^2))

/-- The main theorem stating the minimal number of discs needed -/
theorem minimal_discs_to_separate (points : Finset Point) 
  (h : points.card = 2019) :
  ∃ (discs : Finset Disc), discs.card = 1010 ∧
    ∀ p1 p2 : Point, p1 ∈ points → p2 ∈ points → p1 ≠ p2 →
      ∃ d ∈ discs, separates d p1 p2 :=
sorry

end minimal_discs_to_separate_l199_19901


namespace number_exceeds_fraction_l199_19930

theorem number_exceeds_fraction : ∃ x : ℚ, x = (3/8) * x + 30 ∧ x = 48 := by
  sorry

end number_exceeds_fraction_l199_19930


namespace car_travel_distance_l199_19998

/-- Proves that Car X travels 98 miles from when Car Y starts until both cars stop -/
theorem car_travel_distance (speed_x speed_y : ℝ) (head_start_time : ℝ) : 
  speed_x = 35 →
  speed_y = 50 →
  head_start_time = 72 / 60 →
  ∃ (travel_time : ℝ), 
    travel_time > 0 ∧
    speed_x * (head_start_time + travel_time) = speed_y * travel_time ∧
    speed_x * travel_time = 98 := by
  sorry

end car_travel_distance_l199_19998


namespace sticker_distribution_l199_19970

/-- The number of ways to partition n identical objects into at most k parts -/
def partitions (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 30 ways to partition 10 identical objects into at most 5 parts -/
theorem sticker_distribution : partitions 10 5 = 30 := by sorry

end sticker_distribution_l199_19970


namespace batsman_performance_theorem_l199_19921

/-- Represents a batsman's performance in a cricket tournament -/
structure BatsmanPerformance where
  innings : ℕ
  runsBeforeLastInning : ℕ
  runsInLastInning : ℕ
  averageIncrease : ℚ
  boundariesBeforeLastInning : ℕ
  boundariesInLastInning : ℕ

/-- Calculates the batting average after the last inning -/
def battingAverage (performance : BatsmanPerformance) : ℚ :=
  (performance.runsBeforeLastInning + performance.runsInLastInning) / performance.innings

/-- Calculates the batting efficiency factor -/
def battingEfficiencyFactor (performance : BatsmanPerformance) : ℚ :=
  (performance.boundariesBeforeLastInning + performance.boundariesInLastInning) / performance.innings

theorem batsman_performance_theorem (performance : BatsmanPerformance) 
  (h1 : performance.innings = 17)
  (h2 : performance.runsInLastInning = 84)
  (h3 : performance.averageIncrease = 5/2)
  (h4 : performance.boundariesInLastInning = 12)
  (h5 : performance.boundariesBeforeLastInning + performance.boundariesInLastInning = 72) :
  battingAverage performance = 44 ∧ battingEfficiencyFactor performance = 72/17 := by
  sorry

#eval (72 : ℚ) / 17

end batsman_performance_theorem_l199_19921


namespace paving_stone_width_l199_19928

/-- Given a rectangular courtyard and paving stones with specified dimensions,
    prove that the width of each paving stone is 2 meters. -/
theorem paving_stone_width
  (courtyard_length : ℝ)
  (courtyard_width : ℝ)
  (stone_length : ℝ)
  (num_stones : ℕ)
  (h1 : courtyard_length = 70)
  (h2 : courtyard_width = 16.5)
  (h3 : stone_length = 2.5)
  (h4 : num_stones = 231)
  : ∃ (stone_width : ℝ),
    stone_width = 2 ∧
    courtyard_length * courtyard_width = (stone_length * stone_width) * num_stones :=
by sorry

end paving_stone_width_l199_19928


namespace min_value_abc_l199_19959

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1 / 2) :
  ∃ (min : ℝ), min = 18 ∧ ∀ x y z, x > 0 → y > 0 → z > 0 → x * y * z = 1 / 2 →
    x^2 + 4*x*y + 12*y^2 + 8*y*z + 3*z^2 ≥ min :=
by sorry

end min_value_abc_l199_19959


namespace parabola_decreasing_range_l199_19949

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem parabola_decreasing_range :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → (f x₁ > f x₂ ↔ x₁ < 1 ∧ x₂ < 1) :=
by sorry

end parabola_decreasing_range_l199_19949


namespace antibiotics_cost_l199_19936

/-- The cost of Antibiotic A per dose in dollars -/
def cost_A : ℚ := 3

/-- The number of doses of Antibiotic A per day -/
def doses_per_day_A : ℕ := 2

/-- The number of days Antibiotic A is taken per week -/
def days_per_week_A : ℕ := 3

/-- The cost of Antibiotic B per dose in dollars -/
def cost_B : ℚ := 9/2

/-- The number of doses of Antibiotic B per day -/
def doses_per_day_B : ℕ := 1

/-- The number of days Antibiotic B is taken per week -/
def days_per_week_B : ℕ := 4

/-- The total cost of antibiotics for Archie for one week -/
def total_cost : ℚ := cost_A * doses_per_day_A * days_per_week_A + cost_B * doses_per_day_B * days_per_week_B

theorem antibiotics_cost : total_cost = 36 := by
  sorry

end antibiotics_cost_l199_19936


namespace definite_integral_2x_minus_3x_squared_l199_19971

theorem definite_integral_2x_minus_3x_squared : 
  ∫ x in (0 : ℝ)..2, (2 * x - 3 * x^2) = -4 := by sorry

end definite_integral_2x_minus_3x_squared_l199_19971


namespace family_gathering_arrangements_l199_19909

theorem family_gathering_arrangements (n : ℕ) (h : n = 6) : 
  Nat.choose n (n / 2) = 20 := by
  sorry

end family_gathering_arrangements_l199_19909


namespace line_bisecting_circle_min_value_l199_19916

/-- Given a line that bisects a circle, prove the minimum value of a certain expression -/
theorem line_bisecting_circle_min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, 2*a*x + b*y - 2 = 0 → x^2 + y^2 - 2*x - 4*y - 6 = 0) →
  (∃ x₀ y₀ : ℝ, 2*a*x₀ + b*y₀ - 2 = 0 ∧ x₀ = 1 ∧ y₀ = 2) →
  (∀ k : ℝ, 2/a + 1/b ≥ k) →
  k = 3 + 2*Real.sqrt 2 :=
sorry

end line_bisecting_circle_min_value_l199_19916


namespace optimal_distribution_theorem_l199_19961

/-- Represents the total value of the estate in talents -/
def estate_value : ℚ := 210

/-- Represents the fraction of the estate allocated to the son if only a son is born -/
def son_fraction : ℚ := 2/3

/-- Represents the fraction of the estate allocated to the daughter if only a daughter is born -/
def daughter_fraction : ℚ := 1/3

/-- Represents the optimal fraction of the estate allocated to the son when twins are born -/
def optimal_son_fraction : ℚ := 4/7

/-- Represents the optimal fraction of the estate allocated to the daughter when twins are born -/
def optimal_daughter_fraction : ℚ := 1/7

/-- Represents the optimal fraction of the estate allocated to the mother when twins are born -/
def optimal_mother_fraction : ℚ := 2/7

/-- Theorem stating that the optimal distribution is the best approximation of the will's conditions -/
theorem optimal_distribution_theorem :
  optimal_son_fraction + optimal_daughter_fraction + optimal_mother_fraction = 1 ∧
  optimal_son_fraction * estate_value + 
  optimal_daughter_fraction * estate_value + 
  optimal_mother_fraction * estate_value = estate_value ∧
  optimal_son_fraction > optimal_daughter_fraction ∧
  optimal_son_fraction < son_fraction ∧
  optimal_daughter_fraction < daughter_fraction :=
sorry

end optimal_distribution_theorem_l199_19961


namespace min_color_changes_l199_19929

/-- Represents a 10x10 board with colored chips -/
def Board := Fin 10 → Fin 10 → Fin 100

/-- Checks if a chip is unique in its row or column -/
def is_unique (b : Board) (i j : Fin 10) : Prop :=
  (∀ k : Fin 10, k ≠ j → b i k ≠ b i j) ∨
  (∀ k : Fin 10, k ≠ i → b k j ≠ b i j)

/-- Represents a valid color change operation -/
def valid_change (b1 b2 : Board) : Prop :=
  ∃ i j : Fin 10, 
    (∀ x y : Fin 10, (x ≠ i ∨ y ≠ j) → b1 x y = b2 x y) ∧
    is_unique b1 i j ∧
    b1 i j ≠ b2 i j

/-- Represents a sequence of valid color changes -/
def valid_sequence (n : ℕ) : Prop :=
  ∃ (seq : Fin (n + 1) → Board),
    (∀ i : Fin 10, ∀ j : Fin 10, seq 0 i j = i.val * 10 + j.val) ∧
    (∀ k : Fin n, valid_change (seq k) (seq (k + 1))) ∧
    (∀ i j : Fin 10, ¬is_unique (seq n) i j)

/-- The main theorem stating the minimum number of color changes -/
theorem min_color_changes : 
  (∃ n : ℕ, valid_sequence n) ∧ 
  (∀ m : ℕ, m < 75 → ¬valid_sequence m) :=
sorry

end min_color_changes_l199_19929


namespace pet_store_profit_percentage_l199_19903

-- Define the types of animals
inductive AnimalType
| Gecko
| Parrot
| Tarantula

-- Define the structure for animal sales
structure AnimalSale where
  animalType : AnimalType
  quantity : Nat
  purchasePrice : Nat

-- Define the bulk discount function
def bulkDiscount (quantity : Nat) : Rat :=
  if quantity ≥ 5 then 0.1 else 0

-- Define the selling price function
def sellingPrice (animalType : AnimalType) (purchasePrice : Nat) : Nat :=
  match animalType with
  | AnimalType.Gecko => 3 * purchasePrice + 5
  | AnimalType.Parrot => 2 * purchasePrice + 10
  | AnimalType.Tarantula => 4 * purchasePrice + 15

-- Define the profit percentage calculation
def profitPercentage (sales : List AnimalSale) : Rat :=
  let totalCost := sales.foldl (fun acc sale =>
    acc + sale.quantity * sale.purchasePrice * (1 - bulkDiscount sale.quantity)) 0
  let totalRevenue := sales.foldl (fun acc sale =>
    acc + sale.quantity * sellingPrice sale.animalType sale.purchasePrice) 0
  let profit := totalRevenue - totalCost
  (profit / totalCost) * 100

-- Theorem statement
theorem pet_store_profit_percentage :
  let sales := [
    { animalType := AnimalType.Gecko, quantity := 6, purchasePrice := 100 },
    { animalType := AnimalType.Parrot, quantity := 3, purchasePrice := 200 },
    { animalType := AnimalType.Tarantula, quantity := 10, purchasePrice := 50 }
  ]
  abs (profitPercentage sales - 227.67) < 0.01 := by
  sorry

end pet_store_profit_percentage_l199_19903


namespace fraction_equality_l199_19952

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (3 * x + y) / (x - 3 * y) = -2) : 
  (x + 3 * y) / (3 * x - y) = 2 := by
  sorry

end fraction_equality_l199_19952


namespace arithmetic_sequence_13th_term_l199_19968

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence (α : Type*) [Field α] where
  a : ℕ → α
  d : α
  h_nonzero : d ≠ 0
  h_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- The 13th term of the arithmetic sequence is 28 -/
theorem arithmetic_sequence_13th_term
  {α : Type*} [Field α] (seq : ArithmeticSequence α)
  (h_geometric : seq.a 9 * seq.a 1 = (seq.a 5)^2)
  (h_sum : seq.a 1 + 3 * seq.a 5 + seq.a 9 = 20) :
  seq.a 13 = 28 :=
sorry

end arithmetic_sequence_13th_term_l199_19968


namespace farmer_tomatoes_l199_19920

theorem farmer_tomatoes (picked : ℕ) (left : ℕ) (h1 : picked = 83) (h2 : left = 14) :
  picked + left = 97 := by
  sorry

end farmer_tomatoes_l199_19920


namespace unique_quadratic_solution_l199_19931

theorem unique_quadratic_solution (k : ℝ) (x : ℝ) :
  (∀ y : ℝ, 8 * y^2 + 36 * y + k = 0 ↔ y = x) →
  k = 40.5 ∧ x = -2.25 := by
  sorry

end unique_quadratic_solution_l199_19931


namespace sqrt_ln_relation_l199_19955

theorem sqrt_ln_relation (a b : ℝ) :
  (∀ a b, (Real.log a > Real.log b) → (Real.sqrt a > Real.sqrt b)) ∧
  (∃ a b, (Real.sqrt a > Real.sqrt b) ∧ ¬(Real.log a > Real.log b)) := by
  sorry

end sqrt_ln_relation_l199_19955


namespace geometric_progression_a5_l199_19960

-- Define a geometric progression
def isGeometricProgression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_progression_a5 (a : ℕ → ℝ) :
  isGeometricProgression a →
  (a 3) ^ 2 - 5 * (a 3) + 4 = 0 →
  (a 7) ^ 2 - 5 * (a 7) + 4 = 0 →
  a 5 = 2 :=
by
  sorry


end geometric_progression_a5_l199_19960


namespace chromosome_replication_not_in_prophase_i_l199_19944

-- Define the events that can occur during cell division
inductive CellDivisionEvent
  | ChromosomeReplication
  | ChromosomeShortening
  | HomologousPairing
  | CrossingOver

-- Define the phases of meiosis
inductive MeiosisPhase
  | Interphase
  | ProphaseI
  | OtherPhases

-- Define a function that determines if an event occurs in a given phase
def occurs_in (event : CellDivisionEvent) (phase : MeiosisPhase) : Prop := sorry

-- State the theorem
theorem chromosome_replication_not_in_prophase_i :
  occurs_in CellDivisionEvent.ChromosomeReplication MeiosisPhase.Interphase →
  occurs_in CellDivisionEvent.ChromosomeShortening MeiosisPhase.ProphaseI →
  occurs_in CellDivisionEvent.HomologousPairing MeiosisPhase.ProphaseI →
  occurs_in CellDivisionEvent.CrossingOver MeiosisPhase.ProphaseI →
  ¬ occurs_in CellDivisionEvent.ChromosomeReplication MeiosisPhase.ProphaseI :=
by
  sorry

end chromosome_replication_not_in_prophase_i_l199_19944


namespace basketball_reach_theorem_l199_19963

/-- Represents the height a basketball player can reach above their head using their arms -/
def reachAboveHead (playerHeight rimHeight jumpHeight : ℕ) : ℕ :=
  rimHeight * 12 + 6 - (playerHeight * 12 + jumpHeight)

/-- Theorem stating that a 6-foot tall player who can jump 32 inches high needs to reach 22 inches above their head to dunk on a 10-foot rim -/
theorem basketball_reach_theorem :
  reachAboveHead 6 10 32 = 22 := by
  sorry

end basketball_reach_theorem_l199_19963


namespace odd_numbers_mean_contradiction_l199_19989

theorem odd_numbers_mean_contradiction (a b c d e f g : ℤ) :
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧ f < g ∧  -- Ordered
  Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ Odd e ∧ Odd f ∧ Odd g ∧  -- All odd
  (a + b + c + d + e + f + g) / 7 - d = 3 / 7  -- Mean minus middle equals 3/7
  → False := by sorry

end odd_numbers_mean_contradiction_l199_19989


namespace mask_digit_assignment_l199_19994

/-- Represents the four masks in the problem -/
inductive Mask
| elephant
| mouse
| pig
| panda

/-- Assigns a digit to each mask -/
def digit_assignment : Mask → Nat
| Mask.elephant => 6
| Mask.mouse => 4
| Mask.pig => 8
| Mask.panda => 1

/-- Checks if a number is two digits -/
def is_two_digit (n : Nat) : Prop := n ≥ 10 ∧ n < 100

/-- The main theorem statement -/
theorem mask_digit_assignment :
  (∀ m : Mask, digit_assignment m ≤ 9) ∧ 
  (∀ m1 m2 : Mask, m1 ≠ m2 → digit_assignment m1 ≠ digit_assignment m2) ∧
  (∀ m : Mask, is_two_digit ((digit_assignment m) * (digit_assignment m))) ∧
  (∀ m : Mask, (digit_assignment m) * (digit_assignment m) % 10 ≠ digit_assignment m) ∧
  ((digit_assignment Mask.mouse) * (digit_assignment Mask.mouse) % 10 = digit_assignment Mask.elephant) :=
by sorry

end mask_digit_assignment_l199_19994


namespace complement_A_intersect_B_l199_19908

def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {-1, 0}
def B : Set Int := {0, 1, 2}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {1, 2} := by sorry

end complement_A_intersect_B_l199_19908


namespace equation_solution_l199_19915

theorem equation_solution : ∃ x : ℚ, (5 * x - 3 * (x + 2) = 450 - 9 * (x - 4)) ∧ (x = 492 / 11) := by
  sorry

end equation_solution_l199_19915


namespace class_size_l199_19933

/-- Represents the number of students excelling in various combinations of sports -/
structure SportExcellence where
  sprint : ℕ
  swimming : ℕ
  basketball : ℕ
  sprint_swimming : ℕ
  swimming_basketball : ℕ
  sprint_basketball : ℕ
  all_three : ℕ

/-- The total number of students in the class -/
def total_students (se : SportExcellence) (non_excellent : ℕ) : ℕ :=
  se.sprint + se.swimming + se.basketball
  - se.sprint_swimming - se.swimming_basketball - se.sprint_basketball
  + se.all_three + non_excellent

/-- The theorem stating the total number of students in the class -/
theorem class_size (se : SportExcellence) (non_excellent : ℕ) : 
  se.sprint = 17 → se.swimming = 18 → se.basketball = 15 →
  se.sprint_swimming = 6 → se.swimming_basketball = 6 →
  se.sprint_basketball = 5 → se.all_three = 2 → non_excellent = 4 →
  total_students se non_excellent = 39 := by
  sorry

/-- Example usage of the theorem -/
example : ∃ (se : SportExcellence) (non_excellent : ℕ), 
  total_students se non_excellent = 39 := by
  sorry

end class_size_l199_19933


namespace binomial_sum_divides_power_of_two_l199_19937

theorem binomial_sum_divides_power_of_two (n : ℕ) :
  n > 3 →
  (1 + Nat.choose n 1 + Nat.choose n 2 + Nat.choose n 3) ∣ 2^2000 ↔
  n = 7 ∨ n = 23 := by
  sorry

end binomial_sum_divides_power_of_two_l199_19937


namespace simplify_trig_expression_l199_19983

theorem simplify_trig_expression (x : ℝ) :
  (Real.sqrt 2 / 4) * Real.sin (π / 4 - x) + (Real.sqrt 6 / 4) * Real.cos (π / 4 - x) =
  (Real.sqrt 2 / 2) * Real.sin (7 * π / 12 - x) := by
  sorry

end simplify_trig_expression_l199_19983


namespace area_of_triangle_range_of_sum_a_c_l199_19986

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 * Real.sqrt 3 ∧
  Real.sqrt 3 * Real.cos t.B = t.b * Real.sin t.C

-- Theorem 1: Area of triangle ABC
theorem area_of_triangle (t : Triangle) (h : triangle_conditions t) (ha : t.a = 2) :
  (1/2) * t.a * t.c * Real.sin t.B = 2 * Real.sqrt 3 :=
sorry

-- Theorem 2: Range of a + c
theorem range_of_sum_a_c (t : Triangle) (h : triangle_conditions t) (acute : t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧ t.A + t.B + t.C = π) :
  2 * Real.sqrt 3 < t.a + t.c ∧ t.a + t.c ≤ 4 * Real.sqrt 3 :=
sorry

end area_of_triangle_range_of_sum_a_c_l199_19986


namespace range_of_x_plus_3y_l199_19958

theorem range_of_x_plus_3y (x y : ℝ) 
  (h1 : -1 ≤ x + y ∧ x + y ≤ 4) 
  (h2 : 2 ≤ x - y ∧ x - y ≤ 3) : 
  -5 ≤ x + 3*y ∧ x + 3*y ≤ 6 := by
  sorry

end range_of_x_plus_3y_l199_19958


namespace prime_cube_difference_equation_l199_19964

theorem prime_cube_difference_equation :
  ∀ p q r : ℕ,
    Prime p → Prime q → Prime r →
    5 * p = q^3 - r^3 →
    p = 67 ∧ q = 7 ∧ r = 2 :=
by sorry

end prime_cube_difference_equation_l199_19964


namespace sandwich_cost_is_90_cents_l199_19927

/-- The cost of making a sandwich with two slices of bread, one slice of ham, and one slice of cheese -/
def sandwich_cost (bread_cost cheese_cost ham_cost : ℚ) : ℚ :=
  2 * bread_cost + cheese_cost + ham_cost

/-- Theorem stating that the cost of making a sandwich is 90 cents -/
theorem sandwich_cost_is_90_cents :
  sandwich_cost 0.15 0.35 0.25 * 100 = 90 := by
  sorry

end sandwich_cost_is_90_cents_l199_19927


namespace onions_left_on_scale_l199_19973

/-- Represents the problem of calculating the number of onions left on a scale. -/
def OnionProblem (initial_count : ℕ) (total_weight : ℝ) (removed_count : ℕ) (remaining_avg_weight : ℝ) (removed_avg_weight : ℝ) : Prop :=
  let remaining_count := initial_count - removed_count
  let total_weight_grams := total_weight * 1000
  let remaining_weight := remaining_count * remaining_avg_weight
  let removed_weight := removed_count * removed_avg_weight
  (remaining_weight + removed_weight = total_weight_grams) ∧
  (remaining_count = 35)

/-- Theorem stating that given the problem conditions, 35 onions are left on the scale. -/
theorem onions_left_on_scale :
  OnionProblem 40 7.68 5 190 206 :=
by
  sorry

end onions_left_on_scale_l199_19973


namespace floor_sqrt_120_l199_19902

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end floor_sqrt_120_l199_19902


namespace tangent_line_problem_l199_19993

theorem tangent_line_problem (a : ℝ) :
  (∃ (m : ℝ), 
    (∀ x y : ℝ, y = x^3 → (y - 0 = m * (x - 1) → x = 1 ∨ (y - x^3) = 3 * x^2 * (x - x))) ∧
    (∀ x y : ℝ, y = a * x^2 + (15/4) * x - 9 → (y - 0 = m * (x - 1) → x = 1 ∨ (y - (a * x^2 + (15/4) * x - 9)) = (2 * a * x + 15/4) * (x - x))))
  → a = -1 ∨ a = -25/64 := by
  sorry

end tangent_line_problem_l199_19993


namespace triangle_angle_and_area_l199_19918

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove the measure of angle B and the area of the triangle. -/
theorem triangle_angle_and_area 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a = 6)
  (h2 : b = 5)
  (h3 : Real.cos A = -4/5) :
  B = π/6 ∧ 
  (1/2 * a * b * Real.sin C = (9 * Real.sqrt 3 - 12) / 2) := by
  sorry

end triangle_angle_and_area_l199_19918


namespace expression_simplification_l199_19957

theorem expression_simplification (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 2) (h3 : a ≠ -2) :
  (a - 1) / (a + 2) / ((a^2 - 2*a) / (a^2 - 4)) - (a + 1) / a = -2 / a := by
  sorry

end expression_simplification_l199_19957


namespace soldier_arrangement_l199_19981

theorem soldier_arrangement (x : ℕ) 
  (h1 : x % 2 = 1)
  (h2 : x % 3 = 2)
  (h3 : x % 5 = 3) :
  x % 30 = 23 := by
  sorry

end soldier_arrangement_l199_19981
