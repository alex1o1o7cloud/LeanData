import Mathlib

namespace quadratic_roots_property_l3156_315659

theorem quadratic_roots_property (m n : ℝ) : 
  (m^2 + m - 2021 = 0) → (n^2 + n - 2021 = 0) → (m^2 + 2*m + n = 2020) := by
  sorry

end quadratic_roots_property_l3156_315659


namespace geometric_sum_n1_l3156_315679

theorem geometric_sum_n1 (a : ℝ) (h : a ≠ 1) :
  1 + a + a^2 = (1 - a^3) / (1 - a) := by
  sorry

end geometric_sum_n1_l3156_315679


namespace square_division_square_coverage_l3156_315656

/-- A square is a shape with four equal sides and four right angles -/
structure Square where
  side : ℝ
  side_positive : side > 0

/-- A larger square can be divided into four equal smaller squares -/
theorem square_division (large : Square) : 
  ∃ (small : Square), 
    4 * small.side^2 = large.side^2 ∧ 
    small.side > 0 :=
sorry

/-- Four smaller squares can completely cover a larger square without gaps or overlaps -/
theorem square_coverage (large : Square) 
  (h : ∃ (small : Square), 4 * small.side^2 = large.side^2 ∧ small.side > 0) : 
  ∃ (small : Square), 
    4 * small.side^2 = large.side^2 ∧ 
    small.side > 0 ∧
    (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ large.side ∧ 0 ≤ y ∧ y ≤ large.side → 
      ∃ (i j : Fin 2), 
        i * small.side ≤ x ∧ x < (i + 1) * small.side ∧
        j * small.side ≤ y ∧ y < (j + 1) * small.side) :=
sorry

end square_division_square_coverage_l3156_315656


namespace square_side_length_difference_l3156_315645

theorem square_side_length_difference (area_A area_B : ℝ) 
  (h_A : area_A = 25) (h_B : area_B = 81) : 
  Real.sqrt area_B - Real.sqrt area_A = 4 := by
  sorry

end square_side_length_difference_l3156_315645


namespace career_preference_graph_degrees_l3156_315639

theorem career_preference_graph_degrees 
  (total_students : ℕ) 
  (male_ratio female_ratio : ℚ) 
  (male_preference female_preference : ℚ) :
  male_ratio / (male_ratio + female_ratio) = 2 / 5 →
  female_ratio / (male_ratio + female_ratio) = 3 / 5 →
  male_preference = 1 / 4 →
  female_preference = 1 / 2 →
  (male_ratio * male_preference + female_ratio * female_preference) / (male_ratio + female_ratio) * 360 = 144 := by
  sorry

#check career_preference_graph_degrees

end career_preference_graph_degrees_l3156_315639


namespace integral_rational_function_l3156_315672

open Real

theorem integral_rational_function (x : ℝ) :
  deriv (fun x => (1/2) * log (x^2 + 2*x + 5) + (1/2) * arctan ((x + 1)/2)) x
  = (x + 2) / (x^2 + 2*x + 5) := by sorry

end integral_rational_function_l3156_315672


namespace wendy_bought_four_tables_l3156_315634

/-- The number of chairs Wendy bought -/
def num_chairs : ℕ := 4

/-- The time spent on each piece of furniture (in minutes) -/
def time_per_piece : ℕ := 6

/-- The total assembly time (in minutes) -/
def total_time : ℕ := 48

/-- The number of tables Wendy bought -/
def num_tables : ℕ := (total_time - num_chairs * time_per_piece) / time_per_piece

theorem wendy_bought_four_tables : num_tables = 4 := by
  sorry

end wendy_bought_four_tables_l3156_315634


namespace triangle_centroid_distance_sum_l3156_315657

/-- Given a triangle ABC with centroid G, if GA^2 + GB^2 + GC^2 = 88, 
    then AB^2 + AC^2 + BC^2 = 396 -/
theorem triangle_centroid_distance_sum (A B C G : ℝ × ℝ) : 
  (G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) →
  ((G.1 - A.1)^2 + (G.2 - A.2)^2 + 
   (G.1 - B.1)^2 + (G.2 - B.2)^2 + 
   (G.1 - C.1)^2 + (G.2 - C.2)^2 = 88) →
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 + 
   (A.1 - C.1)^2 + (A.2 - C.2)^2 + 
   (B.1 - C.1)^2 + (B.2 - C.2)^2 = 396) := by
sorry

end triangle_centroid_distance_sum_l3156_315657


namespace smallest_m_proof_l3156_315649

/-- The smallest positive integer m such that 15m - 3 is divisible by 11 -/
def smallest_m : ℕ := 9

theorem smallest_m_proof :
  smallest_m = 9 ∧
  ∀ k : ℕ, k > 0 → (15 * k - 3) % 11 = 0 → k ≥ 9 :=
by sorry

end smallest_m_proof_l3156_315649


namespace male_students_count_l3156_315620

theorem male_students_count (total_students sample_size female_in_sample : ℕ) 
  (h1 : total_students = 1000)
  (h2 : sample_size = 100)
  (h3 : female_in_sample = 51)
  (h4 : female_in_sample < sample_size) :
  (total_students : ℚ) * ((sample_size - female_in_sample) : ℚ) / (sample_size : ℚ) = 490 := by
  sorry

end male_students_count_l3156_315620


namespace integer_decimal_parts_theorem_l3156_315618

theorem integer_decimal_parts_theorem :
  ∀ (a b : ℝ),
  (a = ⌊7 - Real.sqrt 13⌋) →
  (b = 7 - Real.sqrt 13 - a) →
  (2 * a - b = 2 + Real.sqrt 13) := by
sorry

end integer_decimal_parts_theorem_l3156_315618


namespace gilbert_crickets_l3156_315690

/-- The number of crickets Gilbert eats per week at 90°F -/
def crickets_90 : ℕ := 4

/-- The number of crickets Gilbert eats per week at 100°F -/
def crickets_100 : ℕ := 2 * crickets_90

/-- The total number of weeks -/
def total_weeks : ℕ := 15

/-- The fraction of time at 90°F -/
def fraction_90 : ℚ := 4/5

/-- The fraction of time at 100°F -/
def fraction_100 : ℚ := 1 - fraction_90

theorem gilbert_crickets :
  (↑crickets_90 * (fraction_90 * total_weeks) +
   ↑crickets_100 * (fraction_100 * total_weeks)).floor = 72 := by
  sorry

end gilbert_crickets_l3156_315690


namespace distance_to_chord_equals_half_chord_l3156_315670

-- Define the circle and points
structure Circle :=
  (center : Point)
  (radius : ℝ)

structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the distance function
def distance (p q : Point) : ℝ := sorry

-- Define the function to calculate distance from a point to a line segment
def distancePointToSegment (p : Point) (a b : Point) : ℝ := sorry

-- Define the theorem
theorem distance_to_chord_equals_half_chord (O A B C D E : Point) (circle : Circle) :
  O = circle.center →
  distance A E = 2 * circle.radius →
  (∀ p ∈ [A, B, C, E], distance O p = circle.radius) →
  distancePointToSegment O A B = (distance C D) / 2 := by
  sorry

end distance_to_chord_equals_half_chord_l3156_315670


namespace current_rate_calculation_l3156_315624

/-- Given a boat with speed in still water and its downstream travel details, 
    calculate the rate of the current. -/
theorem current_rate_calculation 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : boat_speed = 20) 
  (h2 : downstream_distance = 5.2) 
  (h3 : downstream_time = 0.2) : 
  ∃ (current_rate : ℝ), 
    current_rate = 6 ∧ 
    downstream_distance = (boat_speed + current_rate) * downstream_time :=
by sorry

end current_rate_calculation_l3156_315624


namespace sin_squared_minus_2sin_range_l3156_315662

theorem sin_squared_minus_2sin_range :
  ∀ x : ℝ, -1 ≤ Real.sin x ^ 2 - 2 * Real.sin x ∧ Real.sin x ^ 2 - 2 * Real.sin x ≤ 3 := by
  sorry

end sin_squared_minus_2sin_range_l3156_315662


namespace first_number_of_sequence_l3156_315629

/-- A sequence with specific properties -/
structure Sequence where
  second : ℕ
  increment : ℕ
  final : ℕ

/-- The first number in the sequence -/
def firstNumber (s : Sequence) : ℕ := s.second - s.increment

/-- Theorem stating the properties of the sequence and the first number -/
theorem first_number_of_sequence (s : Sequence) 
  (h1 : s.second = 45)
  (h2 : s.increment = 11)
  (h3 : s.final = 89) :
  firstNumber s = 34 := by
  sorry

#check first_number_of_sequence

end first_number_of_sequence_l3156_315629


namespace sphere_volume_equals_surface_area_l3156_315693

theorem sphere_volume_equals_surface_area (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * r^3 = 4 * Real.pi * r^2) → r = 3 := by
  sorry

end sphere_volume_equals_surface_area_l3156_315693


namespace casino_solution_l3156_315636

def casino_problem (money_A money_B money_C : ℕ) : Prop :=
  (money_B = 2 * money_C) ∧
  (money_A = 40) ∧
  (money_A + money_B + money_C = 220)

theorem casino_solution :
  ∀ money_A money_B money_C,
    casino_problem money_A money_B money_C →
    money_C - money_A = 20 := by
  sorry

end casino_solution_l3156_315636


namespace decimal_places_of_fraction_l3156_315664

theorem decimal_places_of_fraction : ∃ (n : ℕ), 
  (5^5 : ℚ) / (10^3 * 8) = n / 10 ∧ n % 10 ≠ 0 ∧ n < 100 := by
  sorry

end decimal_places_of_fraction_l3156_315664


namespace polynomial_value_l3156_315651

theorem polynomial_value (a b : ℝ) (h : a^2 - 2*b - 1 = 0) :
  -2*a^2 + 4*b + 2025 = 2023 := by
  sorry

end polynomial_value_l3156_315651


namespace equation_solution_l3156_315632

theorem equation_solution : 
  ∃ (x : ℝ), ((3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1)) ∧ 
  (x = 6 ∨ x = -2) := by
  sorry

end equation_solution_l3156_315632


namespace prob_girl_given_boy_specific_l3156_315615

/-- Represents a club with members -/
structure Club where
  total_members : ℕ
  girls : ℕ
  boys : ℕ

/-- The probability of choosing a girl given that at least one boy is chosen -/
def prob_girl_given_boy (c : Club) : ℚ :=
  (c.girls * c.boys : ℚ) / ((c.girls * c.boys + (c.boys * (c.boys - 1)) / 2) : ℚ)

theorem prob_girl_given_boy_specific :
  let c : Club := { total_members := 12, girls := 7, boys := 5 }
  prob_girl_given_boy c = 7/9 := by
  sorry


end prob_girl_given_boy_specific_l3156_315615


namespace determinant_equality_l3156_315609

theorem determinant_equality (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = -3 →
  Matrix.det !![x + z, y + w; z, w] = -3 := by
sorry

end determinant_equality_l3156_315609


namespace shelby_total_stars_l3156_315697

/-- The number of gold stars Shelby earned yesterday -/
def yesterday_stars : ℕ := 4

/-- The number of gold stars Shelby earned today -/
def today_stars : ℕ := 3

/-- The total number of gold stars Shelby earned -/
def total_stars : ℕ := yesterday_stars + today_stars

theorem shelby_total_stars : total_stars = 7 := by
  sorry

end shelby_total_stars_l3156_315697


namespace intersection_line_circle_chord_length_l3156_315627

theorem intersection_line_circle_chord_length (k : ℝ) :
  (∃ M N : ℝ × ℝ, 
    (M.1^2 - 4*M.1 + M.2^2 = 0) ∧ 
    (N.1^2 - 4*N.1 + N.2^2 = 0) ∧
    (M.2 = k*M.1 + 1) ∧ 
    (N.2 = k*N.1 + 1) ∧
    ((M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 12)) →
  -4/3 ≤ k ∧ k ≤ 0 :=
by sorry

end intersection_line_circle_chord_length_l3156_315627


namespace community_population_l3156_315616

/-- Represents the number of people in each category of a community --/
structure Community where
  babies : ℝ
  seniors : ℝ
  children : ℝ
  teenagers : ℝ
  women : ℝ
  men : ℝ

/-- The total number of people in the community --/
def totalPeople (c : Community) : ℝ :=
  c.babies + c.seniors + c.children + c.teenagers + c.women + c.men

/-- Theorem stating the relationship between the number of babies and the total population --/
theorem community_population (c : Community) 
  (h1 : c.men = 1.5 * c.women)
  (h2 : c.women = 3 * c.teenagers)
  (h3 : c.teenagers = 2.5 * c.children)
  (h4 : c.children = 4 * c.seniors)
  (h5 : c.seniors = 3.5 * c.babies) :
  totalPeople c = 316 * c.babies := by
  sorry


end community_population_l3156_315616


namespace opposite_to_turquoise_is_pink_l3156_315602

/-- Represents the colors of the squares --/
inductive Color
  | Pink
  | Violet
  | Turquoise
  | Orange

/-- Represents a face of the cube --/
structure Face where
  color : Color

/-- Represents the cube formed by folding the squares --/
structure Cube where
  faces : List Face
  opposite : Face → Face

/-- The configuration of the cube --/
def cube_config : Cube :=
  { faces := [
      Face.mk Color.Pink,
      Face.mk Color.Pink,
      Face.mk Color.Pink,
      Face.mk Color.Violet,
      Face.mk Color.Violet,
      Face.mk Color.Turquoise,
      Face.mk Color.Orange
    ],
    opposite := sorry  -- The actual implementation of the opposite function
  }

/-- Theorem stating that the face opposite to Turquoise is Pink --/
theorem opposite_to_turquoise_is_pink :
  ∃ (f : Face), f ∈ cube_config.faces ∧ 
    f.color = Color.Turquoise ∧ 
    (cube_config.opposite f).color = Color.Pink :=
  sorry


end opposite_to_turquoise_is_pink_l3156_315602


namespace tinas_oranges_l3156_315647

/-- The number of oranges in Tina's bag -/
def oranges : ℕ := sorry

/-- The number of apples in Tina's bag -/
def apples : ℕ := 9

/-- The number of tangerines in Tina's bag -/
def tangerines : ℕ := 17

/-- The number of oranges removed -/
def oranges_removed : ℕ := 2

/-- The number of tangerines removed -/
def tangerines_removed : ℕ := 10

/-- Theorem stating that the number of oranges in Tina's bag is 5 -/
theorem tinas_oranges : oranges = 5 := by
  have h1 : tangerines - tangerines_removed = (oranges - oranges_removed) + 4 := by sorry
  sorry

end tinas_oranges_l3156_315647


namespace muffins_per_box_l3156_315658

theorem muffins_per_box (total_muffins : ℕ) (num_boxes : ℕ) 
  (h1 : total_muffins = 96) (h2 : num_boxes = 8) :
  total_muffins / num_boxes = 12 := by
  sorry

end muffins_per_box_l3156_315658


namespace square_hexagon_area_l3156_315691

theorem square_hexagon_area (s : ℝ) (square_area : ℝ) (hex_area : ℝ) : 
  square_area = Real.sqrt 3 →
  square_area = s^2 →
  hex_area = 3 * Real.sqrt 3 * s^2 / 2 →
  hex_area = 9 / 2 := by
sorry

end square_hexagon_area_l3156_315691


namespace sin_product_equality_l3156_315676

theorem sin_product_equality : 
  Real.sin (π / 14) * Real.sin (3 * π / 14) * Real.sin (5 * π / 14) = 1 / 8 := by
  sorry

end sin_product_equality_l3156_315676


namespace array_sum_proof_l3156_315640

def grid := [[1, 0, 0, 0], [0, 9, 0, 5], [0, 0, 14, 0]]
def available_numbers := [2, 3, 4, 7, 10, 11, 12, 13, 15]

theorem array_sum_proof :
  ∃ (arrangement : List (List Nat)),
    (∀ row ∈ arrangement, row.sum = 32) ∧
    (∀ col ∈ arrangement.transpose, col.sum = 32) ∧
    (arrangement.join.toFinset = (available_numbers.toFinset \ {10}) ∪ grid.join.toFinset) :=
  by sorry

end array_sum_proof_l3156_315640


namespace exists_increasing_arithmetic_seq_exists_perm_without_long_increasing_seq_l3156_315642

-- Define the set of natural numbers (positive integers)
def N : Set Nat := {n : Nat | n > 0}

-- Define a permutation of N
def isPerm (f : Nat → Nat) : Prop := Function.Bijective f ∧ ∀ n, f n ∈ N

-- Theorem 1
theorem exists_increasing_arithmetic_seq (f : Nat → Nat) (h : isPerm f) :
  ∃ a d : Nat, d > 0 ∧ a ∈ N ∧ (a + d) ∈ N ∧ (a + 2*d) ∈ N ∧
    f a < f (a + d) ∧ f (a + d) < f (a + 2*d) := by sorry

-- Theorem 2
theorem exists_perm_without_long_increasing_seq :
  ∃ f : Nat → Nat, isPerm f ∧
    ∀ a d : Nat, d > 0 → a ∈ N →
      ¬(∀ k : Nat, k ≤ 2003 → f (a + k*d) < f (a + (k+1)*d)) := by sorry

end exists_increasing_arithmetic_seq_exists_perm_without_long_increasing_seq_l3156_315642


namespace radical_axes_property_l3156_315680

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line in 2D space (ax + by + c = 0)
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the radical axis of two circles
def radical_axis (c1 c2 : Circle) : Line :=
  sorry

-- Define the property of lines being coincident
def coincident (l1 l2 l3 : Line) : Prop :=
  sorry

-- Define the property of lines being concurrent
def concurrent (l1 l2 l3 : Line) : Prop :=
  sorry

-- Define the property of lines being parallel
def parallel (l1 l2 l3 : Line) : Prop :=
  sorry

-- Theorem statement
theorem radical_axes_property (Γ₁ Γ₂ Γ₃ : Circle) :
  let Δ₁ := radical_axis Γ₁ Γ₂
  let Δ₂ := radical_axis Γ₂ Γ₃
  let Δ₃ := radical_axis Γ₃ Γ₁
  coincident Δ₁ Δ₂ Δ₃ ∨ concurrent Δ₁ Δ₂ Δ₃ ∨ parallel Δ₁ Δ₂ Δ₃ :=
by
  sorry

end radical_axes_property_l3156_315680


namespace calculation_proof_l3156_315666

theorem calculation_proof : 
  let sin_30 : ℝ := 1/2
  let sqrt_2_gt_1 : 1 < Real.sqrt 2 := by sorry
  let power_zero (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry
  2 * sin_30 - |1 - Real.sqrt 2| + (π - 2022)^0 = 3 - Real.sqrt 2 := by sorry

end calculation_proof_l3156_315666


namespace equation_solution_l3156_315674

theorem equation_solution : 
  let x₁ : ℝ := (3 + Real.sqrt 17) / 2
  let x₂ : ℝ := (-3 - Real.sqrt 17) / 2
  (x₁^2 - 3 * |x₁| - 2 = 0) ∧ 
  (x₂^2 - 3 * |x₂| - 2 = 0) ∧ 
  (∀ x : ℝ, x^2 - 3 * |x| - 2 = 0 → x = x₁ ∨ x = x₂) :=
by sorry

end equation_solution_l3156_315674


namespace every_second_sum_of_arithmetic_sequence_l3156_315635

def sequence_sum (first : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * first + (n - 1)) / 2

def every_second_sum (first : ℚ) (n : ℕ) : ℚ :=
  sequence_sum first ((n + 1) / 2)

theorem every_second_sum_of_arithmetic_sequence 
  (first : ℚ) (n : ℕ) (h1 : n = 3015) (h2 : sequence_sum first n = 8010) :
  every_second_sum first (n - 1) = 3251.5 := by
  sorry

end every_second_sum_of_arithmetic_sequence_l3156_315635


namespace arithmetic_geometric_sequence_properties_l3156_315648

theorem arithmetic_geometric_sequence_properties
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- a_n is arithmetic with common difference d
  (d ≠ 0) →  -- nonzero common difference
  (∀ n, b (n + 1) = q * b n) →  -- b_n is geometric with common ratio q
  (b 1 = a 1 ^ 2) →  -- b₁ = a₁²
  (b 2 = a 2 ^ 2) →  -- b₂ = a₂²
  (b 3 = a 3 ^ 2) →  -- b₃ = a₃²
  (a 2 = -1) →  -- a₂ = -1
  (a 1 < a 2) →  -- a₁ < a₂
  (q = 3 - 2 * Real.sqrt 2 ∧ d = Real.sqrt 2) :=
by sorry

end arithmetic_geometric_sequence_properties_l3156_315648


namespace max_intersections_circle_sine_l3156_315685

/-- The maximum number of intersection points between a circle and sine curve --/
theorem max_intersections_circle_sine (h k : ℝ) : 
  (k ≥ -2 ∧ k ≤ 2) → 
  (∃ (n : ℕ), n ≤ 8 ∧ 
    (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = 4 ∧ y = Real.sin x → 
      (∃ (m : ℕ), m ≤ n ∧ 
        (∀ (p q : ℝ), (p - h)^2 + (q - k)^2 = 4 ∧ q = Real.sin p → 
          (x = p ∧ y = q) ∨ m > 1)))) ∧
  (∀ (m : ℕ), m > 8 → 
    (∃ (x y : ℝ), (x - h)^2 + (y - k)^2 = 4 ∧ y = Real.sin x ∧
      (∀ (p q : ℝ), (p - h)^2 + (q - k)^2 = 4 ∧ q = Real.sin p → 
        (x ≠ p ∨ y ≠ q)))) := by
  sorry

end max_intersections_circle_sine_l3156_315685


namespace triangle_cutting_theorem_l3156_315677

theorem triangle_cutting_theorem (x : ℝ) : 
  (∀ a b c : ℝ, a = 6 - x ∧ b = 8 - x ∧ c = 10 - x → a + b ≤ c) →
  x ≥ 4 :=
by sorry

end triangle_cutting_theorem_l3156_315677


namespace square_of_negative_two_m_cubed_l3156_315623

theorem square_of_negative_two_m_cubed (m : ℝ) : (-2 * m^3)^2 = 4 * m^6 := by
  sorry

end square_of_negative_two_m_cubed_l3156_315623


namespace determine_new_harvest_l3156_315643

/-- Represents the harvest data for two plots of land before and after applying new agricultural techniques. -/
structure HarvestData where
  initial_total : ℝ
  yield_increase_plot1 : ℝ
  yield_increase_plot2 : ℝ
  new_total : ℝ

/-- Represents the harvest amounts for each plot after applying new techniques. -/
structure NewHarvest where
  plot1 : ℝ
  plot2 : ℝ

/-- Theorem stating that given the initial conditions, the new harvest amounts can be determined. -/
theorem determine_new_harvest (data : HarvestData) 
  (h1 : data.initial_total = 14.7)
  (h2 : data.yield_increase_plot1 = 0.8)
  (h3 : data.yield_increase_plot2 = 0.24)
  (h4 : data.new_total = 21.42) :
  ∃ (new_harvest : NewHarvest),
    new_harvest.plot1 = 10.26 ∧
    new_harvest.plot2 = 11.16 ∧
    new_harvest.plot1 + new_harvest.plot2 = data.new_total ∧
    new_harvest.plot1 / (1 + data.yield_increase_plot1) + 
    new_harvest.plot2 / (1 + data.yield_increase_plot2) = data.initial_total :=
  sorry

end determine_new_harvest_l3156_315643


namespace stratified_sample_theorem_l3156_315619

/-- Represents the number of students selected from each year in a stratified sample. -/
structure StratifiedSample where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- Calculates the stratified sample given the total number of students and sample size. -/
def calculate_stratified_sample (total_students : ℕ) (first_year : ℕ) (second_year : ℕ) (third_year : ℕ) (sample_size : ℕ) : StratifiedSample :=
  { first_year := (first_year * sample_size) / total_students,
    second_year := (second_year * sample_size) / total_students,
    third_year := (third_year * sample_size) / total_students }

theorem stratified_sample_theorem :
  let total_students : ℕ := 900
  let first_year : ℕ := 300
  let second_year : ℕ := 200
  let third_year : ℕ := 400
  let sample_size : ℕ := 45
  let result := calculate_stratified_sample total_students first_year second_year third_year sample_size
  result.first_year = 15 ∧ result.second_year = 10 ∧ result.third_year = 20 := by
  sorry

end stratified_sample_theorem_l3156_315619


namespace opposite_of_2023_l3156_315673

theorem opposite_of_2023 : -(2023 : ℤ) = -2023 := by sorry

end opposite_of_2023_l3156_315673


namespace wall_width_proof_l3156_315621

/-- Proves that the width of a wall is 22.5 cm given specific dimensions and number of bricks -/
theorem wall_width_proof (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
  (wall_length : ℝ) (wall_height : ℝ) (num_bricks : ℕ) :
  brick_length = 25 →
  brick_width = 11.25 →
  brick_height = 6 →
  wall_length = 800 →
  wall_height = 600 →
  num_bricks = 6400 →
  ∃ (wall_width : ℝ), wall_width = 22.5 ∧
    wall_length * wall_height * wall_width = 
    (brick_length * brick_width * brick_height * num_bricks) :=
by sorry

end wall_width_proof_l3156_315621


namespace simplify_sqrt_fraction_l3156_315637

theorem simplify_sqrt_fraction : 
  (Real.sqrt ((7:ℝ)^2 + 24^2)) / (Real.sqrt (49 + 16)) = (25 * Real.sqrt 65) / 65 := by
  sorry

end simplify_sqrt_fraction_l3156_315637


namespace line_segment_endpoint_l3156_315601

/-- Given a line segment from (2, 5) to (x, 15) with length 13 and x > 0, prove x = 2 + √69 -/
theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  ((x - 2)^2 + 10^2)^(1/2 : ℝ) = 13 → 
  x = 2 + (69 : ℝ)^(1/2 : ℝ) := by
sorry

end line_segment_endpoint_l3156_315601


namespace first_player_win_prob_correct_l3156_315652

/-- Represents the probability of winning for the first player in a three-player sequential game -/
def first_player_win_probability : ℚ :=
  729 / 5985

/-- The probability of a successful hit on any turn -/
def hit_probability : ℚ := 1 / 3

/-- The number of players in the game -/
def num_players : ℕ := 3

/-- Theorem stating the probability of the first player winning the game -/
theorem first_player_win_prob_correct :
  let p := hit_probability
  let n := num_players
  (p^2 * (1 - p^(2*n))⁻¹ : ℚ) = first_player_win_probability :=
by sorry

end first_player_win_prob_correct_l3156_315652


namespace milk_sales_l3156_315650

theorem milk_sales : 
  let morning_packets : ℕ := 150
  let morning_250ml : ℕ := 60
  let morning_300ml : ℕ := 40
  let morning_350ml : ℕ := morning_packets - morning_250ml - morning_300ml
  let evening_packets : ℕ := 100
  let evening_400ml : ℕ := evening_packets / 2
  let evening_500ml : ℕ := evening_packets / 4
  let evening_450ml : ℕ := evening_packets - evening_400ml - evening_500ml
  let ml_per_ounce : ℕ := 30
  let remaining_ml : ℕ := 42000
  let total_ml : ℕ := 
    morning_250ml * 250 + morning_300ml * 300 + morning_350ml * 350 +
    evening_400ml * 400 + evening_500ml * 500 + evening_450ml * 450
  let sold_ml : ℕ := total_ml - remaining_ml
  let sold_ounces : ℚ := sold_ml / ml_per_ounce
  sold_ounces = 1541.67 := by
    sorry

end milk_sales_l3156_315650


namespace circle_op_range_theorem_l3156_315646

/-- Custom operation ⊙ on real numbers -/
def circle_op (a b : ℝ) : ℝ := a * b - 2 * a - b

/-- Theorem stating the range of x for which x ⊙ (x+2) < 0 -/
theorem circle_op_range_theorem :
  ∀ x : ℝ, circle_op x (x + 2) < 0 ↔ -1 < x ∧ x < 2 := by
  sorry

end circle_op_range_theorem_l3156_315646


namespace differential_equation_solution_l3156_315626

open Real

/-- The differential equation (x^3 + xy^2) dx + (x^2y + y^3) dy = 0 has a solution F(x, y) = x^4 + 2(xy)^2 + y^4 -/
theorem differential_equation_solution (x y : ℝ) :
  let F : ℝ × ℝ → ℝ := fun (x, y) ↦ x^4 + 2*(x*y)^2 + y^4
  let dFdx : ℝ × ℝ → ℝ := fun (x, y) ↦ 4*x^3 + 4*x*y^2
  let dFdy : ℝ × ℝ → ℝ := fun (x, y) ↦ 4*x^2*y + 4*y^3
  (x^3 + x*y^2) * dFdx (x, y) + (x^2*y + y^3) * dFdy (x, y) = 0 := by
  sorry

end differential_equation_solution_l3156_315626


namespace sqrt_sum_difference_l3156_315678

theorem sqrt_sum_difference (x : ℝ) : 
  Real.sqrt ((5 - 4 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 4 * Real.sqrt 2) ^ 2) - Real.sqrt 16 = 6 := by
  sorry

end sqrt_sum_difference_l3156_315678


namespace unique_function_existence_l3156_315663

theorem unique_function_existence : 
  ∃! f : ℕ → ℕ, f 1 = 1 ∧ ∀ n : ℕ, f n * f (n + 2) = f (n + 1) ^ 2 + 1997 := by
  sorry

end unique_function_existence_l3156_315663


namespace hamburger_combinations_count_l3156_315665

/-- The number of condiment choices available. -/
def num_condiments : ℕ := 9

/-- The number of options for meat patties. -/
def patty_options : ℕ := 3

/-- Calculates the number of different hamburger combinations. -/
def hamburger_combinations : ℕ := patty_options * 2^num_condiments

/-- Theorem stating that the number of different hamburger combinations is 1536. -/
theorem hamburger_combinations_count : hamburger_combinations = 1536 := by
  sorry

end hamburger_combinations_count_l3156_315665


namespace sphere_stack_ratio_l3156_315608

theorem sphere_stack_ratio (n : ℕ) (sphere_volume_ratio : ℚ) 
  (h1 : n = 5)
  (h2 : sphere_volume_ratio = 2/3) : 
  (n : ℚ) * (1 - sphere_volume_ratio) / (n * sphere_volume_ratio) = 1/2 := by
  sorry

end sphere_stack_ratio_l3156_315608


namespace proportion_ones_is_42_233_l3156_315692

/-- The number of three-digit integers -/
def num_three_digit_ints : ℕ := 999 - 100 + 1

/-- The total number of digits in all three-digit integers -/
def total_digits : ℕ := num_three_digit_ints * 3

/-- The number of times each digit (1-9) appears in the three-digit integers -/
def digit_occurrences : ℕ := 100 + 90 + 90

/-- The number of times zero appears in the three-digit integers -/
def zero_occurrences : ℕ := 90 + 90

/-- The total number of digits after squaring -/
def total_squared_digits : ℕ := 
  (4 * digit_occurrences + zero_occurrences) + (6 * digit_occurrences * 2)

/-- The number of ones after squaring -/
def num_ones : ℕ := 3 * digit_occurrences

/-- The proportion of ones in the squared digits -/
def proportion_ones : ℚ := num_ones / total_squared_digits

theorem proportion_ones_is_42_233 : proportion_ones = 42 / 233 := by sorry

end proportion_ones_is_42_233_l3156_315692


namespace right_triangle_hypotenuse_l3156_315641

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 60 →
  b = 100 →
  c^2 = a^2 + b^2 →
  c = 20 * Real.sqrt 34 :=
by sorry

end right_triangle_hypotenuse_l3156_315641


namespace exponential_linear_inequalities_l3156_315611

/-- The exponential function -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x

/-- A linear function with slope k -/
def g (k : ℝ) (x : ℝ) : ℝ := k * x + 1

theorem exponential_linear_inequalities (k : ℝ) :
  (∃ (y : ℝ), ∀ (x : ℝ), f x - (x + 1) ≥ y ∧ ∃ (x : ℝ), f x - (x + 1) = y) ∧
  (k > 1 → ∃ (x₀ : ℝ), x₀ > 0 ∧ ∀ (x : ℝ), 0 < x ∧ x < x₀ → f x < g k x) ∧
  (∃ (m : ℝ), m > 0 ∧ ∀ (x : ℝ), 0 < x ∧ x < m → |f x - g k x| > x) ↔ (k ≤ 0 ∨ k > 2) := by
  sorry

end exponential_linear_inequalities_l3156_315611


namespace horner_rule_equality_f_at_two_equals_62_l3156_315687

/-- Horner's Rule representation of a polynomial -/
def horner_form (a b c d e : ℝ) (x : ℝ) : ℝ :=
  x * (x * (x * (a * x + b) + c) + d) + e

/-- Original polynomial function -/
def f (x : ℝ) : ℝ :=
  2 * x^4 + 3 * x^3 + 5 * x - 4

theorem horner_rule_equality :
  ∀ x : ℝ, f x = horner_form 2 3 0 5 (-4) x :=
sorry

theorem f_at_two_equals_62 : f 2 = 62 := by
  sorry

end horner_rule_equality_f_at_two_equals_62_l3156_315687


namespace original_wage_calculation_l3156_315630

/-- The worker's original daily wage -/
def original_wage : ℝ := 242.83

/-- The new total weekly salary -/
def new_total_salary : ℝ := 1457

/-- The percentage increases for each day of the work week -/
def wage_increases : List ℝ := [0.20, 0.30, 0.40, 0.50, 0.60]

theorem original_wage_calculation :
  (wage_increases.map (λ i => (1 + i) * original_wage)).sum = new_total_salary :=
sorry

end original_wage_calculation_l3156_315630


namespace sum_remainder_mod_seven_l3156_315633

theorem sum_remainder_mod_seven (n : ℤ) : ((7 + n) + (n + 5)) % 7 = (5 + 2*n) % 7 := by
  sorry

end sum_remainder_mod_seven_l3156_315633


namespace imaginary_part_of_z_l3156_315606

theorem imaginary_part_of_z (z : ℂ) (h : z * (Complex.I + 1) + Complex.I = 1 + 3 * Complex.I) : 
  z.im = (1 : ℝ) / 2 := by
  sorry

end imaginary_part_of_z_l3156_315606


namespace field_length_proof_l3156_315612

theorem field_length_proof (width : ℝ) (length : ℝ) (pond_side : ℝ) :
  length = 2 * width →
  pond_side = 9 →
  pond_side^2 = (1/8) * (length * width) →
  length = 36 := by
sorry

end field_length_proof_l3156_315612


namespace maria_number_transformation_l3156_315667

theorem maria_number_transformation (x : ℚ) : 
  (2 * (x + 3) - 2) / 3 = 8 → x = 10 := by sorry

end maria_number_transformation_l3156_315667


namespace math_competition_nonparticipants_l3156_315688

theorem math_competition_nonparticipants (total_students : ℕ) 
  (h1 : total_students = 39) 
  (h2 : ∃ participants : ℕ, participants = total_students / 3) : 
  ∃ nonparticipants : ℕ, nonparticipants = 26 ∧ nonparticipants = total_students - (total_students / 3) :=
by sorry

end math_competition_nonparticipants_l3156_315688


namespace first_hour_rate_is_25_l3156_315683

/-- Represents the rental cost structure for a power tool -/
structure RentalCost where
  firstHourRate : ℕ
  additionalHourRate : ℕ

/-- Represents the rental details for Ashwin -/
structure RentalDetails where
  totalCost : ℕ
  totalHours : ℕ

/-- Theorem stating that given the rental conditions, the first hour rate was $25 -/
theorem first_hour_rate_is_25 (rental : RentalCost) (details : RentalDetails) :
  rental.additionalHourRate = 10 ∧
  details.totalCost = 125 ∧
  details.totalHours = 11 →
  rental.firstHourRate = 25 := by
  sorry

#check first_hour_rate_is_25

end first_hour_rate_is_25_l3156_315683


namespace multiply_72518_by_9999_l3156_315689

theorem multiply_72518_by_9999 : 72518 * 9999 = 725107482 := by
  sorry

end multiply_72518_by_9999_l3156_315689


namespace otimes_inequality_implies_a_range_l3156_315655

-- Define the custom operation ⊗
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- Define the theorem
theorem otimes_inequality_implies_a_range :
  (∀ x ∈ Set.Icc 1 2, otimes (x - a) (x + a) < 2) →
  -1 < a ∧ a < 2 := by
  sorry

end otimes_inequality_implies_a_range_l3156_315655


namespace circle_center_coordinates_l3156_315625

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- The center of a circle passing through three given points -/
def circleCenterThroughThreePoints (A B C : Point) : Point :=
  sorry

/-- The three given points -/
def A : Point := ⟨2, 2⟩
def B : Point := ⟨6, 2⟩
def C : Point := ⟨4, 5⟩

/-- Theorem stating that the center of the circle passing through A, B, and C is (4, 17/6) -/
theorem circle_center_coordinates : 
  let center := circleCenterThroughThreePoints A B C
  center.x = 4 ∧ center.y = 17/6 := by
  sorry

end circle_center_coordinates_l3156_315625


namespace product_of_smallest_primes_l3156_315654

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 0 → m < n → n % m ≠ 0

/-- A one-digit number is a natural number less than 10. -/
def isOneDigit (n : ℕ) : Prop := n < 10

/-- A two-digit number is a natural number greater than or equal to 10 and less than 100. -/
def isTwoDigit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

/-- The two smallest one-digit primes are 2 and 3. -/
axiom smallest_one_digit_primes : ∀ n : ℕ, isPrime n → isOneDigit n → n = 2 ∨ n = 3

/-- The smallest two-digit prime is 11. -/
axiom smallest_two_digit_prime : ∀ n : ℕ, isPrime n → isTwoDigit n → n ≥ 11

theorem product_of_smallest_primes : 
  ∃ p q r : ℕ, 
    isPrime p ∧ isOneDigit p ∧
    isPrime q ∧ isOneDigit q ∧
    isPrime r ∧ isTwoDigit r ∧
    p * q * r = 66 :=
sorry

end product_of_smallest_primes_l3156_315654


namespace diagonals_25_sided_polygon_l3156_315686

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 25 sides has 275 diagonals -/
theorem diagonals_25_sided_polygon : num_diagonals 25 = 275 := by
  sorry

end diagonals_25_sided_polygon_l3156_315686


namespace sum_distinct_prime_divisors_of_1728_l3156_315668

theorem sum_distinct_prime_divisors_of_1728 : 
  (Finset.sum (Finset.filter Nat.Prime (Nat.divisors 1728)) id) = 5 := by
  sorry

end sum_distinct_prime_divisors_of_1728_l3156_315668


namespace rectangle_area_puzzle_l3156_315604

/-- Given a rectangle divided into six smaller rectangles, if five of the rectangles
    have areas 126, 63, 161, 20, and 40, then the area of the remaining rectangle is 101. -/
theorem rectangle_area_puzzle (A B C D E F : ℝ) :
  A = 126 →
  B = 63 →
  C = 161 →
  D = 20 →
  E = 40 →
  A + B + C + D + E + F = (A + B) + C →
  F = 101 := by
  sorry

end rectangle_area_puzzle_l3156_315604


namespace system_solution_l3156_315698

theorem system_solution :
  let x : ℝ := (133 - Real.sqrt 73) / 48
  let y : ℝ := (-1 + Real.sqrt 73) / 12
  2 * x - 3 * y^2 = 4 ∧ 4 * x + y = 11 := by
  sorry

end system_solution_l3156_315698


namespace arithmetic_triangle_theorem_l3156_315600

/-- Triangle with sides a, b, c and angles A, B, C in arithmetic sequence --/
structure ArithmeticTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_arithmetic_sequence : True  -- represents that angles are in arithmetic sequence

/-- The theorem to be proved --/
theorem arithmetic_triangle_theorem (t : ArithmeticTriangle) : 
  1 / (t.a + t.b) + 1 / (t.b + t.c) = 3 / (t.a + t.b + t.c) := by
  sorry

end arithmetic_triangle_theorem_l3156_315600


namespace floor_ceil_sum_l3156_315699

theorem floor_ceil_sum : ⌊(0.998 : ℝ)⌋ + ⌈(3.002 : ℝ)⌉ = 4 := by
  sorry

end floor_ceil_sum_l3156_315699


namespace bake_sale_goal_l3156_315614

def brownie_count : ℕ := 4
def brownie_price : ℕ := 3
def lemon_square_count : ℕ := 5
def lemon_square_price : ℕ := 2
def cookie_count : ℕ := 7
def cookie_price : ℕ := 4

def total_goal : ℕ := 50

theorem bake_sale_goal :
  brownie_count * brownie_price +
  lemon_square_count * lemon_square_price +
  cookie_count * cookie_price = total_goal :=
by
  sorry

end bake_sale_goal_l3156_315614


namespace power_multiplication_l3156_315653

theorem power_multiplication (a b : ℕ) : (10 : ℝ) ^ a * (10 : ℝ) ^ b = (10 : ℝ) ^ (a + b) := by
  sorry

end power_multiplication_l3156_315653


namespace lukes_trivia_score_l3156_315628

/-- Luke's trivia game score calculation -/
theorem lukes_trivia_score (rounds : ℕ) (points_per_round : ℕ) (h1 : rounds = 177) (h2 : points_per_round = 46) :
  rounds * points_per_round = 8142 := by
  sorry

end lukes_trivia_score_l3156_315628


namespace waiting_by_stump_is_random_event_l3156_315694

-- Define the type for idioms
inductive Idiom
| WaitingByStump
| MarkingBoat
| ScoopingMoon
| MendingMirror

-- Define the property of being a random event
def isRandomEvent (i : Idiom) : Prop :=
  match i with
  | Idiom.WaitingByStump => true
  | _ => false

-- Theorem statement
theorem waiting_by_stump_is_random_event :
  isRandomEvent Idiom.WaitingByStump = true :=
by sorry

end waiting_by_stump_is_random_event_l3156_315694


namespace function_value_at_five_l3156_315644

theorem function_value_at_five (f : ℝ → ℝ) 
  (h : ∀ x, f x + 3 * f (1 - x) = 2 * x^2 + x) : 
  f 5 = 29/8 := by
  sorry

end function_value_at_five_l3156_315644


namespace factor_expression_l3156_315684

theorem factor_expression (a b : ℝ) : 9 * a^2 * b - b = b * (3 * a + 1) * (3 * a - 1) := by
  sorry

end factor_expression_l3156_315684


namespace least_non_lucky_multiple_of_7_l3156_315661

def sumOfDigits (n : ℕ) : ℕ := sorry

def isLucky (n : ℕ) : Prop := n % (sumOfDigits n) = 0

def isMultipleOf7 (n : ℕ) : Prop := ∃ k : ℕ, n = 7 * k

theorem least_non_lucky_multiple_of_7 : 
  (isMultipleOf7 14) ∧ 
  ¬(isLucky 14) ∧ 
  ∀ n : ℕ, 0 < n ∧ n < 14 ∧ (isMultipleOf7 n) → (isLucky n) := by sorry

end least_non_lucky_multiple_of_7_l3156_315661


namespace hyperbola_center_l3156_315631

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 360 * y - 1001 = 0

/-- The center of a hyperbola -/
def is_center (c : ℝ × ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), ∀ (x y : ℝ), 
    eq x y ↔ ((x - c.1)^2 / a^2) - ((y - c.2)^2 / b^2) = 1

/-- Theorem: The center of the given hyperbola is (3, 5) -/
theorem hyperbola_center : is_center (3, 5) hyperbola_equation :=
sorry

end hyperbola_center_l3156_315631


namespace clara_age_in_five_years_l3156_315613

/-- Given the conditions about Alice and Clara's pens and ages, prove Clara's age in 5 years. -/
theorem clara_age_in_five_years
  (alice_pens : ℕ)
  (clara_pens_ratio : ℚ)
  (alice_age : ℕ)
  (clara_older : Prop)
  (pen_diff_equals_age_diff : Prop)
  (h1 : alice_pens = 60)
  (h2 : clara_pens_ratio = 2 / 5)
  (h3 : alice_age = 20)
  (h4 : clara_older)
  (h5 : pen_diff_equals_age_diff) :
  ∃ (clara_age : ℕ), clara_age + 5 = 61 :=
by sorry

end clara_age_in_five_years_l3156_315613


namespace reflected_ray_equation_l3156_315617

/-- Given a point M and its reflection N across the y-axis, and a point P on the y-axis,
    this theorem states that the line passing through P and N has the equation x - y + 1 = 0. -/
theorem reflected_ray_equation (M P N : ℝ × ℝ) : 
  M.1 = 3 ∧ M.2 = -2 ∧   -- M(3, -2)
  P.1 = 0 ∧ P.2 = 1 ∧    -- P(0, 1) on y-axis
  N.1 = -M.1 ∧ N.2 = M.2 -- N is reflection of M across y-axis
  → (∀ x y : ℝ, (x - y + 1 = 0) ↔ (∃ t : ℝ, x = N.1 * t + P.1 * (1 - t) ∧ y = N.2 * t + P.2 * (1 - t))) :=
by sorry


end reflected_ray_equation_l3156_315617


namespace special_square_area_l3156_315696

/-- A square with one side on a line and two vertices on a parabola -/
structure SpecialSquare where
  /-- The y-coordinate of vertex C -/
  y1 : ℝ
  /-- The y-coordinate of vertex D -/
  y2 : ℝ
  /-- C and D lie on the parabola y^2 = x -/
  h1 : y1^2 = (y1 : ℝ)
  h2 : y2^2 = (y2 : ℝ)
  /-- Side AB lies on the line y = x + 4 -/
  h3 : y2^2 - y1^2 + y1 = y1^2 + y1 - y2 + 4
  /-- The slope condition -/
  h4 : y1 - y2 = y1^2 - y2^2

/-- The area of a SpecialSquare is either 18 or 50 -/
theorem special_square_area (s : SpecialSquare) : (s.y2 - s.y1)^2 = 18 ∨ (s.y2 - s.y1)^2 = 50 := by
  sorry

end special_square_area_l3156_315696


namespace tan_double_angle_l3156_315669

theorem tan_double_angle (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2) : 
  Real.tan (2 * α) = 3/4 := by
  sorry

end tan_double_angle_l3156_315669


namespace binary_111_equals_7_l3156_315622

def binary_to_decimal (b₂ b₁ b₀ : Nat) : Nat :=
  b₂ * 2^2 + b₁ * 2^1 + b₀ * 2^0

theorem binary_111_equals_7 : binary_to_decimal 1 1 1 = 7 := by
  sorry

end binary_111_equals_7_l3156_315622


namespace radio_cost_price_l3156_315681

/-- Calculates the cost price of an item given its selling price and loss percentage. -/
def cost_price (selling_price : ℚ) (loss_percentage : ℚ) : ℚ :=
  selling_price / (1 - loss_percentage / 100)

/-- Proves that the cost price of a radio sold for 1305 with a 13% loss is 1500. -/
theorem radio_cost_price : cost_price 1305 13 = 1500 := by
  sorry

end radio_cost_price_l3156_315681


namespace product_of_numbers_l3156_315675

theorem product_of_numbers (x y : ℝ) (h1 : x^2 + y^2 = 289) (h2 : x + y = 23) : x * y = 120 := by
  sorry

end product_of_numbers_l3156_315675


namespace min_balloons_required_l3156_315603

/-- Represents a balloon color -/
inductive Color
| A | B | C | D | E

/-- Represents a row of balloons -/
def BalloonRow := List Color

/-- Checks if two colors are adjacent in a balloon row -/
def areAdjacent (row : BalloonRow) (c1 c2 : Color) : Prop :=
  ∃ i, (row.get? i = some c1 ∧ row.get? (i+1) = some c2) ∨
       (row.get? i = some c2 ∧ row.get? (i+1) = some c1)

/-- Checks if all pairs of colors are adjacent in a balloon row -/
def allPairsAdjacent (row : BalloonRow) : Prop :=
  ∀ c1 c2, c1 ≠ c2 → areAdjacent row c1 c2

/-- The main theorem: minimum number of balloons required is 11 -/
theorem min_balloons_required :
  ∀ row : BalloonRow,
    allPairsAdjacent row →
    row.length ≥ 11 ∧
    (∃ row' : BalloonRow, allPairsAdjacent row' ∧ row'.length = 11) :=
by sorry

end min_balloons_required_l3156_315603


namespace multiplication_error_correction_l3156_315610

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Checks if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem multiplication_error_correction 
  (c d : ℕ) 
  (h1 : is_two_digit c) 
  (h2 : (reverse_digits c) * d = 143) : 
  c * d = 341 := by
sorry

end multiplication_error_correction_l3156_315610


namespace sum_of_k_for_minimum_area_l3156_315660

/-- The sum of k values that minimize the triangle area --/
def sum_of_k_values : ℤ := 24

/-- Point type --/
structure Point where
  x : ℚ
  y : ℚ

/-- Triangle defined by three points --/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Function to calculate the area of a triangle --/
def triangle_area (t : Triangle) : ℚ :=
  sorry

/-- Function to check if a triangle has minimum area --/
def has_minimum_area (t : Triangle) : Prop :=
  sorry

/-- Theorem stating the sum of k values that minimize the triangle area --/
theorem sum_of_k_for_minimum_area :
  ∃ (k1 k2 : ℤ),
    k1 ≠ k2 ∧
    has_minimum_area (Triangle.mk
      (Point.mk 2 9)
      (Point.mk 14 18)
      (Point.mk 6 k1)) ∧
    has_minimum_area (Triangle.mk
      (Point.mk 2 9)
      (Point.mk 14 18)
      (Point.mk 6 k2)) ∧
    k1 + k2 = sum_of_k_values :=
  sorry

end sum_of_k_for_minimum_area_l3156_315660


namespace triangle_angle_sum_l3156_315671

theorem triangle_angle_sum (A B : Real) (hA : 0 < A ∧ A < π/2) (hB : 0 < B ∧ B < π/2)
  (hsinA : Real.sin A = Real.sqrt 5 / 5) (hsinB : Real.sin B = Real.sqrt 10 / 10) :
  A + B = π/4 := by
sorry

end triangle_angle_sum_l3156_315671


namespace fraction_inequality_l3156_315695

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  c / a < c / b := by
  sorry

end fraction_inequality_l3156_315695


namespace sin_cos_fourth_power_range_l3156_315607

theorem sin_cos_fourth_power_range :
  ∀ x : ℝ, (1/2 : ℝ) ≤ Real.sin x ^ 4 + Real.cos x ^ 4 ∧ Real.sin x ^ 4 + Real.cos x ^ 4 ≤ 1 := by
  sorry

end sin_cos_fourth_power_range_l3156_315607


namespace total_team_score_l3156_315682

def team_score (team_size : ℕ) (faye_score : ℕ) (other_player_score : ℕ) : ℕ :=
  faye_score + (team_size - 1) * other_player_score

theorem total_team_score :
  team_score 5 28 8 = 60 := by
  sorry

end total_team_score_l3156_315682


namespace percentage_gain_calculation_l3156_315638

/-- Calculates the percentage gain when selling an article --/
theorem percentage_gain_calculation (cost_price selling_price : ℚ) : 
  cost_price = 160 → 
  selling_price = 192 → 
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end percentage_gain_calculation_l3156_315638


namespace clock_hands_90_degree_times_l3156_315605

/-- The angle (in degrees) that the minute hand moves per minute -/
def minute_hand_speed : ℚ := 6

/-- The angle (in degrees) that the hour hand moves per minute -/
def hour_hand_speed : ℚ := 1/2

/-- The relative speed (in degrees per minute) at which the minute hand moves compared to the hour hand -/
def relative_speed : ℚ := minute_hand_speed - hour_hand_speed

/-- The time (in minutes) when the clock hands first form a 90° angle after 12:00 -/
def first_90_degree_time : ℚ := 90 / relative_speed

/-- The time (in minutes) when the clock hands form a 90° angle for the second time after 12:00 -/
def second_90_degree_time : ℚ := 270 / relative_speed

theorem clock_hands_90_degree_times :
  (first_90_degree_time = 180/11) ∧ 
  (second_90_degree_time = 540/11) := by
  sorry

end clock_hands_90_degree_times_l3156_315605
