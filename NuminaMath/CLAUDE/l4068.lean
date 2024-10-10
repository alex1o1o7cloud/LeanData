import Mathlib

namespace min_marked_cells_l4068_406878

/-- Represents a board with dimensions m × n -/
structure Board (m n : ℕ) where
  cells : Fin m → Fin n → Bool

/-- Represents an L-shaped piece -/
inductive LPiece
  | mk : Fin 2 → Fin 2 → LPiece

/-- Checks if an L-piece touches a marked cell on the board -/
def touchesMarkedCell (b : Board m n) (l : LPiece) (i j : ℕ) : Prop :=
  ∃ (x y : Fin 2), b.cells ⟨i + x.val, sorry⟩ ⟨j + y.val, sorry⟩ = true

/-- A marking strategy for the board -/
def markingStrategy (b : Board m n) : Prop :=
  ∀ (i : Fin m) (j : Fin n), i.val % 2 = 0 → b.cells i j = true

/-- The main theorem stating that 50 is the smallest number of marked cells
    that ensures any L-shaped piece touches at least one marked cell on a 10 × 11 board -/
theorem min_marked_cells :
  ∀ (b : Board 10 11),
    (∃ (k : ℕ), k < 50 ∧
      (∀ (l : LPiece) (i j : ℕ), i < 9 ∧ j < 10 →
        touchesMarkedCell b l i j)) →
    (∃ (b' : Board 10 11),
      markingStrategy b' ∧
      (∀ (l : LPiece) (i j : ℕ), i < 9 ∧ j < 10 →
        touchesMarkedCell b' l i j) ∧
      (∃ (k : ℕ), k = 50 ∧
        k = (Finset.filter (fun i => b'.cells i.1 i.2) (Finset.product (Finset.range 10) (Finset.range 11))).card)) :=
by
  sorry


end min_marked_cells_l4068_406878


namespace solution_set_equivalence_minimum_value_l4068_406896

-- Define the function f
def f (m n : ℝ) (x : ℝ) : ℝ := m * x^2 - n * x

-- Part 1
theorem solution_set_equivalence
  (m n t : ℝ)
  (h1 : ∀ x, f m n x ≥ t ↔ -3 ≤ x ∧ x ≤ 2) :
  ∀ x, n * x^2 + m * x + t ≤ 0 ↔ -2 ≤ x ∧ x ≤ 3 :=
sorry

-- Part 2
theorem minimum_value
  (m n : ℝ)
  (h1 : f m n 1 > 0)
  (h2 : 1 ≤ m ∧ m ≤ 3) :
  ∃ (m₀ n₀ : ℝ), 1/(m₀-n₀) + 9/m₀ - n₀ = 2 ∧
    ∀ m n, f m n 1 > 0 → 1 ≤ m ∧ m ≤ 3 → 1/(m-n) + 9/m - n ≥ 2 :=
sorry

end solution_set_equivalence_minimum_value_l4068_406896


namespace total_emails_received_l4068_406831

def morning_emails : ℕ := 3
def afternoon_emails : ℕ := 5

theorem total_emails_received : morning_emails + afternoon_emails = 8 := by
  sorry

end total_emails_received_l4068_406831


namespace ribbon_length_proof_l4068_406874

/-- Calculates the total length of a ribbon before division, given the number of students,
    length per student, and leftover length. -/
def totalRibbonLength (numStudents : ℕ) (lengthPerStudent : ℝ) (leftover : ℝ) : ℝ :=
  (numStudents : ℝ) * lengthPerStudent + leftover

/-- Proves that for 10 students, 0.84 meters per student, and 0.50 meters leftover,
    the total ribbon length before division was 8.9 meters. -/
theorem ribbon_length_proof :
  totalRibbonLength 10 0.84 0.50 = 8.9 :=
by sorry

end ribbon_length_proof_l4068_406874


namespace bus_trip_distance_l4068_406839

theorem bus_trip_distance (v : ℝ) (d : ℝ) : 
  v = 40 → 
  d / v - d / (v + 5) = 1 → 
  d = 360 := by sorry

end bus_trip_distance_l4068_406839


namespace total_red_stripes_on_ten_flags_l4068_406843

/-- Represents an American flag -/
structure AmericanFlag where
  stripes : ℕ
  firstStripeRed : Bool
  halfRemainingRed : Bool

/-- Calculates the number of red stripes on a single American flag -/
def redStripesPerFlag (flag : AmericanFlag) : ℕ :=
  if flag.firstStripeRed ∧ flag.halfRemainingRed then
    1 + (flag.stripes - 1) / 2
  else
    0

/-- Theorem stating the total number of red stripes on 10 American flags -/
theorem total_red_stripes_on_ten_flags :
  ∀ (flag : AmericanFlag),
    flag.stripes = 13 →
    flag.firstStripeRed = true →
    flag.halfRemainingRed = true →
    (redStripesPerFlag flag * 10 = 70) :=
by
  sorry

end total_red_stripes_on_ten_flags_l4068_406843


namespace sum_of_coordinates_A_l4068_406809

/-- Given three points A, B, and C in a plane, where C divides AB in a 1:3 ratio,
    prove that the sum of coordinates of A is 21 when B and C are known. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (C.1 - A.1) / (B.1 - A.1) = 1/3 →
  (C.2 - A.2) / (B.2 - A.2) = 1/3 →
  B = (2, 10) →
  C = (5, 4) →
  A.1 + A.2 = 21 :=
by sorry

end sum_of_coordinates_A_l4068_406809


namespace cube_volume_problem_l4068_406805

theorem cube_volume_problem : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a^3 - ((a + 2) * (a - 2) * (a + 3)) = 7) ∧ 
  (a^3 = 27) := by
  sorry

end cube_volume_problem_l4068_406805


namespace abc_sum_l4068_406898

theorem abc_sum (a b c : ℕ+) (h : (139 : ℚ) / 22 = a + 1 / (b + 1 / c)) : 
  (a : ℕ) + b + c = 16 := by
  sorry

end abc_sum_l4068_406898


namespace similar_triangles_side_length_l4068_406819

-- Define the triangles and their sides
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the similarity relation between triangles
def similar (t1 t2 : Triangle) : Prop := sorry

-- Given values
def ABC : Triangle := { A := 15, B := 0, C := 24 }
def FGH : Triangle := { A := 0, B := 0, C := 18 }

-- Theorem statement
theorem similar_triangles_side_length :
  similar ABC FGH →
  FGH.A = 11.25 := by
  sorry

end similar_triangles_side_length_l4068_406819


namespace hyperbola_properties_l4068_406802

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

-- Define eccentricity
def eccentricity (e : ℝ) : Prop := e = 2

-- Define asymptotes
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

theorem hyperbola_properties :
  ∃ (e : ℝ), eccentricity e ∧
  ∀ (x y : ℝ), hyperbola x y → asymptotes x y :=
sorry

end hyperbola_properties_l4068_406802


namespace basketball_series_probability_l4068_406863

/-- The probability of at least k successes in n independent trials with probability p -/
def prob_at_least (n k : ℕ) (p : ℝ) : ℝ := sorry

theorem basketball_series_probability :
  prob_at_least 9 5 (1/2) = 1/2 := by sorry

end basketball_series_probability_l4068_406863


namespace number_problem_l4068_406821

theorem number_problem : ∃ x : ℚ, x - (3/5) * x = 56 ∧ x = 140 := by sorry

end number_problem_l4068_406821


namespace seashell_collection_l4068_406844

theorem seashell_collection (stefan vail aiguo fatima : ℕ) : 
  stefan = vail + 16 →
  vail + 5 = aiguo →
  aiguo = 20 →
  fatima = 2 * aiguo →
  stefan + vail + aiguo + fatima = 106 := by
sorry

end seashell_collection_l4068_406844


namespace program_result_l4068_406850

theorem program_result : ∀ (x : ℕ), 
  x = 51 → 
  9 < x → 
  x < 100 → 
  let a := x / 10
  let b := x % 10
  10 * b + a = 15 := by
sorry

end program_result_l4068_406850


namespace roses_flats_is_three_l4068_406868

/-- Represents the plant shop inventory and fertilizer requirements --/
structure PlantShop where
  petunia_flats : ℕ
  petunias_per_flat : ℕ
  roses_per_flat : ℕ
  venus_flytraps : ℕ
  petunia_fertilizer : ℕ
  rose_fertilizer : ℕ
  venus_flytrap_fertilizer : ℕ
  total_fertilizer : ℕ

/-- Calculates the number of flats of roses in the shop --/
def roses_flats (shop : PlantShop) : ℕ :=
  let petunia_total := shop.petunia_flats * shop.petunias_per_flat * shop.petunia_fertilizer
  let venus_total := shop.venus_flytraps * shop.venus_flytrap_fertilizer
  let roses_total := shop.total_fertilizer - petunia_total - venus_total
  roses_total / (shop.roses_per_flat * shop.rose_fertilizer)

/-- Theorem stating that the number of rose flats is 3 --/
theorem roses_flats_is_three (shop : PlantShop)
  (h1 : shop.petunia_flats = 4)
  (h2 : shop.petunias_per_flat = 8)
  (h3 : shop.roses_per_flat = 6)
  (h4 : shop.venus_flytraps = 2)
  (h5 : shop.petunia_fertilizer = 8)
  (h6 : shop.rose_fertilizer = 3)
  (h7 : shop.venus_flytrap_fertilizer = 2)
  (h8 : shop.total_fertilizer = 314) :
  roses_flats shop = 3 := by
  sorry

end roses_flats_is_three_l4068_406868


namespace number_operation_proof_l4068_406853

theorem number_operation_proof (x : ℝ) : x = 115 → (((x + 45) / 2) / 2) + 45 = 85 := by
  sorry

end number_operation_proof_l4068_406853


namespace marbles_lost_l4068_406864

/-- 
Given that Josh initially had 9 marbles and now has 4 marbles,
prove that the number of marbles he lost is 5.
-/
theorem marbles_lost (initial : ℕ) (current : ℕ) (lost : ℕ) : 
  initial = 9 → current = 4 → lost = initial - current → lost = 5 := by
  sorry

end marbles_lost_l4068_406864


namespace cut_square_equation_l4068_406895

/-- Represents the dimensions of a rectangular sheet and the side length of squares cut from its corners. -/
structure SheetDimensions where
  length : ℝ
  width : ℝ
  cutSide : ℝ

/-- Calculates the area of the base of a box formed by cutting squares from a sheet's corners. -/
def baseArea (d : SheetDimensions) : ℝ :=
  (d.length - 2 * d.cutSide) * (d.width - 2 * d.cutSide)

/-- Calculates the original area of a rectangular sheet. -/
def originalArea (d : SheetDimensions) : ℝ :=
  d.length * d.width

/-- Theorem stating the relationship between the cut side length and the resulting box dimensions. -/
theorem cut_square_equation (d : SheetDimensions) 
    (h1 : d.length = 8)
    (h2 : d.width = 6)
    (h3 : baseArea d = (2/3) * originalArea d) :
  d.cutSide ^ 2 - 7 * d.cutSide + 4 = 0 := by
  sorry

end cut_square_equation_l4068_406895


namespace min_sum_squared_distances_l4068_406876

/-- Given five collinear points A, B, C, D, and E in that order, with specified distances between them,
    this function calculates the sum of squared distances from these points to any point P on the line. -/
def sum_of_squared_distances (x : ℝ) : ℝ :=
  x^2 + (x - 3)^2 + (x - 4)^2 + (x - 9)^2 + (x - 13)^2

/-- The theorem states that the minimum value of the sum of squared distances
    from five collinear points to any point on their line is 170.24,
    given specific distances between the points. -/
theorem min_sum_squared_distances :
  ∃ (min : ℝ), min = 170.24 ∧
  ∀ (x : ℝ), sum_of_squared_distances x ≥ min :=
by sorry

end min_sum_squared_distances_l4068_406876


namespace work_completion_time_increase_l4068_406811

theorem work_completion_time_increase 
  (initial_men : ℕ) 
  (initial_days : ℕ) 
  (removed_men : ℕ) 
  (h1 : initial_men = 100) 
  (h2 : initial_days = 20) 
  (h3 : removed_men = 50) : 
  (initial_men * initial_days) / (initial_men - removed_men) - initial_days = 20 := by
  sorry

end work_completion_time_increase_l4068_406811


namespace arithmetic_sequence_properties_l4068_406832

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_arithmetic_sequence (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n : ℤ) * (a 1 + a n) / 2

theorem arithmetic_sequence_properties
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = -7)
  (h_s3 : sum_arithmetic_sequence a 3 = -15) :
  (∀ n : ℕ, a n = 2 * n - 9) ∧
  (∀ n : ℕ, sum_arithmetic_sequence a n = (n - 4)^2 - 16) ∧
  (∀ n : ℕ, sum_arithmetic_sequence a n ≥ -16) ∧
  (sum_arithmetic_sequence a 4 = -16) :=
sorry

end arithmetic_sequence_properties_l4068_406832


namespace one_match_theorem_one_empty_theorem_l4068_406817

/-- The number of ways to arrange 4 balls in 4 boxes with exactly one match -/
def arrange_one_match : ℕ := 8

/-- The number of ways to arrange 4 balls in 4 boxes with exactly one empty box -/
def arrange_one_empty : ℕ := 144

/-- Theorem for the number of arrangements with exactly one match -/
theorem one_match_theorem : arrange_one_match = 8 := by sorry

/-- Theorem for the number of arrangements with exactly one empty box -/
theorem one_empty_theorem : arrange_one_empty = 144 := by sorry

end one_match_theorem_one_empty_theorem_l4068_406817


namespace min_points_in_S_l4068_406892

-- Define a point in the xy-plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the set S
def S : Set Point := sorry

-- Define symmetry conditions
def symmetric_origin (s : Set Point) : Prop :=
  ∀ p : Point, p ∈ s → Point.mk (-p.x) (-p.y) ∈ s

def symmetric_x_axis (s : Set Point) : Prop :=
  ∀ p : Point, p ∈ s → Point.mk p.x (-p.y) ∈ s

def symmetric_y_axis (s : Set Point) : Prop :=
  ∀ p : Point, p ∈ s → Point.mk (-p.x) p.y ∈ s

def symmetric_y_eq_x (s : Set Point) : Prop :=
  ∀ p : Point, p ∈ s → Point.mk p.y p.x ∈ s

-- Theorem statement
theorem min_points_in_S :
  symmetric_origin S ∧
  symmetric_x_axis S ∧
  symmetric_y_axis S ∧
  symmetric_y_eq_x S ∧
  Point.mk 2 3 ∈ S →
  ∃ (points : Finset Point), points.card = 8 ∧ ↑points ⊆ S ∧
    (∀ (subset : Finset Point), ↑subset ⊆ S → subset.card < 8 → subset ≠ points) :=
sorry

end min_points_in_S_l4068_406892


namespace sugar_required_for_cake_l4068_406871

/-- Given a recipe for a cake, prove the amount of sugar required -/
theorem sugar_required_for_cake (total_flour : ℕ) (flour_added : ℕ) (extra_sugar : ℕ) : 
  total_flour = 9 → 
  flour_added = 4 → 
  extra_sugar = 6 → 
  (total_flour - flour_added) + extra_sugar = 11 := by
  sorry

end sugar_required_for_cake_l4068_406871


namespace factorization_proof_l4068_406842

theorem factorization_proof (a b x y : ℝ) : x * (a + b) - 2 * y * (a + b) = (a + b) * (x - 2 * y) := by
  sorry

end factorization_proof_l4068_406842


namespace probability_two_specific_people_obtain_items_l4068_406872

-- Define the number of people and items
def num_people : ℕ := 4
def num_items : ℕ := 3

-- Define the probability function
noncomputable def probability_both_obtain (n_people n_items : ℕ) : ℚ :=
  (n_items.choose 2 * (n_people - 2).choose 1) / n_people.choose n_items

-- State the theorem
theorem probability_two_specific_people_obtain_items :
  probability_both_obtain num_people num_items = 1/2 := by
  sorry

end probability_two_specific_people_obtain_items_l4068_406872


namespace equation_solution_l4068_406806

/-- Given positive real numbers a, b, c ≤ 1, the equation 
    min{√((ab+1)/(abc)), √((bc+1)/(abc)), √((ac+1)/(abc))} = √((1-a)/a) + √((1-b)/b) + √((1-c)/c)
    is satisfied if and only if (a, b, c) = (1/(-t^2 + t + 1), t, 1 - t) for 1/2 ≤ t < 1 or its permutations. -/
theorem equation_solution (a b c : ℝ) (ha : 0 < a ∧ a ≤ 1) (hb : 0 < b ∧ b ≤ 1) (hc : 0 < c ∧ c ≤ 1) :
  (min (Real.sqrt ((a*b+1)/(a*b*c))) (min (Real.sqrt ((b*c+1)/(a*b*c))) (Real.sqrt ((a*c+1)/(a*b*c)))) =
   Real.sqrt ((1-a)/a) + Real.sqrt ((1-b)/b) + Real.sqrt ((1-c)/c)) ↔
  (∃ t : ℝ, (1/2 ≤ t ∧ t < 1) ∧
   ((a = 1/(-t^2 + t + 1) ∧ b = t ∧ c = 1 - t) ∨
    (a = t ∧ b = 1 - t ∧ c = 1/(-t^2 + t + 1)) ∨
    (a = 1 - t ∧ b = 1/(-t^2 + t + 1) ∧ c = t))) :=
by sorry

end equation_solution_l4068_406806


namespace divisibility_of_factorials_l4068_406858

theorem divisibility_of_factorials (n : ℕ+) : 
  ∃ k : ℤ, 2 * (3 * n.val).factorial = k * n.val.factorial * (n.val + 1).factorial * (n.val + 2).factorial := by
  sorry

end divisibility_of_factorials_l4068_406858


namespace min_value_implies_a_possible_a_set_l4068_406861

/-- The quadratic function f(x) = x^2 - 2x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The theorem stating the possible values of a -/
theorem min_value_implies_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (a - 2) (a + 2), f x ≥ 6) ∧ 
  (∃ x ∈ Set.Icc (a - 2) (a + 2), f x = 6) →
  a = -3 ∨ a = 5 := by
  sorry

/-- The set of possible values for a -/
def possible_a : Set ℝ := {-3, 5}

/-- The theorem stating that the set of possible values for a is {-3, 5} -/
theorem possible_a_set : 
  ∀ a : ℝ, (∀ x ∈ Set.Icc (a - 2) (a + 2), f x ≥ 6) ∧ 
            (∃ x ∈ Set.Icc (a - 2) (a + 2), f x = 6) ↔ 
            a ∈ possible_a := by
  sorry

end min_value_implies_a_possible_a_set_l4068_406861


namespace unique_solution_condition_l4068_406891

theorem unique_solution_condition (a b : ℝ) : 
  (∃! x : ℝ, 5 * x - 7 + a = 2 * b * x + 3) ↔ (a ≠ 10 ∧ b ≠ 5/2) :=
sorry

end unique_solution_condition_l4068_406891


namespace age_ratio_in_two_years_l4068_406828

def son_age : ℕ := 22
def man_age : ℕ := son_age + 24

def son_age_in_two_years : ℕ := son_age + 2
def man_age_in_two_years : ℕ := man_age + 2

theorem age_ratio_in_two_years :
  man_age_in_two_years / son_age_in_two_years = 2 :=
by sorry

end age_ratio_in_two_years_l4068_406828


namespace least_value_quadratic_l4068_406859

theorem least_value_quadratic (y : ℝ) : 
  (5 * y^2 + 7 * y + 3 = 6) → y ≥ -3 := by
  sorry

end least_value_quadratic_l4068_406859


namespace probability_to_reach_3_3_l4068_406830

/-- Represents a point in a 2D grid --/
structure Point where
  x : Int
  y : Int

/-- Represents a direction of movement --/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- Calculates the probability of reaching the target point from the start point
    in the given number of steps or fewer --/
def probability_to_reach (start : Point) (target : Point) (max_steps : Nat) : Rat :=
  sorry

/-- The main theorem stating the probability of reaching (3,3) from (0,0) in 8 or fewer steps --/
theorem probability_to_reach_3_3 :
  probability_to_reach ⟨0, 0⟩ ⟨3, 3⟩ 8 = 55 / 4096 := by
  sorry

end probability_to_reach_3_3_l4068_406830


namespace green_room_fraction_l4068_406884

theorem green_room_fraction (total_rooms : ℕ) (walls_per_room : ℕ) (purple_walls : ℕ) :
  total_rooms = 10 →
  walls_per_room = 8 →
  purple_walls = 32 →
  (total_rooms : ℚ) - (purple_walls / walls_per_room : ℚ) = 3/5 * total_rooms :=
by sorry

end green_room_fraction_l4068_406884


namespace triangle_inequality_ratio_l4068_406860

theorem triangle_inequality_ratio (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + b^2 + c^2) / (a*b + b*c + c*a) ≥ 1 ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' + b' > c' ∧ b' + c' > a' ∧ c' + a' > b' ∧
    (a'^2 + b'^2 + c'^2) / (a'*b' + b'*c' + c'*a') = 1 :=
by sorry

end triangle_inequality_ratio_l4068_406860


namespace quadratic_always_positive_l4068_406833

theorem quadratic_always_positive (a : ℝ) (h : a > (1/2)) :
  ∀ x : ℝ, a * x^2 + x + (1/2) > 0 := by
sorry

end quadratic_always_positive_l4068_406833


namespace square_shadow_not_trapezoid_l4068_406889

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define a shadow as a quadrilateral
structure Shadow where
  vertices : Fin 4 → ℝ × ℝ

-- Define a uniform light source
structure UniformLight where
  direction : ℝ × ℝ
  direction_nonzero : direction ≠ (0, 0)

-- Define a trapezoid
def is_trapezoid (s : Shadow) : Prop :=
  ∃ (i j : Fin 4), i ≠ j ∧ 
    (s.vertices i).1 - (s.vertices j).1 ≠ 0 ∧
    (s.vertices ((i + 1) % 4)).1 - (s.vertices ((j + 1) % 4)).1 ≠ 0 ∧
    ((s.vertices i).2 - (s.vertices j).2) / ((s.vertices i).1 - (s.vertices j).1) =
    ((s.vertices ((i + 1) % 4)).2 - (s.vertices ((j + 1) % 4)).2) / 
    ((s.vertices ((i + 1) % 4)).1 - (s.vertices ((j + 1) % 4)).1)

-- State the theorem
theorem square_shadow_not_trapezoid 
  (square : Square) (light : UniformLight) (shadow : Shadow) :
  (∃ (projection : Square → UniformLight → Shadow), 
    projection square light = shadow) →
  ¬ is_trapezoid shadow :=
sorry

end square_shadow_not_trapezoid_l4068_406889


namespace symmetric_polynomial_value_l4068_406862

/-- Given a function f(x) = (x² + 3x)(x² + ax + b) where f(x) = f(2-x) for all real x, prove f(3) = -18 -/
theorem symmetric_polynomial_value (a b : ℝ) :
  (∀ x : ℝ, (x^2 + 3*x) * (x^2 + a*x + b) = ((2-x)^2 + 3*(2-x)) * ((2-x)^2 + a*(2-x) + b)) →
  (3^2 + 3*3) * (3^2 + a*3 + b) = -18 := by
  sorry

end symmetric_polynomial_value_l4068_406862


namespace bicycle_discount_l4068_406849

theorem bicycle_discount (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  original_price = 200 ∧ 
  discount1 = 0.4 ∧ 
  discount2 = 0.25 → 
  original_price * (1 - discount1) * (1 - discount2) = 90 := by
  sorry

end bicycle_discount_l4068_406849


namespace point_k_value_l4068_406836

theorem point_k_value (A B C K : ℝ) : 
  A = -3 → B = -5 → C = 6 → 
  (A + B + C + K = -A - B - C - K) → 
  K = 2 := by sorry

end point_k_value_l4068_406836


namespace sum_of_xyz_l4068_406841

theorem sum_of_xyz (x y z : ℝ) 
  (h1 : (x - 4)^2 + (y - 3)^2 + (z - 2)^2 = 0)
  (h2 : 3*x + 2*y - z = 12) : 
  x + y + z = 9 := by
  sorry

end sum_of_xyz_l4068_406841


namespace circumcircle_incircle_diameter_implies_equilateral_l4068_406875

-- Define a triangle
structure Triangle where
  -- We don't need to specify the vertices, just that it's a triangle
  is_triangle : Bool

-- Define the circumcircle and incircle of a triangle
def circumcircle (t : Triangle) : ℝ := sorry
def incircle (t : Triangle) : ℝ := sorry

-- Define what it means for a triangle to be equilateral
def is_equilateral (t : Triangle) : Prop := sorry

-- State the theorem
theorem circumcircle_incircle_diameter_implies_equilateral (t : Triangle) :
  circumcircle t = 2 * incircle t → is_equilateral t := by
  sorry


end circumcircle_incircle_diameter_implies_equilateral_l4068_406875


namespace sum_of_distances_to_intersection_points_l4068_406882

noncomputable section

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x + y = 3
def C₂ (x y : ℝ) : Prop := y^2 = 2*x

-- Define point P
def P : ℝ × ℝ := (1, 2)

-- Define the distance function
def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Theorem statement
theorem sum_of_distances_to_intersection_points :
  ∃ (A B : ℝ × ℝ),
    C₁ A.1 A.2 ∧ C₂ A.1 A.2 ∧
    C₁ B.1 B.2 ∧ C₂ B.1 B.2 ∧
    distance P A + distance P B = 6 * Real.sqrt 2 :=
sorry

end

end sum_of_distances_to_intersection_points_l4068_406882


namespace octal_arithmetic_equality_l4068_406880

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Addition operation for octal numbers --/
def octal_add : OctalNumber → OctalNumber → OctalNumber := sorry

/-- Subtraction operation for octal numbers --/
def octal_sub : OctalNumber → OctalNumber → OctalNumber := sorry

/-- Conversion from decimal to octal --/
def to_octal : ℕ → OctalNumber := sorry

/-- Theorem: In base 8, 5234₈ - 127₈ + 235₈ = 5344₈ --/
theorem octal_arithmetic_equality :
  octal_sub (octal_add (to_octal 5234) (to_octal 235)) (to_octal 127) = to_octal 5344 := by
  sorry

end octal_arithmetic_equality_l4068_406880


namespace brandon_lost_skittles_l4068_406820

theorem brandon_lost_skittles (initial : ℕ) (final : ℕ) (lost : ℕ) : 
  initial = 96 → final = 87 → initial = final + lost → lost = 9 := by sorry

end brandon_lost_skittles_l4068_406820


namespace sum_of_squares_coefficients_l4068_406885

theorem sum_of_squares_coefficients 
  (a b c d e f : ℤ) 
  (h : ∀ x : ℝ, 729 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) : 
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 8210 := by
sorry

end sum_of_squares_coefficients_l4068_406885


namespace largest_y_coordinate_l4068_406899

theorem largest_y_coordinate (x y : ℝ) :
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 :=
by sorry

end largest_y_coordinate_l4068_406899


namespace vector_equation_holds_l4068_406852

variable {V : Type*} [AddCommGroup V]

/-- Given points A, B, C, M, O in a vector space, 
    prove that AB + MB + BC + OM + CO = AB --/
theorem vector_equation_holds (A B C M O : V) :
  (A - B) + (M - B) + (B - C) + (O - M) + (C - O) = A - B :=
by sorry

end vector_equation_holds_l4068_406852


namespace line_equation_l4068_406897

/-- Circle with center (3, 5) and radius √5 -/
def C : Set (ℝ × ℝ) :=
  {p | (p.1 - 3)^2 + (p.2 - 5)^2 = 5}

/-- Line passing through the center of circle C -/
structure Line where
  k : ℝ
  eq : ℝ × ℝ → Prop := fun p => p.2 - 5 = k * (p.1 - 3)

/-- Point where the line intersects the y-axis -/
def P (l : Line) : ℝ × ℝ := (0, 5 - 3 * l.k)

/-- Intersection points of the line and the circle -/
def intersectionPoints (l : Line) : Set (ℝ × ℝ) :=
  {p ∈ C | l.eq p}

/-- A is the midpoint of PB -/
def isMidpoint (A B : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  A.1 = (P.1 + B.1) / 2 ∧ A.2 = (P.2 + B.2) / 2

theorem line_equation (l : Line) (A B : ℝ × ℝ) 
  (hA : A ∈ intersectionPoints l) (hB : B ∈ intersectionPoints l)
  (hMid : isMidpoint A B (P l)) :
  (l.k = 2 ∧ l.eq = fun p => 2 * p.1 - p.2 - 1 = 0) ∨
  (l.k = -2 ∧ l.eq = fun p => 2 * p.1 + p.2 + 11 = 0) := by
  sorry

end line_equation_l4068_406897


namespace min_value_theorem_l4068_406869

theorem min_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 27) :
  18 ≤ 3 * a + 2 * b + c ∧ ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ * b₀ * c₀ = 27 ∧ 3 * a₀ + 2 * b₀ + c₀ = 18 :=
by
  sorry

end min_value_theorem_l4068_406869


namespace greatest_difference_is_nine_l4068_406822

/-- A three-digit integer in the form 84x that is a multiple of 3 -/
def ValidNumber (x : ℕ) : Prop :=
  x < 10 ∧ (840 + x) % 3 = 0

/-- The set of all valid x values -/
def ValidXSet : Set ℕ :=
  {x | ValidNumber x}

/-- The greatest possible difference between two valid x values -/
theorem greatest_difference_is_nine :
  ∃ (a b : ℕ), a ∈ ValidXSet ∧ b ∈ ValidXSet ∧
    ∀ (x y : ℕ), x ∈ ValidXSet → y ∈ ValidXSet →
      (a - b : ℤ).natAbs ≥ (x - y : ℤ).natAbs ∧
      (a - b : ℤ).natAbs = 9 :=
sorry

end greatest_difference_is_nine_l4068_406822


namespace trajectory_of_Q_equation_l4068_406835

/-- The trajectory of point Q given the conditions in the problem -/
def trajectory_of_Q (x y : ℝ) : Prop :=
  2 * x - y + 5 = 0

/-- The line on which point P moves -/
def line_of_P (x y : ℝ) : Prop :=
  2 * x - y + 3 = 0

/-- Point M is fixed at (-1, 2) -/
def point_M : ℝ × ℝ := (-1, 2)

/-- Q is on the extension line of PM and PM = MQ -/
def Q_condition (P Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ Q.1 - point_M.1 = t * (point_M.1 - P.1) ∧ 
                    Q.2 - point_M.2 = t * (point_M.2 - P.2)

theorem trajectory_of_Q_equation :
  ∀ x y : ℝ, 
    (∃ P : ℝ × ℝ, line_of_P P.1 P.2 ∧ Q_condition P (x, y)) →
    trajectory_of_Q x y :=
sorry

end trajectory_of_Q_equation_l4068_406835


namespace extremum_maximum_at_negative_one_l4068_406881

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) --/
def f_derivative (x : ℝ) : ℝ := 3*x^2 - 3

/-- Theorem stating that x = -1 is the extremum maximum point of f(x) --/
theorem extremum_maximum_at_negative_one :
  ∃ (a : ℝ), a = -1 ∧ 
  (∀ x : ℝ, f x ≤ f a) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - a| ∧ |x - a| < δ → f x < f a) :=
sorry

end extremum_maximum_at_negative_one_l4068_406881


namespace basketball_win_rate_l4068_406838

theorem basketball_win_rate (games_won : ℕ) (first_games : ℕ) (remaining_games : ℕ) (target_win_rate : ℚ) : 
  games_won = 45 →
  first_games = 60 →
  remaining_games = 54 →
  target_win_rate = 3/4 →
  ∃ (additional_wins : ℕ), 
    (games_won + additional_wins : ℚ) / (first_games + remaining_games : ℚ) = target_win_rate ∧
    additional_wins = 41 :=
by sorry

end basketball_win_rate_l4068_406838


namespace periodic_odd_function_sum_l4068_406818

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem periodic_odd_function_sum (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 2)
  (h_odd : is_odd f)
  (h_def : ∀ x, 0 < x → x < 1 → f x = 4^x) :
  f (-5/2) + f 2 = -2 := by
  sorry

end periodic_odd_function_sum_l4068_406818


namespace rectangle_length_l4068_406837

/-- Given a rectangle with perimeter 680 meters and breadth 82 meters, its length is 258 meters. -/
theorem rectangle_length (perimeter breadth : ℝ) (h1 : perimeter = 680) (h2 : breadth = 82) :
  (perimeter / 2) - breadth = 258 :=
by sorry

end rectangle_length_l4068_406837


namespace factor_sum_l4068_406856

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 7) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 54 := by sorry

end factor_sum_l4068_406856


namespace fraction_sum_to_decimal_l4068_406867

theorem fraction_sum_to_decimal : 3/8 + 5/32 = 0.53125 := by
  sorry

end fraction_sum_to_decimal_l4068_406867


namespace median_inequality_l4068_406807

-- Define a right triangle with medians
structure RightTriangle where
  c : ℝ  -- length of hypotenuse
  sa : ℝ  -- length of median to one leg
  sb : ℝ  -- length of median to the other leg
  c_pos : c > 0  -- hypotenuse length is positive

-- State the theorem
theorem median_inequality (t : RightTriangle) : 
  (3/2) * t.c < t.sa + t.sb ∧ t.sa + t.sb ≤ (Real.sqrt 10 / 2) * t.c := by
  sorry

end median_inequality_l4068_406807


namespace chip_paper_usage_l4068_406812

/-- Calculates the number of packs of paper Chip will use during the semester --/
def calculate_packs_of_paper (pages_per_pack : ℕ) (regular_weeks : ℕ) (short_weeks : ℕ) 
  (pages_per_regular_week : ℕ) (pages_per_short_week : ℕ) : ℕ :=
  let total_pages := regular_weeks * pages_per_regular_week + short_weeks * pages_per_short_week
  ((total_pages + pages_per_pack - 1) / pages_per_pack : ℕ)

/-- Theorem stating that Chip will use 6 packs of paper during the semester --/
theorem chip_paper_usage : 
  calculate_packs_of_paper 100 13 3 40 24 = 6 := by
  sorry

end chip_paper_usage_l4068_406812


namespace cube_root_sixteen_over_thirtytwo_l4068_406814

theorem cube_root_sixteen_over_thirtytwo : 
  (16 / 32 : ℝ)^(1/3) = 1 / 2^(1/3) := by sorry

end cube_root_sixteen_over_thirtytwo_l4068_406814


namespace spherical_to_rectangular_conversion_l4068_406846

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 4
  let θ : ℝ := π / 3
  let φ : ℝ := π / 6
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (1, Real.sqrt 3, 2 * Real.sqrt 3) := by sorry

end spherical_to_rectangular_conversion_l4068_406846


namespace power_division_l4068_406851

theorem power_division (a : ℝ) (h : a ≠ 0) : a^3 / a = a^2 := by
  sorry

end power_division_l4068_406851


namespace new_average_after_grace_marks_l4068_406854

/-- Represents the grace marks distribution for different percentile ranges -/
structure GraceMarksDistribution where
  below_25th : ℕ
  between_25th_50th : ℕ
  between_50th_75th : ℕ
  above_75th : ℕ

/-- Represents the class statistics -/
structure ClassStats where
  size : ℕ
  original_average : ℝ
  standard_deviation : ℝ
  percentile_25th : ℝ
  percentile_50th : ℝ
  percentile_75th : ℝ

def calculate_new_average (stats : ClassStats) (grace_marks : GraceMarksDistribution) : ℝ :=
  sorry

theorem new_average_after_grace_marks
  (stats : ClassStats)
  (grace_marks : GraceMarksDistribution)
  (h_size : stats.size = 35)
  (h_original_avg : stats.original_average = 37)
  (h_std_dev : stats.standard_deviation = 6)
  (h_25th : stats.percentile_25th = 32)
  (h_50th : stats.percentile_50th = 37)
  (h_75th : stats.percentile_75th = 42)
  (h_grace_below_25th : grace_marks.below_25th = 6)
  (h_grace_25th_50th : grace_marks.between_25th_50th = 4)
  (h_grace_50th_75th : grace_marks.between_50th_75th = 2)
  (h_grace_above_75th : grace_marks.above_75th = 0) :
  abs (calculate_new_average stats grace_marks - 40.09) < 0.01 := by
  sorry

end new_average_after_grace_marks_l4068_406854


namespace S_subset_T_l4068_406840

open Set Real

def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ k : ℤ, p.1^2 - p.2^2 = 2*k + 1}

def T : Set (ℝ × ℝ) := {p : ℝ × ℝ | sin (2*π*p.1^2) - sin (2*π*p.2^2) = cos (2*π*p.1^2) - cos (2*π*p.2^2)}

theorem S_subset_T : S ⊆ T := by
  sorry

end S_subset_T_l4068_406840


namespace complement_A_intersect_B_l4068_406826

def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}

def B : Set ℝ := {x | Real.exp (x * Real.log 3) > 9}

theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {x : ℝ | x > 2} := by sorry

end complement_A_intersect_B_l4068_406826


namespace xf_inequality_solution_l4068_406823

noncomputable section

variable (f : ℝ → ℝ)

-- f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- When x < 0, f(x) + xf'(x) < 0
def condition_negative (f : ℝ → ℝ) : Prop :=
  ∀ x, x < 0 → f x + x * (deriv f x) < 0

-- f(3) = 0
def f_3_is_0 (f : ℝ → ℝ) : Prop := f 3 = 0

-- The solution set
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | x < -3 ∨ (0 < x ∧ x < 3)}

theorem xf_inequality_solution
  (heven : even_function f)
  (hneg : condition_negative f)
  (hf3 : f_3_is_0 f) :
  {x : ℝ | x * f x > 0} = solution_set f :=
sorry

end

end xf_inequality_solution_l4068_406823


namespace percentage_of_160_l4068_406888

theorem percentage_of_160 : (3 / 8 : ℚ) / 100 * 160 = 3 / 5 := by sorry

end percentage_of_160_l4068_406888


namespace rupert_age_rupert_candles_l4068_406813

-- Define Peter's age
def peter_age : ℕ := 10

-- Define the ratio of Rupert's age to Peter's age
def age_ratio : ℚ := 7/2

-- Theorem to prove Rupert's age
theorem rupert_age : ℕ := by
  -- The proof goes here
  sorry

-- Theorem to prove the number of candles on Rupert's cake
theorem rupert_candles : ℕ := by
  -- The proof goes here
  sorry

end rupert_age_rupert_candles_l4068_406813


namespace least_five_digit_congruent_to_8_mod_17_l4068_406845

theorem least_five_digit_congruent_to_8_mod_17 :
  ∃ (n : ℕ), 
    (n ≥ 10000 ∧ n < 100000) ∧ 
    (n % 17 = 8) ∧
    (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 17 = 8 → m ≥ n) ∧
    n = 10004 :=
by sorry

end least_five_digit_congruent_to_8_mod_17_l4068_406845


namespace fred_total_cards_l4068_406834

def initial_cards : ℕ := 26
def cards_given_away : ℕ := 18
def new_cards_found : ℕ := 40

theorem fred_total_cards : 
  initial_cards - cards_given_away + new_cards_found = 48 := by
  sorry

end fred_total_cards_l4068_406834


namespace tan_sqrt3_implies_sin2theta_over_1_plus_cos2theta_eq_sqrt3_l4068_406870

theorem tan_sqrt3_implies_sin2theta_over_1_plus_cos2theta_eq_sqrt3 (θ : Real) 
  (h : Real.tan θ = Real.sqrt 3) : 
  (Real.sin (2 * θ)) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 := by
  sorry

end tan_sqrt3_implies_sin2theta_over_1_plus_cos2theta_eq_sqrt3_l4068_406870


namespace product_of_numbers_l4068_406848

theorem product_of_numbers (x y : ℝ) : x + y = 60 → x - y = 10 → x * y = 875 := by
  sorry

end product_of_numbers_l4068_406848


namespace unique_number_of_children_l4068_406873

theorem unique_number_of_children : ∃! n : ℕ, 
  100 ≤ n ∧ n ≤ 150 ∧ n % 8 = 5 ∧ n % 10 = 7 := by sorry

end unique_number_of_children_l4068_406873


namespace cubic_polynomial_d_value_l4068_406894

/-- Represents a cubic polynomial of the form 3x^3 + dx^2 + ex - 6 -/
structure CubicPolynomial where
  d : ℝ
  e : ℝ

def CubicPolynomial.eval (p : CubicPolynomial) (x : ℝ) : ℝ :=
  3 * x^3 + p.d * x^2 + p.e * x - 6

def CubicPolynomial.productOfZeros (p : CubicPolynomial) : ℝ := 2

def CubicPolynomial.sumOfCoefficients (p : CubicPolynomial) : ℝ :=
  3 + p.d + p.e - 6

theorem cubic_polynomial_d_value (p : CubicPolynomial) :
  p.productOfZeros = 9 →
  p.sumOfCoefficients = 9 →
  p.d = -18 := by
  sorry

end cubic_polynomial_d_value_l4068_406894


namespace smallest_repeating_block_length_l4068_406877

/-- The number of digits in the smallest repeating block of the decimal expansion of 3/11 -/
def repeating_block_length : ℕ := 2

/-- The fraction we are considering -/
def fraction : ℚ := 3 / 11

theorem smallest_repeating_block_length :
  repeating_block_length = 2 ∧
  ∀ n : ℕ, n < repeating_block_length →
    ¬∃ (a b : ℕ), fraction = (a : ℚ) / (10^n : ℚ) + (b : ℚ) / (10^n - 1 : ℚ) := by
  sorry

end smallest_repeating_block_length_l4068_406877


namespace norm_scale_vector_l4068_406890

theorem norm_scale_vector (u : ℝ × ℝ) : ‖u‖ = 7 → ‖(5 : ℝ) • u‖ = 35 := by
  sorry

end norm_scale_vector_l4068_406890


namespace rectangular_plot_ratio_l4068_406893

/-- A rectangular plot with given perimeter and short side length has a specific ratio of long to short sides -/
theorem rectangular_plot_ratio (perimeter : ℝ) (short_side : ℝ) 
  (h_perimeter : perimeter = 640) 
  (h_short_side : short_side = 80) : 
  (perimeter / 2 - short_side) / short_side = 3 := by
  sorry

end rectangular_plot_ratio_l4068_406893


namespace special_sequence_growth_l4068_406857

/-- A sequence of positive integers satisfying the given condition -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ i ≥ 1, Nat.gcd (a i) (a (i + 1)) > a (i - 1))

/-- The main theorem: for any special sequence, each term is at least 2^n -/
theorem special_sequence_growth (a : ℕ → ℕ) (h : SpecialSequence a) :
    ∀ n, a n ≥ 2^n := by
  sorry

end special_sequence_growth_l4068_406857


namespace sqrt_144_divided_by_6_l4068_406829

theorem sqrt_144_divided_by_6 : Real.sqrt 144 / 6 = 2 := by
  sorry

end sqrt_144_divided_by_6_l4068_406829


namespace cookie_problem_l4068_406815

/-- The number of cookies in each bag -/
def cookies_per_bag : ℕ := 7

/-- The number of cookies in each box -/
def cookies_per_box : ℕ := 12

/-- The number of boxes -/
def num_boxes : ℕ := 8

/-- The additional number of cookies in boxes compared to bags -/
def additional_cookies : ℕ := 33

/-- The number of bags -/
def num_bags : ℕ := 9

theorem cookie_problem :
  cookies_per_box * num_boxes = cookies_per_bag * num_bags + additional_cookies :=
by sorry

end cookie_problem_l4068_406815


namespace notebook_price_is_3_l4068_406825

-- Define the prices as real numbers
variable (pencil_price notebook_price : ℝ)

-- Define the purchase equations
def xiaohong_purchase : Prop :=
  4 * pencil_price + 5 * notebook_price = 15.8

def xiaoliang_purchase : Prop :=
  4 * pencil_price + 7 * notebook_price = 21.8

-- Theorem statement
theorem notebook_price_is_3
  (h1 : xiaohong_purchase pencil_price notebook_price)
  (h2 : xiaoliang_purchase pencil_price notebook_price) :
  notebook_price = 3 := by sorry

end notebook_price_is_3_l4068_406825


namespace work_left_after_14_days_l4068_406810

/-- The fraction of work left for the first task after 14 days -/
def first_task_left : ℚ := 11/60

/-- The fraction of work left for the second task after 14 days -/
def second_task_left : ℚ := 0

/-- A's work rate per day -/
def rate_A : ℚ := 1/15

/-- B's work rate per day -/
def rate_B : ℚ := 1/20

/-- C's work rate per day -/
def rate_C : ℚ := 1/25

/-- The number of days A and B work on the first task -/
def days_first_task : ℕ := 7

/-- The total number of days -/
def total_days : ℕ := 14

theorem work_left_after_14_days :
  let work_AB_7_days := (rate_A + rate_B) * days_first_task
  let work_C_7_days := rate_C * days_first_task
  let work_ABC_7_days := (rate_A + rate_B + rate_C) * (total_days - days_first_task)
  (1 - work_AB_7_days = first_task_left) ∧
  (max 0 (1 - work_C_7_days - work_ABC_7_days) = second_task_left) := by
  sorry

end work_left_after_14_days_l4068_406810


namespace rectangle_area_rectangle_area_is_240_l4068_406866

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  rectangle_length * rectangle_breadth

theorem rectangle_area_is_240 :
  rectangle_area 3600 10 = 240 := by
  sorry

end rectangle_area_rectangle_area_is_240_l4068_406866


namespace six_meter_logs_more_advantageous_l4068_406865

-- Define the length of logs and the target length of chunks
def log_length_6 : ℕ := 6
def log_length_7 : ℕ := 7
def chunk_length : ℕ := 1
def total_length : ℕ := 42

-- Define the number of cuts needed for each log type
def cuts_per_log_6 : ℕ := log_length_6 - 1
def cuts_per_log_7 : ℕ := log_length_7 - 1

-- Define the number of logs needed for each type
def logs_needed_6 : ℕ := (total_length + log_length_6 - 1) / log_length_6
def logs_needed_7 : ℕ := (total_length + log_length_7 - 1) / log_length_7

-- Define the total number of cuts for each log type
def total_cuts_6 : ℕ := logs_needed_6 * cuts_per_log_6
def total_cuts_7 : ℕ := logs_needed_7 * cuts_per_log_7

-- Theorem statement
theorem six_meter_logs_more_advantageous :
  total_cuts_6 < total_cuts_7 :=
by sorry

end six_meter_logs_more_advantageous_l4068_406865


namespace parabola_and_hyperbola_equations_l4068_406883

-- Define the parabola and hyperbola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the conditions
axiom parabola_vertex_origin : ∃ p > 0, parabola p 0 0
axiom axis_of_symmetry : ∃ p > 0, ∀ x, parabola p x 0 → x = 1
axiom intersection_point : ∃ p a b, parabola p (3/2) (Real.sqrt 6) ∧ hyperbola a b (3/2) (Real.sqrt 6)

-- Theorem to prove
theorem parabola_and_hyperbola_equations :
  ∃ p a b, (∀ x y, parabola p x y ↔ y^2 = 4*x) ∧
           (∀ x y, hyperbola a b x y ↔ 4*x^2 - (4/3)*y^2 = 1) :=
sorry

end parabola_and_hyperbola_equations_l4068_406883


namespace sum_and_sum_squares_bound_equality_conditions_l4068_406801

theorem sum_and_sum_squares_bound (a b c : ℝ) 
  (h1 : a ≥ -1) (h2 : b ≥ -1) (h3 : c ≥ -1)
  (h4 : a^3 + b^3 + c^3 = 1) :
  a + b + c + a^2 + b^2 + c^2 ≤ 4 := by
  sorry

theorem equality_conditions (a b c : ℝ) 
  (h1 : a ≥ -1) (h2 : b ≥ -1) (h3 : c ≥ -1)
  (h4 : a^3 + b^3 + c^3 = 1) :
  (a + b + c + a^2 + b^2 + c^2 = 4) ↔ 
  ((a, b, c) = (1, 1, -1) ∨ (a, b, c) = (1, -1, 1) ∨ (a, b, c) = (-1, 1, 1)) := by
  sorry

end sum_and_sum_squares_bound_equality_conditions_l4068_406801


namespace worker_payment_l4068_406847

/-- Calculate the total amount paid to a worker for a week -/
theorem worker_payment (daily_wage : ℝ) (days_worked : List ℝ) : 
  daily_wage = 20 →
  days_worked = [11, 32, 31, 8.3, 4] →
  (daily_wage * (days_worked.sum)) = 1726 := by
sorry

end worker_payment_l4068_406847


namespace triangle_side_range_l4068_406887

theorem triangle_side_range (a b c : ℝ) : 
  c = 4 → -- Given condition: one side has length 4
  a > 0 → -- Positive length
  b > 0 → -- Positive length
  a ≤ b → -- Assume a is the shorter of the two variable sides
  a^2 + b^2 = c^2 → -- Pythagorean theorem
  a < 4 * Real.sqrt 2 -- Upper bound of the range
  ∧ a > 0 -- Lower bound of the range
  := by sorry

end triangle_side_range_l4068_406887


namespace bells_toll_together_l4068_406855

def bell_intervals : List Nat := [2, 4, 6, 8, 10, 12]
def period_minutes : Nat := 30
def period_seconds : Nat := period_minutes * 60

def lcm_list (list : List Nat) : Nat :=
  list.foldl Nat.lcm 1

theorem bells_toll_together : 
  (period_seconds / lcm_list bell_intervals) + 1 = 16 := by
  sorry

end bells_toll_together_l4068_406855


namespace arithmetic_geometric_sequence_property_l4068_406824

-- Define the arithmetic sequence
def arithmetic_seq (A d : ℝ) (k : ℕ) : ℝ := A + k * d

-- Define the geometric sequence
def geometric_seq (B q : ℝ) (k : ℕ) : ℝ := B * q ^ k

-- Main theorem
theorem arithmetic_geometric_sequence_property
  (a b c : ℝ) (m n p : ℕ) (A d B q : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (ha_arith : a = arithmetic_seq A d m)
  (hb_arith : b = arithmetic_seq A d n)
  (hc_arith : c = arithmetic_seq A d p)
  (ha_geom : a = geometric_seq B q m)
  (hb_geom : b = geometric_seq B q n)
  (hc_geom : c = geometric_seq B q p) :
  a ^ (b - c) * b ^ (c - a) * c ^ (a - b) = 1 := by
  sorry

end arithmetic_geometric_sequence_property_l4068_406824


namespace rectangle_area_l4068_406827

/-- The area of a rectangle with perimeter 60 and width 10 is 200 -/
theorem rectangle_area (perimeter width : ℝ) (h1 : perimeter = 60) (h2 : width = 10) :
  2 * (perimeter / 2 - width) * width = 200 :=
by sorry

end rectangle_area_l4068_406827


namespace rachel_envelope_stuffing_l4068_406803

/-- Rachel's envelope stuffing problem -/
theorem rachel_envelope_stuffing 
  (total_hours : ℕ) 
  (total_envelopes : ℕ) 
  (first_hour_envelopes : ℕ) 
  (h1 : total_hours = 8) 
  (h2 : total_envelopes = 1500) 
  (h3 : first_hour_envelopes = 135) :
  ∃ (second_hour_envelopes : ℕ),
    second_hour_envelopes = 195 ∧ 
    (total_envelopes - first_hour_envelopes - second_hour_envelopes) / (total_hours - 2) = 
    (total_envelopes - first_hour_envelopes - second_hour_envelopes) / (total_hours - 2) :=
by sorry


end rachel_envelope_stuffing_l4068_406803


namespace rectangle_dimensions_l4068_406816

theorem rectangle_dimensions (length width : ℝ) : 
  (2 * length + 2 * width = 16) →  -- Perimeter is 16 cm
  (length - width = 1) →           -- Difference between length and width is 1 cm
  (length = 4.5 ∧ width = 3.5) :=  -- Length is 4.5 cm and width is 3.5 cm
by
  sorry

#check rectangle_dimensions

end rectangle_dimensions_l4068_406816


namespace fifteenSidedFigureArea_l4068_406879

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A polygon defined by a list of vertices -/
structure Polygon where
  vertices : List Point

/-- The area of a polygon -/
noncomputable def area (p : Polygon) : ℝ := sorry

/-- The fifteen-sided figure defined in the problem -/
def fifteenSidedFigure : Polygon :=
  { vertices := [
      {x := 1, y := 1}, {x := 1, y := 3}, {x := 3, y := 5}, {x := 4, y := 5},
      {x := 5, y := 4}, {x := 5, y := 3}, {x := 6, y := 3}, {x := 6, y := 2},
      {x := 5, y := 1}, {x := 4, y := 1}, {x := 3, y := 2}, {x := 2, y := 2},
      {x := 1, y := 1}
    ]
  }

/-- Theorem stating that the area of the fifteen-sided figure is 11 cm² -/
theorem fifteenSidedFigureArea : area fifteenSidedFigure = 11 := by sorry

end fifteenSidedFigureArea_l4068_406879


namespace quadratic_inequality_bc_l4068_406804

/-- Given a quadratic inequality x^2 + bx + c ≤ 0 with solution set [-2, 5], 
    prove that bc = 30 -/
theorem quadratic_inequality_bc (b c : ℝ) : 
  (∀ x, x^2 + b*x + c ≤ 0 ↔ -2 ≤ x ∧ x ≤ 5) → b*c = 30 := by
  sorry

end quadratic_inequality_bc_l4068_406804


namespace lemon_cupcakes_left_at_home_l4068_406886

/-- Proves that the number of lemon cupcakes left at home is 2 -/
theorem lemon_cupcakes_left_at_home 
  (total_baked : ℕ) 
  (boxes_given : ℕ) 
  (cupcakes_per_box : ℕ) 
  (h1 : total_baked = 53) 
  (h2 : boxes_given = 17) 
  (h3 : cupcakes_per_box = 3) : 
  total_baked - (boxes_given * cupcakes_per_box) = 2 := by
  sorry

end lemon_cupcakes_left_at_home_l4068_406886


namespace largest_integer_solution_l4068_406808

theorem largest_integer_solution (x : ℤ) : x ≤ 2 ↔ x / 3 + 4 / 5 < 5 / 3 := by sorry

end largest_integer_solution_l4068_406808


namespace system_solution_l4068_406800

def solution_set : Set (ℝ × ℝ) := {(3, 2)}

theorem system_solution :
  {(x, y) : ℝ × ℝ | x + y = 5 ∧ x - y = 1} = solution_set :=
by sorry

end system_solution_l4068_406800
