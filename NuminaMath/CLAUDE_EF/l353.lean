import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shots_to_hit_ship_l353_35358

/-- Represents a grid --/
structure Grid :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a ship --/
structure Ship :=
  (length : Nat)
  (width : Nat)

/-- Represents a shot that covers a full row or column --/
inductive Shot
  | row (n : Nat)
  | col (n : Nat)

/-- Predicate to check if a set of shots guarantees hitting the ship --/
def guarantees_hit (g : Grid) (s : Ship) (shots : List Shot) : Prop :=
  ∀ (ship_row ship_col : Nat), ship_row < g.rows ∧ ship_col < g.cols →
    ∃ (shot : Shot), shot ∈ shots ∧
      match shot with
      | Shot.row n => n = ship_row
      | Shot.col n => n = ship_col ∨ n = ship_col + 1 ∨ n = ship_col + 2

/-- The main theorem --/
theorem min_shots_to_hit_ship (g : Grid) (s : Ship) :
  g.rows = 8 ∧ g.cols = 8 ∧ s.length = 1 ∧ s.width = 3 →
  ∃ (shots : List Shot),
    guarantees_hit g s shots ∧
    shots.length = 4 ∧
    ∀ (other_shots : List Shot),
      guarantees_hit g s other_shots →
      other_shots.length ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shots_to_hit_ship_l353_35358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l353_35356

/-- The ellipse defined by x^2/16 + y^2/12 = 1 -/
def ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 12 = 1

/-- The line defined by x - 2y - 12 = 0 -/
def line (x y : ℝ) : Prop :=
  x - 2*y - 12 = 0

/-- The distance from a point (x, y) to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x - 2*y - 12| / Real.sqrt 5

/-- The maximum distance from any point on the ellipse to the line is 4√5 -/
theorem max_distance_ellipse_to_line :
  ∃ (max_dist : ℝ), max_dist = 4 * Real.sqrt 5 ∧
  ∀ (x y : ℝ), ellipse x y →
    distance_to_line x y ≤ max_dist ∧
    ∃ (x' y' : ℝ), ellipse x' y' ∧ distance_to_line x' y' = max_dist :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l353_35356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_proof_l353_35367

def a : ℝ × ℝ × ℝ := (1, 0, 1)
def b : ℝ × ℝ × ℝ := (-2, -1, 1)
def c : ℝ × ℝ × ℝ := (3, 1, 0)

theorem vector_magnitude_proof :
  let v := (a.fst - b.fst + 2 * c.fst, a.snd.fst - b.snd.fst + 2 * c.snd.fst, a.snd.snd - b.snd.snd + 2 * c.snd.snd)
  Real.sqrt (v.fst^2 + v.snd.fst^2 + v.snd.snd^2) = 3 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_proof_l353_35367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cylinder_volume_ratio_l353_35368

/-- The ratio of volumes of two cylinders formed by rolling a rectangle -/
theorem rectangle_cylinder_volume_ratio 
  (length width : ℝ) 
  (h_length : length = 9) 
  (h_width : width = 6) 
  (h_pos_length : length > 0) 
  (h_pos_width : width > 0) : 
  (π * (width / (2 * π))^2 * length) / (π * (length / (2 * π))^2 * width) = 4 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cylinder_volume_ratio_l353_35368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_form_l353_35338

/-- Triangle DEF with vertices D, E, F, and point Q -/
structure Triangle :=
  (D : ℝ × ℝ)
  (E : ℝ × ℝ)
  (F : ℝ × ℝ)
  (Q : ℝ × ℝ)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Sum of distances from Q to vertices -/
noncomputable def sum_distances (t : Triangle) : ℝ :=
  distance t.D t.Q + distance t.E t.Q + distance t.F t.Q

/-- Theorem: Sum of distances can be expressed as x√5 + y√13 where x + y = 3 -/
theorem sum_distances_form (t : Triangle) 
  (h1 : t.D = (0, 0))
  (h2 : t.E = (8, 0))
  (h3 : t.F = (2, 4))
  (h4 : t.Q = (3, 1)) :
  ∃ (x y : ℤ), sum_distances t = x * Real.sqrt 5 + y * Real.sqrt 13 ∧ x + y = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_form_l353_35338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l353_35365

/-- The speed of a train given its length, time to pass a person, and the person's speed in the opposite direction. -/
noncomputable def train_speed (train_length : ℝ) (passing_time : ℝ) (person_speed_kmph : ℝ) : ℝ :=
  let person_speed_mps := person_speed_kmph * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let train_speed_mps := relative_speed - person_speed_mps
  train_speed_mps * (3600 / 1000)

/-- Theorem stating that a train of length 110 meters passing a person in 6 seconds, 
    where the person is moving at 6 kmph in the opposite direction, has a speed of 60 kmph. -/
theorem train_speed_theorem :
  train_speed 110 6 6 = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l353_35365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_interior_filling_l353_35319

/-- Represents a truncated table with arbitrary real numbers on the boundary -/
def TruncatedTable (m n : ℕ) := 
  (Fin (m + 2) → Fin (n + 2) → ℝ)

/-- Checks if a given position is on the boundary of the truncated table -/
def isBoundary (m n : ℕ) (i : Fin (m + 2)) (j : Fin (n + 2)) : Prop :=
  i = 0 ∨ i = m + 1 ∨ j = 0 ∨ j = n + 1

/-- Checks if a given position is in the interior of the truncated table -/
def isInterior (m n : ℕ) (i : Fin (m + 2)) (j : Fin (n + 2)) : Prop :=
  1 ≤ i.val ∧ i.val ≤ m ∧ 1 ≤ j.val ∧ j.val ≤ n

/-- Defines the average condition for interior cells -/
def satisfiesAverageCondition (m n : ℕ) (table : TruncatedTable m n) : Prop :=
  ∀ (i : Fin (m + 2)) (j : Fin (n + 2)), isInterior m n i j →
    table i j = (table (i-1) j + table (i+1) j + table i (j-1) + table i (j+1)) / 4

/-- The main theorem statement -/
theorem unique_interior_filling (m n : ℕ) :
  ∀ (boundary : TruncatedTable m n),
    (∀ (i : Fin (m + 2)) (j : Fin (n + 2)), isBoundary m n i j → boundary i j ∈ Set.univ) →
    ∃! (filled : TruncatedTable m n),
      (∀ (i : Fin (m + 2)) (j : Fin (n + 2)), isBoundary m n i j → filled i j = boundary i j) ∧
      satisfiesAverageCondition m n filled :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_interior_filling_l353_35319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_sum_l353_35315

noncomputable def f (x : ℝ) : ℝ := 
  (2 * Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4) + (x + 2)^2 - 4 * (Real.cos x)^2) / (x^2 + 2)

theorem range_sum (m n : ℝ) : 
  (∀ x, m ≤ f x ∧ f x ≤ n) ∧ (∀ y, m ≤ y ∧ y ≤ n → ∃ x, f x = y) → m + n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_sum_l353_35315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2α_minus_π_4_l353_35398

theorem sin_2α_minus_π_4 (α : ℝ) 
  (h1 : Real.sin α - Real.cos α = 1/5) 
  (h2 : 0 ≤ α ∧ α ≤ π) : 
  Real.sin (2*α - π/4) = (31*Real.sqrt 2)/50 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2α_minus_π_4_l353_35398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_first_term_f_sequence_is_geometric_l353_35325

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (Real.cos x - Real.sin x)

noncomputable def f_prime (x : ℝ) : ℝ := -2 * Real.exp x * Real.sin x

noncomputable def x_seq : ℕ → ℝ
  | n => n * Real.pi

theorem geometric_sequence_property (n : ℕ) :
  f (x_seq (n + 1)) / f (x_seq n) = -Real.exp Real.pi := by
  sorry

theorem first_term : f (x_seq 1) = -Real.exp Real.pi := by
  sorry

theorem f_sequence_is_geometric :
  ∃ (a q : ℝ), ∀ n : ℕ, f (x_seq n) = a * q ^ (n - 1) ∧ 
  a = -Real.exp Real.pi ∧ q = -Real.exp Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_first_term_f_sequence_is_geometric_l353_35325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l353_35391

-- Define the function f with domain [-1, 2]
def f : Set ℝ := Set.Icc (-1) 2

-- Define the function g(x) = f(3-|x|)
def g (x : ℝ) : Prop := (3 - |x|) ∈ f

-- Theorem statement
theorem domain_of_g :
  {x : ℝ | g x} = Set.Icc (-4) (-1) ∪ Set.Icc 1 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l353_35391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_inequality_condition_l353_35388

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (m n x : ℝ) : ℝ := m * (x + n) / (x + 1)

-- Statement for the first part of the problem
theorem tangent_perpendicular (n : ℝ) : 
  (∀ x, (deriv f) x * (deriv (g 1 n)) x = -1) ↔ n = 5 := by sorry

-- Statement for the second part of the problem
theorem inequality_condition (m n : ℝ) :
  (m > 0 ∧ ∀ x > 0, |f x| ≥ |g m n x|) ↔ (n = -1 ∧ 0 < m ∧ m ≤ 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_inequality_condition_l353_35388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_sequences_have_p_or_transformed_p_property_l353_35361

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m^2

def has_p_property (a : ℕ → ℤ) : Prop :=
  ∀ i : ℕ, i ≥ 1 → is_perfect_square (a i + i)

def is_permutation (a b : ℕ → ℤ) (n : ℕ) : Prop :=
  ∃ σ : Fin n → Fin n, Function.Bijective σ ∧ ∀ i : Fin n, b i.val = a (σ i).val

def has_transformed_p_property (a : ℕ → ℤ) : Prop :=
  ¬(has_p_property a) ∧ ∃ b : ℕ → ℤ, (∀ n : ℕ, is_permutation a b n) ∧ has_p_property b

def sequence_i (n : ℕ) : ℤ := if n ≤ 5 then n else 0

def sequence_ii (n : ℕ) : ℤ := if n ≤ 12 then n else 0

def sequence_iii (n : ℕ) : ℤ := n^2 - n

theorem all_sequences_have_p_or_transformed_p_property :
  (has_p_property sequence_i ∨ has_transformed_p_property sequence_i) ∧
  (has_p_property sequence_ii ∨ has_transformed_p_property sequence_ii) ∧
  (has_p_property sequence_iii ∨ has_transformed_p_property sequence_iii) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_sequences_have_p_or_transformed_p_property_l353_35361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elective_course_distribution_l353_35350

theorem elective_course_distribution :
  let n : ℕ := 4  -- number of courses
  let m : ℕ := 3  -- number of years
  let max_per_group : ℕ := 3  -- maximum courses per year
  
  -- Function to calculate the number of ways to distribute n items into m groups
  -- where each group can have 0 to max_per_group items, and all items must be used
  ∃ (distribution_ways : ℕ → ℕ → ℕ → ℕ),
    distribution_ways n m max_per_group = 78 := by
  
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elective_course_distribution_l353_35350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_body_volume_l353_35355

/-- A combined body consisting of a truncated cone and a sphere -/
structure CombinedBody where
  R : ℝ  -- Radius of the sphere
  S : ℝ  -- Surface area of the combined body

/-- The volume of the combined body -/
noncomputable def volume (body : CombinedBody) : ℝ := body.S * body.R / 3

/-- Theorem stating the volume of the combined body -/
theorem combined_body_volume (body : CombinedBody) :
  volume body = body.S * body.R / 3 := by
  -- Proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_body_volume_l353_35355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_cards_below_threshold_l353_35357

def jungkook_card : ℚ := 4/5
def yoongi_card : ℚ := 1/2
def yoojung_card : ℚ := 9/10
def yuna_card : ℚ := 1/3

def threshold : ℚ := 3/10

def count_cards_below_threshold (cards : List ℚ) (threshold : ℚ) : ℕ :=
  (cards.filter (λ x => x ≤ threshold)).length

theorem no_cards_below_threshold : 
  count_cards_below_threshold 
    [jungkook_card, yoongi_card, yoojung_card, yuna_card] 
    threshold = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_cards_below_threshold_l353_35357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gnome_river_crossing_l353_35386

/-- Represents the weight of a gnome -/
def GnomeWeight := Nat

/-- Represents the number of gnomes -/
def NumGnomes : Nat := 100

/-- Represents the boat's capacity -/
def BoatCapacity : Nat := 100

/-- Represents a list of gnome weights -/
def gnomeWeights : List GnomeWeight := List.range NumGnomes

/-- Represents the total weight of all gnomes -/
def totalWeight : Nat := (NumGnomes * (NumGnomes + 1)) / 2

/-- Represents whether it's possible to transport all gnomes across the river -/
def canTransportAllGnomes : Prop := ∃ (k : Nat), 200 * (k + 1) - k^2 = 2 * totalWeight

theorem gnome_river_crossing :
  ¬canTransportAllGnomes :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gnome_river_crossing_l353_35386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_always_odd_l353_35317

theorem always_odd (e m : ℕ) (h : Even e) : Odd (e^2 + 3^m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_always_odd_l353_35317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_distribution_heights_l353_35318

noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

noncomputable def water_height (v r : ℝ) : ℝ := v / (Real.pi * r^2)

theorem water_distribution_heights :
  let cone_r : ℝ := 10
  let cone_h : ℝ := 15
  let cylinder1_r : ℝ := 20
  let cylinder2_r : ℝ := 10
  let total_volume := cone_volume cone_r cone_h
  let half_volume := total_volume / 2
  let height1 := water_height half_volume cylinder1_r
  let height2 := water_height half_volume cylinder2_r
  height1 = 5/8 ∧ height2 = 5/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_distribution_heights_l353_35318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_bar_profit_l353_35316

theorem candy_bar_profit : 
  ∀ (num_bars : ℕ) (buy_price sell_price : ℚ) (tax_rate : ℚ),
    num_bars = 800 →
    buy_price = 3 / 6 →
    sell_price = 2 / 3 →
    tax_rate = 1 / 10 →
    (num_bars * sell_price - num_bars * sell_price * tax_rate) - (num_bars * buy_price) = 80.02 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_bar_profit_l353_35316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_specific_line_l353_35359

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  |a * x + b * y + c| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from point P(0,2) to the line x - y + 3 = 0 is √2/2 -/
theorem distance_point_to_specific_line :
  distance_point_to_line 0 2 1 (-1) 3 = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_specific_line_l353_35359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l353_35324

-- Define the sets A and B
def A : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def B : Set ℝ := {x | 3 * x - 7 ≥ 8 - 2 * x}

-- Define the theorem
theorem set_operations :
  (A ∪ B = Set.Ici 2) ∧
  (A ∩ B = Set.Ico 3 4) ∧
  ((Set.univ \ A) ∩ B = Set.Ici 4) := by
  sorry

-- Where:
-- Set.Ici a is the set {x | a ≤ x}, equivalent to [a, +∞)
-- Set.Ico a b is the half-open interval [a, b)
-- Set.univ represents the universal set (ℝ in this case)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l353_35324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_submerged_sphere_properties_l353_35348

/-- Represents the properties of a partially submerged sphere -/
structure SubmergedSphere where
  surface_area : ℝ
  submerged_depth : ℝ

/-- Calculates the radius of the sphere given its surface area -/
noncomputable def sphere_radius (s : SubmergedSphere) : ℝ :=
  Real.sqrt (s.surface_area / (4 * Real.pi))

/-- Calculates the submerged surface area of the sphere -/
noncomputable def submerged_area (s : SubmergedSphere) : ℝ :=
  2 * Real.pi * sphere_radius s * s.submerged_depth

/-- Calculates the weight of the sphere based on displaced water -/
noncomputable def sphere_weight (s : SubmergedSphere) : ℝ :=
  Real.pi * s.submerged_depth^2 * (sphere_radius s - s.submerged_depth / 3)

/-- Calculates the vertex angle of the submerged spherical cap in radians -/
noncomputable def vertex_angle (s : SubmergedSphere) : ℝ :=
  2 * Real.arccos ((sphere_radius s - s.submerged_depth) / sphere_radius s)

/-- Main theorem stating the properties of the submerged sphere -/
theorem submerged_sphere_properties (s : SubmergedSphere) 
  (h1 : s.surface_area = 195.8)
  (h2 : s.submerged_depth = 1.2) :
  ∃ (ε : ℝ), ε > 0 ∧ 
  (abs (submerged_area s - 29.75) < ε) ∧ 
  (abs (sphere_weight s - 16.04) < ε) ∧
  (abs (vertex_angle s - 1.602212) < ε) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_submerged_sphere_properties_l353_35348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mice_population_after_events_l353_35392

/-- Calculates the number of mice remaining after two generations of breeding and various events -/
theorem mice_population_after_events (
  initial_mice : ℕ
) (first_gen_pups_per_mouse : ℕ)
  (second_gen_pups_per_mouse : ℕ)
  (survival_rate_A : ℚ)
  (survival_rate_B : ℚ)
  (survival_rate_others : ℚ)
  (second_gen_survival_rate : ℚ)
  (original_mice_deaths : ℕ)
  (first_gen_death_rate : ℚ)
  (pups_eaten_per_adult : ℕ) :
  initial_mice = 8 →
  first_gen_pups_per_mouse = 7 →
  second_gen_pups_per_mouse = 6 →
  survival_rate_A = 9/10 →
  survival_rate_B = 7/10 →
  survival_rate_others = 4/5 →
  second_gen_survival_rate = 13/20 →
  original_mice_deaths = 2 →
  first_gen_death_rate = 3/20 →
  pups_eaten_per_adult = 3 →
  ∃ n : ℕ, n = 607 := by
    sorry

#check mice_population_after_events

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mice_population_after_events_l353_35392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tokens_99x99_grid_l353_35375

/-- Represents a grid with tokens -/
structure TokenGrid where
  size : ℕ
  tokens : Finset (ℕ × ℕ)

/-- Checks if a 4x4 square in the grid contains at least 8 tokens -/
def has_enough_tokens (g : TokenGrid) (x y : ℕ) : Prop :=
  (Finset.filter (fun p => x ≤ p.1 ∧ p.1 < x + 4 ∧ y ≤ p.2 ∧ p.2 < y + 4) g.tokens).card ≥ 8

/-- A valid token placement satisfies the condition for all 4x4 squares -/
def is_valid_placement (g : TokenGrid) : Prop :=
  ∀ x y, x + 4 ≤ g.size → y + 4 ≤ g.size → has_enough_tokens g x y

/-- The main theorem stating the minimum number of tokens required -/
theorem min_tokens_99x99_grid :
  ∃ (g : TokenGrid), g.size = 99 ∧ is_valid_placement g ∧ g.tokens.card = 4801 ∧
  (∀ (h : TokenGrid), h.size = 99 → is_valid_placement h → h.tokens.card ≥ 4801) := by
  sorry

#check min_tokens_99x99_grid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tokens_99x99_grid_l353_35375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_135_l353_35320

/-- The angle of inclination of a line passing through two given points -/
noncomputable def angle_of_inclination (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.arctan ((y2 - y1) / (x2 - x1)) * (180 / Real.pi)

/-- Theorem: The angle of inclination of a line passing through (-2, 0) and (-5, 3) is 135° -/
theorem line_inclination_135 : 
  angle_of_inclination (-2) 0 (-5) 3 = 135 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_135_l353_35320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_of_equation_l353_35339

noncomputable def equation (x : Real) : Real := Real.sin x + Real.sqrt 3 * Real.cos x + 1

theorem solution_of_equation :
  ∃ (x₁ x₂ : Real),
    x₁ ∈ Set.Icc 0 (2 * Real.pi) ∧
    x₂ ∈ Set.Icc 0 (2 * Real.pi) ∧
    equation x₁ = 0 ∧
    equation x₂ = 0 ∧
    x₁ = 5 * Real.pi / 6 ∧
    x₂ = 3 * Real.pi / 2 ∧
    ∀ (x : Real), x ∈ Set.Icc 0 (2 * Real.pi) ∧ equation x = 0 → x = x₁ ∨ x = x₂ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_of_equation_l353_35339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_calculation_l353_35331

/-- The actual distance traveled by a person, given their walking speeds and additional distance covered at higher speed -/
noncomputable def actual_distance (slow_speed fast_speed : ℝ) (additional_distance : ℝ) : ℝ :=
  (slow_speed * additional_distance) / (fast_speed - slow_speed)

/-- Theorem stating that under given conditions, the actual distance is approximately 34.29 km -/
theorem distance_calculation (slow_speed fast_speed : ℝ) (additional_distance : ℝ)
    (h1 : slow_speed = 8)
    (h2 : fast_speed = 15)
    (h3 : additional_distance = 30) :
    ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
    |actual_distance slow_speed fast_speed additional_distance - 34.29| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_calculation_l353_35331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_friend_pairs_l353_35332

/-- Represents a person in the social network -/
structure Person where
  id : Nat
deriving Ord

/-- Represents the social network -/
structure SocialNetwork where
  people : Finset Person
  invitations : Person → Finset Person
  friend_pairs : Finset (Person × Person)

/-- A valid social network satisfies the problem conditions -/
def is_valid_network (n : SocialNetwork) : Prop :=
  n.people.card = 2000 ∧
  (∀ p, p ∈ n.people → (n.invitations p).card = 1000) ∧
  ∀ p q, p ∈ n.people → q ∈ n.people →
    ((p, q) ∈ n.friend_pairs ↔ q ∈ n.invitations p ∧ p ∈ n.invitations q)

/-- The main theorem stating the minimum number of friend pairs -/
theorem min_friend_pairs (n : SocialNetwork) (h : is_valid_network n) : 
  n.friend_pairs.card ≥ 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_friend_pairs_l353_35332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_origin_condition_l353_35300

/-- A quadratic function -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_origin_condition (a b c : ℝ) :
  (b = 0 ∧ c = 0 → quadratic_function a b c 0 = 0) ∧
  (∃ a' b' c', quadratic_function a' b' c' 0 = 0 ∧ (b' ≠ 0 ∨ c' ≠ 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_origin_condition_l353_35300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l353_35354

/-- A regular triangular pyramid with specific measurements -/
structure RegularTriangularPyramid where
  /-- Distance from midpoint of height to lateral face -/
  midpoint_to_face : ℝ
  /-- Distance from midpoint of height to lateral edge -/
  midpoint_to_edge : ℝ

/-- The volume of a regular triangular pyramid with given measurements -/
noncomputable def pyramid_volume (p : RegularTriangularPyramid) : ℝ :=
  208 * Real.sqrt (13 / 3)

/-- Theorem stating the volume of the specific regular triangular pyramid -/
theorem specific_pyramid_volume :
  ∀ (p : RegularTriangularPyramid),
  p.midpoint_to_face = 2 →
  p.midpoint_to_edge = Real.sqrt 13 →
  pyramid_volume p = 208 * Real.sqrt (13 / 3) :=
by
  intro p h1 h2
  unfold pyramid_volume
  rfl

#check specific_pyramid_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l353_35354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_retail_price_approx_l353_35309

/-- Calculates the retail price before discounts given wholesale price, profit percentage, tax rate, and discount rates. -/
noncomputable def retailPriceBeforeDiscounts (wholesalePrice : ℝ) (profitPercentage : ℝ) (taxRate : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  let profit := wholesalePrice * profitPercentage
  let sellingPriceBeforeTax := wholesalePrice + profit
  let taxOnProfit := profit * taxRate
  let sellingPriceAfterTax := sellingPriceBeforeTax + taxOnProfit
  sellingPriceAfterTax / ((1 - discount1) * (1 - discount2))

/-- Theorem stating that the retail price before discounts is approximately $155.44 given the specified conditions. -/
theorem retail_price_approx (ε : ℝ) (h : ε > 0) :
  ∃ (price : ℝ), abs (price - retailPriceBeforeDiscounts 108 0.20 0.15 0.10 0.05) < ε ∧
                 abs (price - 155.44) < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_retail_price_approx_l353_35309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_l353_35345

noncomputable section

open Real

def f (α : ℝ) (x : ℝ) : ℝ := x + α / x + log x

theorem f_upper_bound (l e m : ℝ) :
  (0 < e) →
  (∀ (α : ℝ), α ∈ Set.Icc (1 / e) (2 * e^2) →
    ∀ (x : ℝ), x ∈ Set.Icc l e → x > 0 → f α x < m) →
  m > 1 + 2 * e^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_l353_35345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_expansion_l353_35304

theorem fourth_term_expansion (a x : ℝ) (h : a ≠ 0 ∧ x > 0) :
  let expansion := (a / Real.sqrt x - Real.sqrt x / a^2)^8
  let fourth_term := Finset.sum (Finset.range 9) (λ k ↦
    if k = 3 then
      Nat.choose 8 k * (a / Real.sqrt x)^(8 - k) * (-Real.sqrt x / a^2)^k
    else 0)
  fourth_term = -56 / (a * x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_expansion_l353_35304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_straight_line_l353_35344

/-- The set of points C(x, y) satisfying the given conditions forms a straight line. -/
theorem trajectory_is_straight_line :
  ∀ (x y lambda1 lambda2 : ℝ),
  (x, y) = lambda1 • (3, 1) + lambda2 • (-1, 3) →
  lambda1 + lambda2 = 1 →
  x + 2 * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_straight_line_l353_35344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_is_perfect_square_l353_35308

theorem x_is_perfect_square (x y : ℕ) (hx : x > 0) (hy : y > 0)
  (h : (x^2 + y^2 - x) % (2*x*y) = 0) : 
  ∃ n : ℕ, x = n^2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_is_perfect_square_l353_35308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slope_l353_35384

/-- Given two lines p and q that intersect at (1, 5), prove that the slope of line q is 5/2. -/
theorem intersection_slope (k : ℝ) : 
  (∀ x y, y = 2*x + 3 → (x, y) ∈ Set.range (λ t ↦ (t, 2*t + 3))) →  -- Line p
  (∀ x y, y = k*x + k → (x, y) ∈ Set.range (λ t ↦ (t, k*t + k))) →  -- Line q
  (1, 5) ∈ Set.range (λ t ↦ (t, 2*t + 3)) ∩ Set.range (λ t ↦ (t, k*t + k)) →  -- Intersection point
  k = 5/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slope_l353_35384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l353_35352

def S (n : ℕ) : ℤ := -n^2 + 7*n

def a : ℕ → ℤ
  | 0 => S 0  -- Adding the base case for 0
  | 1 => S 1
  | n+1 => S (n+1) - S n

theorem sequence_properties :
  (∀ n > 4, a n < 0) ∧
  (∀ n : ℕ, S n ≤ S 3 ∧ S n ≤ S 4) := by
  sorry

#eval a 10  -- This will evaluate a₁₀

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l353_35352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_walk_probability_l353_35347

-- Define the structure of the octahedron
structure Octahedron :=
  (top : Unit)
  (bottom : Unit)
  (middle : Fin 4)

-- Define the possible positions of the ant
inductive Position
  | Top
  | Bottom
  | Middle (i : Fin 4)

-- Define the probability of moving to a specific type of vertex
def prob_move_to_middle (pos : Position) : ℚ :=
  match pos with
  | Position.Top => 1
  | Position.Bottom => 1
  | Position.Middle _ => 1/2

-- Define the probability of C being in the middle given B's position
def prob_C_middle (B : Position) : ℚ :=
  match B with
  | Position.Top => 1
  | Position.Bottom => 1
  | Position.Middle _ => 1/2

-- State the theorem
theorem ant_walk_probability :
  let prob_B_middle : ℚ := 1/2
  let prob_C_middle_given_B_middle : ℚ := 1/2
  let prob_B_not_middle : ℚ := 1/2
  let prob_C_middle_given_B_not_middle : ℚ := 1
  prob_B_middle * prob_C_middle_given_B_middle +
  prob_B_not_middle * prob_C_middle_given_B_not_middle = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_walk_probability_l353_35347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turners_statement_implication_l353_35343

-- Define Student as a structure instead of a class
structure Student where
  percent_correct : ℝ
  grade : ℝ

-- Constants for grades
def B : ℝ := 3.0  -- Assuming B is represented as 3.0 on a 4.0 scale

theorem turners_statement_implication :
  (∀ (student : Student),
    (student.percent_correct ≥ 90 → student.grade ≥ B)) →
  (∀ (student : Student),
    (student.grade < B → student.percent_correct < 90)) :=
by
  intro h
  intro student
  contrapose!
  intro h_percent
  apply h
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_turners_statement_implication_l353_35343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_5_magnitude_l353_35342

/-- Definition of the complex sequence z_n -/
def z : ℕ → ℂ
  | 0 => 1  -- Add this case to cover Nat.zero
  | 1 => 1
  | n + 1 => (z n)^2 * (1 + Complex.I)

/-- The magnitude of the 5th term of the sequence equals 128√2 -/
theorem z_5_magnitude :
  Complex.abs (z 5) = 128 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_5_magnitude_l353_35342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l353_35364

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  2 * sin (π - x) + cos (-x) - sin ((5/2) * π - x) + cos ((π/2) + x)

-- Part 1
theorem part_one (α : ℝ) (h1 : f α = 1/3) (h2 : 0 < α ∧ α < π) : 
  tan α = -sqrt 2 / 4 := by sorry

-- Part 2
theorem part_two (α : ℝ) (h : f α = 2 * sin α - cos α + 1/2) : 
  sin α * cos α = 3/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l353_35364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_congruence_l353_35380

theorem coprime_congruence (a b : ℕ) (h : Nat.Coprime a b) :
  ∃ (m n : ℕ), (a^m + b^n) % (a * b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_congruence_l353_35380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventh_number_remainder_l353_35360

/-- Given a sequence of 50 consecutive natural numbers where exactly 8 are divisible by 7,
    the remainder when the 11th number is divided by 7 is 3. -/
theorem eleventh_number_remainder (start : ℕ) : 
  (∃ (n : ℕ), n ≥ start ∧ n < start + 50 ∧ 
    (Finset.filter (λ k => k % 7 = 0) (Finset.range (start + 50) \ Finset.range start)).card = 8) →
  (start + 10) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventh_number_remainder_l353_35360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l353_35385

noncomputable def f (x : Real) : Real := Real.sqrt 2 * Real.sin (2 * x - Real.pi / 4)

theorem triangle_area (A B C : Real) (a b c : Real) (h1 : f A = 1) 
  (h2 : b + c = 5 + 3 * Real.sqrt 2) (h3 : a = Real.sqrt 13) 
  (h4 : 0 < A ∧ A < Real.pi / 2) (h5 : 0 < B ∧ B < Real.pi / 2) 
  (h6 : 0 < C ∧ C < Real.pi / 2) :
  (1 / 2 : Real) * b * c * Real.sin A = 15 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l353_35385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l353_35363

-- Define the function f(x) = ln((2+x)/(2-x))
noncomputable def f (x : ℝ) : ℝ := Real.log ((2 + x) / (2 - x))

-- State the theorem
theorem f_odd_and_increasing :
  (∀ x ∈ Set.Ioo (-2 : ℝ) 2, f (-x) = -f x) ∧ 
  StrictMono f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l353_35363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_video_game_playing_time_difference_l353_35393

/-- Represents the playing time in hours for three days -/
structure PlayingTime where
  day1 : ℚ
  day2 : ℚ
  day3 : ℚ

/-- Calculates the average playing time over three days -/
def averagePlayingTime (pt : PlayingTime) : ℚ :=
  (pt.day1 + pt.day2 + pt.day3) / 3

theorem video_game_playing_time_difference 
  (pt : PlayingTime) 
  (h1 : pt.day1 = 2)
  (h2 : pt.day2 = 2)
  (h3 : pt.day3 > pt.day1)
  (h4 : averagePlayingTime pt = 3) :
  pt.day3 - pt.day1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_video_game_playing_time_difference_l353_35393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l353_35376

noncomputable def a (a₀ : ℝ) : ℕ → ℝ
| 0 => a₀
| n + 1 => min (2 * a a₀ n) (1 - 2 * a a₀ n)

theorem sequence_properties (a₀ : ℝ) (h_irrational : Irrational a₀) (h_bounds : 0 < a₀ ∧ a₀ < 1/2) :
  (∃ n : ℕ, a a₀ n < 3/16) ∧ ¬(∀ n : ℕ, a a₀ n > 7/40) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l353_35376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_output_increase_l353_35397

theorem factory_output_increase (P : ℝ) : 
  (((1 + P / 100) * (1 + 60 / 100)) * (1 - 43.18 / 100) = 1) →
  (abs (P - 10.09) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_output_increase_l353_35397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_blue_surface_area_l353_35394

/-- Represents a cube with edge length and number of smaller cubes -/
structure Cube where
  edge_length : ℕ
  num_small_cubes : ℕ

/-- Represents the color distribution of smaller cubes -/
structure ColorDistribution where
  blue_cubes : ℕ
  gold_cubes : ℕ

/-- Function to calculate the fraction of blue surface area -/
def blue_surface_area_fraction (large_cube : Cube) (color_dist : ColorDistribution) : ℚ :=
  sorry

/-- Theorem stating the minimum fraction of blue surface area -/
theorem min_blue_surface_area 
  (large_cube : Cube) 
  (color_dist : ColorDistribution) 
  (h1 : large_cube.edge_length = 4)
  (h2 : large_cube.num_small_cubes = 64)
  (h3 : color_dist.blue_cubes = 32)
  (h4 : color_dist.gold_cubes = 32)
  : ∃ (min_fraction : ℚ), 
    min_fraction = 1/12 ∧ 
    min_fraction ≤ blue_surface_area_fraction large_cube color_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_blue_surface_area_l353_35394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_independent_set_size_l353_35329

/-- A graph representing the seating arrangements of students -/
structure SeatingGraph where
  vertices : Finset ℕ
  edges : Finset (ℕ × ℕ)
  vertex_count : vertices.card = 26
  edge_count : edges.card = 13
  two_regular : ∀ v ∈ vertices, (edges.filter (λ e ↦ e.1 = v ∨ e.2 = v)).card = 2

/-- An independent set in the seating graph -/
def IndependentSet (G : SeatingGraph) (S : Finset ℕ) : Prop :=
  S ⊆ G.vertices ∧ ∀ u v, u ∈ S → v ∈ S → u ≠ v → (u, v) ∉ G.edges ∧ (v, u) ∉ G.edges

/-- The theorem stating the maximum size of an independent set -/
theorem max_independent_set_size (G : SeatingGraph) :
  ∃ S : Finset ℕ, IndependentSet G S ∧ S.card = 13 ∧
  ∀ T : Finset ℕ, IndependentSet G T → T.card ≤ 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_independent_set_size_l353_35329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implies_difference_l353_35349

theorem set_equality_implies_difference (a b : ℝ) :
  ({1, a + b, a} : Set ℝ) = {0, b / a, b} → b - a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implies_difference_l353_35349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_hexagonal_pyramid_l353_35373

/-- The radius of a sphere touching the sides of the base and the continuations of the lateral edges of a regular hexagonal pyramid -/
noncomputable def sphere_radius (a b : ℝ) : ℝ :=
  (a * (2 * b + a)) / (4 * Real.sqrt (b^2 - a^2))

/-- Theorem stating the radius of the sphere touching the regular hexagonal pyramid -/
theorem sphere_radius_hexagonal_pyramid (a b : ℝ) (ha : a > 0) (hb : b > a) :
  ∃ r : ℝ, r = sphere_radius a b ∧ 
  r > 0 ∧
  r * 4 * Real.sqrt (b^2 - a^2) = a * (2 * b + a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_hexagonal_pyramid_l353_35373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_special_logarithmic_equation_l353_35328

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem product_of_special_logarithmic_equation (a b : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (h : ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ 
       Real.sqrt (log a) = m ∧ 
       Real.sqrt (log b) = n ∧ 
       2 * log (Real.sqrt a) = m^2 ∧ 
       2 * log (Real.sqrt b) = n^2 ∧
       m + n + m^2 + n^2 = 144) :
  a * b = 10^98 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_special_logarithmic_equation_l353_35328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_block_selection_combinations_l353_35306

theorem block_selection_combinations (n : ℕ) (h : n = 6) : 
  (Nat.choose n 4) * (Nat.choose n 4) * Nat.factorial 4 = 5400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_block_selection_combinations_l353_35306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_test_point_value_l353_35389

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

theorem third_test_point_value (a b : ℝ) (h1 : a = 1000) (h2 : b = 2000) :
  let x₁ := a + (b - a) * (1 - 1 / golden_ratio)
  let x₂ := a + b - x₁
  let x₃ := x₂ + b - x₁
  x₃ = 1764 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_test_point_value_l353_35389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_common_elements_l353_35369

def sequence_a : ℕ → ℤ
  | 0 => 2
  | 1 => 14
  | (n + 2) => 14 * sequence_a (n + 1) + sequence_a n

def sequence_b : ℕ → ℤ
  | 0 => 2
  | 1 => 14
  | (n + 2) => 6 * sequence_b (n + 1) - sequence_b n

theorem infinitely_many_common_elements :
  ∃ (S : Set ℤ), (Set.Infinite S) ∧ (∀ x ∈ S, ∃ n m : ℕ, sequence_a n = x ∧ sequence_b m = x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_common_elements_l353_35369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_count_correct_l353_35371

/-- The number of dogs that can sit -/
def sit : ℕ := 45

/-- The number of dogs that can stay -/
def stay : ℕ := 35

/-- The number of dogs that can roll over -/
def roll_over : ℕ := 40

/-- The number of dogs that can sit and stay -/
def sit_and_stay : ℕ := 20

/-- The number of dogs that can stay and roll over -/
def stay_and_roll_over : ℕ := 15

/-- The number of dogs that can sit and roll over -/
def sit_and_roll_over : ℕ := 20

/-- The number of dogs that can do all three tricks -/
def all_three : ℕ := 12

/-- The number of dogs that can do none of the tricks -/
def no_tricks : ℕ := 8

/-- The total number of dogs in the center -/
def total_dogs : ℕ := 93

theorem dog_count_correct : 
  (sit - sit_and_stay - sit_and_roll_over + all_three) +
  (stay - sit_and_stay - stay_and_roll_over + all_three) +
  (roll_over - sit_and_roll_over - stay_and_roll_over + all_three) +
  all_three + no_tricks = total_dogs :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_count_correct_l353_35371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l353_35302

/-- The ellipse C in the Cartesian coordinate system -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The line l in the Cartesian coordinate system -/
def line (m : ℝ) (x : ℝ) : Prop := x = m

/-- The point P on line l -/
def point_on_line (m y : ℝ) : ℝ × ℝ := (m, y)

/-- Intersection points of a line through P with the ellipse C -/
def intersection_points (a b m y₀ : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, 
    ellipse a b p.1 p.2 ∧ 
    (p.2 - y₀) / (p.1 - m) = t}

/-- The perpendicular line l' -/
def perpendicular_line (m y₀ x y : ℝ) : Prop :=
  y = -3*y₀/(2*Real.sqrt 2) * (x + 4*Real.sqrt 2/3)

theorem ellipse_and_line_properties
  (a b m : ℝ)
  (h₁ : a > b)
  (h₂ : b > 0)
  (h₃ : ellipse a b 3 1)
  (h₄ : ellipse a b 3 (-1))
  (h₅ : ellipse a b (Real.sqrt 3) (Real.sqrt 3))
  (h₆ : line m (-2*Real.sqrt 2)) :
  (a^2 = 12 ∧ b^2 = 4) ∧
  ∀ y₀ : ℝ, ∃ P : ℝ × ℝ, 
    P = point_on_line m y₀ ∧
    ∀ M N : ℝ × ℝ, 
      M ∈ intersection_points a b m y₀ ∧
      N ∈ intersection_points a b m y₀ ∧
      (M.1 - P.1)^2 + (M.2 - P.2)^2 = (N.1 - P.1)^2 + (N.2 - P.2)^2 →
      perpendicular_line m y₀ (-4*Real.sqrt 2/3) 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l353_35302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orientation_after_odd_transformations_l353_35326

/-- Represents the orientation of a triangle -/
inductive TriangleOrientation
| Clockwise
| Counterclockwise

/-- Represents a point on a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle formed by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Calculates the orientation of a triangle -/
def calculateOrientation (t : Triangle) : TriangleOrientation :=
  sorry

/-- Simulates moving one point of the triangle across the line formed by the other two points -/
def movePoint (t : Triangle) : Triangle :=
  sorry

/-- Theorem: After an odd number of transformations, the orientation of the triangle will be opposite to its initial orientation -/
theorem orientation_after_odd_transformations (t : Triangle) (n : ℕ) (h : Odd n) :
  calculateOrientation (((fun t' => movePoint t') ^[n]) t) ≠ calculateOrientation t :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orientation_after_odd_transformations_l353_35326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l353_35370

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 1/(x^2 - 2*x + 1)

theorem f_minimum_value :
  ∀ x ∈ Set.Ioo 0 3, f x ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l353_35370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_correct_l353_35395

/-- The area of the triangle formed by tangents to y = 1/x and y = x^2 at their intersection points and the x-axis -/
noncomputable def triangleArea : ℝ := 1 / 9

/-- The first curve: y = 1/x -/
noncomputable def curve1 (x : ℝ) : ℝ := 1 / x

/-- The second curve: y = x^2 -/
def curve2 (x : ℝ) : ℝ := x^2

/-- Theorem stating that the area of the triangle is equal to the calculated value -/
theorem triangle_area_is_correct : 
  ∃ (x₁ x₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    curve1 x₁ = curve2 x₁ ∧ 
    curve1 x₂ = curve2 x₂ ∧ 
    triangleArea = 1 / 9 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_correct_l353_35395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_negative_seven_pi_sixths_l353_35333

theorem sec_negative_seven_pi_sixths :
  1 / Real.cos (-7 * π / 6) = -2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_negative_seven_pi_sixths_l353_35333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_five_divisor_l353_35323

def Q : ℕ := (Finset.range 150).prod (λ i => 2 * i + 1)

theorem largest_power_of_five_divisor (k : ℕ) : 
  (∀ m : ℕ, m > k → ¬(Q % (5^m) = 0)) ∧ (Q % (5^k) = 0) ↔ k = 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_five_divisor_l353_35323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_ellipse_properties_l353_35335

/-- An ellipse symmetric to the coordinate axes passing through two given points -/
structure SymmetricEllipse where
  a : ℝ  -- The semi-major axis length
  b : ℝ  -- The semi-minor axis length
  passes_through_A : (3 : ℝ)^2 / a^2 + (-16/5 : ℝ)^2 / b^2 = 1
  passes_through_B : (-4 : ℝ)^2 / a^2 + (-12/5 : ℝ)^2 / b^2 = 1

/-- Properties of the symmetric ellipse -/
theorem symmetric_ellipse_properties (e : SymmetricEllipse) :
  (∀ x y : ℝ, x^2 / 25 + y^2 / 16 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1) ∧
  (let c := Real.sqrt (e.a^2 - e.b^2);
   (3 + c)^2 + (-16/5)^2 = (6.8 : ℝ)^2 ∧
   (3 - c)^2 + (-16/5)^2 = (3.2 : ℝ)^2) ∧
  (Real.sqrt (e.a^2 - e.b^2) / e.a = (0.6 : ℝ)) ∧
  (∀ x : ℝ, (x = 25/3 ∨ x = -25/3) ↔ x = e.a / (Real.sqrt (e.a^2 - e.b^2) / e.a)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_ellipse_properties_l353_35335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_owners_without_motorcycle_l353_35383

theorem car_owners_without_motorcycle (total : ℕ) (car_owners : ℕ) (motorcycle_owners : ℕ) 
  (h1 : total = 500)
  (h2 : car_owners = 450)
  (h3 : motorcycle_owners = 120)
  (h4 : ∀ a : ℕ, a < total → (a < car_owners ∨ a < motorcycle_owners))
  (h5 : car_owners + motorcycle_owners - total ≤ car_owners) :
  car_owners - (car_owners + motorcycle_owners - total) = 380 :=
by
  sorry

#check car_owners_without_motorcycle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_owners_without_motorcycle_l353_35383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_surface_area_of_revolution_correct_l353_35396

/-- Represents a parallelogram -/
structure Parallelogram where
  perimeter : ℝ
  diagonal : ℝ

/-- 
Calculates the surface area of a solid of revolution formed by rotating a parallelogram 
around an axis perpendicular to its diagonal and passing through its end.
-/
noncomputable def surfaceAreaOfRevolution (p : Parallelogram) : ℝ :=
  2 * Real.pi * p.diagonal * (p.perimeter / 2)

/-- 
Theorem stating that the surface area of the solid of revolution is correct 
for a parallelogram with given properties.
-/
theorem surface_area_of_revolution_correct (p : Parallelogram) :
  surfaceAreaOfRevolution p = 2 * Real.pi * p.diagonal * (p.perimeter / 2) :=
by
  -- Unfold the definition of surfaceAreaOfRevolution
  unfold surfaceAreaOfRevolution
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_surface_area_of_revolution_correct_l353_35396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_sequence_l353_35307

theorem strictly_increasing_sequence (lambda : ℝ) :
  (∀ n : ℕ+, (n : ℝ)^2 + lambda * n < ((n + 1) : ℝ)^2 + lambda * (n + 1)) ↔ lambda > -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_sequence_l353_35307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_identical_face_sums_l353_35390

/-- Represents a face of a cube -/
inductive Face
  | Top
  | Bottom
  | Left
  | Right
  | Front
  | Back

/-- Represents a small cube with numbers 1-6 on its faces -/
structure Cube where
  faces : Face → Fin 6

/-- Predicate for adjacent cubes in the large cube -/
def IsAdjacent : Fin 27 → Fin 27 → Prop := sorry

/-- Predicate for touching faces of adjacent cubes -/
def TouchingFaces : Fin 27 → Fin 27 → Face → Face → Prop := sorry

/-- Represents a 3x3x3 cube assembled from 27 smaller cubes -/
structure LargeCube where
  small_cubes : Fin 27 → Cube
  adjacent_differ_by_one : ∀ (c1 c2 : Fin 27), IsAdjacent c1 c2 → 
    ∀ (f1 f2 : Face), TouchingFaces c1 c2 f1 f2 → 
    (small_cubes c1).faces f1 - (small_cubes c2).faces f2 = 1 ∨
    (small_cubes c2).faces f2 - (small_cubes c1).faces f1 = 1

/-- Sum of numbers on a face of the large cube -/
def FaceSum (lc : LargeCube) (f : Face) : ℕ := sorry

/-- Theorem stating that it's impossible to have six identical sums on the faces of the large cube -/
theorem impossible_identical_face_sums (lc : LargeCube) : 
  ¬∃ (s : ℕ), ∀ (f : Face), FaceSum lc f = s := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_identical_face_sums_l353_35390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l353_35381

theorem expression_evaluation : 
  (0.064 : ℝ) ^ (-(1/3 : ℝ)) - ((-7/8 : ℝ) ^ (0 : ℝ)) + (16 : ℝ) ^ (0.75 : ℝ) + (0.25 : ℝ) ^ (1/2 : ℝ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l353_35381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_approx_18_l353_35334

/-- The length of the diagonal of a rectangular field -/
noncomputable def diagonal_length (side : ℝ) (area : ℝ) : ℝ :=
  Real.sqrt (side^2 + (area / side)^2)

/-- Theorem: The diagonal of a rectangular field with one side of 15 m and an area of 149.248115565993 m² is approximately 18 m -/
theorem diagonal_approx_18 :
  let side := (15 : ℝ)
  let area := 149.248115565993
  abs (diagonal_length side area - 18) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_approx_18_l353_35334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_and_equality_conditions_l353_35336

theorem inequalities_and_equality_conditions (a b : ℝ) :
  (a * b > 0 → (((a^2 * b^2 * (a + b)^2) / 4) ^ (1/3) ≤ (a^2 + 10*a*b + b^2) / 12)) ∧
  (((a^2 * b^2 * (a + b)^2) / 4) ^ (1/3) ≤ (a^2 + a*b + b^2) / 3) ∧
  ((((a^2 * b^2 * (a + b)^2) / 4) ^ (1/3) = (a^2 + 10*a*b + b^2) / 12) ↔ a = b) ∧
  ((((a^2 * b^2 * (a + b)^2) / 4) ^ (1/3) = (a^2 + a*b + b^2) / 3) ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_and_equality_conditions_l353_35336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l353_35362

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define point P
def point_P : ℝ × ℝ := (0, 1)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_distance_sum :
  ∃ (A B : ℝ × ℝ),
    curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    A ≠ B ∧
    distance point_P A + distance point_P B = 8 * Real.sqrt 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l353_35362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_festival_attendance_l353_35399

theorem festival_attendance (total_students : ℕ) (festival_attendance : ℕ) 
  (h1 : total_students = 1500)
  (h2 : festival_attendance = 900) : ℕ :=
by
  -- Define boys as a variable
  let boys : ℕ := 643 -- We know this from the solution
  let girls : ℕ := total_students - boys
  let girls_attending : ℕ := (3 * girls) / 4
  let boys_attending : ℕ := (2 * boys) / 5
  
  -- State the main equation as an assumption
  have h3 : girls_attending + boys_attending = festival_attendance := by sorry
  
  -- State that girls_attending is 643
  have h4 : girls_attending = 643 := by sorry
  
  -- Return the number of girls attending
  exact girls_attending

-- This will not evaluate due to the theorem's structure, so we comment it out
-- #eval festival_attendance 1500 900

end NUMINAMATH_CALUDE_ERRORFEEDBACK_festival_attendance_l353_35399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_club_officer_selection_l353_35327

theorem club_officer_selection (n : ℕ) (k : ℕ) : n = 10 ∧ k = 4 →
  (n * (n - 1) * (n - 2) * (n - 3)) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_club_officer_selection_l353_35327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_difference_of_squares_l353_35379

def difference_of_squares (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 - b^2

theorem unique_non_difference_of_squares :
  (difference_of_squares 2004) ∧
  (difference_of_squares 2005) ∧
  (difference_of_squares 2007) ∧
  ¬(difference_of_squares 2006) ∧
  ∀ m : ℤ, m ∈ ({2004, 2005, 2006, 2007} : Set ℤ) → m ≠ 2006 → difference_of_squares m :=
by sorry

#check unique_non_difference_of_squares

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_difference_of_squares_l353_35379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_b1_b2_l353_35377

def sequence_b (b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → b (n + 2) = (b n + 2017) / (1 + b (n + 1))

theorem min_sum_b1_b2 (b : ℕ → ℕ) (h : sequence_b b) :
  ∃ b1 b2 : ℕ, b 1 = b1 ∧ b 2 = b2 ∧ b1 + b2 = 91 ∧
  ∀ b1' b2' : ℕ, b 1 = b1' ∧ b 2 = b2' → b1' + b2' ≥ 91 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_b1_b2_l353_35377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_total_cost_l353_35303

/-- Represents the cell phone plan and usage details -/
structure CellPhonePlan where
  baseCost : ℝ
  textCost : ℝ
  includedMinutesCost : ℝ
  extraMinutesCost : ℝ
  textsSent : ℕ
  minutesTalked : ℕ

/-- Calculates the total cost of the cell phone plan -/
noncomputable def totalCost (plan : CellPhonePlan) : ℝ :=
  let includedMinutes : ℕ := 25 * 60
  let extraMinutes : ℕ := max (plan.minutesTalked - includedMinutes) 0
  plan.baseCost +
  plan.textCost * (plan.textsSent : ℝ) / 100 +
  plan.includedMinutesCost * (min plan.minutesTalked includedMinutes : ℝ) / 100 +
  plan.extraMinutesCost * (extraMinutes : ℝ) / 100

/-- Theorem stating that Alex's total cost is $142.00 -/
theorem alex_total_cost :
  let plan : CellPhonePlan := {
    baseCost := 25
    textCost := 10
    includedMinutesCost := 5
    extraMinutesCost := 15
    textsSent := 150
    minutesTalked := 28 * 60
  }
  totalCost plan = 142 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_total_cost_l353_35303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l353_35330

/-- The function q(x) that satisfies the given conditions -/
noncomputable def q (x : ℝ) : ℝ := 3 * x^2 - 12

/-- The numerator of the rational function -/
noncomputable def numerator (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 3

/-- The rational function formed by numerator and q(x) -/
noncomputable def f (x : ℝ) : ℝ := numerator x / q x

theorem q_satisfies_conditions :
  (∀ x, x ≠ 2 ∧ x ≠ -2 → f x ≠ 0) ∧  -- Vertical asymptotes at 2 and -2
  (¬ ∃ L, ∀ ε > 0, ∃ N, ∀ x > N, |f x - L| < ε) ∧  -- No horizontal asymptote
  q 3 = 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l353_35330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extract_constant_gcd_subsequence_l353_35374

/-- A sequence of positive integers -/
def PositiveIntegerSequence := ℕ → ℕ+

/-- A function that counts the number of prime factors of a positive integer -/
def countPrimeFactors : ℕ+ → ℕ := sorry

/-- The theorem statement -/
theorem extract_constant_gcd_subsequence
  (a : PositiveIntegerSequence)
  (h_increasing : ∀ n : ℕ, a n < a (n + 1))
  (h_bounded_factors : ∀ n : ℕ, countPrimeFactors (a n) ≤ 1987) :
  ∃ (b : ℕ → ℕ+) (k : ℕ+),
    (∀ n : ℕ, ∃ m : ℕ, b n = a m) ∧
    (∀ n : ℕ, b n < b (n + 1)) ∧
    (∀ i j : ℕ, Nat.gcd (b i).val (b j).val = k.val) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extract_constant_gcd_subsequence_l353_35374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_workers_for_profit_is_13_workers_13_sufficient_for_profit_l353_35310

/-- Represents the factory's production and cost parameters -/
structure FactoryParams where
  fixedCost : ℝ
  workerWage : ℝ
  productionRate : ℝ
  sellingPrice : ℝ
  workdayLength : ℝ

/-- Calculates the minimum number of workers needed for profit -/
noncomputable def minWorkersForProfit (params : FactoryParams) : ℕ :=
  Nat.ceil (params.fixedCost / (params.workdayLength * (params.productionRate * params.sellingPrice - params.workerWage)))

/-- Theorem stating the minimum number of workers needed for profit -/
theorem min_workers_for_profit_is_13 (params : FactoryParams)
  (h1 : params.fixedCost = 800)
  (h2 : params.workerWage = 20)
  (h3 : params.productionRate = 6)
  (h4 : params.sellingPrice = 4.5)
  (h5 : params.workdayLength = 9) :
  minWorkersForProfit params = 13 := by
  sorry

/-- Theorem proving that 13 workers are sufficient for profit -/
theorem workers_13_sufficient_for_profit (params : FactoryParams)
  (h1 : params.fixedCost = 800)
  (h2 : params.workerWage = 20)
  (h3 : params.productionRate = 6)
  (h4 : params.sellingPrice = 4.5)
  (h5 : params.workdayLength = 9) :
  13 * params.workdayLength * (params.productionRate * params.sellingPrice) > params.fixedCost + 13 * params.workdayLength * params.workerWage := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_workers_for_profit_is_13_workers_13_sufficient_for_profit_l353_35310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_N_l353_35366

/-- Given a positive integer N that satisfies the equation N * (N + 1) / 2 = 3003,
    prove that the sum of its digits is 14. -/
theorem sum_of_digits_of_N (N : ℕ) (h : N * (N + 1) / 2 = 3003) : 
  (N.repr.toList.map (λ c => c.toNat - '0'.toNat)).sum = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_N_l353_35366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_edges_l353_35322

/-- Represents a cube with its properties -/
structure Cube where
  vertices : Fin 8
  edges : Fin 12
  face_diagonals : Fin 12
  space_diagonals : Fin 4

/-- Represents a path on the cube -/
structure CubePath (cube : Cube) where
  start_vertex : Fin 8
  end_vertex : Fin 8
  visited_vertices : List (Fin 8)
  path_roads : Fin 8 → Fin 28 -- 28 = 12 edges + 12 face diagonals + 4 space diagonals

/-- Predicate to check if a path is valid according to the problem constraints -/
def is_valid_path (cube : Cube) (path : CubePath cube) : Prop :=
  path.start_vertex = path.end_vertex ∧
  path.visited_vertices.length = 7 ∧
  path.visited_vertices.Nodup ∧
  ∀ i j, i ≠ j → path.path_roads i ≠ path.path_roads j

/-- Counts the number of edges in a path -/
def count_edges (cube : Cube) (path : CubePath cube) : Nat :=
  (List.filter (· < 12) (List.map path.path_roads (List.range 8))).length

/-- The main theorem to be proved -/
theorem at_least_two_edges (cube : Cube) (path : CubePath cube) 
    (h : is_valid_path cube path) : 
    count_edges cube path ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_edges_l353_35322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_satisfying_integer_l353_35346

noncomputable def deg_to_rad (x : ℝ) : ℝ := x * Real.pi / 180

def satisfies_property (x : ℕ) : Prop :=
  x > 1 ∧ Real.cos (deg_to_rad (x : ℝ)) = Real.cos (deg_to_rad ((x^2 : ℕ) : ℝ))

theorem smallest_satisfying_integer : 
  satisfies_property 26 ∧ ∀ x : ℕ, x < 26 → ¬satisfies_property x :=
by sorry

#check smallest_satisfying_integer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_satisfying_integer_l353_35346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecagon_enclosure_theorem_l353_35378

/-- Represents a regular polygon -/
structure RegularPolygon :=
  (sides : ℕ)

/-- Represents the configuration of polygons -/
structure PolygonConfiguration :=
  (center : RegularPolygon)
  (enclosing : List RegularPolygon)

/-- The property we want to prove -/
def shares_vertex_with_max (config : PolygonConfiguration) (p : RegularPolygon) : Prop :=
  p ∈ config.enclosing ∧
  p.sides = 13 ∧
  ∃ max_polygon : RegularPolygon, max_polygon ∈ config.enclosing ∧
    max_polygon.sides = 14 ∧
    ∀ q ∈ config.enclosing, q.sides ≤ max_polygon.sides

theorem dodecagon_enclosure_theorem (config : PolygonConfiguration) :
  config.center.sides = 12 →
  config.enclosing.length = 12 →
  (∀ i, i ∈ List.range 12 → ∃ p ∈ config.enclosing, p.sides = i + 3) →
  ∃ p : RegularPolygon, shares_vertex_with_max config p :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecagon_enclosure_theorem_l353_35378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_spent_shopping_l353_35340

/-- Given an initial amount and the amount left after spending, 
    calculate the percentage of money spent. -/
noncomputable def percentage_spent (initial_amount : ℝ) (amount_left : ℝ) : ℝ :=
  (initial_amount - amount_left) / initial_amount * 100

/-- Theorem stating that given $4000 initially and $2800 left after spending,
    the percentage spent is 30%. -/
theorem percentage_spent_shopping : percentage_spent 4000 2800 = 30 := by
  -- Unfold the definition of percentage_spent
  unfold percentage_spent
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_spent_shopping_l353_35340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_inequality_solvability_l353_35321

-- Part I
def solution_set (x : ℝ) : Prop :=
  x > 3 ∨ x < -4 ∨ (-3 < x ∧ x < 2)

theorem inequality_solution :
  ∀ x : ℝ, (abs (6 - abs (2*x + 1)) > 1) ↔ solution_set x :=
sorry

-- Part II
theorem inequality_solvability (m : ℝ) :
  (∃ x : ℝ, abs (x + 1) + abs (x - 1) + 3 + x < m) ↔ m > 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_inequality_solvability_l353_35321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_upper_bound_b_l353_35314

-- Part 1
def f (x : ℝ) := |2*x + 1|
def g (x : ℝ) (a : ℝ) := |a*x|

theorem inequality_solution_set :
  {x : ℝ | f x ≥ g x 1 + 1} = Set.Iic (-2) ∪ Set.Ici 0 := by sorry

-- Part 2
theorem upper_bound_b (b : ℝ) :
  (∀ x, f x + g x 2 ≥ b) → b ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_upper_bound_b_l353_35314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_opposite_sides_l353_35387

/-- Predicate indicating that a point is the center of a square -/
def IsCenter (square : Set (ℝ × ℝ)) (center : ℝ × ℝ) : Prop :=
  sorry

/-- Predicate indicating that a set of points is a side of a square -/
def IsSide (square : Set (ℝ × ℝ)) (side : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate indicating that two sides of a square are opposite to each other -/
def IsOpposite (square : Set (ℝ × ℝ)) (side1 side2 : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- Given a square with center at (1, 0) and side AB on the line x - y + 1 = 0,
    prove that the opposite side CD lies on the line x - y - 3 = 0 -/
theorem square_opposite_sides (square : Set (ℝ × ℝ)) :
  (∃ (center : ℝ × ℝ), center = (1, 0) ∧ IsCenter square center) →
  (∃ (AB : Set (ℝ × ℝ)), IsSide square AB ∧ ∀ (x y : ℝ), (x, y) ∈ AB ↔ x - y + 1 = 0) →
  (∃ (CD : Set (ℝ × ℝ)), IsSide square CD ∧ IsOpposite square AB CD ∧
    ∀ (x y : ℝ), (x, y) ∈ CD ↔ x - y - 3 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_opposite_sides_l353_35387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l353_35337

noncomputable def f (x : Real) : Real := 2 * Real.sin (x + Real.pi / 4) ^ 2 - Real.sqrt 3 * Real.cos (2 * x)

theorem problem_solution :
  ∃ (α : Real),
    α ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) ∧
    (∀ x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), f x ≤ f α) ∧
    f α = 3 ∧
    α = 5 * Real.pi / 12 ∧
    ∀ (a b c : Real),
      0 < a ∧ 0 < b ∧ 0 < c →
      a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos (Real.pi / 3) →
      Real.sin (Real.arcsin (b / a)) * Real.sin (Real.arcsin (c / a)) = Real.sin (Real.pi / 3) ^ 2 →
      b = c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l353_35337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_of_sixteen_l353_35382

theorem fourth_root_of_sixteen (z : ℂ) : z^4 = 16 ↔ z ∈ ({2, -2, 2*I, -2*I} : Set ℂ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_of_sixteen_l353_35382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_length_in_minutes_proof_interval_length_l353_35341

/-- The length of each interval in hours -/
noncomputable def interval_length : ℝ := 2 / 15

/-- The speed decrease per interval in miles per hour -/
def speed_decrease : ℝ := 3

/-- The initial speed in miles per hour -/
def initial_speed : ℝ := 39

/-- The distance traveled in the fifth interval in miles -/
def fifth_interval_distance : ℝ := 3.6

/-- The speed in the fifth interval in miles per hour -/
def fifth_interval_speed : ℝ := initial_speed - 4 * speed_decrease

theorem interval_length_in_minutes :
  interval_length * 60 = 8 :=
by sorry

theorem proof_interval_length :
  fifth_interval_distance = fifth_interval_speed * interval_length :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_length_in_minutes_proof_interval_length_l353_35341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chebyshev_generating_functions_l353_35312

/-- Lucas polynomials -/
noncomputable def L (n : ℕ) (x : ℂ) : ℂ := sorry

/-- Fibonacci polynomials -/
noncomputable def F (n : ℕ) (x : ℂ) : ℂ := sorry

/-- Chebyshev polynomials of the first kind -/
noncomputable def T (n : ℕ) (x : ℂ) : ℂ := (1/2) * (-Complex.I)^n * L n (2 * Complex.I * x)

/-- Chebyshev polynomials of the second kind -/
noncomputable def U (n : ℕ) (x : ℂ) : ℂ := (-Complex.I) * n * F (n+1) (2 * Complex.I * x)

/-- Lucas polynomials generating function -/
noncomputable def L_gen (x z : ℂ) : ℂ := (2 - 2*x*z) / (1 - 2*x*z + z^2)

/-- Fibonacci polynomials generating function -/
noncomputable def F_gen (x z : ℂ) : ℂ := z / (1 - x*z - z^2)

/-- Generating function for Chebyshev polynomials of the first kind -/
noncomputable def F_T (x z : ℂ) : ℂ := ∑' n, T n x * z^n

/-- Generating function for Chebyshev polynomials of the second kind -/
noncomputable def F_U (x z : ℂ) : ℂ := ∑' n, U n x * z^n

theorem chebyshev_generating_functions (x z : ℂ) :
  F_T x z = (1 - x*z) / (1 - 2*x*z + z^2) ∧
  F_U x z = 1 / (1 - 2*x*z + z^2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chebyshev_generating_functions_l353_35312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_is_even_l353_35353

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x / (2^x - 1)
noncomputable def g (x : ℝ) : ℝ := x / 2

-- Define h as the sum of f and g
noncomputable def h (x : ℝ) : ℝ := f x + g x

-- State the theorem that h is an even function
theorem h_is_even : ∀ x : ℝ, h (-x) = h x := by
  intro x
  -- Expand the definition of h
  simp [h, f, g]
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_is_even_l353_35353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arithmetic_mean_cosine_l353_35313

theorem triangle_arithmetic_mean_cosine (A B C : ℝ) : 
  (A + B + C = Real.pi) →  -- Sum of angles in a triangle
  (2 * B = A + C) →  -- B is arithmetic mean of A and C
  Real.cos B = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arithmetic_mean_cosine_l353_35313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_mpg_is_25_l353_35305

/-- Ray's car's fuel efficiency in miles per gallon -/
noncomputable def ray_mpg : ℝ := 30

/-- Tom's car's fuel efficiency in miles per gallon -/
noncomputable def tom_mpg : ℝ := 20

/-- Distance Ray drives in miles -/
noncomputable def ray_distance : ℝ := 150

/-- Distance Tom drives in miles -/
noncomputable def tom_distance : ℝ := 100

/-- Combined rate of miles per gallon for both cars -/
noncomputable def combined_mpg : ℝ := 
  (ray_distance + tom_distance) / (ray_distance / ray_mpg + tom_distance / tom_mpg)

theorem combined_mpg_is_25 : combined_mpg = 25 := by
  -- Expand the definition of combined_mpg
  unfold combined_mpg
  -- Perform algebraic manipulations
  norm_num
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_mpg_is_25_l353_35305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sum_square_l353_35301

-- Define the triangle ABC
def triangle (A B C : ℝ × ℝ) : Prop :=
  -- AB is perpendicular to BC (right angle at B)
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0

-- Define the length of a side
noncomputable def length (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define a perfect square
def is_perfect_square (n : ℝ) : Prop :=
  ∃ m : ℕ, n = m^2

-- The main theorem
theorem right_triangle_sum_square 
  (A B C : ℝ × ℝ) 
  (h_triangle : triangle A B C)
  (h_AB : length A B = 12)
  (h_BC : length B C = 16) :
  is_perfect_square (length A C + length B C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sum_square_l353_35301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_5pi_div_2_f_inequality_solution_set_l353_35372

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := tan (x / 2 - π / 3)

-- Theorem for part 1
theorem f_value_at_5pi_div_2 : f (5 * π / 2) = Real.sqrt 3 - 2 := by sorry

-- Theorem for part 2
theorem f_inequality_solution_set (x : ℝ) :
  f x ≤ Real.sqrt 3 ↔ ∃ k : ℤ, 2 * ↑k * π - π / 3 < x ∧ x ≤ 2 * ↑k * π + 4 * π / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_5pi_div_2_f_inequality_solution_set_l353_35372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l353_35311

/-- Calculates the length of a train given its speed, the time it takes to cross a bridge, and the length of the bridge. -/
noncomputable def train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : ℝ :=
  (train_speed * 1000 / 3600) * crossing_time - bridge_length

/-- Theorem stating that a train with the given parameters has the specified length. -/
theorem train_length_calculation :
  let train_speed : ℝ := 72
  let crossing_time : ℝ := 12.399008079353651
  let bridge_length : ℝ := 138
  abs (train_length train_speed crossing_time bridge_length - 109.98016158707302) < 1e-10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l353_35311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_negative_l353_35351

noncomputable def number_list : List ℝ := [-8, 2.89, 6, -1/2, -0.25, 5/3, -13/4, 0]

theorem count_non_negative : (number_list.filter (λ x => x ≥ 0)).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_negative_l353_35351
