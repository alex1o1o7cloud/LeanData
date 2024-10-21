import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_20_l1065_106564

/-- Calculates the position of the hour hand at a given time -/
noncomputable def hour_hand_position (hours : ℕ) (minutes : ℕ) : ℝ :=
  (hours % 12 : ℝ) * 30 + (minutes : ℝ) * 0.5

/-- Calculates the position of the minute hand at a given time -/
noncomputable def minute_hand_position (minutes : ℕ) : ℝ :=
  (minutes : ℝ) * 6

/-- Calculates the angle between the hour and minute hands -/
noncomputable def angle_between_hands (hours : ℕ) (minutes : ℕ) : ℝ :=
  abs (hour_hand_position hours minutes - minute_hand_position minutes)

/-- Theorem: The angle between the hour and minute hands of a clock at 7:20 is 100° -/
theorem clock_angle_at_7_20 :
  angle_between_hands 7 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_20_l1065_106564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l1065_106585

/-- The sum of the infinite series Σ(k=1 to ∞) k^2 / 3^k is equal to 1 -/
theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ)^2 / (3 : ℝ)^k = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l1065_106585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_edges_bethia_graph_bethia_graph_with_7688_edges_l1065_106504

/-- A graph representing the cities and express train connections in Bethia. -/
structure BethiaGraph where
  /-- The set of vertices (cities) in the graph. -/
  V : Type
  /-- The number of vertices is 125. -/
  vertex_count : Fintype V
  card_V : Fintype.card V = 125
  /-- The set of edges (express train connections) in the graph. -/
  E : Finset (V × V)
  /-- The graph is undirected. -/
  symmetric : ∀ a b, (a, b) ∈ E → (b, a) ∈ E
  /-- Any four vertices can be connected in a cycle. -/
  four_cycle : ∀ (a b c d : V), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a →
    ∃ (p : Fin 4 → V), 
      ((p 0, p 1) ∈ E ∧ (p 1, p 2) ∈ E ∧ (p 2, p 3) ∈ E ∧ (p 3, p 0) ∈ E)

/-- The theorem stating the minimum number of edges in a BethiaGraph. -/
theorem min_edges_bethia_graph (G : BethiaGraph) : G.E.card ≥ 7688 := by
  sorry

/-- The theorem stating that 7688 edges is achievable in a BethiaGraph. -/
theorem bethia_graph_with_7688_edges : ∃ (G : BethiaGraph), G.E.card = 7688 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_edges_bethia_graph_bethia_graph_with_7688_edges_l1065_106504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suv_max_distance_l1065_106547

/-- Represents the fuel efficiency of an SUV in miles per gallon -/
structure FuelEfficiency where
  highway : ℝ
  city : ℝ

/-- Calculates the maximum distance an SUV can travel given its fuel efficiency and available fuel -/
noncomputable def maxDistance (efficiency : FuelEfficiency) (fuel : ℝ) : ℝ :=
  max efficiency.highway efficiency.city * fuel

/-- Theorem: The maximum distance of an SUV with given fuel efficiency on 21 gallons of gasoline -/
theorem suv_max_distance (efficiency : FuelEfficiency) 
  (h1 : efficiency.highway = 12.2)
  (h2 : efficiency.city = 7.6) :
  maxDistance efficiency 21 = 256.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_suv_max_distance_l1065_106547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_form_l1065_106561

theorem cosine_sum_product_form :
  ∃ (a b c d : ℕ+), 
    (∀ x : ℝ, Real.cos (2*x) + Real.cos (4*x) + Real.cos (8*x) + Real.cos (10*x) = 
      (a : ℝ) * Real.cos (b*x) * Real.cos (c*x) * Real.cos (d*x)) ∧
    a + b + c + d = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_form_l1065_106561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_radius_l1065_106519

/-- The equation of a circle in the form x^2 + y^2 + ax + by + c = 0 -/
def CircleEquation (a b c : ℝ) : ℝ × ℝ → Prop :=
  fun p => p.1^2 + p.2^2 + a * p.1 + b * p.2 + c = 0

/-- The center of a circle given by its equation coefficients -/
noncomputable def CircleCenter (a b c : ℝ) : ℝ × ℝ :=
  (-a/2, -b/2)

/-- The radius of a circle given by its equation coefficients -/
noncomputable def CircleRadius (a b c : ℝ) : ℝ :=
  Real.sqrt ((a^2 + b^2) / 4 - c)

theorem circle_center_and_radius :
  let eq := CircleEquation (-4) 2 2
  let center := CircleCenter (-4) 2 2
  let radius := CircleRadius (-4) 2 2
  center = (2, -1) ∧ radius = Real.sqrt 3 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_radius_l1065_106519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l1065_106503

/-- Represents the time (in seconds) for a train to cross a bridge -/
noncomputable def time_to_cross_bridge (num_carriages : ℕ) (carriage_length : ℝ) (engine_length : ℝ) 
  (train_speed_kmph : ℝ) (bridge_length_km : ℝ) : ℝ :=
  let train_length := num_carriages * carriage_length + engine_length
  let total_distance := train_length + bridge_length_km * 1000
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  total_distance / train_speed_mps

/-- Theorem stating that the time for a train to cross a bridge is approximately 179.88 seconds -/
theorem train_bridge_crossing_time :
  ∃ ε > 0, |time_to_cross_bridge 24 60 60 60 1.5 - 179.88| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l1065_106503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_vertical_asymptote_l1065_106508

/-- The function g(x) defined by (x^2 - 2x + k) / (x^2 - 3x - 10) -/
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x + k) / (x^2 - 3*x - 10)

/-- Predicate to check if a function has a vertical asymptote at a point -/
def HasVerticalAsymptoteAt (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ M > 0, ∃ δ > 0, ∀ y, 0 < |y - x| ∧ |y - x| < δ → |f y| > M

/-- Theorem stating that g(x) has exactly one vertical asymptote iff k = -15 or k = -8 -/
theorem exactly_one_vertical_asymptote (k : ℝ) :
  (∃! x, HasVerticalAsymptoteAt (g k) x) ↔ (k = -15 ∨ k = -8) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_vertical_asymptote_l1065_106508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_time_difference_l1065_106560

/-- The time difference between when Car X and Car Y start traveling -/
noncomputable def time_difference (v_x v_y d : ℝ) : ℝ :=
  (d / v_y - d / v_x) * 60

/-- Theorem stating the time difference between when Car X and Car Y start traveling -/
theorem car_time_difference :
  let v_x : ℝ := 35  -- speed of Car X in miles per hour
  let v_y : ℝ := 50  -- speed of Car Y in miles per hour
  let d : ℝ := 98    -- distance traveled by Car X after Car Y starts
  time_difference v_x v_y d = 50.4 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_time_difference_l1065_106560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_rooms_required_l1065_106546

def department_sizes : List Nat := [120, 240, 360, 480, 600]

def is_valid_room_size (room_size : Nat) : Bool :=
  department_sizes.all (fun size => size % room_size = 0)

def total_rooms (room_size : Nat) : Nat :=
  department_sizes.foldr (fun size acc => acc + size / room_size) 0

theorem minimum_rooms_required :
  ∃ (room_size : Nat),
    room_size > 0 ∧
    is_valid_room_size room_size ∧
    total_rooms room_size = 15 ∧
    ∀ (other_size : Nat),
      other_size > 0 →
      is_valid_room_size other_size →
      total_rooms other_size ≥ 15 :=
by
  sorry

#eval total_rooms 120

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_rooms_required_l1065_106546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1065_106501

theorem exponential_inequality (a b c : ℝ) :
  a = (2 : ℝ) ^ (3/10) ∧ b = (2 : ℝ) ^ (1/10) ∧ c = (1/5 : ℝ) ^ (13/10) → c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1065_106501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_in_all_classes_l1065_106513

/-- Represents the number of students in each class combination --/
structure ClassEnrollment where
  photography : ℕ
  music : ℕ
  theatre : ℕ
  dance : ℕ
  photography_music : ℕ
  photography_theatre : ℕ
  photography_dance : ℕ
  music_theatre : ℕ
  music_dance : ℕ
  theatre_dance : ℕ
  all_four : ℕ

/-- Theorem stating the number of students enrolled in all four classes --/
theorem students_in_all_classes (e : ClassEnrollment) : e.all_four = 4 :=
  by
    -- Define constants
    let total_students : ℕ := 30
    let students_in_two_or_more : ℕ := 18

    -- Define given conditions
    have photography_total : e.photography = 15 := by sorry
    have music_total : e.music = 18 := by sorry
    have theatre_total : e.theatre = 12 := by sorry
    have dance_total : e.dance = 10 := by sorry

    -- Define equations
    have sum_of_enrollments : 
      e.photography + e.music + e.theatre + e.dance - 
      (e.photography_music + e.photography_theatre + e.photography_dance + 
       e.music_theatre + e.music_dance + e.theatre_dance) + 
      2 * e.all_four = total_students := by sorry
    
    have sum_of_two_or_more : 
      e.photography_music + e.photography_theatre + e.photography_dance + 
      e.music_theatre + e.music_dance + e.theatre_dance + 
      3 * e.all_four = students_in_two_or_more := by sorry

    -- The actual proof would go here
    sorry

#check students_in_all_classes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_in_all_classes_l1065_106513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_starts_with_nines_l1065_106516

theorem square_starts_with_nines (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  a^2 ≥ 1 - 10^(-100 : ℤ) → a ≥ 1 - 5 * 10^(-101 : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_starts_with_nines_l1065_106516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1065_106515

def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + 1 ≤ 0}
def B : Set ℝ := {x : ℝ | x^2 - 3*x + 2 ≤ 0}

def sufficient_not_necessary (p q : Set ℝ) : Prop :=
  (p ⊆ q) ∧ ¬(q ⊆ p)

theorem range_of_a :
  ∀ a : ℝ, (sufficient_not_necessary (A a) B) ↔ (-2 ≤ a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1065_106515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_shift_for_even_function_l1065_106558

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sin (2 * x + Real.pi / 3)

theorem smallest_shift_for_even_function : 
  ∀ φ : ℝ, φ > 0 → 
  (∀ x : ℝ, f (x + φ) = f (-x + φ)) → 
  φ ≥ Real.pi / 6 :=
by
  sorry

#check smallest_shift_for_even_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_shift_for_even_function_l1065_106558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sequence_for_2015_l1065_106577

def is_valid_operation (op : ℕ → ℕ → ℕ) : Prop :=
  (op = (·+·)) ∨ (op = (·*·))

def is_valid_sequence (seq : List (ℕ × (ℕ → ℕ → ℕ))) : Prop :=
  ∀ n, n ∈ seq.map Prod.fst → n ≥ 1 ∧ n ≤ 9 ∧
  (∀ m, m ∈ seq.map Prod.fst → m = n → (seq.map Prod.fst).count m = 1) ∧
  (∀ op, op ∈ seq.map Prod.snd → is_valid_operation op)

def evaluate_sequence (seq : List (ℕ × (ℕ → ℕ → ℕ))) : ℕ :=
  seq.foldl (λ acc (n, op) => op acc n) 0

theorem exists_sequence_for_2015 :
  ∃ seq : List (ℕ × (ℕ → ℕ → ℕ)),
    is_valid_sequence seq ∧
    evaluate_sequence seq = 2015 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sequence_for_2015_l1065_106577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_fourth_rod_l1065_106590

def valid_quadrilateral (a b c d : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a ∧
  a + b > d ∧ a + d > b ∧ b + d > a ∧
  a + c > d ∧ a + d > c ∧ c + d > a ∧
  b + c > d ∧ b + d > c ∧ c + d > b

def rod_lengths : Finset ℕ := Finset.range 25 

theorem count_valid_fourth_rod :
  (rod_lengths.filter (λ x => x ∉ ({4, 9, 18} : Finset ℕ) ∧ x > 5 ∧ x < 31)).card = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_fourth_rod_l1065_106590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1065_106512

noncomputable def f (x : ℝ) : ℝ := (2*x - 3) / (x + 1)

theorem f_properties :
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 9 → f x ≤ 3/2) ∧
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 9 → 1/3 ≤ f x) ∧
  f 9 = 3/2 ∧
  f 2 = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1065_106512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_machine_90_l1065_106568

noncomputable def number_machine (x : ℝ) : ℝ :=
  (((x * 3 + 20) / 2)^2 - 45)

theorem number_machine_90 : number_machine 90 = 20980 := by
  -- Unfold the definition of number_machine
  unfold number_machine
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_machine_90_l1065_106568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l1065_106506

/-- The standard equation of a hyperbola given its asymptotes and a point -/
theorem hyperbola_standard_equation (C : Set (ℝ × ℝ)) 
  (h1 : ∀ (x y : ℝ), (x, y) ∈ C → (2*x = 3*y ∨ 2*x = -3*y))
  (h2 : (3 * Real.sqrt 2, 2) ∈ C) :
  ∃ (f : ℝ × ℝ → ℝ), f = (λ p ↦ p.1^2/9 - p.2^2/4 - 1) ∧ 
  ∀ (p : ℝ × ℝ), p ∈ C ↔ f p = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l1065_106506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_circle_l1065_106511

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the condition for point P
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  distance P A = Real.sqrt 2 * distance P B

-- Define the equation of the trajectory
def trajectory_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 12*x + 4 = 0

-- Theorem statement
theorem trajectory_is_circle :
  ∀ (P : ℝ × ℝ), satisfies_condition P →
  trajectory_equation P.1 P.2 ∧
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), x^2 + y^2 - 12*x + 4 = 0 ↔
      distance (x, y) center = radius) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_circle_l1065_106511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_abc_l1065_106541

-- Define the constants as noncomputable
noncomputable def a : ℝ := (1/4) * Real.log 3 / Real.log 2
noncomputable def b : ℝ := 1/2
noncomputable def c : ℝ := (1/2) * Real.log 3 / Real.log 5

-- State the theorem
theorem ordering_of_abc : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_abc_l1065_106541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_for_negative_x_l1065_106599

-- Define the custom operation
noncomputable def customOp (m n : ℝ) : ℝ := -n / m

-- Define the function y = x ⊕ 2
noncomputable def f (x : ℝ) : ℝ := customOp x 2

-- Theorem statement
theorem f_increasing_for_negative_x :
  ∀ x₁ x₂ : ℝ, x₁ < 0 → x₂ < 0 → x₁ < x₂ → f x₁ < f x₂ := by
  -- Introduce variables and assumptions
  intro x₁ x₂ h1 h2 h3
  -- Unfold the definition of f
  unfold f
  -- Unfold the definition of customOp
  unfold customOp
  -- Use real number properties to prove the inequality
  -- This part would require a more detailed proof, which we'll skip for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_for_negative_x_l1065_106599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_3_neg3_l1065_106505

/-- Conversion of rectangular coordinates (3, -3) to polar coordinates (r, θ) -/
theorem rect_to_polar_3_neg3 :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  r * Real.cos θ = 3 ∧ r * Real.sin θ = -3 ∧
  r = 3 * Real.sqrt 2 ∧ θ = 7 * π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_3_neg3_l1065_106505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_completion_l1065_106584

noncomputable def vertex1 : ℂ := Complex.mk 2 3
noncomputable def vertex2 : ℂ := Complex.mk (-3) 2
noncomputable def vertex3 : ℂ := Complex.mk (-2) (-3)
noncomputable def fourthVertex : ℂ := Complex.mk 2.5 0.5

def IsSquare (s : Set ℂ) : Prop := sorry

theorem square_completion (v1 v2 v3 v4 : ℂ) 
  (h1 : v1 = vertex1) 
  (h2 : v2 = vertex2) 
  (h3 : v3 = vertex3) 
  (h4 : v4 = fourthVertex) : 
  IsSquare {v1, v2, v3, v4} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_completion_l1065_106584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_tsp_l1065_106552

/-- A line with slope m passing through point (x₀, y₀) -/
def Line (m x₀ y₀ : ℝ) : ℝ → ℝ := λ x ↦ m * (x - x₀) + y₀

/-- The intersection point of two lines -/
noncomputable def Intersection (l₁ l₂ : ℝ → ℝ) : ℝ × ℝ := sorry

/-- The perpendicular line to a given line through a point -/
def PerpendicularLine (l : ℝ → ℝ) (x₀ y₀ : ℝ) : ℝ → ℝ := sorry

/-- The area of a triangle given three points -/
def TriangleArea (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ := sorry

theorem triangle_area_tsp :
  let l₁ := Line 3 2 7
  let l₂ := Line (-1) 2 7
  let S := (2, 7)
  let T := Intersection l₁ (λ x ↦ 0)  -- T is on y-axis
  let l₃ := PerpendicularLine l₁ T.1 T.2
  let P := Intersection l₂ l₃
  TriangleArea T.1 T.2 S.1 S.2 P.1 P.2 = 15 * Real.sqrt 17 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_tsp_l1065_106552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_l1065_106570

-- Define the triangle and points
structure RightTriangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  is_right_triangle : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0
  M_on_PQ : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ M = (t * Q.1 + (1 - t) * P.1, t * Q.2 + (1 - t) * P.2)
  N_on_PR : ∃ s : ℝ, 0 < s ∧ s < 1 ∧ N = (s * R.1 + (1 - s) * P.1, s * R.2 + (1 - s) * P.2)

-- Define the ratios
def ratio_condition (t : RightTriangle) : Prop :=
  let PM := Real.sqrt ((t.M.1 - t.P.1)^2 + (t.M.2 - t.P.2)^2)
  let MQ := Real.sqrt ((t.Q.1 - t.M.1)^2 + (t.Q.2 - t.M.2)^2)
  let PN := Real.sqrt ((t.N.1 - t.P.1)^2 + (t.N.2 - t.P.2)^2)
  let NR := Real.sqrt ((t.R.1 - t.N.1)^2 + (t.R.2 - t.N.2)^2)
  PM / MQ = 1 / 3 ∧ PN / NR = 1 / 3

-- Define the given lengths
def length_condition (t : RightTriangle) : Prop :=
  let QN := Real.sqrt ((t.N.1 - t.Q.1)^2 + (t.N.2 - t.Q.2)^2)
  let MR := Real.sqrt ((t.R.1 - t.M.1)^2 + (t.R.2 - t.M.2)^2)
  QN = 18 ∧ MR = 30

-- Theorem statement
theorem hypotenuse_length (t : RightTriangle) 
  (h1 : ratio_condition t) (h2 : length_condition t) : 
  Real.sqrt ((t.Q.1 - t.R.1)^2 + (t.Q.2 - t.R.2)^2) = 8 * Real.sqrt 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_l1065_106570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_when_p_range_of_a_when_p_and_q_l1065_106528

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 / Real.sqrt (a * x^2 - a * x + 1)

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x, f a x ∈ Set.univ

-- Define proposition q
def q (a : ℝ) : Prop := ∀ x > 0, 3^x - 9^x < a - 1

-- Theorem for question 1
theorem range_of_a_when_p (a : ℝ) : 
  p a → a ∈ Set.Icc 0 4 := by sorry

-- Theorem for question 2
theorem range_of_a_when_p_and_q (a : ℝ) : 
  p a ∧ q a → a ∈ Set.Ico 1 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_when_p_range_of_a_when_p_and_q_l1065_106528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_surjective_iff_a_in_range_l1065_106500

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > a then -x + 10 else x^2 + 2*x

-- State the theorem
theorem f_surjective_iff_a_in_range (a : ℝ) :
  Function.Surjective (f a) ↔ a ∈ Set.Icc (-5) 11 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_surjective_iff_a_in_range_l1065_106500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_sufficient_nor_necessary_l1065_106582

theorem not_sufficient_nor_necessary (m n : ℕ) :
  ¬(∀ a b : ℝ, a > b → a^(m+n) + b^(m+n) > a^n * b^m + a^m * b^n) ∧
  ¬(∀ a b : ℝ, a^(m+n) + b^(m+n) > a^n * b^m + a^m * b^n → a > b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_sufficient_nor_necessary_l1065_106582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rooks_on_chessboard_l1065_106576

/-- Represents a chessboard configuration -/
def ChessboardConfiguration := Fin 8 → Fin 8 → Bool

/-- Checks if a rook at position (x, y) attacks more than one other rook -/
def attacks_more_than_one (config : ChessboardConfiguration) (x y : Fin 8) : Bool :=
  let attacked := List.filter (fun i => 
    (i ≠ x ∧ config i y) ∨ (i ≠ y ∧ config x i)) (List.range 8)
  attacked.length > 1

/-- Counts the number of rooks on the board -/
def count_rooks (config : ChessboardConfiguration) : Nat :=
  List.sum (List.map (fun x => List.count (config x) (List.range 8)) (List.range 8))

/-- Checks if the configuration is valid (no rook attacks more than one other rook) -/
def is_valid_configuration (config : ChessboardConfiguration) : Prop :=
  ∀ x y, config x y → ¬(attacks_more_than_one config x y)

/-- The main theorem: The maximum number of rooks that can be placed on an 8x8 chessboard
    such that each rook attacks no more than one other rook is 10 -/
theorem max_rooks_on_chessboard :
  (∃ (config : ChessboardConfiguration), is_valid_configuration config ∧ count_rooks config = 10) ∧
  (∀ (config : ChessboardConfiguration), is_valid_configuration config → count_rooks config ≤ 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rooks_on_chessboard_l1065_106576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1065_106507

noncomputable def f (α : ℝ) : ℝ := Real.sin α * Real.cos α

theorem problem_1 (α : ℝ) (h1 : f α = 1/8) (h2 : π/4 < α) (h3 : α < π/2) :
  Real.cos α - Real.sin α = -Real.sqrt 3 / 2 := by sorry

theorem problem_2 : f (-31*π/3) = -Real.sqrt 3 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1065_106507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_head_start_l1065_106566

/-- A race between two runners with different speeds and a head start. -/
structure Race where
  length : ℕ
  speed_cristina : ℕ
  speed_nicky : ℕ
  catch_up_time : ℕ

/-- Calculate the head start time given to the slower runner. -/
def headStartTime (race : Race) : ℕ :=
  race.catch_up_time - (race.speed_nicky * race.catch_up_time) / race.speed_cristina

/-- The theorem stating the head start time for the given race conditions. -/
theorem race_head_start (race : Race) 
  (h1 : race.length = 200)
  (h2 : race.speed_cristina = 5)
  (h3 : race.speed_nicky = 3)
  (h4 : race.catch_up_time = 30) :
  headStartTime race = 12 := by
  sorry

#eval headStartTime { length := 200, speed_cristina := 5, speed_nicky := 3, catch_up_time := 30 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_head_start_l1065_106566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l1065_106567

/-- A line defined by y = kx + 1 -/
def line (k : ℝ) : ℝ → ℝ := λ x => k * x + 1

/-- A parabola defined by y² = 4x -/
noncomputable def parabola : ℝ → ℝ := λ x => Real.sqrt (4 * x)

/-- The number of intersection points between the line and the parabola -/
def intersectionCount (k : ℝ) : ℕ :=
  -- Definition omitted, as it would require knowledge from the solution
  sorry

theorem unique_intersection (k : ℝ) :
  intersectionCount k = 1 ↔ k = 0 ∨ k = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l1065_106567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_range_l1065_106535

-- Define the circle
def Circle (R : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - 2)^2 = R^2}

-- Define the line
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = Real.sqrt 3 * p.1 - 2}

-- Define the distance function from a point to the line
noncomputable def distToLine (p : ℝ × ℝ) : ℝ :=
  |Real.sqrt 3 * p.1 - p.2 - 2| / Real.sqrt 4

-- Define the condition for exactly two points at distance 1
def twoPointsAtDistance1 (R : ℝ) : Prop :=
  ∃! (p1 p2 : ℝ × ℝ), p1 ∈ Circle R ∧ p2 ∈ Circle R ∧
    p1 ≠ p2 ∧ distToLine p1 = 1 ∧ distToLine p2 = 1

-- The main theorem
theorem circle_line_distance_range :
  ∀ R > 0, twoPointsAtDistance1 R ↔ 1 < R ∧ R < 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_range_l1065_106535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1065_106542

open Real

theorem problem_statement :
  (¬ ∃ x : ℝ, x ∈ Set.Ioo 0 2 ∧ x^3 - x^2 - x + 2 < 0) ∧
  (¬ ∀ x y : ℝ, x + y > 4 → x > 2 ∧ y > 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1065_106542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_l1065_106539

/-- Fixed cost for producing electronic instruments -/
noncomputable def fixed_cost : ℝ := 20000

/-- Variable cost per unit -/
noncomputable def variable_cost : ℝ := 100

/-- Total revenue function -/
noncomputable def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 400 then 400 * x - (1/2) * x^2
  else 80000

/-- Profit function -/
noncomputable def f (x : ℝ) : ℝ :=
  R x - (fixed_cost + variable_cost * x)

/-- The production volume that maximizes profit -/
noncomputable def optimal_production : ℝ := 300

/-- The maximum profit -/
noncomputable def max_profit : ℝ := 25000

/-- Theorem stating that the profit is maximized at the optimal production volume -/
theorem profit_maximized :
  ∀ x, f x ≤ f optimal_production ∧
  f optimal_production = max_profit := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_l1065_106539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_terminating_decimal_l1065_106571

theorem smallest_terminating_decimal (n : ℕ+) : 
  (∀ m : ℕ+, m < n → ¬(∃ (a b : ℕ), (m : ℚ) / ((m : ℚ) + 150) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0 ∧ Nat.Coprime a b)) → 
  (∃ (a b : ℕ), (n : ℚ) / ((n : ℚ) + 150) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0 ∧ Nat.Coprime a b) → 
  n = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_terminating_decimal_l1065_106571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_exterior_possible_l1065_106574

-- Define the number of small cubes
def num_small_cubes : ℕ := 8

-- Define the fraction of blue faces on small cubes
def blue_fraction : ℚ := 1/3

-- Define the fraction of red faces visible on the large cube
def visible_red_fraction : ℚ := 1/3

-- Define the condition that small cubes form a larger cube
def forms_larger_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

-- Define a type for cube arrangements
structure CubeArrangement where
  -- Add necessary fields here
  mk ::

-- Define the property of having an all-red exterior
def exterior_all_red (arrangement : CubeArrangement) : Prop :=
  sorry -- Define the condition for all-red exterior

-- Theorem statement
theorem red_exterior_possible (h1 : forms_larger_cube num_small_cubes) 
  (h2 : blue_fraction + (1 - blue_fraction) = 1) 
  (h3 : visible_red_fraction < 1 - blue_fraction) : 
  ∃ arrangement : CubeArrangement, exterior_all_red arrangement :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_exterior_possible_l1065_106574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_prop4_is_simple_l1065_106521

-- Define the propositions
def prop1 : Prop := ∃ n : ℕ, n = 12 ∧ 4 ∣ n ∧ 3 ∣ n

-- We'll use a more general definition for prop2 without specific geometric types
def prop2 : Prop := ∃ (A B : Set ℝ) (r : ℝ), (∀ x ∈ A, ∃ y ∈ B, y = r * x) ∧ ¬(∀ x ∈ A, ∃ y ∈ B, x = y)

-- We'll use a more general definition for prop3 without specific geometric types
def prop3 : Prop := ∀ (a b c : ℝ), ∃ m : ℝ, m = (a + b) / 2 ∧ m - c = (a - c) / 2

-- We'll use a more general definition for prop4 without specific geometric types
def prop4 : Prop := ∀ a b : ℝ, a = b → a + b = 2 * a

-- Define what it means for a proposition to be simple
def IsSimpleProposition (p : Prop) : Prop := 
  (p ∨ ¬p) ∧ ¬(∃ q r : Prop, p = (q ∧ r) ∨ p = (q ∨ r))

-- State the theorem
theorem only_prop4_is_simple : 
  IsSimpleProposition prop4 ∧ 
  ¬IsSimpleProposition prop1 ∧ 
  ¬IsSimpleProposition prop2 ∧ 
  ¬IsSimpleProposition prop3 := by
  sorry

#check only_prop4_is_simple

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_prop4_is_simple_l1065_106521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_2_3_5_not_geometric_l1065_106550

theorem sqrt_2_3_5_not_geometric : ¬ ∃ r : ℝ, (Real.sqrt 2 * r = Real.sqrt 3) ∧ (Real.sqrt 3 * r = Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_2_3_5_not_geometric_l1065_106550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clean_room_to_homework_ratio_l1065_106531

def movie_time : ℕ := 120 -- 2 hours in minutes
def homework_time : ℕ := 30
def dog_walk_time : ℕ := homework_time + 5
def trash_time : ℕ := homework_time / 6
def time_left : ℕ := 35

theorem clean_room_to_homework_ratio :
  let total_task_time := movie_time - time_left
  let other_tasks_time := homework_time + dog_walk_time + trash_time
  let clean_room_time := total_task_time - other_tasks_time
  (clean_room_time : ℚ) / homework_time = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clean_room_to_homework_ratio_l1065_106531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_amount_after_bets_l1065_106557

/-- Represents the outcome of a single bet --/
inductive BetOutcome
| Win
| Loss
deriving BEq, Repr

/-- Calculates the final amount after a series of bets --/
def finalAmount (initialAmount : Int) (initialBet : Int) (outcomes : List BetOutcome) : Int :=
  outcomes.foldl
    (fun acc outcome =>
      let betAmount := initialBet * (2 ^ (outcomes.length - acc.2 - 1))
      match outcome with
      | BetOutcome.Win => (acc.1 + betAmount, acc.2 + 1)
      | BetOutcome.Loss => (acc.1 - betAmount, acc.2 + 1)
    )
    (initialAmount, 0)
  |>.1

/-- Theorem stating the final amount after 5 bets with 2 wins and 3 losses --/
theorem final_amount_after_bets (outcomes : List BetOutcome) :
  outcomes.length = 5 →
  outcomes.count BetOutcome.Win = 2 →
  outcomes.count BetOutcome.Loss = 3 →
  finalAmount 100 10 outcomes = -110 := by
  sorry

#eval finalAmount 100 10 [BetOutcome.Win, BetOutcome.Loss, BetOutcome.Win, BetOutcome.Loss, BetOutcome.Loss]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_amount_after_bets_l1065_106557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_points_theorem_l1065_106589

/-- Triangle with inscribed and ex-circles -/
structure TriangleWithCircles where
  /-- Side lengths of the triangle -/
  a : ℝ
  b : ℝ
  c : ℝ
  /-- Point where inscribed circle touches side BC -/
  k : ℝ
  /-- Point where ex-circle touches side BC -/
  l : ℝ

/-- Theorem about the relationship between points of tangency and side lengths -/
theorem tangency_points_theorem (t : TriangleWithCircles) : 
  t.k = t.l ∧ t.k = (t.a + t.b - t.c) / 2 := by
  sorry

#check tangency_points_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_points_theorem_l1065_106589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_is_negative_two_l1065_106581

/-- A geometric sequence with four terms -/
structure GeometricSequence :=
  (a : Fin 4 → ℚ)

/-- The common ratio of a geometric sequence -/
def commonRatio (seq : GeometricSequence) : ℚ :=
  seq.a 1 / seq.a 0

/-- Our specific geometric sequence -/
def ourSequence : GeometricSequence :=
  ⟨λ i => match i with
    | 0 => 10
    | 1 => -20
    | 2 => 40
    | 3 => -80⟩

theorem common_ratio_is_negative_two :
  commonRatio ourSequence = -2 := by
  -- Unfold definitions
  unfold commonRatio
  unfold ourSequence
  -- Simplify
  simp
  -- The result should now be obvious
  rfl

#eval commonRatio ourSequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_is_negative_two_l1065_106581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_theorem_l1065_106530

/-- Represents a parabola y² = 2px with p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a point on the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Focus of a parabola -/
noncomputable def focus (c : Parabola) : Point :=
  ⟨c.p / 2, 0⟩

/-- Check if a point lies on a parabola -/
def on_parabola (p : Point) (c : Parabola) : Prop :=
  p.y^2 = 2 * c.p * p.x

/-- Represents a line ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

theorem parabola_and_line_theorem (c : Parabola) (m : ℝ) :
  on_parabola ⟨3, m⟩ c →
  distance ⟨3, m⟩ (focus c) = 5 →
  (∃ (l : Line), l.a = 4 ∧ l.b = 1 ∧ l.c = -8 ∧
    (∃ (A B : Point), 
      on_parabola A c ∧ on_parabola B c ∧
      A.y + B.y = -2 ∧
      l.a * (focus c).x + l.b * (focus c).y + l.c = 0 ∧
      l.a * A.x + l.b * A.y + l.c = 0 ∧
      l.a * B.x + l.b * B.y + l.c = 0)) →
  c.p = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_theorem_l1065_106530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_factor_change_l1065_106553

/-- Given a function q defined in terms of w, m, x, and z, 
    prove that after applying specific changes to these variables, 
    the new value of q is (4/z) times the original value. -/
theorem q_factor_change (w m x z : ℝ) (hw : w ≠ 0) (hm : m ≠ 0) (hx : x ≠ 0) (hz : z ≠ 0) :
  let q := 7 * w / (6 * m * x * z^3)
  let q_new := 7 * (4*w) / (6 * (2*m) * (x/2) * z^4)
  q_new = (4/z) * q := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_factor_change_l1065_106553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n2o_molecular_weight_l1065_106517

/-- The molecular weight of a compound in grams per mole. -/
def molecular_weight (compound : Type) : ℝ := sorry

/-- The number of moles of a compound. -/
def moles (compound : Type) : ℝ := sorry

/-- The total weight of a given number of moles of a compound in grams. -/
def total_weight (compound : Type) (m : ℝ) : ℝ :=
  m * molecular_weight compound

/-- N2O (nitrous oxide) compound -/
def N2O : Type := sorry

theorem n2o_molecular_weight :
  molecular_weight N2O = 44 :=
by
  have h : total_weight N2O 8 = 352 := by sorry
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n2o_molecular_weight_l1065_106517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_difference_l1065_106509

theorem binomial_difference : (Nat.choose 16 4) - (Nat.choose 16 3) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_difference_l1065_106509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_less_than_sum_of_sins_l1065_106563

theorem sin_sum_less_than_sum_of_sins (α β : ℝ) :
  0 < α ∧ α < π/2 → 0 < β ∧ β < π/2 → Real.sin α + Real.sin β > Real.sin (α + β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_less_than_sum_of_sins_l1065_106563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1065_106526

/-- The function f(x) defined on the interval (0,1] --/
noncomputable def f (x : ℝ) : ℝ := x * (1 - x) / ((x + 1) * (x + 2) * (2 * x + 1))

/-- The theorem stating the maximum value of f(x) on (0,1] --/
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 1 ∧
  ∀ (y : ℝ), y ∈ Set.Ioo 0 1 → f y ≤ f x ∧
  f x = (1/3) * (8 * Real.sqrt 2 - 5 * Real.sqrt 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1065_106526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_numbers_l1065_106540

theorem order_of_numbers : ∃ (a b c : ℝ), 
  (a = (7 : ℝ)^(0.3 : ℝ)) ∧ (b = (0.3 : ℝ)^(7 : ℝ)) ∧ (c = Real.log 0.3) ∧ (a > c) ∧ (c > b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_numbers_l1065_106540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l1065_106532

def a : ℝ := 0.76

theorem expression_value : 
  abs ((a * a * a - 0.008) / (a * a + a * 0.2 + 0.04) - 0.560) < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l1065_106532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_theorem_l1065_106579

/-- The area of a regular octagon inscribed in a square -/
noncomputable def octagonArea (squarePerimeter : ℝ) : ℝ :=
  let squareSide := squarePerimeter / 4
  let segmentLength := squareSide / 4
  let innerSquareSide := 2 * segmentLength
  let largeSquareArea := squareSide ^ 2
  let smallSquareArea := innerSquareSide ^ 2
  let triangleArea := segmentLength ^ 2 / 2
  largeSquareArea - 4 * triangleArea

theorem octagon_area_theorem (squarePerimeter : ℝ) :
  squarePerimeter = 216 →
  octagonArea squarePerimeter = 2551.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_theorem_l1065_106579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l1065_106529

-- Define the curves C₁ and C₂
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (Real.cos α, 1 + Real.sin α)
noncomputable def C₂ (α : ℝ) : ℝ × ℝ := (3 * Real.cos α, 3 + 3 * Real.sin α)

-- Define the line y = (√3/3)x
noncomputable def line (x : ℝ) : ℝ := (Real.sqrt 3 / 3) * x

-- Define the polar coordinates of A and B
def ρ_A : ℝ := 1
def ρ_B : ℝ := 3

theorem distance_AB : 
  ∀ (α : ℝ), 
  let A := C₁ α
  let B := C₂ α
  (A.2 = line A.1 ∧ B.2 = line B.1) → |ρ_B - ρ_A| = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l1065_106529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l1065_106565

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (60 / 360 : ℝ) * (2 * Real.pi * r₁) = (30 / 360 : ℝ) * (2 * Real.pi * r₂) →
  (Real.pi * r₁^2) / (Real.pi * r₂^2) = 1 / 4 := by
  intro h_arc_equality
  -- Proof steps would go here
  sorry

#check circle_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l1065_106565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1065_106587

noncomputable def f (x : ℝ) := Real.sin x ^ 2 - Real.cos x ^ 2 - 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (f (2 * Real.pi / 3) = 2) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ Real.pi) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6),
    ∀ y ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6),
    x ≤ y → f x ≥ f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1065_106587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_with_replacement_count_l1065_106594

structure SamplingMethod where
  name : String
  withReplacement : Bool

def samplingWithReplacement (method : SamplingMethod) : Bool :=
  method.withReplacement

def stratifiedSampling : SamplingMethod :=
  { name := "Stratified Sampling", withReplacement := false }

def systematicSampling : SamplingMethod :=
  { name := "Systematic Sampling", withReplacement := false }

def simpleRandomSampling : SamplingMethod :=
  { name := "Simple Random Sampling", withReplacement := false }

def samplingMethods : List SamplingMethod :=
  [stratifiedSampling, systematicSampling, simpleRandomSampling]

theorem sampling_with_replacement_count :
  (samplingMethods.filter samplingWithReplacement).length = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_with_replacement_count_l1065_106594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_for_odd_g_l1065_106598

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (2 * x) - Real.sin (2 * x)

noncomputable def g (t x : ℝ) : ℝ := f (x + t)

def is_odd (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

theorem min_t_for_odd_g :
  ∃ t : ℝ, t > 0 ∧ is_odd (g t) ∧ ∀ s, s > 0 ∧ is_odd (g s) → t ≤ s := by
  sorry

#check min_t_for_odd_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_for_odd_g_l1065_106598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_point_to_line_l1065_106562

/-- The distance from a point in polar coordinates to a line in polar form --/
noncomputable def distance_point_to_line (ρ : ℝ) (θ : ℝ) (α : ℝ) (c : ℝ) : ℝ :=
  let x := ρ * Real.cos θ
  let y := ρ * Real.sin θ
  let a := Real.cos α
  let b := Real.sin α
  |a * x + b * y - c| / Real.sqrt (a^2 + b^2)

/-- The theorem stating the distance from the given point to the line --/
theorem distance_specific_point_to_line :
  distance_point_to_line 2 (11 * Real.pi / 6) (Real.pi / 6) 1 = Real.sqrt 3 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_point_to_line_l1065_106562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_omega_value_l1065_106580

variable (ω k : ℝ)
variable (f : ℝ → ℝ)

theorem possible_omega_value :
  ∃ ω > 0, (∀ x, f x = Real.sin (ω * x - π / 6) + k) ∧
  (∀ x, f x ≤ f (π / 3)) ∧
  ω = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_omega_value_l1065_106580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_maximum_value_minimum_value_l1065_106544

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 4

-- Define the interval
def interval : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

-- Theorem for the tangent line at x = 1
theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ 3*x + y - 10/3 = 0 := by
  sorry

-- Theorem for the maximum value
theorem maximum_value :
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 4 := by
  sorry

-- Theorem for the minimum value
theorem minimum_value :
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≥ f x ∧ f x = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_maximum_value_minimum_value_l1065_106544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_first_inequality_solution_set_second_inequality_l1065_106597

-- First inequality
theorem solution_set_first_inequality :
  {x : ℝ | (2 : ℝ)^(3*x - 1) < 2} = {x : ℝ | x < 2/3} := by sorry

-- Second inequality
theorem solution_set_second_inequality (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  {x : ℝ | a^(3*x^2 + 3*x - 1) < a^(3*x^2 + 3)} = 
    if a > 1 then {x : ℝ | x < 4/3}
    else {x : ℝ | x > 4/3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_first_inequality_solution_set_second_inequality_l1065_106597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1065_106510

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the medians
noncomputable def median (t : Triangle) (vertex : ℝ × ℝ) (opposite : ℝ × ℝ) : ℝ × ℝ := 
  ((vertex.1 + opposite.1) / 2, (vertex.2 + opposite.2) / 2)

-- Define the area function
noncomputable def area (t : Triangle) : ℝ := sorry

-- Define the theorem
theorem triangle_area (ABC : Triangle) 
  (h1 : Real.sqrt ((median ABC ABC.A (median ABC ABC.B ABC.C)).1 - (median ABC ABC.B ABC.C).1)^2 + 
            ((median ABC ABC.A (median ABC ABC.B ABC.C)).2 - (median ABC ABC.B ABC.C).2)^2 = 18)
  (h2 : Real.sqrt ((median ABC ABC.B (median ABC ABC.C ABC.A)).1 - (median ABC ABC.C ABC.A).1)^2 + 
            ((median ABC ABC.B (median ABC ABC.C ABC.A)).2 - (median ABC ABC.C ABC.A).2)^2 = 24)
  (h3 : Real.arccos (((median ABC ABC.A (median ABC ABC.B ABC.C)).1 - (median ABC ABC.B ABC.C).1) * 
                   ((median ABC ABC.B (median ABC ABC.C ABC.A)).1 - (median ABC ABC.C ABC.A).1) +
                   ((median ABC ABC.A (median ABC ABC.B ABC.C)).2 - (median ABC ABC.B ABC.C).2) * 
                   ((median ABC ABC.B (median ABC ABC.C ABC.A)).2 - (median ABC ABC.C ABC.A).2)) /
        (18 * 24) = π/4) :
  area ABC = 144 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1065_106510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_not_common_is_zero_l1065_106583

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ

/-- Represents the configuration of two overlapping 30-60-90 triangles -/
structure OverlappingTriangles where
  triangle1 : Triangle30_60_90
  triangle2 : Triangle30_60_90
  overlap : ℝ

/-- Calculates the area of the region not common to one of the triangles -/
noncomputable def areaNotCommon (ot : OverlappingTriangles) : ℝ := 
  sorry

/-- Theorem stating that the area not common to one of the triangles is 0 -/
theorem area_not_common_is_zero (ot : OverlappingTriangles) 
  (h1 : ot.triangle1.hypotenuse = 10)
  (h2 : ot.triangle2.hypotenuse = 10)
  (h3 : ot.overlap = 8) : 
  areaNotCommon ot = 0 := by
  sorry

#check area_not_common_is_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_not_common_is_zero_l1065_106583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_radii_different_curvatures_exists_equal_curvature_arcs_l1065_106533

noncomputable def curvature (R : ℝ) : ℝ := 1 / R

-- Theorem 1: Curvatures of circles with different radii are not equal
theorem different_radii_different_curvatures (R₁ R₂ : ℝ) (h : R₁ ≠ R₂) (h₁ : R₁ > 0) (h₂ : R₂ > 0) :
  curvature R₁ ≠ curvature R₂ := by sorry

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a hypothetical curvature_at function
noncomputable def curvature_at (p : Point2D) (arc : Set Point2D) : ℝ := sorry

-- Theorem 2: Existence of equal curvature arcs connecting two points
theorem exists_equal_curvature_arcs (A B : Point2D) (h : A ≠ B) :
  ∃ (κ : ℝ), ∃ (arc1 arc2 : Set Point2D),
    (A ∈ arc1 ∧ B ∈ arc1) ∧
    (A ∈ arc2 ∧ B ∈ arc2) ∧
    (∀ p ∈ arc1, curvature_at p arc1 = κ) ∧
    (∀ p ∈ arc2, curvature_at p arc2 = κ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_radii_different_curvatures_exists_equal_curvature_arcs_l1065_106533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1065_106523

/-- Calculates the speed of a train given its length, tunnel length, and time to cross the tunnel. -/
noncomputable def train_speed (train_length : ℝ) (tunnel_length : ℝ) (time_minutes : ℝ) : ℝ :=
  let total_distance := train_length + tunnel_length
  let time_seconds := time_minutes * 60
  let speed_mps := total_distance / time_seconds
  speed_mps * 3.6

/-- Theorem stating that a train of length 800 meters crossing a tunnel of length 500 meters
    in 1 minute has a speed of approximately 78.01 km/hr. -/
theorem train_speed_calculation :
  ∀ (ε : ℝ), ε > 0 →
  ∃ (speed : ℝ), abs (speed - train_speed 800 500 1) < ε ∧ abs (speed - 78.01) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1065_106523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_representation_theorem_l1065_106591

/-- Representation of an integer as a sum of powers of 2 with coefficients -1, 0, or 1 -/
def IntegerRepresentation (A : ℤ) : Prop :=
  ∃ (n : ℕ) (a : ℕ → ℤ),
    (A = (Finset.range n).sum (λ k ↦ 2^k * a k)) ∧
    (∀ k, a k ∈ ({-1, 0, 1} : Set ℤ)) ∧
    (∀ k, k < n → a k * a (k + 1) = 0)

/-- The uniqueness of the representation -/
def UniqueRepresentation (A : ℤ) : Prop :=
  ∀ (n m : ℕ) (a b : ℕ → ℤ),
    (A = (Finset.range n).sum (λ k ↦ 2^k * a k)) ∧
    (A = (Finset.range m).sum (λ k ↦ 2^k * b k)) ∧
    (∀ k, a k ∈ ({-1, 0, 1} : Set ℤ)) ∧
    (∀ k, b k ∈ ({-1, 0, 1} : Set ℤ)) ∧
    (∀ k, k < n → a k * a (k + 1) = 0) ∧
    (∀ k, k < m → b k * b (k + 1) = 0) →
    n = m ∧ (∀ k, a k = b k)

theorem integer_representation_theorem (A : ℤ) :
  IntegerRepresentation A ∧ UniqueRepresentation A :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_representation_theorem_l1065_106591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_perpendicular_distance_product_l1065_106549

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 3*y^2 = 1

-- Define perpendicularity
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem ellipse_perpendicular_distance_product :
  ∃ (max min : ℝ),
    (∀ (x₁ y₁ x₂ y₂ : ℝ),
      ellipse x₁ y₁ → ellipse x₂ y₂ → perpendicular x₁ y₁ x₂ y₂ →
      distance x₁ y₁ x₂ y₂ ≤ max ∧ distance x₁ y₁ x₂ y₂ ≥ min) ∧
    max * min = 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_perpendicular_distance_product_l1065_106549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_combinations_count_l1065_106525

def Card : Type := ℕ × ℕ

def cards : List Card := [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

def is_valid_number (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def can_form_number (c1 c2 c3 : Card) (n : ℕ) : Prop :=
  ∃ d1 d2 d3, (d1 = c1.1 ∨ d1 = c1.2) ∧ 
              (d2 = c2.1 ∨ d2 = c2.2) ∧ 
              (d3 = c3.1 ∨ d3 = c3.2) ∧
              n = 100 * d1 + 10 * d2 + d3

def count_valid_numbers : ℕ := sorry

theorem card_combinations_count :
  count_valid_numbers = 432 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_combinations_count_l1065_106525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cube_volume_l1065_106534

/-- Volume of a truncated cube with edge length a -/
noncomputable def volume_of_truncated_cube (a : ℝ) : ℝ :=
  let x := a * (2 - Real.sqrt 2) / 2
  a^3 - 8 * (1/6) * x^3

/-- A cube is truncated if planes cut its vertices forming regular octagons on each face -/
def is_truncated_cube (a : ℝ) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ x < a/2 ∧ 2*x + x*Real.sqrt 2 = a

/-- The volume of a truncated cube with regular octagonal faces -/
theorem truncated_cube_volume (a : ℝ) (a_pos : a > 0) :
  ∃ (V : ℝ), V = (7/3) * a^3 * (Real.sqrt 2 - 1) ∧
  V = volume_of_truncated_cube a :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cube_volume_l1065_106534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_values_l1065_106537

/-- The function that we're analyzing -/
noncomputable def y (a x : ℝ) : ℝ := a^(2*x) + 2*a^x - 1

/-- The theorem stating the conditions and the result -/
theorem max_value_implies_a_values (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, y a x ≤ 14) ∧ 
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, y a x = 14) →
  a = 1/3 ∨ a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_values_l1065_106537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_square_rotation_intersection_area_l1065_106543

/-- The area of intersection between a unit square and its 30° rotation around a vertex -/
theorem unit_square_rotation_intersection_area : 
  ∃ (A B C D B' C' D' : ℝ × ℝ) (area_of_intersection : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ),
    -- Define the original square ABCD
    A = (0, 0) ∧ 
    B = (1, 0) ∧ 
    C = (1, 1) ∧ 
    D = (0, 1) ∧
    -- Define the rotated square AB'C'D'
    B' = (Real.cos (π/6) - Real.sin (π/6), Real.sin (π/6) + Real.cos (π/6)) ∧
    C' = (-Real.sin (π/6), Real.cos (π/6)) ∧
    D' = (-Real.sin (π/6) - 1, Real.cos (π/6)) ∧
    -- The area of intersection is 1 - √3/3
    area_of_intersection A B C D B' C' D' = 1 - Real.sqrt 3 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_square_rotation_intersection_area_l1065_106543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_juice_concentration_l1065_106502

/-- Calculates the final grape juice concentration after adding pure grape juice to a mixture --/
theorem grape_juice_concentration
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_volume : ℝ)
  (initial_volume_positive : 0 < initial_volume)
  (initial_concentration_percent : 0 ≤ initial_concentration ∧ initial_concentration ≤ 1)
  (added_volume_positive : 0 < added_volume) :
  let initial_grape_juice := initial_volume * initial_concentration
  let total_grape_juice := initial_grape_juice + added_volume
  let final_volume := initial_volume + added_volume
  let final_concentration := total_grape_juice / final_volume
  (initial_volume = 30 ∧ initial_concentration = 0.1 ∧ added_volume = 10) →
  final_concentration = 0.325 := by
  intro h
  sorry

#check grape_juice_concentration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_juice_concentration_l1065_106502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_of_0_3989_l1065_106592

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The original number to be rounded -/
def original : ℝ := 0.3989

/-- Theorem stating that rounding 0.3989 to the nearest hundredth equals 0.40 -/
theorem round_to_hundredth_of_0_3989 :
  roundToHundredth original = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_of_0_3989_l1065_106592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_surface_area_to_circumradius_squared_ratio_l1065_106569

/-- A right tetrahedron with perpendicular edges -/
structure RightTetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The surface area of a right tetrahedron -/
noncomputable def surfaceArea (t : RightTetrahedron) : ℝ :=
  2 * (t.a * t.b + t.b * t.c + t.c * t.a)

/-- The radius of the circumscribed sphere of a right tetrahedron -/
noncomputable def circumradius (t : RightTetrahedron) : ℝ :=
  Real.sqrt (t.a^2 + t.b^2 + t.c^2) / 2

/-- The theorem stating the maximum value of S/R^2 for a right tetrahedron -/
theorem max_surface_area_to_circumradius_squared_ratio (t : RightTetrahedron) :
    (surfaceArea t / circumradius t ^ 2) ≤ (2 / 3) * (3 + Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_surface_area_to_circumradius_squared_ratio_l1065_106569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_judgments_correctness_l1065_106596

-- Define the propositions
def proposition1 : Prop := ∀ a b : ℝ, a + b ≠ 6 → a ≠ 3 ∨ b ≠ 3

def proposition2 : Prop := ∀ p q : Prop, (p ∨ q) → (p ∧ q)

def proposition3 : Prop := ∀ A : ℝ, 
  (A > 30 * Real.pi / 180 → Real.sin A > 1/2) ∧ 
  ¬(Real.sin A > 1/2 → A > 30 * Real.pi / 180)

noncomputable def proposition4 : Prop := ∀ θ : ℝ,
  (∃ k : ℝ, k ≠ 0 ∧ k * Real.sin (2*θ) = k * Real.cos θ ∧ k * Real.cos θ = k * 1) →
  Real.tan θ = 1/2

theorem judgments_correctness : 
  proposition1 ∧ 
  ¬proposition2 ∧ 
  ¬proposition3 ∧ 
  proposition4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_judgments_correctness_l1065_106596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_diameter_proof_l1065_106559

/-- The diameter of a circle circumscribing seven tangent circles of radius 2 -/
noncomputable def largeDiameter : ℝ := 4 / Real.sin (Real.pi / 7) + 4

/-- The configuration of seven small circles and one large circle -/
structure CircleConfiguration where
  smallRadius : ℝ
  smallCircleCount : ℕ
  smallCirclesTouchCount : ℕ

/-- The specific configuration described in the problem -/
def problemConfig : CircleConfiguration :=
  { smallRadius := 2
  , smallCircleCount := 7
  , smallCirclesTouchCount := 2 }

/-- Theorem stating that the diameter of the large circle in the given configuration
    is equal to the calculated largeDiameter -/
theorem large_circle_diameter_proof (config : CircleConfiguration)
  (h1 : config = problemConfig)
  (h2 : config.smallCircleCount = 7)
  (h3 : config.smallRadius = 2)
  (h4 : config.smallCirclesTouchCount = 2) :
  ∃ (d : ℝ), d = largeDiameter ∧ d = 4 / Real.sin (Real.pi / 7) + 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_diameter_proof_l1065_106559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_valuation_l1065_106551

theorem company_valuation (price kw_price a_assets b_assets : ℝ) 
  (h1 : kw_price = 2 * b_assets)
  (h2 : kw_price = 0.8571428571428571 * (a_assets + b_assets)) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0000001 ∧ 
  (kw_price / a_assets - 1) ∈ Set.Icc (0.6666666666667 - ε) (0.6666666666667 + ε) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_valuation_l1065_106551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_problem_l1065_106548

theorem exponent_problem (m n : ℝ) (h1 : (3 : ℝ)^m = 2) (h2 : (3 : ℝ)^n = 5) :
  ((3 : ℝ)^(m-n) = 2/5) ∧ ((9 : ℝ)^m * (27 : ℝ)^n = 500) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_problem_l1065_106548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_ratio_l1065_106524

noncomputable section

-- Define the parabola P
def P : Set (ℝ × ℝ) := {(x, y) | y = (1/2) * x^2}

-- Define the vertex and focus of P
def V₁ : ℝ × ℝ := (0, 0)
def F₁ : ℝ × ℝ := (0, 1/8)

-- Define points C and D on P
def C (c : ℝ) : ℝ × ℝ := (c, (1/2) * c^2)
def D (d : ℝ) : ℝ × ℝ := (d, (1/2) * d^2)

-- Define the condition for the right angle
def right_angle (c d : ℝ) : Prop := c * d = -2

-- Define the midpoint of CD
def midpoint_CD (c d : ℝ) : ℝ × ℝ := ((c + d) / 2, ((c + d)^2) / 8 - 1/2)

-- Define the parabola R (locus of midpoints)
def R : Set (ℝ × ℝ) := {(x, y) | y = (1/2) * x^2 - 1/2}

-- Define the vertex and focus of R
def V₃ : ℝ × ℝ := (0, -1/2)
def F₃ : ℝ × ℝ := (0, -1/4)

-- The theorem to be proved
theorem parabola_focus_ratio : 
  (F₁.2 - F₃.2) / (V₁.2 - V₃.2) = 3/4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_ratio_l1065_106524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_edges_value_l1065_106595

/-- A rectangular solid with specific properties -/
structure RectangularSolid where
  -- Dimensions of the solid
  a : ℝ
  r : ℝ
  -- Volume of the solid is 432 cm³
  volume_eq : a * a * a = 432
  -- Total surface area is 360 cm²
  surface_area_eq : 2 * (a^2/r + a^2 + a^2*r) = 360
  -- Dimensions form a geometric progression
  geometric_progression : r > 0

/-- The sum of the lengths of all edges of the rectangular solid -/
noncomputable def sum_of_edges (solid : RectangularSolid) : ℝ :=
  4 * (solid.a/solid.r + solid.a + solid.a*solid.r)

/-- Theorem stating the sum of edges for the specific rectangular solid -/
theorem sum_of_edges_value (solid : RectangularSolid) :
  sum_of_edges solid = 72 * Real.rpow 2 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_edges_value_l1065_106595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tire_circumference_144kmh_400rpm_l1065_106514

/-- Given a car's speed in km/h and tire rotation in revolutions per minute,
    calculates the circumference of the tire in meters. -/
noncomputable def tire_circumference (car_speed : ℝ) (tire_rpm : ℝ) : ℝ :=
  (car_speed * 1000 / 60) / tire_rpm

/-- Theorem stating that for a car traveling at 144 km/h with tires rotating
    at 400 revolutions per minute, the circumference of the tire is 6 meters. -/
theorem tire_circumference_144kmh_400rpm :
  tire_circumference 144 400 = 6 := by
  -- Unfold the definition of tire_circumference
  unfold tire_circumference
  -- Simplify the arithmetic expression
  simp [div_eq_mul_inv]
  -- The proof is completed by normalization of real number arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tire_circumference_144kmh_400rpm_l1065_106514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marias_flower_purchase_cost_l1065_106555

/-- Calculates the total cost of Maria's purchase at April's discount flowers --/
theorem marias_flower_purchase_cost : 
  (6 : ℕ) * 7 + (4 : ℕ) * 3 + (10 : ℕ) * 2 + (8 : ℕ) * 1 = 82 :=
by
  -- Evaluate the left-hand side of the equation
  calc
    (6 : ℕ) * 7 + (4 : ℕ) * 3 + (10 : ℕ) * 2 + (8 : ℕ) * 1
    = 42 + 12 + 20 + 8 := by ring
    _ = 82 := by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marias_flower_purchase_cost_l1065_106555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_yz_plane_intersection_l1065_106578

/-- Given a line passing through points (1,2,3) and (4,5,-2), 
    prove that (0, 1, 20/3) is the intersection point with the yz-plane -/
theorem line_yz_plane_intersection :
  let p1 : Fin 3 → ℝ := ![1, 2, 3]
  let p2 : Fin 3 → ℝ := ![4, 5, -2]
  let direction : Fin 3 → ℝ := ![p2 0 - p1 0, p2 1 - p1 1, p2 2 - p1 2]
  let t : ℝ := -1/3
  let intersection : Fin 3 → ℝ := ![p1 0 + t * direction 0, 
                                    p1 1 + t * direction 1, 
                                    p1 2 + t * direction 2]
  (intersection 0 = 0 ∧ 
   intersection 1 = 1 ∧ 
   intersection 2 = 20/3) ∧ 
  intersection 0 = 0 -- Condition for being on the yz-plane
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_yz_plane_intersection_l1065_106578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1065_106593

theorem simplify_expression : (5 : Int) - 3 - (-1) + (-5) = 5 - 3 + 1 - 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1065_106593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_parameters_l1065_106572

/-- Represents a circle in the 2D plane -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The equation of the circle -/
def Circle.equation (circle : Circle) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*circle.a*x - circle.b*y + circle.c = 0

/-- The center of the circle -/
noncomputable def Circle.center (circle : Circle) : ℝ × ℝ :=
  (-circle.a, circle.b/2)

/-- The radius of the circle -/
noncomputable def Circle.radius (circle : Circle) : ℝ :=
  Real.sqrt ((circle.b^2)/4 + circle.a^2 - circle.c)

/-- Theorem: For a circle with center (2, 2) and radius 2, 
    the values of a, b, and c are -2, 4, and 4 respectively -/
theorem circle_parameters : 
  ∃ (circle : Circle), 
    circle.center = (2, 2) ∧ 
    circle.radius = 2 ∧ 
    circle.a = -2 ∧ 
    circle.b = 4 ∧ 
    circle.c = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_parameters_l1065_106572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_object_speed_theorem_l1065_106554

/-- Calculates the speed in miles per hour given distance in feet and time in seconds -/
noncomputable def speed_mph (distance_feet : ℝ) (time_seconds : ℝ) : ℝ :=
  (distance_feet / 5280) / (time_seconds / 3600)

/-- Theorem stating that an object traveling 70 feet in 2 seconds has a speed of approximately 23.856 mph -/
theorem object_speed_theorem :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |speed_mph 70 2 - 23.856| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_object_speed_theorem_l1065_106554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_sum_l1065_106575

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Definition of an ellipse -/
structure Ellipse where
  f1 : Point
  f2 : Point
  constant : ℝ

/-- Theorem: For an ellipse with given properties, h + k + a + b = 14 -/
theorem ellipse_property_sum (e : Ellipse) 
  (h1 : e.f1 = ⟨0, 2⟩)
  (h2 : e.f2 = ⟨6, 2⟩)
  (h3 : e.constant = 10)
  : ∃ (h k a b : ℝ), 
    (∀ (p : Point), distance p e.f1 + distance p e.f2 = e.constant → 
      (p.x - h)^2 / a^2 + (p.y - k)^2 / b^2 = 1) ∧
    h + k + a + b = 14 := by
  sorry

#check ellipse_property_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_sum_l1065_106575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1065_106556

theorem exponential_inequality (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (1/2 : ℝ)^x - (1/2 : ℝ)^y < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1065_106556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l1065_106545

/-- Represents a parabola with parameter p -/
structure Parabola where
  p : ℝ

/-- The distance from a point on the parabola to its focus -/
noncomputable def distanceToFocus (para : Parabola) (x : ℝ) : ℝ :=
  x + para.p / 2

theorem parabola_focus_distance (para : Parabola) :
  distanceToFocus para 4 = 5 → para.p = 2 := by
  intro h
  unfold distanceToFocus at h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l1065_106545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l1065_106520

/-- Represents the tank filling problem -/
structure TankProblem where
  capacity : ℚ  -- Tank capacity in liters
  initial_fill : ℚ  -- Initial fill level in liters
  fill_rate : ℚ  -- Fill rate in liters per minute
  drain_rate1 : ℚ  -- Drain 1 rate in liters per minute
  drain_rate2 : ℚ  -- Drain 2 rate in liters per minute

/-- Calculates the time to fill the tank completely -/
def time_to_fill (p : TankProblem) : ℚ :=
  let remaining_volume := p.capacity - p.initial_fill
  let net_flow_rate := p.fill_rate - (p.drain_rate1 + p.drain_rate2)
  remaining_volume / net_flow_rate

/-- Theorem stating that the time to fill the tank is 12 minutes -/
theorem tank_fill_time (p : TankProblem) 
    (h1 : p.capacity = 2000)
    (h2 : p.initial_fill = 1000)
    (h3 : p.fill_rate = 500)
    (h4 : p.drain_rate1 = 250)
    (h5 : p.drain_rate2 = 1000 / 6) :
  time_to_fill p = 12 := by
  sorry

def main : IO Unit := do
  let result := time_to_fill {
    capacity := 2000,
    initial_fill := 1000,
    fill_rate := 500,
    drain_rate1 := 250,
    drain_rate2 := 1000 / 6
  }
  IO.println s!"Time to fill the tank: {result} minutes"

#eval main

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l1065_106520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_line_equation_circle_equation_l1065_106586

noncomputable section

-- Define the points
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (0, -2)
def C : ℝ × ℝ := (-2, 3)
def P : ℝ × ℝ := (0, -6)
def Q : ℝ × ℝ := (1, -5)

-- Define the midpoint M of AB
def M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the line l: x - y + 1 = 0
def l (x y : ℝ) : Prop := x - y + 1 = 0

-- Theorem for the equation of median line CM
theorem median_line_equation : 
  ∀ x y : ℝ, (y - M.2) / (C.2 - M.2) = (x - M.1) / (C.1 - M.1) ↔ 2*x + 3*y - 5 = 0 :=
by sorry

-- Theorem for the standard equation of the circle
theorem circle_equation :
  ∃ E : ℝ × ℝ, l E.1 E.2 ∧ 
  (∀ x y : ℝ, (x - E.1)^2 + (y - E.2)^2 = (P.1 - E.1)^2 + (P.2 - E.2)^2 ↔ 
               (x + 3)^2 + (y + 2)^2 = 25) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_line_equation_circle_equation_l1065_106586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1065_106527

theorem trig_identity (α β : Real) :
  Real.sin α ^ 2 + Real.sin β ^ 2 - Real.sin α ^ 2 * Real.sin β ^ 2 + Real.cos α ^ 2 * Real.cos β ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1065_106527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_pde_solution_is_zero_l1065_106588

/-- A function h: ℝ² → ℝ with continuous partial derivatives satisfying a specific PDE and boundedness condition is identically zero. -/
theorem bounded_pde_solution_is_zero (h : ℝ × ℝ → ℝ) (a b : ℝ) (M : ℝ) 
  (h_cont : ContinuousOn h (Set.univ : Set (ℝ × ℝ))) 
  (h_pde : ∀ (x y : ℝ), h (x, y) = a * (deriv (fun p => h (p, y)) x) + b * (deriv (fun p => h (x, p)) y)) 
  (h_bound : ∀ (x y : ℝ), |h (x, y)| ≤ M) :
  ∀ (x y : ℝ), h (x, y) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_pde_solution_is_zero_l1065_106588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sarahs_house_number_l1065_106538

theorem sarahs_house_number :
  ∃! n : ℕ,
    10 ≤ n ∧ n < 100 ∧
    n % 5 = 0 ∧
    n % 2 = 1 ∧
    n % 3 = 0 ∧
    (¬(n / 10 = 7 ∨ n % 10 = 7)) ∧
    n % 10 = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sarahs_house_number_l1065_106538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_decreasing_f_l1065_106573

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

-- State the theorem
theorem a_range_for_decreasing_f (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  0 < a ∧ a ≤ 1/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_decreasing_f_l1065_106573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1065_106522

/-- Calculates the speed of a train in km/h given its length and time to pass a fixed point. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem stating that a train of length 90 meters passing a point in 7.363047319850775 seconds
    has a speed of approximately 43.9628 km/h. -/
theorem train_speed_calculation :
  let length : ℝ := 90
  let time : ℝ := 7.363047319850775
  abs (train_speed length time - 43.9628) < 0.0001 := by
  sorry

-- Use #eval only for computable functions
def approx_train_speed : Float := (90 / 7.363047319850775) * 3.6

#eval approx_train_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1065_106522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_numbers_greater_than_average_l1065_106518

theorem max_numbers_greater_than_average (n : Nat) (avg : ℚ) (h_n : n = 30) (h_avg : avg = 4) :
  let total_sum := n * avg
  let min_greater := n / 2
  (∃ (count : Nat), count ≤ n ∧ count * avg + (n - count) * avg = total_sum ∧
    count ≥ min_greater ∧
    ∀ (greater_count : Nat), greater_count ≤ n →
      greater_count * avg + (n - greater_count) * avg > total_sum →
        greater_count ≤ count) →
  n / 2 = 15 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_numbers_greater_than_average_l1065_106518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flash_arrow_race_l1065_106536

/-- The distance Flash must cover to catch up to Arrow in a race -/
theorem flash_arrow_race (z u k w : ℝ) (hz : z > 1) (hu : u > 0) (hk : k ≥ 0) (hw : w ≥ 0) :
  (z * (k * u + w)) / (z - 1) = 
    let arrow_speed := u
    let flash_speed := z * u
    let initial_lead := w
    let maintained_lead_time := k
    let total_lead := k * u + w
    let catch_up_time := total_lead / (flash_speed - arrow_speed)
    flash_speed * catch_up_time :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flash_arrow_race_l1065_106536
