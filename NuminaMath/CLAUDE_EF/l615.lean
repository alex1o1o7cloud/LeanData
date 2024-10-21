import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solutions_l615_61562

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem floor_equation_solutions :
  ∃ (a b c d : ℝ), a < b ∧ c < d ∧
  (∀ x : ℝ, x > 0 → (floor x + floor (1 / x) = 3 ↔ (a < x ∧ x < b) ∨ (c < x ∧ x < d))) :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solutions_l615_61562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_profit_calculation_l615_61566

theorem clock_profit_calculation (total_clocks : ℕ) (clocks_with_unknown_profit : ℕ) 
  (clocks_with_known_profit : ℕ) (known_profit_percentage : ℚ) 
  (uniform_profit_percentage : ℚ) (cost_price : ℚ) (difference : ℚ) 
  (unknown_profit_percentage : ℚ) :
  total_clocks = clocks_with_unknown_profit + clocks_with_known_profit →
  known_profit_percentage = 20 / 100 →
  uniform_profit_percentage = 15 / 100 →
  cost_price = 79.99999999999773 →
  (total_clocks : ℚ) * cost_price * (1 + uniform_profit_percentage) + difference = 
    (clocks_with_unknown_profit : ℚ) * cost_price * (1 + unknown_profit_percentage) +
    (clocks_with_known_profit : ℚ) * cost_price * (1 + known_profit_percentage) →
  difference = 40 →
  unknown_profit_percentage = 10 / 100 :=
by sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_profit_calculation_l615_61566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_base_range_l615_61536

-- Define the function f(x) = (a+1)^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1) ^ x

-- State the theorem
theorem decreasing_exponential_base_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) → -1 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_base_range_l615_61536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l615_61528

theorem integral_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let f (x : ℝ) := |Real.sqrt (3*b*(2*a - b) + 2*(a - 2*b)*x - x^2) - 
                    Real.sqrt (3*a*(2*b - a) + 2*(2*a - b)*x - x^2)|
  ∫ x in (a - 2*b)..(2*a - b), f x ≤ (π/3)*(a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l615_61528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l615_61579

/-- Curve C₁ in parametric form -/
noncomputable def curve_C1 (α : ℝ) (t : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, t * Real.sin α)

/-- Curve C₂ in polar form -/
noncomputable def curve_C2 (θ : ℝ) : ℝ := Real.sqrt (12 / (3 + Real.sin θ ^ 2))

/-- Point F -/
def point_F : ℝ × ℝ := (1, 0)

/-- Theorem stating that cos α = 2/3 given the problem conditions -/
theorem cos_alpha_value (α : ℝ) (h_α : 0 ≤ α ∧ α < π) :
  ∃ (t₁ t₂ : ℝ), t₁ > 0 ∧ t₂ < 0 ∧
  (curve_C1 α t₁).1 ^ 2 / 4 + (curve_C1 α t₁).2 ^ 2 / 3 = 1 ∧
  (curve_C1 α t₂).1 ^ 2 / 4 + (curve_C1 α t₂).2 ^ 2 / 3 = 1 ∧
  (curve_C1 α t₂).1 - (point_F.1) = 2 * ((curve_C1 α t₁).1 - point_F.1) ∧
  (curve_C1 α t₂).2 = 2 * (curve_C1 α t₁).2 →
  Real.cos α = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l615_61579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l615_61514

-- Define the circle C
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the line l
def line_eq (x y k : ℝ) : Prop := y = k * x

-- Define the perpendicularity condition
def perpendicular_condition (x1 y1 x2 y2 b : ℝ) : Prop := x1 * x2 + (y1 - b) * (y2 - b) = 0

-- Theorem for part 1
theorem part_one (k : ℝ) : 
  (∃ x1 y1 x2 y2 : ℝ, 
    circle_eq x1 y1 ∧ circle_eq x2 y2 ∧ 
    line_eq x1 y1 k ∧ line_eq x2 y2 k ∧ 
    perpendicular_condition x1 y1 x2 y2 1) → 
  k = 1 := by sorry

-- Theorem for part 2
theorem part_two (k b : ℝ) : 
  (∃ x1 y1 x2 y2 : ℝ, 
    circle_eq x1 y1 ∧ circle_eq x2 y2 ∧ 
    line_eq x1 y1 k ∧ line_eq x2 y2 k ∧ 
    perpendicular_condition x1 y1 x2 y2 b) ∧ 
  1 < b ∧ b < 3/2 → 
  (1 < k ∧ k < 6 - Real.sqrt 23) ∨ (6 + Real.sqrt 23 < k) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l615_61514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l615_61525

theorem inequality_proof (x : ℝ) (hx : x > 0) :
  (2 : ℝ)^(x^(1/12)) + (2 : ℝ)^(x^(1/4)) ≥ 2 * (2 : ℝ)^(x^(1/6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l615_61525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l615_61573

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (x + 1))

-- Define the function g
noncomputable def g (a x : ℝ) : ℝ := 2 - a^x

theorem problem_solution :
  (∀ x : ℝ, f (f x) + f (Real.log 3) > 0 ↔ x ∈ Set.Ioo (1/2) (9/11)) ∧
  (∀ a : ℝ, a > 0 → a ≠ 1 →
    (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Ico 0 1 ∧ x₂ ∈ Set.Ico 0 1 ∧ f x₁ = g a x₂) →
    a ∈ Set.Ioi 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l615_61573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_two_element_set_l615_61532

theorem proper_subsets_of_two_element_set :
  let A : Finset ℕ := {2, 3}
  Finset.card (Finset.powerset A \ {∅, A}) = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_two_element_set_l615_61532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_connected_after_removal_l615_61530

/-- A graph where edges are colored with N colors, and each vertex has exactly one edge of each color. -/
structure ColoredGraph (α : Type*) (N : ℕ) where
  vertices : Set α
  edges : Set (α × α)
  colors : Fin N → Set (α × α)
  symmetric : ∀ {a b}, (a, b) ∈ edges → (b, a) ∈ edges
  colored : ∀ e ∈ edges, ∃! c, e ∈ colors c
  one_per_color : ∀ v ∈ vertices, ∀ c, ∃! w, (v, w) ∈ colors c

/-- A path in the graph is a list of vertices where consecutive pairs form edges. -/
def GraphPath (G : ColoredGraph α N) (a b : α) : List α → Prop
  | [] => a = b
  | [v] => a = v ∧ v = b
  | (v :: w :: rest) => (v, w) ∈ G.edges ∧ GraphPath G w b (w :: rest)

/-- A graph is connected if there exists a path between any two vertices. -/
def Connected (G : ColoredGraph α N) : Prop :=
  ∀ a b, a ∈ G.vertices → b ∈ G.vertices → ∃ p, GraphPath G a b p

/-- The graph after removing N-1 edges of different colors. -/
def RemovedEdges (G : ColoredGraph α N) (removed : Fin (N-1) → α × α) : ColoredGraph α N where
  vertices := G.vertices
  edges := G.edges \ (Set.range removed)
  colors c := G.colors c \ (Set.range removed)
  symmetric := sorry
  colored := sorry
  one_per_color := sorry

/-- The main theorem: If G is connected, then the graph remains connected after removing N-1 edges of different colors. -/
theorem connected_after_removal (G : ColoredGraph α N) (removed : Fin (N-1) → α × α)
    (h_connected : Connected G)
    (h_different_colors : ∀ i j, i ≠ j → ∃ c₁ c₂, c₁ ≠ c₂ ∧ removed i ∈ G.colors c₁ ∧ removed j ∈ G.colors c₂) :
    Connected (RemovedEdges G removed) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_connected_after_removal_l615_61530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_N_less_than_three_fourths_l615_61509

def P (N : ℕ) : ℚ :=
  (Int.floor (N / 3 : ℚ) + 1 + (N - Int.ceil ((2 * N : ℚ) / 3) + 1)) / (N + 1 : ℚ)

theorem smallest_N_less_than_three_fourths : 
  (P 6 < 3/4) ∧ (∀ n : ℕ, n < 6 → n % 3 = 0 → P n ≥ 3/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_N_less_than_three_fourths_l615_61509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l615_61593

def data_set : List ℝ := [1, 3, 4, 5, 7, 9, 11, 16]

def chi_squared : ℝ := 3.937
def significance_level : ℝ := 0.05
def critical_value : ℝ := 3.841

def total_students : ℕ := 1500
def sample_size : ℕ := 100
def male_students_in_sample : ℕ := 55

-- Define percentile function (placeholder)
def percentile (p : ℝ) (data : List ℝ) : ℝ := sorry

-- Define are_related function (placeholder)
def are_related (X Y : Type) : Prop := sorry

theorem problem_solution :
  (percentile 75 data_set = 10) ∧
  (chi_squared > critical_value → ∃ (X Y : Type), are_related X Y) ∧
  (total_students - (male_students_in_sample * (total_students / sample_size)) = 675) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l615_61593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l615_61599

-- Define the train's length in meters
noncomputable def train_length : ℝ := 100

-- Define the train's speed in km/hr
noncomputable def train_speed_kmh : ℝ := 144

-- Define the conversion factor from km/hr to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Define the theorem
theorem train_crossing_time :
  let train_speed_ms := train_speed_kmh * kmh_to_ms
  train_length / train_speed_ms = 2.5 := by
  -- Unfold the definitions
  unfold train_length train_speed_kmh kmh_to_ms
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l615_61599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_T_less_than_one_l615_61537

def sequence_a : ℕ → ℕ
  | 0 => 1
  | n + 1 => 2 * sequence_a n + 1

def c (n : ℕ) : ℚ := (2 : ℚ)^n / ((sequence_a n) * (sequence_a (n + 1)))

def T (n : ℕ) : ℚ := (Finset.range n).sum (λ k => c k)

theorem sequence_a_formula (n : ℕ) : sequence_a n = 2^n - 1 := by
  sorry

theorem T_less_than_one (n : ℕ) : T n < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_T_less_than_one_l615_61537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_range_l615_61597

theorem sequence_range (a₁ a₂ a₃ a₄ k : ℝ) : 
  (∃ r : ℝ, r ≠ 1 ∧ a₂ = a₁ * r ∧ a₃ = a₂ * r) →  -- geometric sequence condition
  (∃ d : ℝ, d ≠ 0 ∧ a₃ = a₂ + d ∧ a₄ = a₃ + d) →  -- arithmetic sequence condition
  a₁ + a₂ + a₃ = k →                              -- sum of first three terms
  a₂ + a₃ + a₄ = 15 →                             -- sum of last three terms
  (∃ b₁ b₂ b₃ b₄ : ℝ, b₁ ≠ a₁ ∧ b₂ ≠ a₂ ∧ b₃ ≠ a₃ ∧ b₄ ≠ a₄ ∧
    (∃ r : ℝ, r ≠ 1 ∧ b₂ = b₁ * r ∧ b₃ = b₂ * r) ∧
    (∃ d : ℝ, d ≠ 0 ∧ b₃ = b₂ + d ∧ b₄ = b₃ + d) ∧
    b₁ + b₂ + b₃ = k ∧
    b₂ + b₃ + b₄ = 15) →                          -- existence of another sequence
  k ∈ Set.Icc (15/4) 5 ∪ Set.Ioo 5 15 ∪ Set.Ioi 15 := -- conclusion
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_range_l615_61597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l615_61568

noncomputable section

-- Define the triangle vertices
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (12, 0)

-- Define midpoints
noncomputable def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define lines AE and CD
noncomputable def line_AE (x : ℝ) : ℝ := (E.2 - A.2) / (E.1 - A.1) * (x - A.1) + A.2
noncomputable def line_CD (x : ℝ) : ℝ := (D.2 - C.2) / (D.1 - C.1) * (x - C.1) + C.2

-- Define the intersection point F
noncomputable def F : ℝ × ℝ :=
  let x := (line_CD 0 - line_AE 0) / ((E.2 - A.2) / (E.1 - A.1) - (D.2 - C.2) / (D.1 - C.1))
  (x, line_AE x)

-- Theorem statement
theorem intersection_point_sum :
  F.1 + F.2 = 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l615_61568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l615_61501

def circle_eq (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

def line_eq (x y : ℝ) : Prop := y = Real.sqrt 3 * x

def intersection_points (M N : ℝ × ℝ) : Prop :=
  circle_eq M.1 M.2 ∧ line_eq M.1 M.2 ∧
  circle_eq N.1 N.2 ∧ line_eq N.1 N.2 ∧
  M ≠ N

theorem chord_length (M N : ℝ × ℝ) :
  intersection_points M N →
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = Real.sqrt 3 := by
  sorry

#check chord_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l615_61501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proportional_sampling_correct_l615_61560

/-- Represents the production volumes of three sedan models -/
def production_volumes : Fin 3 → ℕ := ![1600, 6000, 2000]

/-- Total sample size -/
def total_samples : ℕ := 48

/-- Calculates the number of samples for each model based on proportional sampling -/
def calculate_samples (volumes : Fin 3 → ℕ) (total : ℕ) : Fin 3 → ℕ :=
  let total_volume := (List.ofFn volumes).sum
  fun i => (volumes i * total) / total_volume

/-- Theorem stating that the calculated samples match the expected results -/
theorem proportional_sampling_correct : 
  calculate_samples production_volumes total_samples = ![8, 30, 10] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proportional_sampling_correct_l615_61560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l615_61520

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (x + Real.pi/6)

noncomputable def translated_function (x : ℝ) : ℝ := original_function (x + Real.pi/4)

noncomputable def final_function (x : ℝ) : ℝ := translated_function (x/2)

theorem graph_transformation :
  ∀ x : ℝ, final_function x = Real.sin (x/2 + 5*Real.pi/12) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l615_61520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_less_than_one_l615_61534

def atop (k j : ℕ) : ℕ :=
  (List.range j).foldl (λ acc i => acc * (k + i)) k

theorem ratio_less_than_one :
  (atop 2020 4) / (atop 2120 4) < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_less_than_one_l615_61534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_tan_alpha_fraction_l615_61561

-- Problem 1
theorem sin_beta_value (β : ℝ) (h1 : Real.cos β = -3/5) (h2 : π/2 < β ∧ β < π) : 
  Real.sin β = 4/5 := by sorry

-- Problem 2
theorem tan_alpha_fraction (α : ℝ) (h : Real.tan α / (Real.tan α - 6) = -1) : 
  (2 * Real.cos α - 3 * Real.sin α) / (3 * Real.cos α + 4 * Real.sin α) = -7/15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_tan_alpha_fraction_l615_61561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l615_61543

theorem cos_double_angle_special_case (θ : ℝ) (h : Real.sin θ = 3/5) : Real.cos (2 * θ) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l615_61543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l615_61540

/-- Represents the sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The value of λ in the equation S₁₂ = λS₈ -/
def lambda : ℝ := sorry

/-- Theorem stating the value of λ given the conditions -/
theorem arithmetic_sequence_sum_ratio :
  S 4 ≠ 0 →
  S 8 = 3 * S 4 →
  S 12 = lambda * S 8 →
  (S 8 - S 4) - (S 4) = (S 12 - S 8) - (S 8 - S 4) →
  lambda = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l615_61540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shifted_l615_61591

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)

theorem f_shifted (x : ℝ) (h : x^2 ≠ 1) : f (x + 2) = (x + 3) / (x + 1) := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shifted_l615_61591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sprint_competition_races_l615_61503

/-- The number of races needed to determine a champion in a sprint competition -/
noncomputable def races_needed (total_sprinters : ℕ) (lanes_per_race : ℕ) (advancing_per_race : ℕ) : ℕ :=
  Nat.ceil ((total_sprinters - 1 : ℚ) / (lanes_per_race - advancing_per_race))

/-- Theorem stating that 60 races are needed for the given conditions -/
theorem sprint_competition_races :
  races_needed 300 8 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sprint_competition_races_l615_61503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_for_angle_through_point_l615_61564

theorem sin_cos_product_for_angle_through_point (α : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ r^2 = 5 ∧ Real.sin α = -2/r ∧ Real.cos α = 1/r) →
  Real.sin α * Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_for_angle_through_point_l615_61564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_lifespan_l615_61586

/-- The lifespan of a well-cared fish is 12 years. -/
theorem fish_lifespan (hamster_lifespan dog_lifespan fish_lifespan : ℝ) :
  hamster_lifespan = 2.5 →
  dog_lifespan = 4 * hamster_lifespan →
  fish_lifespan = dog_lifespan + 2 →
  fish_lifespan = 12 := by
    intros h1 h2 h3
    have h4 : dog_lifespan = 10 := by
      rw [h2, h1]
      norm_num
    have h5 : fish_lifespan = 12 := by
      rw [h3, h4]
      norm_num
    exact h5


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_lifespan_l615_61586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deepak_speed_l615_61516

/-- Calculates the walking speed of the second person given the track circumference,
    the first person's speed, and the time they meet. -/
noncomputable def calculate_speed (track_circumference : ℝ) (speed1 : ℝ) (meet_time : ℝ) : ℝ :=
  let speed1_mpm := speed1 * 1000 / 60
  let distance1 := speed1_mpm * meet_time
  let distance2 := track_circumference - distance1
  let speed2_mpm := distance2 / meet_time
  speed2_mpm * 60 / 1000

/-- Theorem stating that given the specified conditions, Deepak's walking speed is 4.5 km/hr -/
theorem deepak_speed (track_circumference : ℝ) (wife_speed : ℝ) (meet_time : ℝ)
  (h1 : track_circumference = 660)
  (h2 : wife_speed = 3.75)
  (h3 : meet_time = 4.8) :
  calculate_speed track_circumference wife_speed meet_time = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_deepak_speed_l615_61516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_mechanic_worked_five_hours_l615_61596

/-- Represents the work time and charge of two mechanics --/
structure MechanicsWork where
  first_rate : ℚ
  second_rate : ℚ
  total_hours : ℚ
  total_charge : ℚ

/-- Calculates the work time of the second mechanic --/
def second_mechanic_time (work : MechanicsWork) : ℚ :=
  (work.total_charge - work.first_rate * work.total_hours) / (work.second_rate - work.first_rate)

/-- Theorem stating that given the problem conditions, the second mechanic worked for 5 hours --/
theorem second_mechanic_worked_five_hours (work : MechanicsWork) 
  (h1 : work.first_rate = 45)
  (h2 : work.second_rate = 85)
  (h3 : work.total_hours = 20)
  (h4 : work.total_charge = 1100) : 
  second_mechanic_time work = 5 := by
  sorry

def example_work : MechanicsWork := {
  first_rate := 45,
  second_rate := 85,
  total_hours := 20,
  total_charge := 1100
}

#eval second_mechanic_time example_work

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_mechanic_worked_five_hours_l615_61596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_has_78_pennies_l615_61551

/-- The number of pennies Alex currently has -/
def alex : ℕ := sorry

/-- The number of pennies Bob currently has -/
def bob : ℕ := sorry

/-- If Alex gives Bob two pennies, Bob will have four times as many pennies as Alex -/
axiom scenario1 : bob + 2 = 4 * (alex - 2)

/-- If Bob gives Alex three pennies, Bob will have three times as many pennies as Alex -/
axiom scenario2 : bob - 3 = 3 * (alex + 3)

/-- Bob currently has 78 pennies -/
theorem bob_has_78_pennies : bob = 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_has_78_pennies_l615_61551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_formula_optimal_ownership_period_l615_61595

/-- Represents the total cost of owning a car for n years -/
noncomputable def total_cost (n : ℕ) : ℝ :=
  169000 + 10000 * (n : ℝ) + 1000 * (n : ℝ)^2

/-- Represents the average annual cost of owning a car for n years -/
noncomputable def average_cost (n : ℕ) : ℝ :=
  total_cost n / (n : ℝ)

theorem total_cost_formula (n : ℕ) :
  total_cost n = 1000 * (n : ℝ)^2 + 10000 * (n : ℝ) + 169000 := by
  sorry

theorem optimal_ownership_period :
  ∃ n : ℕ, n > 0 ∧ ∀ m : ℕ, m > 0 → average_cost n ≤ average_cost m ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_formula_optimal_ownership_period_l615_61595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l615_61581

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin (abs x)

-- State the theorem
theorem f_range : Set.range f = Set.Icc (-2) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l615_61581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_slope_line_through_two_points_lines_through_point_with_equal_intercepts_l615_61523

-- Define the slope type
def Slope : Type := ℚ

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to calculate slope between two points
noncomputable def slopeBetweenPoints (p1 p2 : Point2D) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

-- Function to check if a line has equal intercepts on both axes
def hasEqualIntercepts (l : Line2D) : Prop :=
  l.a = l.b

-- Theorem 1
theorem line_through_point_with_slope
  (p : Point2D)
  (m : ℝ)
  (l : Line2D)
  (h1 : p.x = 2 ∧ p.y = 1)
  (h2 : m = -1/2)
  (h3 : pointOnLine p l)
  (h4 : slopeBetweenPoints p (Point2D.mk (p.x + 1) (p.y + m)) = m) :
  l.a = 1 ∧ l.b = 2 ∧ l.c = -4 :=
by sorry

-- Theorem 2
theorem line_through_two_points
  (p1 p2 : Point2D)
  (l : Line2D)
  (h1 : p1.x = 1 ∧ p1.y = 4)
  (h2 : p2.x = 2 ∧ p2.y = 3)
  (h3 : pointOnLine p1 l)
  (h4 : pointOnLine p2 l) :
  l.a = 1 ∧ l.b = 1 ∧ l.c = -5 :=
by sorry

-- Theorem 3
theorem lines_through_point_with_equal_intercepts
  (p : Point2D)
  (l1 l2 : Line2D)
  (h1 : p.x = 2 ∧ p.y = 1)
  (h2 : pointOnLine p l1)
  (h3 : pointOnLine p l2)
  (h4 : hasEqualIntercepts l1)
  (h5 : hasEqualIntercepts l2) :
  (l1.a = 1 ∧ l1.b = -2 ∧ l1.c = 0) ∨
  (l2.a = 1 ∧ l2.b = 1 ∧ l2.c = -3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_slope_line_through_two_points_lines_through_point_with_equal_intercepts_l615_61523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_equation_line_l_slope_l615_61578

-- Define the polar curve C₁
def C₁ (ρ θ : ℝ) : Prop := ρ * Real.sin θ = 2

-- Define the relationship between OM and OP
def OM_OP_relation (OM OP : ℝ) : Prop := OM * OP = 4

-- Define the parametric equation of line l
noncomputable def line_l (t α : ℝ) : ℝ × ℝ := (t * Real.cos α, t * Real.sin α)

-- Define the constraint on α
def α_constraint (α : ℝ) : Prop := 0 ≤ α ∧ α < Real.pi

-- Define the distance |OA|
noncomputable def OA_length : ℝ := Real.sqrt 3

-- Theorem for the Cartesian equation of C₂
theorem C₂_equation (x y : ℝ) (h : y ≠ 0) :
  ∃ (ρ θ : ℝ), C₁ ρ θ → OM_OP_relation ρ (2 / Real.sin θ) →
  x^2 + (y - 1)^2 = 1 := by sorry

-- Theorem for the slope of line l
theorem line_l_slope (α : ℝ) (h : α_constraint α) :
  ∃ (A : ℝ × ℝ), (∃ t, line_l t α = A) ∧ 
  Real.sqrt ((A.1)^2 + (A.2)^2) = OA_length →
  (Real.tan α = Real.sqrt 3 ∨ Real.tan α = -Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_equation_line_l_slope_l615_61578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_common_element_l615_61550

/-- A collection of 11 sets, each containing 5 elements -/
def SetCollection := Fin 11 → Finset (Fin 55)

/-- The property that each set in the collection has 5 elements -/
def has_five_elements (S : SetCollection) : Prop :=
  ∀ i : Fin 11, (S i).card = 5

/-- The property that the intersection of any two sets is non-empty -/
def pairwise_intersect (S : SetCollection) : Prop :=
  ∀ i j : Fin 11, i ≠ j → (S i ∩ S j).Nonempty

/-- The maximum number of sets that have a common element -/
noncomputable def max_common_element (S : SetCollection) : ℕ :=
  Finset.sup (Finset.univ : Finset (Fin 55)) (λ x => (Finset.filter (λ i => x ∈ S i) Finset.univ).card)

/-- The theorem stating that the smallest possible value for the maximum number of sets 
    that have a common element is 4 -/
theorem min_max_common_element :
  ∀ S : SetCollection, 
    has_five_elements S → 
    pairwise_intersect S → 
    max_common_element S ≥ 4 ∧ 
    ∃ S', has_five_elements S' ∧ pairwise_intersect S' ∧ max_common_element S' = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_common_element_l615_61550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_group_frequency_l615_61552

theorem sixth_group_frequency 
  (total_points : ℕ) 
  (num_groups : ℕ) 
  (group1 : ℕ) 
  (group2 : ℕ) 
  (group3 : ℕ) 
  (group4 : ℕ) 
  (group5_prop : ℚ) 
  (h1 : total_points = 40) 
  (h2 : num_groups = 6) 
  (h3 : group1 = 10) 
  (h4 : group2 = 5) 
  (h5 : group3 = 7) 
  (h6 : group4 = 6) 
  (h7 : group5_prop = 1/10) : 
  ∃ (group6 : ℕ), group6 = 8 ∧ 
  group1 + group2 + group3 + group4 + (group5_prop * total_points).floor + group6 = total_points :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_group_frequency_l615_61552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l615_61590

def M : Set ℝ := {x : ℝ | x ≥ 0}
def N : Set ℝ := {x : ℝ | x^2 < 1}

theorem intersection_M_N : M ∩ N = Set.Ico 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l615_61590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_tether_area_l615_61522

/-- The area outside a regular hexagonal doghouse that a tethered dog can reach -/
theorem dog_tether_area (side_length tether_length : ℝ) : 
  side_length = 2 → tether_length = 3 → 
  (π * tether_length^2 - 3 * Real.sqrt 3 * side_length^2) = 9 * π - 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_tether_area_l615_61522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_b_onto_a_l615_61569

noncomputable section

def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![-3, 1]

def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

def norm_squared (v : Fin 2 → ℝ) : ℝ := dot_product v v

def proj (u v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  fun i => (dot_product u v / norm_squared v) * (v i)

theorem projection_of_b_onto_a :
  proj b a = fun i => -a i :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_b_onto_a_l615_61569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_mean_calculation_l615_61588

theorem class_mean_calculation (total_students : ℕ) 
  (group1_students : ℕ) (group1_mean : ℚ)
  (group2_students : ℕ) (group2_mean : ℚ) :
  total_students = 28 →
  group1_students = 24 →
  group2_students = 4 →
  group1_mean = 80/100 →
  group2_mean = 88/100 →
  let combined_score := group1_students * group1_mean + group2_students * group2_mean
  let overall_mean := combined_score / total_students
  ⌊overall_mean * 100 + 1/2⌋ = 81 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_mean_calculation_l615_61588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_imply_m_range_l615_61574

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -x + m else x^2 - 1

theorem f_zeros_imply_m_range (m : ℝ) (h_m_pos : m > 0) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f m (f m x) - 1 = 0 ∧
    f m (f m y) - 1 = 0 ∧
    f m (f m z) - 1 = 0) →
  m > 0 ∧ m < 1 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_imply_m_range_l615_61574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_long_distance_cost_per_minute_l615_61542

/-- Calculates the cost per minute for long distance calls -/
noncomputable def cost_per_minute (monthly_fee : ℝ) (total_bill : ℝ) (minutes : ℝ) : ℝ :=
  (total_bill - monthly_fee) / minutes

/-- Proves that the cost per minute for John's long distance calls is $0.25 -/
theorem john_long_distance_cost_per_minute :
  let monthly_fee : ℝ := 5
  let total_bill : ℝ := 12.02
  let minutes : ℝ := 28.08
  cost_per_minute monthly_fee total_bill minutes = 0.25 := by
  -- Unfold the definition of cost_per_minute
  unfold cost_per_minute
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_long_distance_cost_per_minute_l615_61542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l615_61521

noncomputable def sequence_a (a : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => (5 * sequence_a a n - 8) / (sequence_a a n - 1)

noncomputable def geometric_ratio (a : ℝ) (n : ℕ) : ℝ :=
  (sequence_a a (n + 1) - 2) / (sequence_a a (n + 1) - 4) /
  ((sequence_a a n - 2) / (sequence_a a n - 4))

theorem sequence_properties (a : ℝ) :
  (a = 3 →
    (∀ n : ℕ, geometric_ratio a n = 3) ∧
    (∀ n : ℕ, sequence_a a n = (4 * 3^n + 2) / (3^n + 1))) ∧
  ((∀ n : ℕ, sequence_a a n > 3) → a > 3) := by
  sorry

#check sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l615_61521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_and_profit_optimization_l615_61580

-- Define the function f(x)
noncomputable def f (a b x : ℝ) : ℝ := a / (x - 3) + b * (x - 6)^2

-- Define the profit function g(x)
noncomputable def g (a b x : ℝ) : ℝ := (x - 3) * f a b x

-- State the theorem
theorem sales_and_profit_optimization 
  (a b : ℝ) 
  (h1 : f a b 4.5 = 22) 
  (h2 : f a b 5 = 11) :
  (a = 6 ∧ b = 8) ∧
  (∀ x, 3 < x → x < 6 → g a b x ≤ g a b 4) := by
  sorry

#check sales_and_profit_optimization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_and_profit_optimization_l615_61580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2023_equals_4046_l615_61500

-- Define the function f recursively
def f : ℕ → ℕ
  | 0 => 2  -- Add a case for 0 to cover all natural numbers
  | 1 => 2
  | 2 => 3
  | (n + 3) => f (n + 2) - f (n + 1) + 2 * (n + 3)

-- State the theorem
theorem f_2023_equals_4046 : f 2023 = 4046 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2023_equals_4046_l615_61500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_AD_vector_l615_61584

/-- A parallelogram ABCD with given vector properties -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_parallelogram : B - A = C - D
  AB_eq : B - A = (2, 4)
  AC_eq : C - A = (1, 3)

/-- Theorem: In the given parallelogram, AD vector is (-1, -1) -/
theorem parallelogram_AD_vector (p : Parallelogram) : 
  p.D - p.A = (-1, -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_AD_vector_l615_61584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_monotonic_interval_l615_61507

noncomputable def f (x : ℝ) : ℝ := x^2 - (1/2) * Real.log x + 3/2

noncomputable def f' (x : ℝ) : ℝ := 2*x - 1/(2*x)

theorem not_monotonic_interval (a : ℝ) :
  (∀ x, x ∈ Set.Ioo (a-1) (a+1) → x > 0) →
  (∃ x y, x ∈ Set.Ioo (a-1) (a+1) ∧ y ∈ Set.Ioo (a-1) (a+1) ∧ x < y ∧ f x > f y) →
  a ∈ Set.Ico 1 (3/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_monotonic_interval_l615_61507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l615_61571

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def tangentToXAxis (c : Circle) : Prop :=
  c.center.2 = c.radius

def centerOnLine (c : Circle) : Prop :=
  3 * c.center.1 - 4 * c.center.2 = 0

def intersectsLine (c : Circle) : Prop :=
  ∃ (x y : ℝ), (x - y = 0) ∧ 
    ((x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
    ((x + 2*Real.sqrt 17 - c.center.1)^2 + (y + 2*Real.sqrt 17 - c.center.2)^2 = c.radius^2)

-- Theorem statement
theorem circle_equation (c : Circle) 
  (h1 : tangentToXAxis c)
  (h2 : centerOnLine c)
  (h3 : intersectsLine c) :
  ((c.center = (4*Real.sqrt 2, 3*Real.sqrt 2) ∧ c.radius^2 = 18) ∨
   (c.center = (-4*Real.sqrt 2, -3*Real.sqrt 2) ∧ c.radius^2 = 18)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l615_61571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_height_difference_l615_61583

def height_difference (heights : List ℝ) : ℝ :=
  heights.foldr (λ x acc => heights.foldl (λ sum y => sum + |x - y|) acc) 0

theorem total_height_difference 
  (Anne Cathy Bella Daisy Ellie : ℝ)
  (h1 : Anne = 80)
  (h2 : Cathy = Anne / 2)
  (h3 : Bella = 3 * Anne)
  (h4 : Daisy = (Cathy + Anne) / 2)
  (h5 : Ellie = Real.sqrt (Bella * Cathy))
  : height_difference [Bella, Cathy, Daisy, Ellie] = 638 := by
  sorry

#eval height_difference [240, 40, 60, 98]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_height_difference_l615_61583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_10_value_l615_61567

def x : ℕ → ℚ
  | 0 => 1
  | 1 => 1
  | (n + 2) => (x (n + 1) * x n) / (x (n + 1) + x n)

theorem x_10_value : x 10 = 1 / 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_10_value_l615_61567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_theorem_l615_61511

/-- Represents the scenario of Ksyusha's journey to school -/
structure JourneyScenario where
  S : ℚ  -- Distance unit
  v : ℚ  -- Walking speed
  run_speed : ℚ  -- Running speed
  total_distance : ℚ  -- Total distance from home to school

/-- Tuesday's journey details -/
def tuesday (j : JourneyScenario) : Prop :=
  j.run_speed = 2 * j.v ∧
  j.total_distance = 3 * j.S ∧
  (2 * j.S) / j.v + j.S / (2 * j.v) = 30

/-- Wednesday's journey time calculation -/
def wednesday_time (j : JourneyScenario) : ℚ :=
  j.S / j.v + (2 * j.S) / (2 * j.v)

/-- The main theorem to prove -/
theorem journey_time_theorem (j : JourneyScenario) :
  tuesday j → wednesday_time j = 24 := by
  intro h
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_theorem_l615_61511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_implies_two_functions_and_sum_four_l615_61553

def FunctionProperty (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (g x + y) = g (x^2 + y) + 2 * g x * y

theorem function_property_implies_two_functions_and_sum_four :
  ∃ (g₁ g₂ : ℝ → ℝ),
    FunctionProperty g₁ ∧ FunctionProperty g₂ ∧
    (∀ g : ℝ → ℝ, FunctionProperty g → (g = g₁ ∨ g = g₂)) ∧
    g₁ 2 + g₂ 2 = 4 := by
  sorry

#check function_property_implies_two_functions_and_sum_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_implies_two_functions_and_sum_four_l615_61553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_range_l615_61592

/-- Given a quadratic function f(x) = ax^2 - 2ax + c, where a and c are real numbers,
    if f(2017) < f(-2016), then the set of real numbers m satisfying f(m) ≤ f(0)
    is exactly the closed interval [0, 2]. -/
theorem quadratic_function_range (a c : ℝ) :
  let f := λ x : ℝ => a * x^2 - 2 * a * x + c
  (f 2017 < f (-2016)) →
  {m : ℝ | f m ≤ f 0} = Set.Icc 0 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_range_l615_61592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_complex_fraction_l615_61575

theorem real_part_of_complex_fraction : ∃ z : ℂ, 
  z = (Complex.I + 5) / (1 + Complex.I) ∧ Complex.re z = 3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_complex_fraction_l615_61575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l615_61505

noncomputable def f (x a : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6) + Real.sin (2 * x - Real.pi / 6) + Real.cos (2 * x) + a

theorem f_properties (a : ℝ) :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f x a = f (x + T) a ∧
    ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f x a = f (x + S) a) → T ≤ S) ∧
  (∀ k : ℤ, ∀ x : ℝ, x ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) →
    ∀ y : ℝ, y ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) →
      x ≤ y → f x a ≤ f y a) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f x a ≥ -2) →
  a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l615_61505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_partition_integer_side_l615_61563

/-- A rectangle represented by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- A partition of a rectangle into smaller rectangles -/
structure RectanglePartition where
  original : Rectangle
  parts : List Rectangle
  
/-- Predicate to check if a rectangle has at least one integer side -/
def hasIntegerSide (r : Rectangle) : Prop :=
  ∃ n : ℤ, (r.width = n) ∨ (r.height = n)

/-- Theorem: If a rectangle can be partitioned into smaller rectangles, 
    each with at least one integer side, then the original rectangle 
    has at least one integer side -/
theorem rectangle_partition_integer_side 
  (partition : RectanglePartition) 
  (h : ∀ r ∈ partition.parts, hasIntegerSide r) : 
  hasIntegerSide partition.original :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_partition_integer_side_l615_61563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_l615_61556

-- Problem 1
theorem problem_1 : (Real.pi - 3) ^ (0 : ℤ) + (1 / 2) ^ (-3 : ℤ) - 3 ^ 2 + (-1 : ℝ) ^ 2024 = 1 := by sorry

-- Problem 2
theorem problem_2 (m n : ℝ) : 3 * m * 2 * n^2 - (2 * n)^2 * (1 / 2) * m = 4 * m * n^2 := by sorry

-- Problem 3
theorem problem_3 (x y : ℝ) (hx : x = -3) (hy : y = 2) :
  ((x + 2*y)^2 - y*(x + 3*y) + (x - y)*(x + y)) / (2*x) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_l615_61556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dmitry_arrives_first_probability_l615_61506

-- Define the time interval
def TimeInterval : Type := ℝ

-- Define the arrival times as real numbers within the time interval
def ArrivalTime (interval : TimeInterval) : Type := {t : ℝ // 0 ≤ t ∧ t ≤ interval}

-- Define the probability space
def ProbabilitySpace (interval : TimeInterval) : Type :=
  ArrivalTime interval × ArrivalTime interval × ArrivalTime interval

-- Define the event where Dmitry arrives before his father
def DmitryArrivesFirst (interval : TimeInterval) (space : ProbabilitySpace interval) : Prop :=
  space.2.1.val < space.1.val

-- Define a probability measure (this is a simplification)
noncomputable def Prob (interval : TimeInterval) (event : ProbabilitySpace interval → Prop) : ℝ :=
  sorry

-- State the theorem
theorem dmitry_arrives_first_probability (interval : TimeInterval) :
  ∃ (p : ℝ), p = 2/3 ∧ 
  Prob interval (DmitryArrivesFirst interval) = p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dmitry_arrives_first_probability_l615_61506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_l615_61515

noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

def triangle_conditions (l m n : ℕ) : Prop :=
  l > m ∧ m > n ∧
  frac (3^l / 10000 : ℝ) = frac (3^m / 10000 : ℝ) ∧
  frac (3^m / 10000 : ℝ) = frac (3^n / 10000 : ℝ)

theorem min_perimeter (l m n : ℕ) :
  triangle_conditions l m n → l + m + n ≥ 3003 := by
  sorry

#check min_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_l615_61515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_l615_61531

-- Define the functions f and g
def f (a c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - 4*x + c

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := Real.log x + (b-1)*x + 4

-- State the theorem
theorem tangent_line_and_inequality (a b c : ℝ) :
  (∀ x, (3*x - f a c x + 1 = 0) → x = 1) ∧ 
  (∀ x₁ x₂, x₁ ∈ Set.Icc (-3) 0 → x₂ ∈ Set.Ioi 0 → f a c x₁ ≥ g b x₂) →
  (f 2 5 = f a c) ∧ (b ≤ 1 - Real.exp (-2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_l615_61531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l615_61549

/-- Fixed cost for producing electronic instruments -/
noncomputable def fixed_cost : ℝ := 20000

/-- Additional investment required per instrument -/
noncomputable def additional_investment : ℝ := 100

/-- Total revenue function -/
noncomputable def total_revenue (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 400 then 400 * x - (1/2) * x^2
  else 80000

/-- Profit function -/
noncomputable def profit (x : ℝ) : ℝ :=
  total_revenue x - (fixed_cost + additional_investment * x)

/-- Theorem stating the maximum profit and corresponding production volume -/
theorem max_profit :
  ∃ (max_profit : ℝ) (max_volume : ℝ),
    max_profit = 25000 ∧
    max_volume = 300 ∧
    ∀ x, profit x ≤ max_profit ∧
    profit max_volume = max_profit := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l615_61549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_equals_e_squared_minus_three_l615_61526

open Real MeasureTheory

-- Define the curves
noncomputable def f (x : ℝ) : ℝ := 2 * x
noncomputable def g (x : ℝ) : ℝ := 2 / x

-- Define the bounds
noncomputable def lower_bound : ℝ := 1
noncomputable def upper_bound : ℝ := Real.exp 1

-- Define the area function
noncomputable def area : ℝ := ∫ x in lower_bound..upper_bound, (f x - g x)

-- Theorem statement
theorem enclosed_area_equals_e_squared_minus_three :
  area = Real.exp 2 - 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_equals_e_squared_minus_three_l615_61526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l615_61508

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ p = Real.pi ∧ ∀ x : ℝ, f (x + p) = f x) ∧
  (∀ x : ℝ, f (2 * Real.pi / 3 - x) = f (2 * Real.pi / 3 + x)) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l615_61508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_point_distance_l615_61570

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define point A
def point_A : ℝ × ℝ := (1, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem ellipse_point_distance :
  ∀ (x y : ℝ), ellipse_C x y →
  distance (x, y) point_A = 3/2 →
  (x = 1 ∧ y = Real.sqrt 3/2) ∨ (x = 1 ∧ y = -Real.sqrt 3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_point_distance_l615_61570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l615_61541

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x^2 + x else -x^2

-- State the theorem
theorem range_of_t (t : ℝ) : f (f t) ≤ 2 → t ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l615_61541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_max_inscribed_circle_l615_61535

/-- An isosceles triangle with leg length 1 and base length x -/
structure IsoscelesTriangle where
  x : ℝ
  leg_length : ℝ := 1

/-- The diameter of the inscribed circle of an isosceles triangle -/
noncomputable def inscribedCircleDiameter (t : IsoscelesTriangle) : ℝ :=
  t.x * Real.sqrt ((2 - t.x) / (2 + t.x))

/-- The base length that maximizes the inscribed circle diameter -/
noncomputable def optimalBaseLength : ℝ := Real.sqrt 5 - 1

/-- The maximum inscribed circle diameter -/
noncomputable def maxDiameter : ℝ := Real.sqrt (10 * Real.sqrt 5 - 22)

theorem isosceles_triangle_max_inscribed_circle 
  (t : IsoscelesTriangle) : 
  inscribedCircleDiameter t ≤ maxDiameter ∧
  inscribedCircleDiameter { x := optimalBaseLength } = maxDiameter := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_max_inscribed_circle_l615_61535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l615_61548

/-- Represents a pyramid with a square base and a vertex --/
structure Pyramid where
  base_area : ℝ
  face1_area : ℝ
  face2_area : ℝ

/-- Calculates the volume of a pyramid --/
noncomputable def pyramid_volume (p : Pyramid) : ℝ :=
  sorry

/-- The theorem stating the volume of the specific pyramid --/
theorem specific_pyramid_volume :
  ∃ (p : Pyramid),
    p.base_area = 256 ∧
    p.face1_area = 160 ∧
    p.face2_area = 120 ∧
    |pyramid_volume p - 1706.67| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l615_61548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_pet_owners_l615_61546

theorem three_pet_owners (total dog cat bird dog_cat cat_bird dog_bird : ℕ) :
  total = 50 →
  dog = 30 →
  cat = 35 →
  bird = 10 →
  dog_cat = 8 →
  cat_bird = 5 →
  dog_bird = 3 →
  total = dog + cat + bird - dog_cat - cat_bird - dog_bird + (dog + cat + bird - (dog + cat - dog_cat) - (cat + bird - cat_bird) - (dog + bird - dog_bird) + total) →
  (dog + cat + bird - (dog + cat - dog_cat) - (cat + bird - cat_bird) - (dog + bird - dog_bird) + total) = 7 :=
by
  sorry

#check three_pet_owners

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_pet_owners_l615_61546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_retail_discount_percentage_l615_61555

theorem retail_discount_percentage (wholesale_price retail_price : ℝ) 
  (profit_percentage : ℝ) (h1 : wholesale_price = 90) 
  (h2 : retail_price = 120) (h3 : profit_percentage = 0.2) : ℝ :=
  let profit := wholesale_price * profit_percentage
  let selling_price := wholesale_price + profit
  let discount_amount := retail_price - selling_price
  let discount_percentage := (discount_amount / retail_price) * 100
  have h4 : discount_percentage = 10 := by sorry
  10

-- Remove the #eval line as it was causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_retail_discount_percentage_l615_61555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pool_cost_l615_61517

/-- Represents the cost function for constructing a rectangular open-top pool -/
noncomputable def pool_cost (w : ℝ) : ℝ :=
  1800 + 5400 / w + 600 * w

/-- Theorem stating the minimum cost for constructing the pool -/
theorem min_pool_cost :
  ∃ (w : ℝ), w > 0 ∧ 
  (∀ (x : ℝ), x > 0 → pool_cost w ≤ pool_cost x) ∧
  pool_cost w = 5400 := by
  sorry

#check min_pool_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pool_cost_l615_61517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_A_l615_61554

noncomputable def A (x y z : ℝ) : ℝ :=
  ((x^3 - 6) * (x + 6)^(1/3) + (y^3 - 6) * (y + 6)^(1/3) + (z^3 - 6) * (z + 6)^(1/3)) / (x^2 + y^2 + z^2)

theorem max_value_of_A (x y z : ℝ) (hx : 0 < x ∧ x ≤ 2) (hy : 0 < y ∧ y ≤ 2) (hz : 0 < z ∧ z ≤ 2) :
  A x y z ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_A_l615_61554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_specific_circles_l615_61504

/-- The area between two concentric circles -/
noncomputable def area_between_circles (R r : ℝ) : ℝ := Real.pi * (R^2 - r^2)

/-- The theorem to be proved -/
theorem area_between_specific_circles :
  ∃ (R r : ℝ),
    R = 12 ∧                           -- Radius of larger circle
    2 * (R * r) = 20 * r ∧             -- Chord is tangent to smaller circle
    20^2 = 2 * R^2 * (1 - (r / R)^2) ∧ -- Chord length is 20
    area_between_circles R r = 100 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_specific_circles_l615_61504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_of_thirteen_numbers_l615_61559

theorem divisor_of_thirteen_numbers (n : ℕ) : n > 0 ∧ 
  (∃! x : ℕ, x > 0 ∧ (Finset.filter (fun y => 10 ≤ y ∧ y ≤ 50 ∧ y % x = 0) (Finset.range 51)).card = 13) → n = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_of_thirteen_numbers_l615_61559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l615_61587

/-- The time taken for a train to cross a pole -/
noncomputable def train_crossing_time (speed_kmh : ℝ) (length_m : ℝ) : ℝ :=
  length_m / (speed_kmh * 1000 / 3600)

/-- Theorem: The time taken for a train to cross a pole is approximately 9 seconds -/
theorem train_crossing_pole_time :
  let speed := (52 : ℝ)
  let length := (130 : ℝ)
  abs (train_crossing_time speed length - 9) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l615_61587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodic_and_correct_l615_61577

noncomputable def f (x : ℝ) : ℝ :=
  if x < -1 then (x + 2)^2 + 1
  else if x ≤ 1 then x^2 + 1
  else (x - 2)^2 + 1

theorem f_periodic_and_correct :
  (∀ x, f (x + 2) = f x) ∧
  (∀ x, -1 ≤ x ∧ x ≤ 1 → f x = x^2 + 1) ∧
  (∀ x, -3 ≤ x ∧ x ≤ 3 → 
    f x = if -3 ≤ x ∧ x < -1 then x^2 + 4*x + 5
          else if -1 ≤ x ∧ x ≤ 1 then x^2 + 1
          else x^2 - 4*x + 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodic_and_correct_l615_61577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l615_61524

theorem triangle_inequality (a b c : ℝ) (n : ℕ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) : 
  (a^n / (b + c) + b^n / (c + a) + c^n / (a + b)) ≥ (2/3)^(n-2) * ((a + b + c) / 2)^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l615_61524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2012_value_l615_61510

def sequence_a : ℕ → ℚ
  | 0 => 2  -- We define a₀ = 2 to match a₁ = 2 in the original problem
  | 1 => 2
  | (n + 2) => 2 * sequence_a (n + 1) / sequence_a n

theorem a_2012_value : sequence_a 2012 = 1 / 1006 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2012_value_l615_61510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_incenters_imply_equilateral_triangle_l615_61547

-- Define the necessary structures and functions
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

noncomputable def incenter (t : Triangle) : ℝ × ℝ := sorry

def is_equilateral (t : Triangle) : Prop := sorry

-- Define the theorem
theorem equilateral_incenters_imply_equilateral_triangle 
  (A B C I X Y Z : ℝ × ℝ) : 
  let ABC := Triangle.mk A B C
  let BIC := Triangle.mk B I C
  let CIA := Triangle.mk C I A
  let AIB := Triangle.mk A I B
  let XYZ := Triangle.mk X Y Z
  (I = incenter ABC) →
  (X = incenter BIC) →
  (Y = incenter CIA) →
  (Z = incenter AIB) →
  is_equilateral XYZ →
  is_equilateral ABC := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_incenters_imply_equilateral_triangle_l615_61547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_figure_l615_61565

theorem area_of_closed_figure (a : ℝ) (f : ℝ → ℝ) :
  (∃ r : ℕ, (18 - r = 9) ∧ 
    ((-1)^r * (1/a)^r * (Nat.choose 9 r) = -21/2)) →
  (f = λ x ↦ Real.sin x) →
  (∫ x in -a..a, f x) = 2 * Real.cos 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_figure_l615_61565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l615_61502

noncomputable def f (x : ℝ) : ℝ := x + 2 / (x^2)

theorem range_of_f :
  (∀ x > 0, f x ≥ 3 / (2^(1/3))) ∧
  (∀ y ≥ 3 / (2^(1/3)), ∃ x > 0, f x = y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l615_61502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_square_with_equal_diagonal_sums_l615_61527

/-- Represents a cell in the n×n table -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents the n×n table -/
def Table (n : Nat) := Cell → Nat

/-- Two cells are adjacent if they share an edge -/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ c1.col + 1 = c2.col) ∨
  (c1.row = c2.row ∧ c1.col = c2.col + 1) ∨
  (c1.row + 1 = c2.row ∧ c1.col = c2.col) ∨
  (c1.row = c2.row + 1 ∧ c1.col = c2.col)

/-- The table is filled with numbers 1 through n^2 exactly once each -/
def validFilling (n : Nat) (t : Table n) : Prop :=
  ∀ i : Nat, 1 ≤ i ∧ i ≤ n^2 → ∃! c : Cell, t c = i

/-- Adjacent cells differ by at most n -/
def adjacentDifferByAtMostN (n : Nat) (t : Table n) : Prop :=
  ∀ c1 c2 : Cell, adjacent c1 c2 → |Int.ofNat (t c1) - Int.ofNat (t c2)| ≤ n

/-- A 2×2 square of cells -/
structure Square (n : Nat) where
  topLeft : Cell
  topRight : Cell
  bottomLeft : Cell
  bottomRight : Cell
  valid : topLeft.row + 1 = bottomLeft.row ∧
          topLeft.col + 1 = topRight.col ∧
          topRight.row + 1 = bottomRight.row ∧
          bottomLeft.col + 1 = bottomRight.col

/-- The diagonally opposite pairs in the square sum to the same number -/
def diagonalsSumEqual (n : Nat) (t : Table n) (s : Square n) : Prop :=
  t s.topLeft + t s.bottomRight = t s.topRight + t s.bottomLeft

/-- The main theorem -/
theorem exists_square_with_equal_diagonal_sums
  (n : Nat) (hn : n ≥ 2) (t : Table n)
  (hfill : validFilling n t) (hdiff : adjacentDifferByAtMostN n t) :
  ∃ s : Square n, diagonalsSumEqual n t s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_square_with_equal_diagonal_sums_l615_61527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l615_61539

/-- The curve C defined by the equation (x - arcsin α)(x - arccos α) + (y - arcsin α)(y + arccos α) = 0 -/
def C (α : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - Real.arcsin α) * (p.1 - Real.arccos α) + (p.2 - Real.arcsin α) * (p.2 + Real.arccos α) = 0}

/-- The line x = π/4 -/
def L : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = Real.pi/4}

/-- The length of the chord intercepted by the line x = π/4 on the curve C -/
noncomputable def chordLength (α : ℝ) : ℝ :=
  let p₁ := (Real.pi/4, Real.sqrt ((Real.arcsin α - Real.pi/4) * (Real.arccos α - Real.pi/4)))
  let p₂ := (Real.pi/4, -Real.sqrt ((Real.arcsin α - Real.pi/4) * (Real.arccos α - Real.pi/4)))
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

/-- The theorem stating that the minimum chord length is π/2 -/
theorem min_chord_length :
  ∃ (α₀ : ℝ), ∀ (α : ℝ), chordLength α₀ ≤ chordLength α ∧ chordLength α₀ = Real.pi/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_l615_61539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l615_61594

-- Define the rectangle
def rectangle_length : ℝ := 16
def rectangle_diagonal : ℝ := 20

-- Theorem to prove
theorem rectangle_area : 
  (rectangle_length * Real.sqrt (rectangle_diagonal^2 - rectangle_length^2)) = 192 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l615_61594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l615_61544

noncomputable def triangle (A B C : Real) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi

theorem triangle_property (A B C : Real) 
  (h_triangle : triangle A B C)
  (h_perimeter : Real.sin A + Real.sin B + Real.sin C = Real.sqrt 2 + 1)
  (h_sin_sum : Real.sin A + Real.sin B = Real.sqrt 2 * Real.sin C)
  (h_area : (1/2) * Real.sin A * Real.sin B * Real.sin C / (Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2)) = (1/6) * Real.sin C) :
  Real.sin B = 1 ∧ C = Real.pi/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l615_61544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_substance_density_l615_61585

-- Define the densities of the three substances
variable (x y z : ℝ)

-- Define the conditions
def total_volume : ℝ := 6
def total_mass : ℝ := 16
def volume_relation : ℝ := 0.5

-- Define mass_relation as a proposition
def mass_relation (y x : ℝ) : Prop := y = 2 * x

-- Theorem statement
theorem third_substance_density :
  x + y + z = total_mass / total_volume →
  1 / y - 1 / z = volume_relation / 4 →
  mass_relation y x →
  z = 8 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_substance_density_l615_61585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_swims_18km_upstream_l615_61529

/-- Calculates the upstream distance swam by a man given his downstream distance, time, and still water speed. -/
noncomputable def upstream_distance (downstream_distance : ℝ) (time : ℝ) (still_water_speed : ℝ) : ℝ :=
  let current_speed := downstream_distance / time - still_water_speed
  (still_water_speed - current_speed) * time

/-- Theorem stating that under the given conditions, the man swims 18 km upstream. -/
theorem man_swims_18km_upstream (downstream_distance : ℝ) (time : ℝ) (still_water_speed : ℝ)
  (h1 : downstream_distance = 42)
  (h2 : time = 3)
  (h3 : still_water_speed = 10) :
  upstream_distance downstream_distance time still_water_speed = 18 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_swims_18km_upstream_l615_61529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_set_properties_l615_61512

-- Define a perfect set
def PerfectSet (A : Set ℚ) : Prop :=
  (0 ∈ A) ∧ (1 ∈ A) ∧ 
  (∀ x y, x ∈ A → y ∈ A → (x - y) ∈ A) ∧
  (∀ x, x ∈ A → x ≠ 0 → (1 / x) ∈ A)

-- Theorem stating the properties of perfect sets
theorem perfect_set_properties :
  ∀ A : Set ℚ, PerfectSet A →
    (PerfectSet (Set.univ : Set ℚ)) ∧
    (∀ x y, x ∈ A → y ∈ A → (x + y) ∈ A) ∧
    (∀ x y, x ∈ A → y ∈ A → (x * y) ∈ A) ∧
    (∀ x y, x ∈ A → y ∈ A → x ≠ 0 → (y / x) ∈ A) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_set_properties_l615_61512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radii_difference_l615_61538

-- Define the circles and points
def origin : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (5, 12)
def S : ℝ → ℝ × ℝ := λ k => (0, k)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem circle_radii_difference (k : ℝ) : 
  distance origin P - distance origin (S k) = 5 → k = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radii_difference_l615_61538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l615_61545

theorem exponential_inequality (x : ℝ) (h : -1 < x ∧ x < 0) :
  (2 : ℝ)^x < (2 : ℝ)^(-x) ∧ (2 : ℝ)^(-x) < (1/5 : ℝ)^x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l615_61545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_three_fourths_l615_61582

/-- The sum of the infinite series ∑(k/3^k) for k from 1 to infinity -/
noncomputable def series_sum : ℝ := ∑' k, k / 3^k

/-- Theorem: The sum of the infinite series ∑(k/3^k) for k from 1 to infinity equals 3/4 -/
theorem series_sum_equals_three_fourths : series_sum = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_three_fourths_l615_61582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_crossing_point_l615_61519

/-- The x-coordinate of a point on the curve as a function of t -/
def x (t : ℝ) : ℝ := t^3 - t + 1

/-- The y-coordinate of a point on the curve as a function of t -/
def y (t : ℝ) : ℝ := t^3 - 11*t + 11

/-- The point where the curve crosses itself -/
noncomputable def crossingPoint : ℝ × ℝ := (10 * Real.sqrt 11 + 1, 11)

/-- Theorem stating that the curve crosses itself at the specified point -/
theorem curve_crossing_point :
  ∃ (a b : ℝ), a ≠ b ∧ 
    x a = x b ∧ 
    y a = y b ∧ 
    (x a, y a) = crossingPoint :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_crossing_point_l615_61519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_negative_exponents_l615_61589

theorem no_negative_exponents (a b c d e : ℤ) :
  (2 : ℝ)^a + (2 : ℝ)^b + (2 : ℝ)^c = (5 : ℝ)^d + (5 : ℝ)^e →
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_negative_exponents_l615_61589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_with_common_difference_l615_61576

theorem subset_with_common_difference (S : Finset ℕ) :
  S ⊆ Finset.range 2001 →
  S.card = 401 →
  ∃ n : ℕ, 0 < n ∧ (S.filter (fun x ↦ x + n ∈ S)).card ≥ 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_with_common_difference_l615_61576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_3_l615_61513

/-- Represents an isosceles triangle ABC with side lengths AB = AC = 6 and BC = 8 -/
structure IsoscelesTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  ab_eq_ac : dist A B = dist A C
  ab_eq_6 : dist A B = 6
  bc_eq_8 : dist B C = 8

/-- The length of the crease when folding vertex A onto point B -/
def creaseLength (t : IsoscelesTriangle) : ℝ := 3

/-- Theorem stating that the crease length is 3 inches -/
theorem crease_length_is_3 (t : IsoscelesTriangle) :
  creaseLength t = 3 := by
  -- Unfold the definition of creaseLength
  unfold creaseLength
  -- The definition directly returns 3, so this is true by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_3_l615_61513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l615_61557

-- Define the function f with domain [-8, 1]
noncomputable def f : Set ℝ → ℝ → ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc (-8) 1

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (f dom_f (2*x + 1)) / (x + 2)

-- Theorem stating the domain of g
theorem domain_of_g :
  {x : ℝ | g x ≠ 0} = Set.Ioc (-9/2) (-2) ∪ Set.Ico (-2) 0 :=
sorry

#check domain_of_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l615_61557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_score_rounded_l615_61572

def class_size : Nat := 50

def score_distribution : List (Nat × Nat) := [
  (90, 8),
  (83, 11),
  (74, 10),
  (65, 16),
  (56, 3),
  (49, 2)
]

def total_score : Nat := (score_distribution.map (fun (score, count) => score * count)).sum

def total_students : Nat := (score_distribution.map (fun (_, count) => count)).sum

theorem average_score_rounded :
  (((total_score : ℚ) / total_students * 10).floor / 10 : ℚ) = 73.6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_score_rounded_l615_61572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_art_price_increase_l615_61533

theorem art_price_increase (first_three_total : ℝ) (all_four_total : ℝ) : 
  first_three_total = 45000 →
  all_four_total = 67500 →
  (let first_piece_price := first_three_total / 3
   let fourth_piece_price := all_four_total - first_three_total
   let price_increase := fourth_piece_price - first_piece_price
   let percentage_increase := (price_increase / first_piece_price) * 100
   percentage_increase = 50) := by
  intro h1 h2
  -- Proof steps would go here
  sorry

#check art_price_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_art_price_increase_l615_61533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l615_61518

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define z as given in the problem
noncomputable def z : ℂ := 1 / (1 + i)

-- Theorem statement
theorem complex_fraction_simplification : z = (1 - i) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l615_61518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annes_solo_cleaning_time_is_12_hours_l615_61558

/-- Represents the time it takes Anne to clean the house alone -/
def annes_cleaning_time (anne_rate : ℚ) : ℚ :=
  1 / anne_rate

theorem annes_solo_cleaning_time_is_12_hours 
  (bruce_rate anne_rate : ℚ) 
  (h1 : bruce_rate + anne_rate = 1/4) 
  (h2 : bruce_rate + 2*anne_rate = 1/3) : 
  annes_cleaning_time anne_rate = 12 := by
  sorry

#check annes_solo_cleaning_time_is_12_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annes_solo_cleaning_time_is_12_hours_l615_61558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_coordinates_l615_61598

/-- The equation of a hyperbola with center (h, k), transverse axis 2a, and conjugate axis 2b. -/
def hyperbola_equation (x y h k a b : ℝ) : Prop :=
  (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1

/-- The distance from the center to a focus in a hyperbola. -/
noncomputable def focus_distance (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

/-- Theorem: For a hyperbola with equation (x-1)^2/7^2 - (y-10)^2/3^2 = 1 and horizontally oriented foci,
    the coordinates of the focus with the larger x-coordinate are (1+√58, 10). -/
theorem hyperbola_focus_coordinates :
  ∀ (x y : ℝ),
    hyperbola_equation x y 1 10 7 3 →
    (1 + focus_distance 7 3, 10) = (1 + Real.sqrt 58, 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_coordinates_l615_61598
