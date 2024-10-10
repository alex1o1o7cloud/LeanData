import Mathlib

namespace sum_of_22_and_62_l2125_212506

theorem sum_of_22_and_62 : 22 + 62 = 84 := by
  sorry

end sum_of_22_and_62_l2125_212506


namespace scientific_notation_equality_l2125_212576

theorem scientific_notation_equality : 935000000 = 9.35 * (10 ^ 8) := by
  sorry

end scientific_notation_equality_l2125_212576


namespace d_necessary_not_sufficient_for_a_l2125_212586

-- Define propositions
variable (A B C D : Prop)

-- Define the relationships between propositions
variable (h1 : A → B ∧ ¬(B → A))
variable (h2 : B → C ∧ ¬(C → B))
variable (h3 : D ↔ C)

-- Theorem to prove
theorem d_necessary_not_sufficient_for_a :
  (D → A) ∧ ¬(A → D) :=
sorry

end d_necessary_not_sufficient_for_a_l2125_212586


namespace intersection_range_l2125_212582

-- Define the points M and N
def M : ℝ × ℝ := (1, 0)
def N : ℝ × ℝ := (-1, 0)

-- Define the line equation
def line (x y b : ℝ) : Prop := 2 * x + y = b

-- Define the line segment MN
def on_segment (x y : ℝ) : Prop :=
  x ≥ -1 ∧ x ≤ 1 ∧ y = 0

-- Theorem statement
theorem intersection_range :
  ∀ b : ℝ,
  (∃ x y : ℝ, line x y b ∧ on_segment x y) ↔
  b ≥ -2 ∧ b ≤ 2 :=
sorry

end intersection_range_l2125_212582


namespace jackies_free_time_l2125_212504

def hours_in_day : ℕ := 24

def working_hours : ℕ := 8
def exercise_hours : ℕ := 3
def sleep_hours : ℕ := 8

def scheduled_hours : ℕ := working_hours + exercise_hours + sleep_hours

theorem jackies_free_time : hours_in_day - scheduled_hours = 5 := by
  sorry

end jackies_free_time_l2125_212504


namespace exists_congruent_triangle_with_same_color_on_sides_l2125_212594

/-- A color type with 1992 different colors -/
inductive Color : Type
| mk : Fin 1992 → Color

/-- A point in the plane -/
structure Point : Type :=
  (x y : ℝ)

/-- A triangle in the plane -/
structure Triangle : Type :=
  (a b c : Point)

/-- A coloring of the plane -/
def Coloring : Type := Point → Color

/-- A predicate to check if a point is on a line segment -/
def OnSegment (p q r : Point) : Prop := sorry

/-- A predicate to check if two triangles are congruent -/
def Congruent (t1 t2 : Triangle) : Prop := sorry

/-- Main theorem -/
theorem exists_congruent_triangle_with_same_color_on_sides
  (coloring : Coloring)
  (all_colors_used : ∀ c : Color, ∃ p : Point, coloring p = c)
  (t : Triangle) :
  ∃ t' : Triangle, Congruent t t' ∧
    ∃ (p1 p2 p3 : Point) (c : Color),
      OnSegment p1 t'.a t'.b ∧
      OnSegment p2 t'.b t'.c ∧
      OnSegment p3 t'.c t'.a ∧
      coloring p1 = c ∧
      coloring p2 = c ∧
      coloring p3 = c :=
sorry

end exists_congruent_triangle_with_same_color_on_sides_l2125_212594


namespace investment_ratio_l2125_212510

/-- Given two investors P and Q who divide their profit in the ratio 2:3,
    where P invested 30000, prove that Q invested 45000. -/
theorem investment_ratio (p q : ℕ) (profit_ratio : ℚ) :
  profit_ratio = 2 / 3 →
  p = 30000 →
  q * profit_ratio = p * (1 - profit_ratio) →
  q = 45000 := by
sorry

end investment_ratio_l2125_212510


namespace egyptian_fraction_representation_l2125_212546

theorem egyptian_fraction_representation : ∃! (b₂ b₃ b₄ b₅ b₆ b₇ : ℕ),
  (17 : ℚ) / 23 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 + b₇ / 5040 ∧
  b₂ < 2 ∧ b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧ b₆ < 6 ∧ b₇ < 7 ∧
  b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 11 := by
  sorry

end egyptian_fraction_representation_l2125_212546


namespace expression_evaluation_l2125_212587

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 2) :
  3 * x^(y + 1) + 4 * y^(x + 1) = 145 := by
  sorry

end expression_evaluation_l2125_212587


namespace xyz_mod_seven_l2125_212589

theorem xyz_mod_seven (x y z : ℕ) (hx : x < 7) (hy : y < 7) (hz : z < 7)
  (h1 : x + 3*y + 2*z ≡ 0 [ZMOD 7])
  (h2 : 3*x + 2*y + z ≡ 2 [ZMOD 7])
  (h3 : 2*x + y + 3*z ≡ 3 [ZMOD 7]) :
  x * y * z ≡ 1 [ZMOD 7] := by
  sorry

end xyz_mod_seven_l2125_212589


namespace polynomial_coefficient_sum_l2125_212561

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₄ + a₂ + a₀ = -121 := by
sorry

end polynomial_coefficient_sum_l2125_212561


namespace trees_in_gray_areas_trees_in_gray_areas_proof_l2125_212552

/-- Given three pictures with an equal number of trees, where the white areas
contain 82, 82, and 100 trees respectively, the total number of trees in the
gray areas is 26. -/
theorem trees_in_gray_areas : ℕ → ℕ → ℕ → Prop :=
  fun (total : ℕ) (x : ℕ) (y : ℕ) =>
    (total = 82 + x) ∧
    (total = 82 + y) ∧
    (total = 100) →
    x + y = 26

/-- Proof of the theorem -/
theorem trees_in_gray_areas_proof : ∃ (total : ℕ), trees_in_gray_areas total 18 8 := by
  sorry

end trees_in_gray_areas_trees_in_gray_areas_proof_l2125_212552


namespace arithmetic_sequence_calculation_l2125_212593

theorem arithmetic_sequence_calculation : 
  let n := 2023
  let sum_to_n (k : ℕ) := k * (k + 1) / 2
  let diff_from_one_to (k : ℕ) := 1 - (sum_to_n k - 1)
  (diff_from_one_to (n - 1)) * (sum_to_n n - 1) - 
  (diff_from_one_to n) * (sum_to_n (n - 1) - 1) = n :=
by sorry

end arithmetic_sequence_calculation_l2125_212593


namespace product_sum_inequality_l2125_212524

theorem product_sum_inequality (p : ℝ) (hp : p > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x * y = p ∧
  ∀ (a b : ℝ), a > 0 → b > 0 → a * b = p → x + y ≤ a + b :=
sorry

end product_sum_inequality_l2125_212524


namespace kennedy_benedict_house_difference_l2125_212595

theorem kennedy_benedict_house_difference (kennedy_house : ℕ) (benedict_house : ℕ)
  (h1 : kennedy_house = 10000)
  (h2 : benedict_house = 2350) :
  kennedy_house - 4 * benedict_house = 600 :=
by sorry

end kennedy_benedict_house_difference_l2125_212595


namespace train_length_problem_l2125_212555

/-- The length of two trains passing each other -/
theorem train_length_problem (speed_kmh : ℝ) (crossing_time : ℝ) : 
  speed_kmh = 18 ∧ crossing_time = 24 →
  ∃ (train_length : ℝ), train_length = 120 :=
by
  sorry

end train_length_problem_l2125_212555


namespace perpendicular_parallel_implication_l2125_212547

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_implication 
  (a b c : Line) (α β γ : Plane) 
  (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h2 : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) 
  (h3 : perpendicular a α) 
  (h4 : perpendicular b β) 
  (h5 : parallel_lines a b) : 
  parallel_planes α β :=
sorry

end perpendicular_parallel_implication_l2125_212547


namespace profit_calculation_correct_l2125_212538

/-- Represents the demand scenario --/
inductive DemandScenario
  | High
  | Moderate
  | Low

/-- Calculates the profit for a given demand scenario --/
def calculate_profit (scenario : DemandScenario) : ℚ :=
  let total_cloth : ℕ := 40
  let profit_per_meter : ℚ := 35
  let high_discount : ℚ := 0.1
  let moderate_discount : ℚ := 0.05
  let sales_tax : ℚ := 0.05
  let low_demand_cloth : ℕ := 30

  match scenario with
  | DemandScenario.High =>
    let original_profit := total_cloth * profit_per_meter
    let discounted_profit := original_profit * (1 - high_discount)
    discounted_profit * (1 - sales_tax)
  | DemandScenario.Moderate =>
    let original_profit := total_cloth * profit_per_meter
    let discounted_profit := original_profit * (1 - moderate_discount)
    discounted_profit * (1 - sales_tax)
  | DemandScenario.Low =>
    let original_profit := low_demand_cloth * profit_per_meter
    original_profit * (1 - sales_tax)

theorem profit_calculation_correct :
  (calculate_profit DemandScenario.High = 1197) ∧
  (calculate_profit DemandScenario.Moderate = 1263.5) ∧
  (calculate_profit DemandScenario.Low = 997.5) :=
by sorry

end profit_calculation_correct_l2125_212538


namespace megaTek_circle_graph_error_l2125_212553

theorem megaTek_circle_graph_error :
  let total_degrees : ℕ := 360
  let manufacturing_degrees : ℕ := 252
  let administration_degrees : ℕ := 68
  let research_degrees : ℕ := 40
  manufacturing_degrees + administration_degrees + research_degrees = total_degrees :=
by
  sorry

end megaTek_circle_graph_error_l2125_212553


namespace initial_liquid_A_amount_l2125_212560

/-- Proves that the initial amount of liquid A in a can is 36.75 litres given the specified conditions -/
theorem initial_liquid_A_amount
  (initial_ratio_A : ℚ)
  (initial_ratio_B : ℚ)
  (drawn_off_amount : ℚ)
  (new_ratio_A : ℚ)
  (new_ratio_B : ℚ)
  (h1 : initial_ratio_A = 7)
  (h2 : initial_ratio_B = 5)
  (h3 : drawn_off_amount = 18)
  (h4 : new_ratio_A = 7)
  (h5 : new_ratio_B = 9) :
  ∃ (initial_A : ℚ),
    initial_A = 36.75 ∧
    (initial_A / (initial_A * initial_ratio_B / initial_ratio_A) = initial_ratio_A / initial_ratio_B) ∧
    ((initial_A - drawn_off_amount * initial_ratio_A / (initial_ratio_A + initial_ratio_B)) /
     (initial_A * initial_ratio_B / initial_ratio_A - drawn_off_amount * initial_ratio_B / (initial_ratio_A + initial_ratio_B) + drawn_off_amount) =
     new_ratio_A / new_ratio_B) :=
by sorry

end initial_liquid_A_amount_l2125_212560


namespace min_vertices_is_six_l2125_212598

/-- A graph where each vertex knows exactly three others -/
def KnowledgeGraph (V : Type*) := V → Finset V

/-- Predicate to check if a vertex has exactly 3 neighbors -/
def has_three_neighbors (G : KnowledgeGraph V) (v : V) : Prop :=
  (G v).card = 3

/-- Predicate to check if among any three vertices, two are not connected -/
def has_non_connected_pair (G : KnowledgeGraph V) : Prop :=
  ∀ (a b c : V), a ≠ b ∧ b ≠ c ∧ a ≠ c →
    ¬(a ∈ G b ∧ b ∈ G c ∧ c ∈ G a)

/-- The main theorem stating the minimum number of vertices is 6 -/
theorem min_vertices_is_six (V : Type*) [Fintype V] :
  (∃ (G : KnowledgeGraph V), (∀ v, has_three_neighbors G v) ∧ has_non_connected_pair G) →
  Fintype.card V ≥ 6 :=
sorry

end min_vertices_is_six_l2125_212598


namespace area_difference_l2125_212542

/-- A right isosceles triangle with base length 1 -/
structure RightIsoscelesTriangle where
  base : ℝ
  base_eq_one : base = 1

/-- Configuration of two identical squares in the triangle (Figure 2) -/
structure SquareConfig2 (t : RightIsoscelesTriangle) where
  side_length : ℝ
  side_length_eq : side_length = 1 / 4

/-- Configuration of two identical squares in the triangle (Figure 3) -/
structure SquareConfig3 (t : RightIsoscelesTriangle) where
  side_length : ℝ
  side_length_eq : side_length = Real.sqrt 2 / 6

/-- Total area of squares in Configuration 2 -/
def totalArea2 (t : RightIsoscelesTriangle) (c : SquareConfig2 t) : ℝ :=
  2 * c.side_length ^ 2

/-- Total area of squares in Configuration 3 -/
def totalArea3 (t : RightIsoscelesTriangle) (c : SquareConfig3 t) : ℝ :=
  2 * c.side_length ^ 2

/-- The main theorem stating the difference in areas -/
theorem area_difference (t : RightIsoscelesTriangle) 
  (c2 : SquareConfig2 t) (c3 : SquareConfig3 t) : 
  totalArea2 t c2 - totalArea3 t c3 = 1 / 72 := by
  sorry

end area_difference_l2125_212542


namespace matrix_with_unequal_rank_and_square_rank_l2125_212543

theorem matrix_with_unequal_rank_and_square_rank
  (n : ℕ)
  (h_n : n ≥ 2)
  (A : Matrix (Fin n) (Fin n) ℂ)
  (h_rank : Matrix.rank A ≠ Matrix.rank (A * A)) :
  ∃ (B : Matrix (Fin n) (Fin n) ℂ), B ≠ 0 ∧ A * B = 0 ∧ B * A = 0 ∧ B * B = 0 := by
sorry

end matrix_with_unequal_rank_and_square_rank_l2125_212543


namespace travel_probability_is_two_thirds_l2125_212591

/-- Represents the probability of a bridge being destroyed in an earthquake -/
def p : ℝ := 0.5

/-- Represents the probability of a bridge surviving an earthquake -/
def q : ℝ := 1 - p

/-- Represents the probability of traveling from the first island to the shore after an earthquake -/
noncomputable def travel_probability : ℝ := q / (1 - p * q)

/-- Theorem stating that the probability of traveling from the first island to the shore
    after an earthquake is 2/3 -/
theorem travel_probability_is_two_thirds :
  travel_probability = 2 / 3 := by sorry

end travel_probability_is_two_thirds_l2125_212591


namespace jungkook_age_relation_l2125_212518

theorem jungkook_age_relation :
  ∃ (x : ℕ), 
    (46 - x : ℤ) = 4 * (16 - x : ℤ) ∧ 
    x ≤ 16 ∧ 
    x ≤ 46 ∧ 
    x = 6 :=
by sorry

end jungkook_age_relation_l2125_212518


namespace exam_pass_count_l2125_212575

theorem exam_pass_count (total_candidates : ℕ) (avg_all : ℚ) (avg_pass : ℚ) (avg_fail : ℚ) : 
  total_candidates = 120 →
  avg_all = 35 →
  avg_pass = 39 →
  avg_fail = 15 →
  ∃ (pass_count : ℕ), pass_count = 100 ∧ pass_count ≤ total_candidates :=
by sorry

end exam_pass_count_l2125_212575


namespace matrix_product_50_l2125_212572

def matrix_product (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  (List.range n).foldl
    (fun acc k => acc * !![1, 2*(k+1); 0, 1])
    !![1, 0; 0, 1]

theorem matrix_product_50 :
  matrix_product 50 = !![1, 2550; 0, 1] := by
  sorry

end matrix_product_50_l2125_212572


namespace factor_count_l2125_212533

/-- The number of positive factors of 180 that are also multiples of 15 -/
def count_factors : ℕ :=
  (Finset.filter (λ x => x ∣ 180 ∧ 15 ∣ x) (Finset.range 181)).card

theorem factor_count : count_factors = 6 := by
  sorry

end factor_count_l2125_212533


namespace cylinder_surface_area_and_volume_l2125_212515

/-- Given a cylinder with cross-sectional area M and axial section area N,
    prove its surface area and volume. -/
theorem cylinder_surface_area_and_volume (M N : ℝ) (M_pos : M > 0) (N_pos : N > 0) :
  ∃ (surface_area volume : ℝ),
    surface_area = N * Real.pi + 2 * M ∧
    volume = (N / 2) * Real.sqrt (M * Real.pi) := by
  sorry

end cylinder_surface_area_and_volume_l2125_212515


namespace fisherman_pelican_difference_l2125_212539

/-- The number of fish caught by the pelican -/
def pelican_fish : ℕ := 13

/-- The number of fish caught by the kingfisher -/
def kingfisher_fish : ℕ := pelican_fish + 7

/-- The total number of fish caught by the pelican and kingfisher -/
def total_fish : ℕ := pelican_fish + kingfisher_fish

/-- The number of fish caught by the fisherman -/
def fisherman_fish : ℕ := 3 * total_fish

theorem fisherman_pelican_difference :
  fisherman_fish - pelican_fish = 86 := by sorry

end fisherman_pelican_difference_l2125_212539


namespace count_hollow_circles_l2125_212525

/-- The length of the repeating sequence of circles -/
def sequence_length : ℕ := 24

/-- The number of hollow circles in each repetition of the sequence -/
def hollow_circles_per_sequence : ℕ := 5

/-- The total number of circles we're considering -/
def total_circles : ℕ := 2003

/-- The number of hollow circles in the first 2003 circles -/
def hollow_circles_count : ℕ := 446

theorem count_hollow_circles :
  (total_circles / sequence_length) * hollow_circles_per_sequence +
  (hollow_circles_per_sequence * (total_circles % sequence_length) / sequence_length) =
  hollow_circles_count :=
sorry

end count_hollow_circles_l2125_212525


namespace most_colored_pencils_l2125_212548

theorem most_colored_pencils (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) : 
  total = 24 →
  red = total / 4 →
  blue = red + 6 →
  yellow = total - red - blue →
  blue > red ∧ blue > yellow :=
by
  sorry

end most_colored_pencils_l2125_212548


namespace suzie_reading_rate_l2125_212528

/-- The number of pages Liza reads in an hour -/
def liza_pages_per_hour : ℕ := 20

/-- The number of additional pages Liza reads compared to Suzie in 3 hours -/
def liza_additional_pages : ℕ := 15

/-- The number of hours considered -/
def hours : ℕ := 3

/-- The number of pages Suzie reads in an hour -/
def suzie_pages_per_hour : ℕ := 15

theorem suzie_reading_rate :
  suzie_pages_per_hour = (liza_pages_per_hour * hours - liza_additional_pages) / hours :=
by sorry

end suzie_reading_rate_l2125_212528


namespace horner_rule_V₁_l2125_212509

-- Define the polynomial coefficients
def a₄ : ℝ := 3
def a₃ : ℝ := 0
def a₂ : ℝ := 2
def a₁ : ℝ := 1
def a₀ : ℝ := 4

-- Define the x value
def x : ℝ := 10

-- Define Horner's Rule first step
def V₀ : ℝ := a₄

-- Define Horner's Rule second step (V₁)
def V₁ : ℝ := V₀ * x + a₃

-- Theorem statement
theorem horner_rule_V₁ : V₁ = 32 := by
  sorry

end horner_rule_V₁_l2125_212509


namespace factorization_equality_l2125_212580

theorem factorization_equality (a b : ℝ) :
  2*a*b^2 - 6*a^2*b^2 + 4*a^3*b^2 = 2*a*b^2*(2*a - 1)*(a - 1) := by sorry

end factorization_equality_l2125_212580


namespace distribution_problem_l2125_212545

theorem distribution_problem (total : ℕ) (a b c : ℕ) : 
  total = 370 →
  total = a + b + c →
  b + c = a + 50 →
  (a : ℚ) / b = (b : ℚ) / c →
  a = 160 ∧ b = 120 ∧ c = 90 := by
  sorry

end distribution_problem_l2125_212545


namespace bowlfuls_in_box_l2125_212554

/-- Represents the number of clusters of oats in each spoonful -/
def clusters_per_spoonful : ℕ := 4

/-- Represents the number of spoonfuls in each bowl of cereal -/
def spoonfuls_per_bowl : ℕ := 25

/-- Represents the total number of clusters of oats in each box -/
def clusters_per_box : ℕ := 500

/-- Calculates the number of bowlfuls of cereal in each box -/
def bowlfuls_per_box : ℕ :=
  clusters_per_box / (clusters_per_spoonful * spoonfuls_per_bowl)

/-- Theorem stating that the number of bowlfuls of cereal in each box is 5 -/
theorem bowlfuls_in_box : bowlfuls_per_box = 5 := by
  sorry

end bowlfuls_in_box_l2125_212554


namespace equilateral_triangle_area_l2125_212508

theorem equilateral_triangle_area (p : ℝ) (h : p > 0) :
  let perimeter := 3 * p
  let side_length := perimeter / 3
  let area := (Real.sqrt 3 / 4) * side_length ^ 2
  area = (Real.sqrt 3 / 4) * p ^ 2 :=
by sorry

end equilateral_triangle_area_l2125_212508


namespace gift_cost_per_parent_l2125_212526

-- Define the given values
def total_spent : ℝ := 150
def siblings_count : ℕ := 3
def cost_per_sibling : ℝ := 30

-- Define the theorem
theorem gift_cost_per_parent :
  let spent_on_siblings := siblings_count * cost_per_sibling
  let spent_on_parents := total_spent - spent_on_siblings
  let cost_per_parent := spent_on_parents / 2
  cost_per_parent = 30 := by sorry

end gift_cost_per_parent_l2125_212526


namespace buddys_gym_class_size_l2125_212507

theorem buddys_gym_class_size (group1 : ℕ) (group2 : ℕ) 
  (h1 : group1 = 34) (h2 : group2 = 37) : group1 + group2 = 71 := by
  sorry

end buddys_gym_class_size_l2125_212507


namespace intersection_A_complement_B_l2125_212519

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | x > 1}

theorem intersection_A_complement_B : A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end intersection_A_complement_B_l2125_212519


namespace problem_solution_l2125_212556

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log x + (2 / x) * Real.exp (x - 1)

noncomputable def g (x : ℝ) : ℝ := x * Real.exp (-x) - 2 / Real.exp 1

theorem problem_solution (x : ℝ) (hx : x > 0) :
  f 1 = 2 ∧ 
  (deriv f) 1 = Real.exp 1 ∧
  (∀ y > 0, g y ≤ -1 / Real.exp 1) ∧
  (∀ y > 0, g y = -1 / Real.exp 1 → y = 1) ∧
  f x > 1 :=
sorry

end problem_solution_l2125_212556


namespace parallel_lines_c_value_l2125_212570

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_iff_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of c for which the lines y = 5x - 3 and y = (3c)x + 1 are parallel -/
theorem parallel_lines_c_value :
  (∀ x y : ℝ, y = 5 * x - 3 ↔ y = (3 * c) * x + 1) ↔ c = 5 / 3 :=
sorry

end parallel_lines_c_value_l2125_212570


namespace binomial_inequality_l2125_212512

theorem binomial_inequality (n k : ℕ) (h1 : n > k) (h2 : k > 0) :
  (1 : ℝ) / (n + 1 : ℝ) * (n^n : ℝ) / ((k^k : ℝ) * ((n - k)^(n - k) : ℝ)) <
  (n.factorial : ℝ) / ((k.factorial : ℝ) * ((n - k).factorial : ℝ)) ∧
  (n.factorial : ℝ) / ((k.factorial : ℝ) * ((n - k).factorial : ℝ)) <
  (n^n : ℝ) / ((k^k : ℝ) * ((n - k)^(n - k) : ℝ)) :=
by sorry

end binomial_inequality_l2125_212512


namespace linear_equation_condition_l2125_212573

/-- The equation (a-2)x^|a-1| + 3 = 9 is linear in x if and only if a = 0 -/
theorem linear_equation_condition (a : ℝ) : 
  (∃ b c : ℝ, ∀ x : ℝ, (a - 2) * x^(|a - 1|) + 3 = b * x + c) ↔ a = 0 :=
sorry

end linear_equation_condition_l2125_212573


namespace wireless_internet_percentage_l2125_212584

/-- The percentage of major airline companies that offer free on-board snacks -/
def snacks_percentage : ℝ := 70

/-- The greatest possible percentage of major airline companies that offer both wireless internet and free on-board snacks -/
def both_services_percentage : ℝ := 50

/-- The percentage of major airline companies that equip their planes with wireless internet access -/
def wireless_percentage : ℝ := 50

theorem wireless_internet_percentage :
  wireless_percentage = 50 :=
sorry

end wireless_internet_percentage_l2125_212584


namespace largest_digit_divisible_by_six_l2125_212585

theorem largest_digit_divisible_by_six :
  ∀ N : ℕ, N ≤ 9 → (4517 * 10 + N) % 6 = 0 → N ≤ 4 :=
by sorry

end largest_digit_divisible_by_six_l2125_212585


namespace geometric_sequence_sum_l2125_212534

/-- Given a geometric sequence {a_n} with the specified conditions, prove that a₆ + a₇ + a₈ = 32 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q)
  (h_sum1 : a 1 + a 2 + a 3 = 1)
  (h_sum2 : a 2 + a 3 + a 4 = 2) :
  a 6 + a 7 + a 8 = 32 := by
  sorry

end geometric_sequence_sum_l2125_212534


namespace ndfl_calculation_l2125_212529

/-- Calculates the total NDFL (personal income tax) on securities income -/
def calculate_ndfl (dividend_income : ℕ) (ofz_income : ℕ) (corporate_bond_income : ℕ) 
                   (shares_sold : ℕ) (sale_price_per_share : ℕ) (purchase_price_per_share : ℕ) : ℕ :=
  let dividend_tax := dividend_income * 13 / 100
  let corporate_bond_tax := corporate_bond_income * 13 / 100
  let capital_gain := shares_sold * (sale_price_per_share - purchase_price_per_share)
  let capital_gain_tax := capital_gain * 13 / 100
  dividend_tax + corporate_bond_tax + capital_gain_tax

/-- The total NDFL on securities income is 11,050 rubles -/
theorem ndfl_calculation : 
  calculate_ndfl 50000 40000 30000 100 200 150 = 11050 := by
  sorry

end ndfl_calculation_l2125_212529


namespace vasya_always_wins_l2125_212503

/-- Represents the state of the game with the number of piles -/
structure GameState :=
  (piles : ℕ)

/-- Represents a player in the game -/
inductive Player
| Petya
| Vasya

/-- Defines a single move in the game -/
def move (state : GameState) : GameState :=
  { piles := state.piles + 2 }

/-- Determines if a given state is a winning state for the current player -/
def is_winning_state (state : GameState) : Prop :=
  ∃ (n : ℕ), state.piles = 2 * n + 1

/-- The main theorem stating that Vasya (second player) always wins -/
theorem vasya_always_wins :
  ∀ (initial_state : GameState),
  initial_state.piles = 3 →
  is_winning_state (move initial_state) :=
sorry

end vasya_always_wins_l2125_212503


namespace g_of_f_3_l2125_212571

def f (x : ℝ) : ℝ := x^3 + 3

def g (x : ℝ) : ℝ := 2*x^2 + 2*x + x^3 + 1

theorem g_of_f_3 : g (f 3) = 28861 := by
  sorry

end g_of_f_3_l2125_212571


namespace constant_water_level_l2125_212583

/-- Represents a water pipe that can fill or empty a tank -/
structure Pipe where
  rate : ℚ  -- Rate of fill/empty (positive for fill, negative for empty)

/-- Represents a water tank system with multiple pipes -/
structure TankSystem where
  pipes : List Pipe

def TankSystem.netRate (system : TankSystem) : ℚ :=
  system.pipes.map (λ p => p.rate) |>.sum

theorem constant_water_level (pipeA pipeB pipeC : Pipe) 
  (hA : pipeA.rate = 1 / 15)
  (hB : pipeB.rate = -1 / 6)  -- Negative because it empties the tank
  (hC : pipeC.rate = 1 / 10) :
  TankSystem.netRate { pipes := [pipeA, pipeB, pipeC] } = 0 := by
  sorry

#check constant_water_level

end constant_water_level_l2125_212583


namespace systematic_sampling_interval_l2125_212521

/-- The sampling interval for systematic sampling. -/
def sampling_interval (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

/-- Theorem: The sampling interval for a systematic sampling of 30 students
    from a population of 1200 students is 40. -/
theorem systematic_sampling_interval :
  sampling_interval 1200 30 = 40 := by
  sorry

end systematic_sampling_interval_l2125_212521


namespace solution_satisfies_equation_l2125_212581

theorem solution_satisfies_equation :
  let x : ℝ := 1
  let y : ℝ := -1
  x - 2 * y = 3 := by sorry

end solution_satisfies_equation_l2125_212581


namespace stratified_sample_o_blood_type_l2125_212579

/-- Calculates the number of students with blood type O in a stratified sample -/
def stratifiedSampleO (totalStudents : ℕ) (oTypeStudents : ℕ) (sampleSize : ℕ) : ℕ :=
  (oTypeStudents * sampleSize) / totalStudents

/-- Theorem: In a stratified sample of 40 students from a population of 500 students, 
    where 200 students have blood type O, the number of students with blood type O 
    in the sample should be 16. -/
theorem stratified_sample_o_blood_type 
  (totalStudents : ℕ) 
  (oTypeStudents : ℕ) 
  (sampleSize : ℕ) 
  (h1 : totalStudents = 500) 
  (h2 : oTypeStudents = 200) 
  (h3 : sampleSize = 40) :
  stratifiedSampleO totalStudents oTypeStudents sampleSize = 16 := by
  sorry

#eval stratifiedSampleO 500 200 40

end stratified_sample_o_blood_type_l2125_212579


namespace quadratic_function_properties_l2125_212501

/-- A quadratic function with specific properties -/
def f (a b : ℝ) : ℝ → ℝ := λ x ↦ a * (x + 2)^2 + b

/-- The chord length intercepted by the x-axis is 2√3 -/
def chord_length (a b : ℝ) : Prop := ∃ x₁ x₂, x₁ ≠ x₂ ∧ f a b x₁ = 0 ∧ f a b x₂ = 0 ∧ |x₁ - x₂| = 2 * Real.sqrt 3

/-- The function passes through (0, 1) -/
def passes_through_origin (a b : ℝ) : Prop := f a b 0 = 1

/-- The function passes through (-2+√3, 0) -/
def passes_through_intercept (a b : ℝ) : Prop := f a b (-2 + Real.sqrt 3) = 0

theorem quadratic_function_properties :
  ∀ a b : ℝ, chord_length a b → passes_through_origin a b → passes_through_intercept a b →
  (∀ x, f a b x = (x + 2)^2 - 3) ∧
  (∀ k, k < 13/4 → ∃ x ∈ Set.Icc (-1 : ℝ) 1, f a b ((1/2 : ℝ)^x) > k) :=
sorry


end quadratic_function_properties_l2125_212501


namespace vector_v_satisfies_conditions_l2125_212513

/-- Two-dimensional vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Parametric line in 2D space -/
structure Line2D where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

def line_l : Line2D where
  x := λ t => 2 + 2*t
  y := λ t => 3 + t

def line_m : Line2D where
  x := λ s => -3 + 2*s
  y := λ s => 5 + s

def point_A : Point2D :=
  { x := line_l.x 1, y := line_l.y 1 }

def point_B : Point2D :=
  { x := line_m.x 2, y := line_m.y 2 }

def vector_BA : Vector2D :=
  { x := point_B.x - point_A.x, y := point_B.y - point_A.y }

def direction_m : Vector2D :=
  { x := 2, y := 1 }

def perpendicular_m : Vector2D :=
  { x := 1, y := -2 }

def projection (v w : Vector2D) : Vector2D :=
  { x := 0, y := 0 } -- Placeholder definition

theorem vector_v_satisfies_conditions :
  ∃ (v : Vector2D),
    v.x + v.y = 3 ∧
    ∃ (P : Point2D),
      projection vector_BA v = { x := P.x - point_A.x, y := P.y - point_A.y } ∧
      (P.x - point_A.x) * direction_m.x + (P.y - point_A.y) * direction_m.y = 0 ∧
      v = { x := 3, y := -6 } :=
sorry

end vector_v_satisfies_conditions_l2125_212513


namespace f_always_positive_l2125_212578

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 + 4*x^3 + a*x^2 - 4*x + 1

/-- Theorem stating that f(x) is always positive if and only if a > 2 -/
theorem f_always_positive (a : ℝ) : (∀ x : ℝ, f a x > 0) ↔ a > 2 := by
  sorry

end f_always_positive_l2125_212578


namespace bus_children_difference_l2125_212596

/-- Given the initial number of children on a bus, the number of children who got on,
    and the final number of children on the bus, this theorem proves that
    2 more children got on than got off. -/
theorem bus_children_difference (initial : ℕ) (got_on : ℕ) (final : ℕ)
    (h1 : initial = 28)
    (h2 : got_on = 82)
    (h3 : final = 30)
    (h4 : final = initial + got_on - (initial + got_on - final)) :
  got_on - (initial + got_on - final) = 2 :=
by sorry

end bus_children_difference_l2125_212596


namespace four_periods_required_l2125_212517

/-- The number of periods required for all students to present their projects -/
def required_periods (num_students : ℕ) (presentation_time : ℕ) (period_length : ℕ) : ℕ :=
  (num_students * presentation_time + period_length - 1) / period_length

/-- Proof that 4 periods are required for the given conditions -/
theorem four_periods_required :
  required_periods 32 5 40 = 4 := by
  sorry

end four_periods_required_l2125_212517


namespace intersection_k_value_l2125_212567

/-- Given two lines that intersect at x = 5, prove that k = 10 -/
theorem intersection_k_value (k : ℝ) : 
  (∃ y : ℝ, 3 * 5 - y = k ∧ -5 - y = -10) → k = 10 := by
  sorry

end intersection_k_value_l2125_212567


namespace horner_operations_degree_5_l2125_212537

/-- The number of operations required to evaluate a polynomial using Horner's method -/
def horner_operations (degree : ℕ) : ℕ :=
  2 * degree

/-- Theorem: The number of operations to evaluate a polynomial of degree 5 using Horner's method is 10 -/
theorem horner_operations_degree_5 :
  horner_operations 5 = 10 := by
  sorry

#eval horner_operations 5

end horner_operations_degree_5_l2125_212537


namespace potato_bag_weight_l2125_212514

theorem potato_bag_weight : ∀ w : ℝ, w = 12 / (w / 2) → w = 12 := by
  sorry

end potato_bag_weight_l2125_212514


namespace total_pizza_slices_l2125_212559

theorem total_pizza_slices : 
  let number_of_pizzas : ℕ := 17
  let slices_per_pizza : ℕ := 4
  number_of_pizzas * slices_per_pizza = 68 := by
sorry

end total_pizza_slices_l2125_212559


namespace angle_is_90_degrees_l2125_212566

def vector1 : ℝ × ℝ := (4, -3)
def vector2 : ℝ × ℝ := (6, 8)

def angle_between_vectors (v1 v2 : ℝ × ℝ) : ℝ := sorry

theorem angle_is_90_degrees :
  angle_between_vectors vector1 vector2 = 90 := by sorry

end angle_is_90_degrees_l2125_212566


namespace prop1_prop2_prop3_l2125_212588

-- Define the function f(x)
def f (p q x : ℝ) : ℝ := x * abs x + p * x + q

-- Proposition 1: When q = 0, f(x) is an odd function
theorem prop1 (p : ℝ) : 
  ∀ x : ℝ, f p 0 (-x) = -(f p 0 x) := by sorry

-- Proposition 2: The graph of y = f(x) is symmetric with respect to the point (0,q)
theorem prop2 (p q : ℝ) :
  ∀ x : ℝ, f p q x - q = -(f p q (-x) - q) := by sorry

-- Proposition 3: When p = 0 and q > 0, the equation f(x) = 0 has exactly one real root
theorem prop3 (q : ℝ) (hq : q > 0) :
  ∃! x : ℝ, f 0 q x = 0 := by sorry

end prop1_prop2_prop3_l2125_212588


namespace third_side_length_l2125_212597

/-- A triangle with known perimeter and two side lengths -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  perimeter : ℝ
  perimeter_eq : side1 + side2 + side3 = perimeter

/-- The theorem stating that for a triangle with two sides 7 and 15, and perimeter 32, the third side is 10 -/
theorem third_side_length (t : Triangle) 
    (h1 : t.side1 = 7)
    (h2 : t.side2 = 15)
    (h3 : t.perimeter = 32) : 
  t.side3 = 10 := by
  sorry


end third_side_length_l2125_212597


namespace intersection_point_l2125_212590

def L₁ (x : ℝ) : ℝ := 3 * x + 9
def L₂ (x : ℝ) : ℝ := -x + 6

def parameterization_L₁ (t : ℝ) : ℝ × ℝ := (t, 3 * t + 9)
def parameterization_L₂ (s : ℝ) : ℝ × ℝ := (s, -s + 6)

theorem intersection_point :
  ∃ (x y : ℝ), L₁ x = y ∧ L₂ x = y ∧ x = -3/4 ∧ y = 15/4 :=
by sorry

end intersection_point_l2125_212590


namespace intersection_when_m_is_two_subset_condition_l2125_212530

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}
def B : Set ℝ := {x | -4 ≤ x ∧ x ≤ 2}

-- Theorem 1: When m = 2, A ∩ B = [1, 2]
theorem intersection_when_m_is_two : 
  A 2 ∩ B = {x | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem 2: A ⊆ (A ∩ B) if and only if -2 ≤ m ≤ 1/2
theorem subset_condition (m : ℝ) : 
  A m ⊆ (A m ∩ B) ↔ -2 ≤ m ∧ m ≤ 1/2 := by sorry

end intersection_when_m_is_two_subset_condition_l2125_212530


namespace jellybean_count_jellybean_count_proof_l2125_212592

theorem jellybean_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun black green orange total =>
    (black = 8) →
    (green = black + 2) →
    (orange = green - 1) →
    (total = black + green + orange) →
    (total = 27)

-- The proof is omitted
theorem jellybean_count_proof : jellybean_count 8 10 9 27 := by sorry

end jellybean_count_jellybean_count_proof_l2125_212592


namespace total_lifting_capacity_is_250_l2125_212500

/-- Calculates the new combined total lifting capacity given initial weights and increases -/
def new_total_lifting_capacity (initial_clean_and_jerk : ℝ) (initial_snatch : ℝ) : ℝ :=
  (2 * initial_clean_and_jerk) + (initial_snatch * 1.8)

/-- Proves that the new combined total lifting capacity is 250 kg -/
theorem total_lifting_capacity_is_250 :
  new_total_lifting_capacity 80 50 = 250 := by
  sorry

end total_lifting_capacity_is_250_l2125_212500


namespace team_win_percentage_l2125_212505

theorem team_win_percentage (first_games : ℕ) (first_wins : ℕ) (remaining_games : ℕ) (remaining_wins : ℕ) :
  first_games = 50 →
  first_wins = 40 →
  remaining_games = 40 →
  remaining_wins = 23 →
  (first_wins + remaining_wins : ℚ) / (first_games + remaining_games : ℚ) = 7 / 10 := by
  sorry

end team_win_percentage_l2125_212505


namespace jiaotong_primary_school_students_l2125_212563

theorem jiaotong_primary_school_students (b g : ℕ) : 
  b = 7 * g ∧ b = g + 900 → b + g = 1200 := by
  sorry

end jiaotong_primary_school_students_l2125_212563


namespace polynomial_identity_l2125_212540

/-- For any real numbers a, b, and c, 
    a(b - c)^4 + b(c - a)^4 + c(a - b)^4 = (a - b)(b - c)(c - a)(a + b + c) -/
theorem polynomial_identity (a b c : ℝ) : 
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 = (a - b) * (b - c) * (c - a) * (a + b + c) := by
  sorry

end polynomial_identity_l2125_212540


namespace nineteenth_triangular_number_l2125_212574

/-- The n-th triangular number -/
def triangular_number (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

/-- The 19th triangular number is 210 -/
theorem nineteenth_triangular_number : triangular_number 19 = 210 := by
  sorry

end nineteenth_triangular_number_l2125_212574


namespace rental_company_properties_l2125_212551

/-- Represents the rental company's car rental scenario. -/
structure RentalCompany where
  totalCars : ℕ := 100
  initialRent : ℕ := 3000
  rentIncrement : ℕ := 50
  rentedCarMaintenance : ℕ := 150
  nonRentedCarMaintenance : ℕ := 50

/-- Calculates the number of cars rented given a specific rent. -/
def carsRented (company : RentalCompany) (rent : ℕ) : ℕ :=
  company.totalCars - (rent - company.initialRent) / company.rentIncrement

/-- Calculates the monthly revenue given a specific rent. -/
def monthlyRevenue (company : RentalCompany) (rent : ℕ) : ℕ :=
  let rented := carsRented company rent
  rent * rented - company.rentedCarMaintenance * rented - 
    company.nonRentedCarMaintenance * (company.totalCars - rented)

/-- Theorem stating the properties of the rental company scenario. -/
theorem rental_company_properties (company : RentalCompany) : 
  carsRented company 3600 = 88 ∧ 
  (∃ (maxRent : ℕ), maxRent = 4050 ∧ 
    (∀ (rent : ℕ), monthlyRevenue company rent ≤ monthlyRevenue company maxRent) ∧
    monthlyRevenue company maxRent = 307050) := by
  sorry


end rental_company_properties_l2125_212551


namespace perpendicular_to_plane_false_l2125_212565

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)  -- line is subset of plane
variable (perp : Line → Line → Prop)     -- line is perpendicular to line
variable (perpPlane : Line → Plane → Prop)  -- line is perpendicular to plane

-- State the theorem
theorem perpendicular_to_plane_false
  (l m n : Line) (α : Plane)
  (h1 : subset m α)
  (h2 : subset n α)
  (h3 : perp l m)
  (h4 : perp l n) :
  ¬ (perpPlane l α) :=
sorry

end perpendicular_to_plane_false_l2125_212565


namespace least_positive_integer_congruence_l2125_212599

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 5419 : ℤ) ≡ 3789 [ZMOD 15] ∧
  ∀ (y : ℕ), y > 0 ∧ (y + 5419 : ℤ) ≡ 3789 [ZMOD 15] → x ≤ y :=
by sorry

end least_positive_integer_congruence_l2125_212599


namespace sin_75_plus_sin_15_l2125_212520

theorem sin_75_plus_sin_15 : Real.sin (75 * π / 180) + Real.sin (15 * π / 180) = Real.sqrt 6 / 2 := by
  sorry

end sin_75_plus_sin_15_l2125_212520


namespace second_term_is_plus_minus_one_l2125_212511

/-- A geometric sequence with a_1 = 1/5 and a_3 = 5 -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1/5 ∧ a 3 = 5 ∧ ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The second term of the geometric sequence is either 1 or -1 -/
theorem second_term_is_plus_minus_one (a : ℕ → ℚ) (h : geometric_sequence a) :
  a 2 = 1 ∨ a 2 = -1 :=
sorry

end second_term_is_plus_minus_one_l2125_212511


namespace percentage_problem_l2125_212549

theorem percentage_problem (x y : ℝ) (P : ℝ) 
  (h1 : 0.7 * (x - y) = (P / 100) * (x + y))
  (h2 : y = 0.4 * x) : 
  P = 30 := by
sorry

end percentage_problem_l2125_212549


namespace connie_tickets_l2125_212557

def ticket_distribution (total_tickets : ℕ) : Prop :=
  let koala := total_tickets * 20 / 100
  let earbuds := 30
  let car := earbuds * 2
  let bracelets := total_tickets * 15 / 100
  let remaining := total_tickets - (koala + earbuds + car + bracelets)
  let poster := (remaining * 4) / 7
  let keychain := (remaining * 3) / 7
  koala = 100 ∧ 
  earbuds = 30 ∧ 
  car = 60 ∧ 
  bracelets = 75 ∧ 
  poster = 135 ∧ 
  keychain = 100 ∧
  koala + earbuds + car + bracelets + poster + keychain = total_tickets

theorem connie_tickets : ticket_distribution 500 := by
  sorry

end connie_tickets_l2125_212557


namespace lattice_points_on_hyperbola_l2125_212562

theorem lattice_points_on_hyperbola :
  let equation := fun (x y : ℤ) => x^2 - y^2 = 1500^2
  (∑' p : ℤ × ℤ, if equation p.1 p.2 then 1 else 0) = 90 :=
sorry

end lattice_points_on_hyperbola_l2125_212562


namespace minimum_yellow_balls_l2125_212544

theorem minimum_yellow_balls
  (g : ℕ) -- number of green balls
  (y : ℕ) -- number of yellow balls
  (o : ℕ) -- number of orange balls
  (h1 : o ≥ g / 3)  -- orange balls at least one-third of green balls
  (h2 : o ≤ y / 4)  -- orange balls at most one-fourth of yellow balls
  (h3 : g + o ≥ 75) -- combined green and orange balls at least 75
  : y ≥ 76 := by
  sorry

#check minimum_yellow_balls

end minimum_yellow_balls_l2125_212544


namespace fixed_point_line_l2125_212502

/-- Given a line that always passes through a fixed point, prove that the line
    passing through this fixed point and the origin has the equation y = 2x -/
theorem fixed_point_line (a : ℝ) : 
  (∃ (x₀ y₀ : ℝ), ∀ (x y : ℝ), a * x + y + a + 2 = 0 → x = x₀ ∧ y = y₀) → 
  ∃ (m : ℝ), ∀ (x y : ℝ), 
    (a * x₀ + y₀ + a + 2 = 0 ∧ 
     y - y₀ = m * (x - x₀) ∧ 
     0 - y₀ = m * (0 - x₀)) → 
    y = 2 * x :=
sorry

end fixed_point_line_l2125_212502


namespace function_symmetry_l2125_212541

/-- A function f : ℝ → ℝ is symmetric with respect to the point (a, b) if f(x) + f(2a - x) = 2b for all x ∈ ℝ -/
def SymmetricAboutPoint (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f x + f (2 * a - x) = 2 * b

/-- The function property given in the problem -/
def FunctionProperty (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -f (-x)

theorem function_symmetry (f : ℝ → ℝ) (h : FunctionProperty f) :
  SymmetricAboutPoint f 1 0 := by
  sorry


end function_symmetry_l2125_212541


namespace reflection_problem_l2125_212523

/-- Reflection of a point across a line --/
def reflect (x y m b : ℝ) : ℝ × ℝ := sorry

/-- The problem statement --/
theorem reflection_problem (m b : ℝ) :
  reflect (-4) 2 m b = (6, 0) → m + b = 1 := by sorry

end reflection_problem_l2125_212523


namespace salary_sum_proof_l2125_212568

/-- Given 5 people with an average salary of 8600 and one person's salary of 5000,
    prove that the sum of the other 4 people's salaries is 38000 -/
theorem salary_sum_proof (average_salary : ℕ) (num_people : ℕ) (one_salary : ℕ) 
  (h1 : average_salary = 8600)
  (h2 : num_people = 5)
  (h3 : one_salary = 5000) :
  average_salary * num_people - one_salary = 38000 := by
  sorry

#check salary_sum_proof

end salary_sum_proof_l2125_212568


namespace complement_A_in_U_l2125_212577

def U : Set ℝ := {x | x < 2}
def A : Set ℝ := {x | x^2 < x}

theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | x ≤ 0 ∨ (1 ≤ x ∧ x < 2)} := by sorry

end complement_A_in_U_l2125_212577


namespace roadster_paving_cement_usage_l2125_212532

/-- The amount of cement used for Lexi's street in tons -/
def lexi_cement : ℝ := 10

/-- The amount of cement used for Tess's street in tons -/
def tess_cement : ℝ := 5.1

/-- The total amount of cement used by Roadster's Paving Company in tons -/
def total_cement : ℝ := lexi_cement + tess_cement

theorem roadster_paving_cement_usage :
  total_cement = 15.1 := by sorry

end roadster_paving_cement_usage_l2125_212532


namespace cubic_root_sum_product_l2125_212531

theorem cubic_root_sum_product (p q r : ℂ) : 
  (5 * p^3 - 10 * p^2 + 17 * p - 7 = 0) →
  (5 * q^3 - 10 * q^2 + 17 * q - 7 = 0) →
  (5 * r^3 - 10 * r^2 + 17 * r - 7 = 0) →
  p * q + q * r + r * p = 17 / 5 := by
sorry

end cubic_root_sum_product_l2125_212531


namespace fixed_charge_is_45_l2125_212527

/-- Represents Chris's internet bill structure and usage -/
structure InternetBill where
  fixed_charge : ℝ  -- Fixed monthly charge for 100 GB
  over_charge_per_gb : ℝ  -- Charge per GB over 100 GB limit
  total_bill : ℝ  -- Total bill amount
  gb_over_limit : ℝ  -- Number of GB over the 100 GB limit

/-- Theorem stating that given the conditions, the fixed monthly charge is $45 -/
theorem fixed_charge_is_45 (bill : InternetBill) 
  (h1 : bill.over_charge_per_gb = 0.25)
  (h2 : bill.total_bill = 65)
  (h3 : bill.gb_over_limit = 80) : 
  bill.fixed_charge = 45 := by
  sorry

#check fixed_charge_is_45

end fixed_charge_is_45_l2125_212527


namespace sandy_book_purchase_l2125_212516

theorem sandy_book_purchase (books_shop1 : ℕ) (cost_shop1 : ℕ) (cost_shop2 : ℕ) (avg_price : ℚ) :
  books_shop1 = 65 →
  cost_shop1 = 1280 →
  cost_shop2 = 880 →
  avg_price = 18 →
  ∃ (books_shop2 : ℕ), 
    (books_shop1 + books_shop2) * avg_price = cost_shop1 + cost_shop2 ∧
    books_shop2 = 55 :=
by sorry

end sandy_book_purchase_l2125_212516


namespace root_equation_value_l2125_212522

theorem root_equation_value (m : ℝ) : m^2 - m - 110 = 0 → (m - 1)^2 + m = 111 := by
  sorry

end root_equation_value_l2125_212522


namespace discount_store_purchase_solution_l2125_212569

/-- Represents the purchase scenario at the discount store -/
structure DiscountStorePurchase where
  totalItems : ℕ
  itemsAt9Yuan : ℕ
  totalCost : ℕ

/-- Theorem stating the number of items priced at 9 yuan -/
theorem discount_store_purchase_solution :
  ∀ (purchase : DiscountStorePurchase),
    purchase.totalItems % 2 = 0 ∧
    purchase.totalCost = 172 ∧
    purchase.totalCost = 8 * (purchase.totalItems - purchase.itemsAt9Yuan) + 9 * purchase.itemsAt9Yuan →
    purchase.itemsAt9Yuan = 12 := by
  sorry

end discount_store_purchase_solution_l2125_212569


namespace two_digit_number_division_l2125_212550

theorem two_digit_number_division (x y : ℕ) : 
  (1 ≤ x) ∧ (x ≤ 9) ∧ (0 ≤ y) ∧ (y ≤ 9) →
  (10 * x + y) / (x + y) = 7 ∧ (10 * x + y) % (x + y) = 6 →
  (10 * x + y = 62) ∨ (10 * x + y = 83) :=
by sorry

end two_digit_number_division_l2125_212550


namespace circle_center_sum_l2125_212558

/-- For a circle with equation x^2 + y^2 = 6x + 8y - 15, if (h, k) is its center, then h + k = 7 -/
theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 6*x + 8*y - 15 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - (6*h + 8*k - 15))) →
  h + k = 7 := by
sorry

end circle_center_sum_l2125_212558


namespace sin_130_equals_sin_50_l2125_212536

theorem sin_130_equals_sin_50 : Real.sin (130 * π / 180) = Real.sin (50 * π / 180) := by
  sorry

end sin_130_equals_sin_50_l2125_212536


namespace isabellas_hair_growth_l2125_212564

/-- Given the initial length of Isabella's hair, the amount it grew, and the final length,
    prove that the initial length plus the growth equals the final length. -/
theorem isabellas_hair_growth (initial_length growth final_length : ℝ) 
    (h1 : growth = 6)
    (h2 : final_length = 24)
    (h3 : initial_length + growth = final_length) : 
  initial_length = 18 := by
  sorry

end isabellas_hair_growth_l2125_212564


namespace sum_of_perimeters_theorem_l2125_212535

/-- The sum of perimeters of all polygons in the sequence formed by repeatedly
    joining mid-points of an n-sided regular polygon with initial side length 60 cm. -/
def sum_of_perimeters (n : ℕ) : ℝ :=
  n * 120

/-- Theorem: The sum of perimeters of all polygons in the sequence formed by repeatedly
    joining mid-points of an n-sided regular polygon with initial side length 60 cm
    is equal to n * 120 cm. -/
theorem sum_of_perimeters_theorem (n : ℕ) (h : n > 0) :
  let initial_side_length : ℝ := 60
  let perimeter_sequence : ℕ → ℝ := λ k => n * (initial_side_length / 2^(k - 1))
  (∑' k, perimeter_sequence k) = sum_of_perimeters n :=
by
  sorry

end sum_of_perimeters_theorem_l2125_212535
