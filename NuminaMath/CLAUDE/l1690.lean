import Mathlib

namespace NUMINAMATH_CALUDE_cylinder_prism_height_equality_l1690_169020

/-- The height of a cylinder is equal to the height of a rectangular prism 
    when they have the same volume and base area. -/
theorem cylinder_prism_height_equality 
  (V : ℝ) -- Volume of both shapes
  (A : ℝ) -- Base area of both shapes
  (h_cylinder : ℝ) -- Height of the cylinder
  (h_prism : ℝ) -- Height of the rectangular prism
  (h_cylinder_def : h_cylinder = V / A) -- Definition of cylinder height
  (h_prism_def : h_prism = V / A) -- Definition of prism height
  : h_cylinder = h_prism := by
  sorry

end NUMINAMATH_CALUDE_cylinder_prism_height_equality_l1690_169020


namespace NUMINAMATH_CALUDE_arithmetic_problem_l1690_169016

theorem arithmetic_problem : (36 / (8 + 2 - 3)) * 7 = 36 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_problem_l1690_169016


namespace NUMINAMATH_CALUDE_x_difference_is_22_l1690_169039

theorem x_difference_is_22 (x : ℝ) (h : (x + 3)^2 / (3*x + 65) = 2) :
  ∃ (x₁ x₂ : ℝ), ((x₁ + 3)^2 / (3*x₁ + 65) = 2) ∧
                 ((x₂ + 3)^2 / (3*x₂ + 65) = 2) ∧
                 (x₁ ≠ x₂) ∧
                 (x₁ - x₂ = 22 ∨ x₂ - x₁ = 22) :=
by sorry

end NUMINAMATH_CALUDE_x_difference_is_22_l1690_169039


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l1690_169082

/-- A quadratic function passing through three given points -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_function_theorem :
  ∃ (a b c : ℝ),
    (QuadraticFunction a b c (-2) = 9) ∧
    (QuadraticFunction a b c 0 = 3) ∧
    (QuadraticFunction a b c 4 = 3) ∧
    (∀ x, QuadraticFunction a b c x = (1/2) * x^2 - 2 * x + 3) ∧
    (let vertex_x := -b / (2*a);
     let vertex_y := QuadraticFunction a b c vertex_x;
     vertex_x = 2 ∧ vertex_y = 1) ∧
    (∀ m : ℝ,
      let y₁ := QuadraticFunction a b c m;
      let y₂ := QuadraticFunction a b c (m+1);
      (m < 3/2 → y₁ > y₂) ∧
      (m = 3/2 → y₁ = y₂) ∧
      (m > 3/2 → y₁ < y₂)) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_theorem_l1690_169082


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l1690_169077

/-- Given a parabola with equation y = (1/m)x^2 where m < 0, 
    its focus has coordinates (0, m/4) -/
theorem parabola_focus_coordinates (m : ℝ) (hm : m < 0) :
  let parabola := {(x, y) : ℝ × ℝ | y = (1/m) * x^2}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (0, m/4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l1690_169077


namespace NUMINAMATH_CALUDE_negative_product_implies_odd_negatives_l1690_169089

theorem negative_product_implies_odd_negatives (a b c : ℝ) : 
  a * b * c < 0 → (a < 0 ∧ b < 0 ∧ c < 0) ∨ (a < 0 ∧ b > 0 ∧ c > 0) ∨ 
                   (a > 0 ∧ b < 0 ∧ c > 0) ∨ (a > 0 ∧ b > 0 ∧ c < 0) := by
  sorry

end NUMINAMATH_CALUDE_negative_product_implies_odd_negatives_l1690_169089


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1690_169078

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {1, 2, 3, 4}

-- Define set B
def B : Set Nat := {1, 3, 5}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ (A ∩ B)) = {2, 4, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1690_169078


namespace NUMINAMATH_CALUDE_problem_solution_l1690_169034

theorem problem_solution (x y : ℚ) (hx : x = 2/3) (hy : y = 3/2) :
  (3/4 : ℚ) * x^4 * y^5 = 9/8 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1690_169034


namespace NUMINAMATH_CALUDE_symmetric_function_domain_l1690_169079

/-- A function with either odd or even symmetry -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∨ (∀ x, f x = -f (-x))

/-- The theorem stating that if a symmetric function is defined on [3-a, 5], then a = -2 -/
theorem symmetric_function_domain (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc (3 - a) 5, f x ≠ 0 → True) →
  SymmetricFunction f →
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_domain_l1690_169079


namespace NUMINAMATH_CALUDE_cylinder_volume_relation_l1690_169094

/-- Given two cylinders A and B, where the radius of A equals the height of B,
    and the height of A equals the radius of B, if the volume of A is three times
    the volume of B, then the volume of A can be expressed as 9πh^3,
    where h is the height of A. -/
theorem cylinder_volume_relation (h r : ℝ) : 
  h > 0 → r > 0 → 
  (π * r^2 * h) = 3 * (π * h^2 * r) → 
  ∃ (N : ℝ), π * r^2 * h = N * π * h^3 ∧ N = 9 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_relation_l1690_169094


namespace NUMINAMATH_CALUDE_investment_calculation_l1690_169061

/-- Calculates the total investment given share details and dividend income -/
def calculate_investment (face_value : ℚ) (quoted_price : ℚ) (dividend_rate : ℚ) (annual_income : ℚ) : ℚ :=
  let dividend_per_share := (dividend_rate / 100) * face_value
  let number_of_shares := annual_income / dividend_per_share
  number_of_shares * quoted_price

/-- Theorem stating that the investment is 4940 given the problem conditions -/
theorem investment_calculation :
  calculate_investment 10 9.5 14 728 = 4940 := by
  sorry

#eval calculate_investment 10 9.5 14 728

end NUMINAMATH_CALUDE_investment_calculation_l1690_169061


namespace NUMINAMATH_CALUDE_remainder_7547_div_11_l1690_169074

theorem remainder_7547_div_11 : 7547 % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7547_div_11_l1690_169074


namespace NUMINAMATH_CALUDE_annas_vegetable_patch_area_l1690_169013

/-- Represents a rectangular enclosure with fence posts -/
structure FencedRectangle where
  total_posts : ℕ
  post_spacing : ℝ
  long_side_post_ratio : ℕ

/-- Calculates the area of a fenced rectangle -/
def calculate_area (fence : FencedRectangle) : ℝ :=
  let short_side_posts := (fence.total_posts + 4) / (2 * (fence.long_side_post_ratio + 1))
  let long_side_posts := fence.long_side_post_ratio * short_side_posts
  let short_side_length := (short_side_posts - 1) * fence.post_spacing
  let long_side_length := (long_side_posts - 1) * fence.post_spacing
  short_side_length * long_side_length

/-- Theorem stating that the area of Anna's vegetable patch is 144 square meters -/
theorem annas_vegetable_patch_area :
  let fence := FencedRectangle.mk 24 3 3
  calculate_area fence = 144 := by sorry

end NUMINAMATH_CALUDE_annas_vegetable_patch_area_l1690_169013


namespace NUMINAMATH_CALUDE_friends_to_movies_l1690_169023

theorem friends_to_movies (total_friends : ℕ) (cant_go : ℕ) (can_go : ℕ) 
  (h1 : total_friends = 15)
  (h2 : cant_go = 7)
  (h3 : can_go = total_friends - cant_go) :
  can_go = 8 := by
  sorry

end NUMINAMATH_CALUDE_friends_to_movies_l1690_169023


namespace NUMINAMATH_CALUDE_pencil_sharpening_l1690_169008

/-- Given a pencil that is shortened from 22 inches to 18 inches over two days
    with equal amounts sharpened each day, the amount sharpened per day is 2 inches. -/
theorem pencil_sharpening (initial_length : ℝ) (final_length : ℝ) (days : ℕ)
  (h1 : initial_length = 22)
  (h2 : final_length = 18)
  (h3 : days = 2) :
  (initial_length - final_length) / days = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencil_sharpening_l1690_169008


namespace NUMINAMATH_CALUDE_painting_price_theorem_l1690_169058

theorem painting_price_theorem (total_cost : ℕ) (price : ℕ) (quantity : ℕ) :
  total_cost = 104 →
  price > 0 →
  quantity * price = total_cost →
  10 < quantity →
  quantity < 60 →
  (price = 2 ∨ price = 4 ∨ price = 8) :=
by sorry

end NUMINAMATH_CALUDE_painting_price_theorem_l1690_169058


namespace NUMINAMATH_CALUDE_all_terms_irrational_l1690_169025

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Define the property of √2 and √3 being in the sequence
def sqrt2_sqrt3_in_sequence (a : ℕ → ℝ) : Prop :=
  ∃ m n : ℕ, a m = Real.sqrt 2 ∧ a n = Real.sqrt 3

-- Theorem statement
theorem all_terms_irrational
  (a : ℕ → ℝ)
  (h1 : is_arithmetic_sequence a)
  (h2 : sqrt2_sqrt3_in_sequence a) :
  ∀ n : ℕ, Irrational (a n) :=
sorry

end NUMINAMATH_CALUDE_all_terms_irrational_l1690_169025


namespace NUMINAMATH_CALUDE_ball_distribution_probability_ratio_l1690_169046

theorem ball_distribution_probability_ratio :
  let total_balls : ℕ := 25
  let num_bins : ℕ := 5
  let prob_6_7_4_4_4 := (Nat.choose num_bins 2 * Nat.choose total_balls 6 * Nat.choose 19 7 * 
                         Nat.choose 12 4 * Nat.choose 8 4 * Nat.choose 4 4) / 
                        (num_bins ^ total_balls : ℚ)
  let prob_5_5_5_5_5 := (Nat.choose total_balls 5 * Nat.choose 20 5 * Nat.choose 15 5 * 
                         Nat.choose 10 5 * Nat.choose 5 5) / 
                        (num_bins ^ total_balls : ℚ)
  prob_6_7_4_4_4 / prob_5_5_5_5_5 = 
    (10 * Nat.choose total_balls 6 * Nat.choose 19 7 * Nat.choose 12 4 * Nat.choose 8 4 * Nat.choose 4 4) / 
    (Nat.choose total_balls 5 * Nat.choose 20 5 * Nat.choose 15 5 * Nat.choose 10 5 * Nat.choose 5 5)
  := by sorry

end NUMINAMATH_CALUDE_ball_distribution_probability_ratio_l1690_169046


namespace NUMINAMATH_CALUDE_min_value_xy_expression_l1690_169033

theorem min_value_xy_expression :
  (∀ x y : ℝ, (x*y - 2)^2 + (x - y)^2 ≥ 0) ∧
  (∃ x y : ℝ, (x*y - 2)^2 + (x - y)^2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_xy_expression_l1690_169033


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_odd_integers_l1690_169044

theorem sum_of_five_consecutive_odd_integers (n : ℤ) : 
  (n + (n + 8) = 156) → (n + (n + 2) + (n + 4) + (n + 6) + (n + 8) = 390) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_odd_integers_l1690_169044


namespace NUMINAMATH_CALUDE_sin_value_from_tan_cos_l1690_169072

theorem sin_value_from_tan_cos (θ : Real) 
  (h1 : 6 * Real.tan θ = 4 * Real.cos θ) 
  (h2 : π < θ) (h3 : θ < 2 * π) : 
  Real.sin θ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_value_from_tan_cos_l1690_169072


namespace NUMINAMATH_CALUDE_bug_prob_after_7_steps_l1690_169068

/-- Probability of a bug being at vertex A after n steps on a regular tetrahedron -/
def prob_at_A (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | k + 1 => 1/3 * (1 - prob_at_A k)

/-- The probability of being at vertex A after 7 steps is 182/729 -/
theorem bug_prob_after_7_steps :
  prob_at_A 7 = 182 / 729 :=
by sorry

end NUMINAMATH_CALUDE_bug_prob_after_7_steps_l1690_169068


namespace NUMINAMATH_CALUDE_triangle_area_and_angle_l1690_169007

-- Define the triangle ABC
def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Add any necessary conditions for a valid triangle
  true

-- Define the dot product of two 2D vectors
def dot_product (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  x₁ * x₂ + y₁ * y₂

-- Define parallel vectors
def parallel (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * y₂ = x₂ * y₁

theorem triangle_area_and_angle (A B C : ℝ) (a b c : ℝ) :
  triangle A B C a b c →
  Real.cos C = 3/10 →
  dot_product c 0 (-a) 0 = 9/2 →
  parallel (2 * Real.sin B) (-Real.sqrt 3) (Real.cos (2 * B)) (1 - 2 * (Real.sin (B/2))^2) →
  (1/2 * a * b * Real.sin C = (3 * Real.sqrt 91)/4) ∧ B = 5*π/6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_and_angle_l1690_169007


namespace NUMINAMATH_CALUDE_pau_total_chicken_l1690_169086

def kobe_order : ℕ := 5

def pau_order (kobe : ℕ) : ℕ := 2 * kobe

def total_pau_order (initial : ℕ) : ℕ := 2 * initial

theorem pau_total_chicken :
  total_pau_order (pau_order kobe_order) = 20 := by
  sorry

end NUMINAMATH_CALUDE_pau_total_chicken_l1690_169086


namespace NUMINAMATH_CALUDE_vector_problem_l1690_169084

/-- Custom vector operation ⊗ -/
def vector_op (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 * b.1, a.2 * b.2)

/-- Theorem statement -/
theorem vector_problem (p q : ℝ × ℝ) 
  (h1 : p = (1, 2)) 
  (h2 : vector_op p q = (-3, -4)) : 
  q = (-3, -2) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l1690_169084


namespace NUMINAMATH_CALUDE_min_value_squared_differences_l1690_169062

theorem min_value_squared_differences (a b c d : ℝ) 
  (h1 : a * b = 3) 
  (h2 : c + 3 * d = 0) : 
  (a - c)^2 + (b - d)^2 ≥ 18/5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_differences_l1690_169062


namespace NUMINAMATH_CALUDE_correct_ranking_l1690_169099

-- Define the set of friends
inductive Friend : Type
| Amy : Friend
| Bill : Friend
| Celine : Friend

-- Define the age relation
def older_than : Friend → Friend → Prop := sorry

-- Define the statements
def statement_I : Prop := older_than Friend.Bill Friend.Amy ∧ older_than Friend.Bill Friend.Celine
def statement_II : Prop := ¬(older_than Friend.Amy Friend.Bill ∧ older_than Friend.Amy Friend.Celine)
def statement_III : Prop := ¬(older_than Friend.Amy Friend.Celine ∧ older_than Friend.Bill Friend.Celine)

-- Define the theorem
theorem correct_ranking :
  -- Conditions
  (∀ (x y : Friend), x ≠ y → (older_than x y ∨ older_than y x)) →
  (∀ (x y z : Friend), older_than x y → older_than y z → older_than x z) →
  (statement_I ∨ statement_II ∨ statement_III) →
  (¬statement_I ∨ ¬statement_II) →
  (¬statement_I ∨ ¬statement_III) →
  (¬statement_II ∨ ¬statement_III) →
  -- Conclusion
  older_than Friend.Amy Friend.Celine ∧ older_than Friend.Celine Friend.Bill :=
by sorry

end NUMINAMATH_CALUDE_correct_ranking_l1690_169099


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1690_169002

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the point that lies on the hyperbola
def point_on_hyperbola (a b : ℝ) : Prop :=
  hyperbola a b (Real.sqrt 6) (Real.sqrt 3)

-- Define the focus of the hyperbola
def focus (a b : ℝ) : Prop :=
  hyperbola a b (-Real.sqrt 6) 0

-- Define the intersection line
def intersection_line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 2

-- Theorem statement
theorem hyperbola_properties :
  ∀ a b : ℝ,
  point_on_hyperbola a b →
  focus a b →
  (∀ x y : ℝ, hyperbola a b x y ↔ x^2 - y^2 = 3) ∧
  (∀ k : ℝ, (∃ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    hyperbola a b x₁ y₁ ∧ hyperbola a b x₂ y₂ ∧
    intersection_line k x₁ y₁ ∧ intersection_line k x₂ y₂)
    ↔ -Real.sqrt (21 / 9) < k ∧ k < -1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1690_169002


namespace NUMINAMATH_CALUDE_probability_in_standard_deck_l1690_169053

/-- Represents a standard deck of 52 cards -/
structure Deck :=
  (total_cards : Nat)
  (diamonds : Nat)
  (spades : Nat)
  (hearts : Nat)

/-- The probability of drawing a diamond, then a spade, then a heart from a standard deck -/
def probability_diamond_spade_heart (d : Deck) : Rat :=
  (d.diamonds : Rat) / d.total_cards *
  (d.spades : Rat) / (d.total_cards - 1) *
  (d.hearts : Rat) / (d.total_cards - 2)

/-- A standard deck of 52 cards -/
def standard_deck : Deck :=
  { total_cards := 52
  , diamonds := 13
  , spades := 13
  , hearts := 13 }

theorem probability_in_standard_deck :
  probability_diamond_spade_heart standard_deck = 13 / 780 := by
  sorry

end NUMINAMATH_CALUDE_probability_in_standard_deck_l1690_169053


namespace NUMINAMATH_CALUDE_product_abcd_l1690_169083

theorem product_abcd (a b c d : ℚ) : 
  (2*a + 4*b + 6*c + 8*d = 48) →
  (4*(d+c) = b) →
  (4*b + 2*c = a) →
  (c + 1 = d) →
  (a * b * c * d = -319603200 / 10503489) := by
sorry

end NUMINAMATH_CALUDE_product_abcd_l1690_169083


namespace NUMINAMATH_CALUDE_meal_price_calculation_l1690_169006

/-- Calculate the entire price of a meal given the costs and tip percentage --/
theorem meal_price_calculation 
  (appetizer_cost : ℚ)
  (entree_cost : ℚ)
  (num_entrees : ℕ)
  (dessert_cost : ℚ)
  (tip_percentage : ℚ)
  (h1 : appetizer_cost = 9)
  (h2 : entree_cost = 20)
  (h3 : num_entrees = 2)
  (h4 : dessert_cost = 11)
  (h5 : tip_percentage = 30 / 100) :
  appetizer_cost + num_entrees * entree_cost + dessert_cost + 
  (appetizer_cost + num_entrees * entree_cost + dessert_cost) * tip_percentage = 78 := by
  sorry

end NUMINAMATH_CALUDE_meal_price_calculation_l1690_169006


namespace NUMINAMATH_CALUDE_marble_remainder_l1690_169021

theorem marble_remainder (r p : ℕ) (hr : r % 8 = 5) (hp : p % 8 = 7) : 
  (r + p) % 8 = 4 := by
sorry

end NUMINAMATH_CALUDE_marble_remainder_l1690_169021


namespace NUMINAMATH_CALUDE_committee_probability_l1690_169017

def total_members : ℕ := 30
def boys : ℕ := 12
def girls : ℕ := 18
def committee_size : ℕ := 6

theorem committee_probability :
  let total_ways := Nat.choose total_members committee_size
  let all_boys := Nat.choose boys committee_size
  let all_girls := Nat.choose girls committee_size
  let favorable_ways := total_ways - (all_boys + all_girls)
  (favorable_ways : ℚ) / total_ways = 574287 / 593775 := by sorry

end NUMINAMATH_CALUDE_committee_probability_l1690_169017


namespace NUMINAMATH_CALUDE_exists_universal_shape_l1690_169066

/-- Represents a tetrimino --/
structure Tetrimino where
  cells : Finset (ℤ × ℤ)
  cell_count : cells.card = 4

/-- Represents the five types of tetriminoes --/
inductive TetriminoType
  | O
  | I
  | L
  | T
  | Z

/-- A shape is a set of cells in the plane --/
def Shape := Finset (ℤ × ℤ)

/-- Rotation of a tetrimino --/
def rotate (t : Tetrimino) : Tetrimino := sorry

/-- Check if a shape can be composed using only one type of tetrimino --/
def canComposeWithType (s : Shape) (type : TetriminoType) : Prop := sorry

/-- The main theorem --/
theorem exists_universal_shape :
  ∃ (s : Shape), ∀ (type : TetriminoType), canComposeWithType s type := by sorry

end NUMINAMATH_CALUDE_exists_universal_shape_l1690_169066


namespace NUMINAMATH_CALUDE_complement_M_union_N_when_a_is_2_M_union_N_equals_M_iff_a_in_range_l1690_169030

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x ≤ 10}
def N (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2*a + 1}

-- Theorem for the first part of the problem
theorem complement_M_union_N_when_a_is_2 :
  (Set.univ \ M) ∪ N 2 = {x | x > 5 ∨ x < -2 ∨ (1 ≤ x ∧ x ≤ 5)} := by sorry

-- Theorem for the second part of the problem
theorem M_union_N_equals_M_iff_a_in_range (a : ℝ) :
  M ∪ N a = M ↔ a < -1 ∨ (-1 ≤ a ∧ a ≤ 2) := by sorry

end NUMINAMATH_CALUDE_complement_M_union_N_when_a_is_2_M_union_N_equals_M_iff_a_in_range_l1690_169030


namespace NUMINAMATH_CALUDE_intersection_point_is_solution_l1690_169015

/-- The intersection point of two lines -/
def intersection_point : ℝ × ℝ := (2, 3)

/-- First line equation -/
def line1 (x y : ℝ) : Prop := 10 * x - 5 * y = 5

/-- Second line equation -/
def line2 (x y : ℝ) : Prop := 8 * x + 2 * y = 22

theorem intersection_point_is_solution :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → x' = x ∧ y' = y :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_is_solution_l1690_169015


namespace NUMINAMATH_CALUDE_jellybean_average_proof_l1690_169069

/-- Proves that the initial average number of jellybeans per bag was 117,
    given the conditions of the problem. -/
theorem jellybean_average_proof 
  (initial_bags : ℕ) 
  (new_bag_jellybeans : ℕ) 
  (average_increase : ℕ) 
  (h1 : initial_bags = 34)
  (h2 : new_bag_jellybeans = 362)
  (h3 : average_increase = 7) :
  ∃ (initial_average : ℕ),
    (initial_average * initial_bags + new_bag_jellybeans) / (initial_bags + 1) = 
    initial_average + average_increase ∧ 
    initial_average = 117 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_average_proof_l1690_169069


namespace NUMINAMATH_CALUDE_f_properties_imply_b_range_l1690_169010

-- Define the function f
noncomputable def f (b : ℝ) : ℝ → ℝ := fun x =>
  if 0 < x ∧ x < 2 then Real.log (x^2 - x + b) else 0  -- placeholder for other x values

-- State the theorem
theorem f_properties_imply_b_range :
  ∀ b : ℝ,
  (∀ x : ℝ, f b (-x) = -(f b x)) →  -- f is odd
  (∀ x : ℝ, f b (x + 4) = f b x) →  -- f has period 4
  (∀ x : ℝ, 0 < x → x < 2 → f b x = Real.log (x^2 - x + b)) →  -- f definition for x ∈ (0, 2)
  (∃ x₁ x₂ x₃ x₄ x₅ : ℝ, -2 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅ ∧ x₅ ≤ 2 ∧
    f b x₁ = 0 ∧ f b x₂ = 0 ∧ f b x₃ = 0 ∧ f b x₄ = 0 ∧ f b x₅ = 0) →  -- 5 zero points in [-2, 2]
  ((1/4 < b ∧ b ≤ 1) ∨ b = 5/4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_imply_b_range_l1690_169010


namespace NUMINAMATH_CALUDE_expression_equals_percentage_of_y_l1690_169063

theorem expression_equals_percentage_of_y (y d : ℝ) : 
  y > 0 → 
  (7 * y) / 20 + (3 * y) / d = 0.6499999999999999 * y → 
  d = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_percentage_of_y_l1690_169063


namespace NUMINAMATH_CALUDE_triangle_side_length_l1690_169040

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < pi ∧ 0 < B ∧ B < pi ∧ 0 < C ∧ C < pi ∧
  A + B + C = pi →
  -- Given conditions
  a = 2 →
  B = pi / 3 →
  b = Real.sqrt 7 →
  -- Conclusion
  c = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1690_169040


namespace NUMINAMATH_CALUDE_walnut_trees_remaining_l1690_169019

/-- The number of walnut trees remaining after removal -/
def remaining_trees (initial : ℕ) (removed : ℕ) : ℕ :=
  initial - removed

theorem walnut_trees_remaining : remaining_trees 6 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_remaining_l1690_169019


namespace NUMINAMATH_CALUDE_island_inhabitants_l1690_169045

theorem island_inhabitants (total : Nat) (blue_eyed : Nat) (brown_eyed : Nat) : 
  total = 100 →
  blue_eyed + brown_eyed = total →
  (blue_eyed * brown_eyed * 2 > (total * (total - 1)) / 2) →
  (∀ (x : Nat), x ≤ blue_eyed → x ≤ brown_eyed → x * (total - x) ≤ blue_eyed * brown_eyed) →
  46 ≤ brown_eyed ∧ brown_eyed ≤ 54 := by
  sorry

end NUMINAMATH_CALUDE_island_inhabitants_l1690_169045


namespace NUMINAMATH_CALUDE_inequalities_always_true_l1690_169048

theorem inequalities_always_true (x y a b : ℝ) (h1 : x > y) (h2 : a > b) :
  (a + x > b + y) ∧ (x - b > y - a) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_always_true_l1690_169048


namespace NUMINAMATH_CALUDE_monic_polynomial_property_l1690_169085

def is_monic_polynomial_with_properties (p : ℝ → ℝ) : Prop :=
  (∀ x, ∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆, p x = x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) ∧
  (∀ i : Fin 8, p i = i)

theorem monic_polynomial_property (p : ℝ → ℝ) 
  (h : is_monic_polynomial_with_properties p) : p 8 = 40328 := by
  sorry

end NUMINAMATH_CALUDE_monic_polynomial_property_l1690_169085


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l1690_169067

-- Define the circle M
def circle_M (m n γ : ℝ) (x y : ℝ) : Prop :=
  (x - m)^2 + (y - n)^2 = γ^2

-- Define point N
def point_N : ℝ × ℝ := (1, 0)

-- Define the conditions for points P, Q, and G
def conditions (m n γ : ℝ) (P Q G : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  let (gx, gy) := G
  circle_M m n γ px py ∧
  (∃ t : ℝ, Q = point_N + t • (P - point_N)) ∧
  (∃ s : ℝ, G = P + s • (P - (m, n))) ∧
  (P - point_N) = 2 • (Q - point_N) ∧
  (G - Q) • (P - point_N) = 0

-- Define the trajectory of point G
def trajectory_G (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Theorem statement
theorem trajectory_is_ellipse :
  ∀ (P Q G : ℝ × ℝ),
    conditions (-1) 0 4 P Q G →
    ∃ (x y : ℝ), G = (x, y) ∧ trajectory_G x y :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l1690_169067


namespace NUMINAMATH_CALUDE_braden_winnings_l1690_169031

/-- Calculates the total amount in Braden's money box after winning a bet -/
def total_amount_after_bet (initial_amount : ℕ) (bet_multiplier : ℕ) : ℕ :=
  initial_amount + bet_multiplier * initial_amount

/-- Theorem stating that given the initial conditions, Braden's final amount is $1200 -/
theorem braden_winnings :
  let initial_amount := 400
  let bet_multiplier := 2
  total_amount_after_bet initial_amount bet_multiplier = 1200 := by
  sorry

end NUMINAMATH_CALUDE_braden_winnings_l1690_169031


namespace NUMINAMATH_CALUDE_emily_necklaces_l1690_169093

def beads_per_necklace : ℕ := 8
def total_beads : ℕ := 16

theorem emily_necklaces :
  total_beads / beads_per_necklace = 2 := by
  sorry

end NUMINAMATH_CALUDE_emily_necklaces_l1690_169093


namespace NUMINAMATH_CALUDE_seven_is_unique_solution_l1690_169018

/-- Product of all prime numbers less than n -/
def n_question_mark (n : ℕ) : ℕ :=
  (Finset.filter Nat.Prime (Finset.range n)).prod id

/-- The theorem stating that 7 is the only solution -/
theorem seven_is_unique_solution :
  ∃! (n : ℕ), n > 3 ∧ n_question_mark n = 2 * n + 16 :=
sorry

end NUMINAMATH_CALUDE_seven_is_unique_solution_l1690_169018


namespace NUMINAMATH_CALUDE_jasmine_remaining_money_l1690_169009

/-- Calculates the remaining amount after spending on fruits --/
def remaining_amount (initial : ℝ) (spent : ℝ) : ℝ :=
  initial - spent

/-- Theorem: The remaining amount after spending $15.00 from an initial $100.00 is $85.00 --/
theorem jasmine_remaining_money :
  remaining_amount 100 15 = 85 := by
  sorry

end NUMINAMATH_CALUDE_jasmine_remaining_money_l1690_169009


namespace NUMINAMATH_CALUDE_watch_loss_percentage_l1690_169047

theorem watch_loss_percentage (CP : ℝ) (SP : ℝ) : 
  CP = 1357.142857142857 →
  SP + 190 = CP * (1 + 4 / 100) →
  (CP - SP) / CP * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_watch_loss_percentage_l1690_169047


namespace NUMINAMATH_CALUDE_permutation_problem_l1690_169049

theorem permutation_problem (n : ℕ) : n * (n - 1) = 132 → n = 12 := by sorry

end NUMINAMATH_CALUDE_permutation_problem_l1690_169049


namespace NUMINAMATH_CALUDE_local_face_value_difference_l1690_169087

def numeral : ℕ := 96348621

theorem local_face_value_difference :
  let digit : ℕ := 8
  let position : ℕ := 5  -- 1-indexed from right, so 8 is in the 5th position
  let local_value : ℕ := digit * (10 ^ (position - 1))
  let face_value : ℕ := digit
  local_value - face_value = 79992 :=
by sorry

end NUMINAMATH_CALUDE_local_face_value_difference_l1690_169087


namespace NUMINAMATH_CALUDE_cubic_roots_geometric_progression_l1690_169098

/-- 
A cubic polynomial with coefficients a, b, and c has roots that form 
a geometric progression if and only if a^3 * c = b^3.
-/
theorem cubic_roots_geometric_progression 
  (a b c : ℝ) : 
  (∃ x y z : ℝ, (x^3 + a*x^2 + b*x + c = 0) ∧ 
                (y^3 + a*y^2 + b*y + c = 0) ∧ 
                (z^3 + a*z^2 + b*z + c = 0) ∧ 
                (y^2 = x*z)) ↔ 
  (a^3 * c = b^3) := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_geometric_progression_l1690_169098


namespace NUMINAMATH_CALUDE_f_sum_l1690_169005

/-- A function satisfying the given properties -/
def f (x : ℝ) : ℝ := sorry

/-- f is an odd function -/
axiom f_odd (x : ℝ) : f (-x) = -f x

/-- f(t) = f(1-t) for all t ∈ ℝ -/
axiom f_symmetry (t : ℝ) : f t = f (1 - t)

/-- f(x) = -x² for x ∈ [0, 1/2] -/
axiom f_def (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1/2) : f x = -x^2

/-- The main theorem to prove -/
theorem f_sum : f 3 + f (-3/2) = -1/4 := by sorry

end NUMINAMATH_CALUDE_f_sum_l1690_169005


namespace NUMINAMATH_CALUDE_arithmetic_mean_sqrt2_l1690_169051

theorem arithmetic_mean_sqrt2 (a b : ℝ) : 
  a = 1 / (Real.sqrt 2 + 1) → 
  b = 1 / (Real.sqrt 2 - 1) → 
  (a + b) / 2 = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_sqrt2_l1690_169051


namespace NUMINAMATH_CALUDE_extended_parallelepiped_volume_calculation_l1690_169080

/-- The volume of the set of points inside or within two units of a rectangular parallelepiped with dimensions 2 by 3 by 4 units -/
def extended_parallelepiped_volume : ℝ := sorry

/-- The dimensions of the rectangular parallelepiped -/
def parallelepiped_dimensions : Fin 3 → ℝ
| 0 => 2
| 1 => 3
| 2 => 4
| _ => 0

/-- The extension distance around the parallelepiped -/
def extension_distance : ℝ := 2

theorem extended_parallelepiped_volume_calculation :
  extended_parallelepiped_volume = (384 + 140 * Real.pi) / 3 := by sorry

end NUMINAMATH_CALUDE_extended_parallelepiped_volume_calculation_l1690_169080


namespace NUMINAMATH_CALUDE_graph_is_parabola_l1690_169054

-- Define the quadratic function
def f (x : ℝ) : ℝ := 3 * (x - 2)^2 + 6

-- Theorem stating that the graph of f is a parabola
theorem graph_is_parabola : 
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c :=
sorry

end NUMINAMATH_CALUDE_graph_is_parabola_l1690_169054


namespace NUMINAMATH_CALUDE_apple_slices_equality_l1690_169057

/-- Represents the number of slices in an apple -/
structure Apple :=
  (slices : ℕ)

/-- Represents the amount of apple eaten -/
def eaten (a : Apple) (s : ℕ) : ℚ :=
  s / a.slices

theorem apple_slices_equality (yeongchan minhyuk : Apple) 
  (h1 : yeongchan.slices = 3)
  (h2 : minhyuk.slices = 12) :
  eaten yeongchan 1 = eaten minhyuk 4 :=
by sorry

end NUMINAMATH_CALUDE_apple_slices_equality_l1690_169057


namespace NUMINAMATH_CALUDE_hash_five_neg_one_l1690_169041

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * (y + 2) + x * y

-- Theorem statement
theorem hash_five_neg_one : hash 5 (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_hash_five_neg_one_l1690_169041


namespace NUMINAMATH_CALUDE_congruence_problem_l1690_169070

theorem congruence_problem : ∃! n : ℤ, 0 ≤ n ∧ n < 31 ∧ -527 ≡ n [ZMOD 31] ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1690_169070


namespace NUMINAMATH_CALUDE_compare_fractions_compare_specific_fractions_l1690_169012

theorem compare_fractions (a b : ℝ) (h1 : 3 * a > b) (h2 : b > 0) : a / b > (a + 1) / (b + 3) := by
  sorry

theorem compare_specific_fractions : (23 : ℝ) / 68 < 22 / 65 := by
  sorry

end NUMINAMATH_CALUDE_compare_fractions_compare_specific_fractions_l1690_169012


namespace NUMINAMATH_CALUDE_square_division_impossible_l1690_169091

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in a 2D plane -/
structure Square where
  sideLength : ℝ
  center : Point

/-- Represents a division of a square by two internal points -/
structure SquareDivision where
  square : Square
  point1 : Point
  point2 : Point

/-- Checks if a point is inside a square -/
def isPointInsideSquare (s : Square) (p : Point) : Prop :=
  abs (p.x - s.center.x) ≤ s.sideLength / 2 ∧ abs (p.y - s.center.y) ≤ s.sideLength / 2

/-- Checks if a square division results in 9 equal parts -/
def isDividedIntoNineEqualParts (sd : SquareDivision) : Prop :=
  ∃ (areas : Finset ℝ), areas.card = 9 ∧ 
  (∀ a ∈ areas, a = sd.square.sideLength^2 / 9) ∧
  (isPointInsideSquare sd.square sd.point1) ∧
  (isPointInsideSquare sd.square sd.point2)

/-- Theorem stating that it's impossible to divide a square into 9 equal parts
    by connecting two internal points to its vertices -/
theorem square_division_impossible :
  ¬ ∃ (sd : SquareDivision), isDividedIntoNineEqualParts sd :=
sorry

end NUMINAMATH_CALUDE_square_division_impossible_l1690_169091


namespace NUMINAMATH_CALUDE_cells_reach_1540_in_9_hours_l1690_169032

/-- The number of cells after n hours -/
def cell_count (n : ℕ) : ℕ :=
  3 * 2^(n-1) + 4

/-- The theorem stating that it takes 9 hours to reach 1540 cells -/
theorem cells_reach_1540_in_9_hours :
  cell_count 9 = 1540 ∧
  ∀ k : ℕ, k < 9 → cell_count k < 1540 :=
by sorry

end NUMINAMATH_CALUDE_cells_reach_1540_in_9_hours_l1690_169032


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1690_169027

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides. -/
theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 18 → n * exterior_angle = 360 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1690_169027


namespace NUMINAMATH_CALUDE_intersection_implies_m_equals_four_l1690_169065

def A : Set ℝ := {x | x ≥ 1}
def B (m : ℝ) : Set ℝ := {x | x^2 - m*x ≤ 0}

theorem intersection_implies_m_equals_four (m : ℝ) : A ∩ B m = {x | 1 ≤ x ∧ x ≤ 4} → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_equals_four_l1690_169065


namespace NUMINAMATH_CALUDE_expression_upper_bound_l1690_169043

theorem expression_upper_bound :
  ∃ (U : ℕ), 
    (∃ (S : Finset ℤ), 
      (Finset.card S = 50) ∧ 
      (∀ n ∈ S, 1 < 4*n + 7 ∧ 4*n + 7 < U) ∧
      (∀ U' < U, ∃ n ∈ S, 4*n + 7 ≥ U')) →
    U = 204 :=
by sorry

end NUMINAMATH_CALUDE_expression_upper_bound_l1690_169043


namespace NUMINAMATH_CALUDE_three_number_ratio_problem_l1690_169055

theorem three_number_ratio_problem (x y z : ℝ) 
  (h_sum : x + y + z = 120)
  (h_ratio1 : x / y = 3 / 4)
  (h_ratio2 : y / z = 5 / 7)
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0) :
  y = 800 / 21 := by
  sorry

end NUMINAMATH_CALUDE_three_number_ratio_problem_l1690_169055


namespace NUMINAMATH_CALUDE_base4_calculation_l1690_169024

/-- Converts a base 4 number to base 10 --/
def base4_to_base10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base 10 number to base 4 --/
def base10_to_base4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Theorem: In base 4, (1230₄ + 32₄) ÷ 13₄ = 111₄ --/
theorem base4_calculation : 
  let a := base4_to_base10 [0, 3, 2, 1]  -- 1230₄
  let b := base4_to_base10 [2, 3]        -- 32₄
  let c := base4_to_base10 [3, 1]        -- 13₄
  base10_to_base4 ((a + b) / c) = [1, 1, 1] := by
  sorry


end NUMINAMATH_CALUDE_base4_calculation_l1690_169024


namespace NUMINAMATH_CALUDE_car_airplane_energy_consumption_ratio_l1690_169075

theorem car_airplane_energy_consumption_ratio :
  ∀ (maglev airplane car : ℝ),
    maglev > 0 → airplane > 0 → car > 0 →
    maglev = (1/3) * airplane →
    maglev = 0.7 * car →
    car = (10/21) * airplane :=
by sorry

end NUMINAMATH_CALUDE_car_airplane_energy_consumption_ratio_l1690_169075


namespace NUMINAMATH_CALUDE_race_time_difference_l1690_169059

/-- Race parameters and result -/
theorem race_time_difference
  (malcolm_speed : ℝ) -- Malcolm's speed in minutes per mile
  (joshua_speed : ℝ)  -- Joshua's speed in minutes per mile
  (race_distance : ℝ) -- Race distance in miles
  (h1 : malcolm_speed = 7)
  (h2 : joshua_speed = 8)
  (h3 : race_distance = 12) :
  joshua_speed * race_distance - malcolm_speed * race_distance = 12 := by
  sorry

end NUMINAMATH_CALUDE_race_time_difference_l1690_169059


namespace NUMINAMATH_CALUDE_smallest_k_for_digit_sum_945_l1690_169073

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The number formed by k repetitions of the digit 7 -/
def repeated_sevens (k : ℕ) : ℕ := sorry

theorem smallest_k_for_digit_sum_945 :
  (∀ k < 312, sum_of_digits (7 * repeated_sevens k) < 945) ∧
  sum_of_digits (7 * repeated_sevens 312) = 945 := by sorry

end NUMINAMATH_CALUDE_smallest_k_for_digit_sum_945_l1690_169073


namespace NUMINAMATH_CALUDE_min_intersection_at_45_deg_l1690_169060

/-- A square in 2D space -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- Rotation of a square around its center -/
def rotate_square (s : Square) (angle : ℝ) : Square :=
  { s with }  -- The internal structure remains the same after rotation

/-- The area of intersection between two squares -/
def intersection_area (s1 s2 : Square) : ℝ := sorry

/-- Theorem: The area of intersection between a square and its rotated version is minimized at 45 degrees -/
theorem min_intersection_at_45_deg (s : Square) :
  ∀ x : ℝ, 0 ≤ x → x ≤ 2 * π →
    intersection_area s (rotate_square s (π/4)) ≤ intersection_area s (rotate_square s x) := by
  sorry

#check min_intersection_at_45_deg

end NUMINAMATH_CALUDE_min_intersection_at_45_deg_l1690_169060


namespace NUMINAMATH_CALUDE_library_growth_rate_l1690_169056

theorem library_growth_rate (initial_collection : ℝ) (final_collection : ℝ) (years : ℝ) :
  initial_collection = 100000 →
  final_collection = 144000 →
  years = 2 →
  let growth_rate := ((final_collection / initial_collection) ^ (1 / years)) - 1
  growth_rate = 0.2 := by
sorry

end NUMINAMATH_CALUDE_library_growth_rate_l1690_169056


namespace NUMINAMATH_CALUDE_age_multiplier_proof_l1690_169035

theorem age_multiplier_proof (matt_age john_age : ℕ) (h1 : matt_age = 41) (h2 : john_age = 11) :
  ∃ x : ℚ, matt_age = x * john_age - 3 :=
by
  sorry

end NUMINAMATH_CALUDE_age_multiplier_proof_l1690_169035


namespace NUMINAMATH_CALUDE_range_of_m_l1690_169028

theorem range_of_m (x : ℝ) (m : ℝ) : 
  (∃ x ∈ Set.Icc (-1) 1, x^2 - x - (m + 1) = 0) →
  m ∈ Set.Icc (-5/4) 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1690_169028


namespace NUMINAMATH_CALUDE_correct_system_of_equations_l1690_169038

theorem correct_system_of_equations :
  ∀ (x y : ℕ),
  (x + y = 12) →
  (4 * x + 3 * y = 40) →
  (∀ (a b : ℕ), (a + b = 12 ∧ 4 * a + 3 * b = 40) → (a = x ∧ b = y)) :=
by sorry

end NUMINAMATH_CALUDE_correct_system_of_equations_l1690_169038


namespace NUMINAMATH_CALUDE_complex_equation_l1690_169090

theorem complex_equation (z : ℂ) (h : z = 1 + I) : z^2 + 2 / z = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_l1690_169090


namespace NUMINAMATH_CALUDE_complex_equality_l1690_169000

theorem complex_equality (a b : ℂ) : a - b = 0 → a = b := by sorry

end NUMINAMATH_CALUDE_complex_equality_l1690_169000


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l1690_169036

/-- A cone with three spheres inside it -/
structure ConeWithSpheres where
  R : ℝ  -- radius of the base of the cone
  r : ℝ  -- radius of each sphere
  slant_height : ℝ  -- slant height of the cone
  spheres_touch : Bool  -- spheres touch each other externally
  two_touch_base : Bool  -- two spheres touch the lateral surface and base
  third_in_plane : Bool  -- third sphere touches at a point in the same plane as centers

/-- The properties of the cone and spheres arrangement -/
def cone_sphere_properties (c : ConeWithSpheres) : Prop :=
  c.R > 0 ∧ c.r > 0 ∧
  c.slant_height = 2 * c.R ∧  -- base diameter equals slant height
  c.spheres_touch ∧
  c.two_touch_base ∧
  c.third_in_plane

/-- The theorem stating the ratio of cone base radius to sphere radius -/
theorem cone_sphere_ratio (c : ConeWithSpheres) 
  (h : cone_sphere_properties c) : c.R / c.r = 5 / 4 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l1690_169036


namespace NUMINAMATH_CALUDE_pencil_rows_l1690_169081

theorem pencil_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h1 : total_pencils = 154) (h2 : pencils_per_row = 11) :
  total_pencils / pencils_per_row = 14 := by
sorry

end NUMINAMATH_CALUDE_pencil_rows_l1690_169081


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l1690_169050

theorem geometric_mean_minimum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_gm : Real.sqrt (a * b) = 2) :
  5 ≤ (b + 1/a) + (a + 1/b) ∧ 
  (∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ Real.sqrt (a₀ * b₀) = 2 ∧ (b₀ + 1/a₀) + (a₀ + 1/b₀) = 5) :=
sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l1690_169050


namespace NUMINAMATH_CALUDE_triangle_cos_C_l1690_169071

/-- Given a triangle ABC where b = 2a and b sin A = c sin C, prove that cos C = 3/4 -/
theorem triangle_cos_C (a b c : ℝ) (A B C : ℝ) : 
  b = 2 * a → b * Real.sin A = c * Real.sin C → Real.cos C = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cos_C_l1690_169071


namespace NUMINAMATH_CALUDE_meal_preparation_assignments_l1690_169022

theorem meal_preparation_assignments (n : ℕ) (h : n = 6) :
  (n.choose 3) * ((n - 3).choose 1) * ((n - 4).choose 2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_meal_preparation_assignments_l1690_169022


namespace NUMINAMATH_CALUDE_remainder_problem_l1690_169064

theorem remainder_problem (m : ℤ) (h : m % 288 = 47) : m % 24 = 23 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1690_169064


namespace NUMINAMATH_CALUDE_green_valley_olympiad_l1690_169042

theorem green_valley_olympiad (j s : ℕ) (hj : j > 0) (hs : s > 0) 
  (h_participation : (1 : ℚ) / 3 * j = (2 : ℚ) / 3 * s) : j = 2 * s :=
sorry

end NUMINAMATH_CALUDE_green_valley_olympiad_l1690_169042


namespace NUMINAMATH_CALUDE_new_student_weight_l1690_169088

/-- 
Given 5 students with an initial total weight W, 
if replacing two students weighing x and y with a new student 
causes the average weight to decrease by 8 kg, 
then the new student's weight is 40 kg less than x + y.
-/
theorem new_student_weight 
  (W : ℝ) -- Initial total weight of 5 students
  (x y : ℝ) -- Weights of the two replaced students
  (new_avg : ℝ) -- New average weight after replacement
  (h1 : new_avg = (W - x - y + (x + y - 40)) / 5) -- New average calculation
  (h2 : W / 5 - new_avg = 8) -- Average weight decrease
  : x + y - 40 = (x + y) - 40 := by sorry

end NUMINAMATH_CALUDE_new_student_weight_l1690_169088


namespace NUMINAMATH_CALUDE_dark_lord_squads_l1690_169096

/-- The number of squads needed to transport swords --/
def num_squads (total_weight : ℕ) (orcs_per_squad : ℕ) (weight_per_orc : ℕ) : ℕ :=
  total_weight / (orcs_per_squad * weight_per_orc)

/-- Proof that 10 squads are needed for the given conditions --/
theorem dark_lord_squads :
  num_squads 1200 8 15 = 10 := by
  sorry

end NUMINAMATH_CALUDE_dark_lord_squads_l1690_169096


namespace NUMINAMATH_CALUDE_derivative_y_l1690_169097

noncomputable def y (x : ℝ) : ℝ := Real.arcsin (1 / (2 * x + 3)) + 2 * Real.sqrt (x^2 + 3 * x + 2)

theorem derivative_y (x : ℝ) (h : 2 * x + 3 > 0) :
  deriv y x = (4 * Real.sqrt (x^2 + 3 * x + 2)) / (2 * x + 3) := by sorry

end NUMINAMATH_CALUDE_derivative_y_l1690_169097


namespace NUMINAMATH_CALUDE_train_speed_l1690_169037

/-- Given a train of length 125 meters crossing a bridge of length 250 meters in 30 seconds,
    its speed is 45 km/hr. -/
theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 125 →
  bridge_length = 250 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1690_169037


namespace NUMINAMATH_CALUDE_f_satisfies_equation_l1690_169092

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x + 1

-- State the theorem
theorem f_satisfies_equation : ∀ x : ℝ, 2 * f x - f (-x) = 3 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_equation_l1690_169092


namespace NUMINAMATH_CALUDE_power_mod_thirteen_l1690_169095

theorem power_mod_thirteen : (6 ^ 1234 : ℕ) % 13 = 10 := by sorry

end NUMINAMATH_CALUDE_power_mod_thirteen_l1690_169095


namespace NUMINAMATH_CALUDE_product_of_odd_negative_integers_l1690_169004

def odd_negative_integers : List ℤ := sorry

theorem product_of_odd_negative_integers :
  let product := (List.prod odd_negative_integers)
  (product < 0) ∧ (product % 10 = -5) := by
  sorry

end NUMINAMATH_CALUDE_product_of_odd_negative_integers_l1690_169004


namespace NUMINAMATH_CALUDE_baker_cakes_sold_l1690_169001

theorem baker_cakes_sold (pastries_made : ℕ) (cakes_made : ℕ) (pastries_sold : ℕ) (cakes_left : ℕ) 
  (h1 : pastries_made = 61)
  (h2 : cakes_made = 167)
  (h3 : pastries_sold = 44)
  (h4 : cakes_left = 59) :
  cakes_made - cakes_left = 108 :=
by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_sold_l1690_169001


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l1690_169029

/-- 
Given a rectangular solid with dimensions x, y, and z,
if the total surface area is 34 cm² and the total length of all edges is 28 cm,
then the length of any interior diagonal is √15 cm.
-/
theorem rectangular_solid_diagonal 
  (x y z : ℝ) 
  (h_surface_area : 2 * (x * y + y * z + z * x) = 34)
  (h_edge_length : 4 * (x + y + z) = 28) :
  Real.sqrt (x^2 + y^2 + z^2) = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l1690_169029


namespace NUMINAMATH_CALUDE_z2_magnitude_range_l1690_169026

theorem z2_magnitude_range (z₁ z₂ : ℂ) 
  (h1 : (z₁ - Complex.I) * (z₂ + Complex.I) = 1)
  (h2 : Complex.abs z₁ = Real.sqrt 2) :
  2 - Real.sqrt 2 ≤ Complex.abs z₂ ∧ Complex.abs z₂ ≤ 2 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_z2_magnitude_range_l1690_169026


namespace NUMINAMATH_CALUDE_obtuse_triangle_x_range_l1690_169076

/-- Given three line segments with lengths x^2+4, 4x, and x^2+8,
    this theorem states the range of x values that can form an obtuse triangle. -/
theorem obtuse_triangle_x_range (x : ℝ) :
  (∃ (a b c : ℝ), a = x^2 + 4 ∧ b = 4*x ∧ c = x^2 + 8 ∧
   a > 0 ∧ b > 0 ∧ c > 0 ∧
   a + b > c ∧ b + c > a ∧ a + c > b ∧
   c^2 > a^2 + b^2) ↔ 
  (1 < x ∧ x < Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_x_range_l1690_169076


namespace NUMINAMATH_CALUDE_league_games_count_l1690_169052

/-- The number of unique games played in a league season --/
def uniqueGamesInSeason (n : ℕ) (g : ℕ) : ℕ :=
  n * (n - 1) * g / 2

/-- Theorem: In a league with 30 teams, where each team plays 15 games against every other team,
    the total number of unique games played in the season is 6,525. --/
theorem league_games_count :
  uniqueGamesInSeason 30 15 = 6525 := by
  sorry

#eval uniqueGamesInSeason 30 15

end NUMINAMATH_CALUDE_league_games_count_l1690_169052


namespace NUMINAMATH_CALUDE_sum_of_squares_l1690_169011

theorem sum_of_squares (a b c : ℕ+) (h1 : a < b) (h2 : b < c)
  (h3 : (b.val * c.val - 1) % a.val = 0)
  (h4 : (a.val * c.val - 1) % b.val = 0)
  (h5 : (a.val * b.val - 1) % c.val = 0) :
  a^2 + b^2 + c^2 = 38 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1690_169011


namespace NUMINAMATH_CALUDE_number_of_subsets_A_l1690_169014

def U : Finset ℕ := {0, 1, 2}

theorem number_of_subsets_A (A : Finset ℕ) (h : U \ A = {2}) : Finset.card (Finset.powerset A) = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_subsets_A_l1690_169014


namespace NUMINAMATH_CALUDE_triangle_side_length_l1690_169003

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a + b + c = 20 →
  (1/2) * b * c * Real.sin A = 10 * Real.sqrt 3 →
  A = π / 3 →
  a = 7 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1690_169003
