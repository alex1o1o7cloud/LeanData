import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1828_182812

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + a*x + a^2 - 1 = 0 ∧ y^2 + a*y + a^2 - 1 = 0) → 
  -1 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1828_182812


namespace NUMINAMATH_CALUDE_paintbrush_cost_l1828_182891

theorem paintbrush_cost (paint_cost easel_cost albert_has albert_needs : ℚ) :
  paint_cost = 4.35 →
  easel_cost = 12.65 →
  albert_has = 6.50 →
  albert_needs = 12 →
  paint_cost + easel_cost + (albert_has + albert_needs - (paint_cost + easel_cost)) = 1.50 := by
sorry

end NUMINAMATH_CALUDE_paintbrush_cost_l1828_182891


namespace NUMINAMATH_CALUDE_pairball_playtime_l1828_182820

/-- Given a game of pairball with the following conditions:
  * There are 12 children participating.
  * Only 2 children can play at a time.
  * The game runs continuously for 120 minutes.
  * Every child has an equal amount of playtime.
  Prove that each child plays for 20 minutes. -/
theorem pairball_playtime (num_children : ℕ) (players_per_game : ℕ) (total_time : ℕ) 
  (h1 : num_children = 12)
  (h2 : players_per_game = 2)
  (h3 : total_time = 120)
  : (total_time * players_per_game) / num_children = 20 := by
  sorry

end NUMINAMATH_CALUDE_pairball_playtime_l1828_182820


namespace NUMINAMATH_CALUDE_banana_permutations_l1828_182831

-- Define the word and its properties
def word : String := "BANANA"
def word_length : Nat := 6
def b_count : Nat := 1
def a_count : Nat := 3
def n_count : Nat := 2

-- Theorem statement
theorem banana_permutations :
  (Nat.factorial word_length) / 
  (Nat.factorial b_count * Nat.factorial a_count * Nat.factorial n_count) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_l1828_182831


namespace NUMINAMATH_CALUDE_linear_equation_exponent_sum_l1828_182833

/-- If x^(a-1) - 3y^(b-2) = 7 is a linear equation in x and y, then a + b = 5 -/
theorem linear_equation_exponent_sum (a b : ℝ) : 
  (∀ x y : ℝ, ∃ m n c : ℝ, x^(a-1) - 3*y^(b-2) = m*x + n*y + c) → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_exponent_sum_l1828_182833


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1828_182844

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-1, 2}
def B : Set Int := {-1, 0, 1}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1828_182844


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_4_meaningful_l1828_182846

theorem sqrt_2x_minus_4_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 4) ↔ x ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_4_meaningful_l1828_182846


namespace NUMINAMATH_CALUDE_velocity_equal_distance_time_l1828_182819

/-- For uniform motion, the velocity that makes the distance equal to time is 1. -/
theorem velocity_equal_distance_time (s t v : ℝ) (h : s = v * t) (h2 : s = t) : v = 1 := by
  sorry

end NUMINAMATH_CALUDE_velocity_equal_distance_time_l1828_182819


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1828_182883

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^5 + 5 * x^4 - 3 * Real.sqrt 2 * x^3 + 8 * x^2 + 2 * x - 6) + 
  (-5 * x^4 + Real.sqrt 2 * x^3 - 3 * x^2 + x + 10) = 
  2 * x^5 - 2 * Real.sqrt 2 * x^3 + 5 * x^2 + 3 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1828_182883


namespace NUMINAMATH_CALUDE_complex_modulus_l1828_182892

theorem complex_modulus (z : ℂ) : z - Complex.I = 1 + Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1828_182892


namespace NUMINAMATH_CALUDE_not_equivalent_fraction_l1828_182850

theorem not_equivalent_fraction : (1 : ℚ) / 20000000 ≠ (48 : ℚ) / 1000000000 := by
  sorry

end NUMINAMATH_CALUDE_not_equivalent_fraction_l1828_182850


namespace NUMINAMATH_CALUDE_division_equality_l1828_182893

theorem division_equality : 250 / (5 + 12 * 3^2) = 250 / 113 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l1828_182893


namespace NUMINAMATH_CALUDE_middle_numbers_average_l1828_182829

theorem middle_numbers_average (a b c d : ℕ+) : 
  a < b ∧ b < c ∧ c < d ∧  -- Four different positive integers
  (a + b + c + d : ℚ) / 4 = 5 ∧  -- Average is 5
  ∀ w x y z : ℕ+, w < x ∧ x < y ∧ y < z ∧ (w + x + y + z : ℚ) / 4 = 5 → (z - w : ℤ) ≤ (d - a : ℤ) →  -- Maximum possible difference
  (b + c : ℚ) / 2 = 5/2 :=
sorry

end NUMINAMATH_CALUDE_middle_numbers_average_l1828_182829


namespace NUMINAMATH_CALUDE_coefficient_x3y2z5_in_expansion_l1828_182862

/-- The coefficient of x³y²z⁵ in the expansion of (2x+y+z)¹⁰ -/
def coefficient : ℕ :=
  2^3 * (Nat.choose 10 3) * (Nat.choose 7 2) * (Nat.choose 5 5)

/-- Theorem stating that the coefficient of x³y²z⁵ in (2x+y+z)¹⁰ is 20160 -/
theorem coefficient_x3y2z5_in_expansion : coefficient = 20160 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y2z5_in_expansion_l1828_182862


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1828_182802

theorem arithmetic_sequence_problem (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 3 = 3 →
  a 6 = 24 →
  a 9 = 45 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1828_182802


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1828_182805

theorem complex_fraction_equality : (2 : ℂ) / (1 - Complex.I) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1828_182805


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l1828_182875

-- Define the quadrilateral ABCD and points E and F
variable (A B C D E F : Point)

-- Define the properties of the quadrilateral
def is_cyclic_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the angle measure
def angle_measure (A B C : Point) : ℝ := sorry

-- Define the distance between two points
def distance (P Q : Point) : ℝ := sorry

-- Define the intersection of two rays
def ray_intersection (P Q R S : Point) : Point := sorry

-- Define the perimeter of a triangle
def triangle_perimeter (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_perimeter 
  (h_cyclic : is_cyclic_quadrilateral A B C D)
  (h_angle : angle_measure B A D = π / 3)
  (h_side1 : distance B C = 1)
  (h_side2 : distance A D = 1)
  (h_intersect1 : E = ray_intersection A B C D)
  (h_intersect2 : F = ray_intersection B C A D)
  (h_perimeter1 : ∃ n : ℕ, triangle_perimeter B C E = n)
  (h_perimeter2 : ∃ m : ℕ, triangle_perimeter C D F = m) :
  distance A B + distance B C + distance C D + distance D A = 38 / 7 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_l1828_182875


namespace NUMINAMATH_CALUDE_gondor_monday_phones_l1828_182824

/-- Represents the earnings and repair information for Gondor --/
structure GondorEarnings where
  phone_repair_cost : ℕ
  laptop_repair_cost : ℕ
  tuesday_phones : ℕ
  wednesday_laptops : ℕ
  thursday_laptops : ℕ
  total_earnings : ℕ

/-- Calculates the number of phones repaired on Monday --/
def monday_phones (g : GondorEarnings) : ℕ :=
  (g.total_earnings - (g.phone_repair_cost * g.tuesday_phones + 
   g.laptop_repair_cost * (g.wednesday_laptops + g.thursday_laptops))) / g.phone_repair_cost

/-- Theorem stating that Gondor repaired 3 phones on Monday --/
theorem gondor_monday_phones (g : GondorEarnings) 
  (h1 : g.phone_repair_cost = 10)
  (h2 : g.laptop_repair_cost = 20)
  (h3 : g.tuesday_phones = 5)
  (h4 : g.wednesday_laptops = 2)
  (h5 : g.thursday_laptops = 4)
  (h6 : g.total_earnings = 200) :
  monday_phones g = 3 := by
  sorry

end NUMINAMATH_CALUDE_gondor_monday_phones_l1828_182824


namespace NUMINAMATH_CALUDE_phone_service_cost_per_minute_l1828_182853

/-- Calculates the cost per minute for a phone service given the total bill, monthly fee, and minutes used. -/
def cost_per_minute (total_bill monthly_fee : ℚ) (minutes_used : ℕ) : ℚ :=
  (total_bill - monthly_fee) / minutes_used

/-- Theorem stating that given the specific conditions, the cost per minute is $0.12. -/
theorem phone_service_cost_per_minute :
  let total_bill : ℚ := 23.36
  let monthly_fee : ℚ := 2
  let minutes_used : ℕ := 178
  cost_per_minute total_bill monthly_fee minutes_used = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_phone_service_cost_per_minute_l1828_182853


namespace NUMINAMATH_CALUDE_greatest_common_divisor_problem_l1828_182864

theorem greatest_common_divisor_problem :
  Nat.gcd 105 (Nat.gcd 1001 (Nat.gcd 2436 (Nat.gcd 10202 49575))) = 7 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_problem_l1828_182864


namespace NUMINAMATH_CALUDE_books_pages_after_move_l1828_182867

theorem books_pages_after_move (initial_books : ℕ) (pages_per_book : ℕ) (lost_books : ℕ) : 
  initial_books = 10 → pages_per_book = 100 → lost_books = 2 →
  (initial_books - lost_books) * pages_per_book = 800 := by
  sorry

end NUMINAMATH_CALUDE_books_pages_after_move_l1828_182867


namespace NUMINAMATH_CALUDE_vector_projection_l1828_182898

/-- The projection of vector a onto vector b is -√5/5 -/
theorem vector_projection (a b : ℝ × ℝ) : 
  a = (3, 1) → b = (-2, 4) → 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_l1828_182898


namespace NUMINAMATH_CALUDE_translation_of_B_l1828_182827

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the translation function
def translate (p : Point) (v : ℝ × ℝ) : Point :=
  (p.1 + v.1, p.2 + v.2)

-- Define the given points
def A : Point := (-1, 0)
def B : Point := (1, 2)
def A₁ : Point := (2, -1)

-- Define the translation vector
def translation_vector : ℝ × ℝ := (A₁.1 - A.1, A₁.2 - A.2)

-- State the theorem
theorem translation_of_B (h : A₁ = translate A translation_vector) :
  translate B translation_vector = (4, 1) := by
  sorry


end NUMINAMATH_CALUDE_translation_of_B_l1828_182827


namespace NUMINAMATH_CALUDE_ab_value_l1828_182886

theorem ab_value (a b : ℕ+) (h1 : a + b = 30) (h2 : 3 * a * b + 5 * a = 4 * b + 180) : a * b = 29 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1828_182886


namespace NUMINAMATH_CALUDE_highest_score_can_be_less_than_16_l1828_182878

/-- Represents a team in the tournament -/
structure Team :=
  (id : Nat)
  (score : Nat)

/-- Represents the tournament -/
structure Tournament :=
  (teams : Finset Team)
  (num_teams : Nat)
  (games_played : Nat)
  (total_points : Nat)

/-- The tournament satisfies the given conditions -/
def valid_tournament (t : Tournament) : Prop :=
  t.num_teams = 16 ∧
  t.games_played = (t.num_teams * (t.num_teams - 1)) / 2 ∧
  t.total_points = 2 * t.games_played

/-- The highest score in the tournament -/
def highest_score (t : Tournament) : Nat :=
  Finset.sup t.teams (fun team => team.score)

/-- Theorem stating that it's possible for the highest score to be less than 16 -/
theorem highest_score_can_be_less_than_16 (t : Tournament) :
  valid_tournament t → ∃ (score : Nat), highest_score t < 16 :=
by
  sorry

end NUMINAMATH_CALUDE_highest_score_can_be_less_than_16_l1828_182878


namespace NUMINAMATH_CALUDE_thabo_hardcover_count_l1828_182861

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfying the given conditions -/
def thabos_books : BookCollection where
  hardcover_nonfiction := 25
  paperback_nonfiction := 45
  paperback_fiction := 90

theorem thabo_hardcover_count :
  ∀ (books : BookCollection),
    books.hardcover_nonfiction + books.paperback_nonfiction + books.paperback_fiction = 160 →
    books.paperback_nonfiction = books.hardcover_nonfiction + 20 →
    books.paperback_fiction = 2 * books.paperback_nonfiction →
    books.hardcover_nonfiction = 25 := by
  sorry

#eval thabos_books.hardcover_nonfiction

end NUMINAMATH_CALUDE_thabo_hardcover_count_l1828_182861


namespace NUMINAMATH_CALUDE_first_jump_exceeding_2000_l1828_182828

-- Define the jump sequence
def jump_sequence : ℕ → ℕ
  | 0 => 2  -- First jump (we use 0-based indexing here)
  | n + 1 => 2 * jump_sequence n + n

-- Define a function to check if a jump exceeds 2000 meters
def exceeds_2000 (n : ℕ) : Prop := jump_sequence n > 2000

-- Theorem statement
theorem first_jump_exceeding_2000 :
  (∀ m : ℕ, m < 14 → ¬(exceeds_2000 m)) ∧ exceeds_2000 14 := by sorry

end NUMINAMATH_CALUDE_first_jump_exceeding_2000_l1828_182828


namespace NUMINAMATH_CALUDE_cubic_factorization_l1828_182840

theorem cubic_factorization (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1828_182840


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1828_182888

theorem inequality_equivalence (x : ℝ) : (x + 2) * (x - 9) < 0 ↔ -2 < x ∧ x < 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1828_182888


namespace NUMINAMATH_CALUDE_S_inter_T_eq_T_l1828_182871

/-- The set of odd integers -/
def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}

/-- The set of integers of the form 4n + 1 -/
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

/-- Theorem stating that the intersection of S and T is equal to T -/
theorem S_inter_T_eq_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_S_inter_T_eq_T_l1828_182871


namespace NUMINAMATH_CALUDE_red_peaches_count_l1828_182826

theorem red_peaches_count (total_baskets : ℕ) (green_per_basket : ℕ) (total_peaches : ℕ) :
  total_baskets = 11 →
  green_per_basket = 18 →
  total_peaches = 308 →
  ∃ red_per_basket : ℕ,
    red_per_basket * total_baskets + green_per_basket * total_baskets = total_peaches ∧
    red_per_basket = 10 :=
by sorry

end NUMINAMATH_CALUDE_red_peaches_count_l1828_182826


namespace NUMINAMATH_CALUDE_ice_cream_combinations_l1828_182872

theorem ice_cream_combinations (n : ℕ) (k : ℕ) : 
  n = 5 → k = 3 → Nat.choose (n + k - 1) (k - 1) = 21 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_combinations_l1828_182872


namespace NUMINAMATH_CALUDE_spheres_in_cone_radius_l1828_182838

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Theorem stating the radius of spheres in a cone under specific conditions -/
theorem spheres_in_cone_radius (c : Cone) (s1 s2 : Sphere) : 
  c.baseRadius = 6 ∧ 
  c.height = 15 ∧ 
  s1.radius = s2.radius ∧
  -- The spheres are tangent to each other, the side, and the base of the cone
  -- (This condition is implicitly assumed in the statement)
  True →
  s1.radius = 12 * Real.sqrt 29 / 29 :=
sorry

end NUMINAMATH_CALUDE_spheres_in_cone_radius_l1828_182838


namespace NUMINAMATH_CALUDE_circle_equation_correct_l1828_182899

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  equation : ℝ → ℝ → Prop

/-- Checks if a point lies on a circle -/
def Point.liesOn (p : Point) (c : Circle) : Prop :=
  c.equation p.x p.y

/-- The circle we want to prove about -/
def ourCircle : Circle :=
  { center := { x := 2, y := -1 }
  , equation := fun x y => (x - 2)^2 + (y + 1)^2 = 2 }

/-- The theorem to prove -/
theorem circle_equation_correct :
  ourCircle.center = { x := 2, y := -1 } ∧
  Point.liesOn { x := 3, y := 0 } ourCircle :=
sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l1828_182899


namespace NUMINAMATH_CALUDE_complex_product_l1828_182804

theorem complex_product (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2) 
  (h2 : Complex.abs z₂ = 3) 
  (h3 : 3 * z₁ - 2 * z₂ = 2 - I) : 
  z₁ * z₂ = -30/13 + 72/13 * I := by
sorry

end NUMINAMATH_CALUDE_complex_product_l1828_182804


namespace NUMINAMATH_CALUDE_apple_distribution_l1828_182879

/-- Represents the number of apples Karen has at the end -/
def karens_final_apples (initial_apples : ℕ) : ℕ :=
  let after_first_transfer := initial_apples - 12
  (after_first_transfer - after_first_transfer / 2)

/-- Represents the number of apples Alphonso has at the end -/
def alphonsos_final_apples (initial_apples : ℕ) : ℕ :=
  let after_first_transfer := initial_apples + 12
  let karens_remaining := initial_apples - 12
  (after_first_transfer + karens_remaining / 2)

theorem apple_distribution (initial_apples : ℕ) 
  (h1 : initial_apples ≥ 12)
  (h2 : alphonsos_final_apples initial_apples = 4 * karens_final_apples initial_apples) :
  karens_final_apples initial_apples = 24 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l1828_182879


namespace NUMINAMATH_CALUDE_chichikov_guarantee_l1828_182842

/-- Represents a distribution of nuts into three boxes -/
def Distribution := (ℕ × ℕ × ℕ)

/-- Checks if a distribution is valid (sum is 1001) -/
def valid_distribution (d : Distribution) : Prop :=
  d.1 + d.2.1 + d.2.2 = 1001

/-- Represents the number of nuts that need to be moved for a given N -/
def nuts_to_move (d : Distribution) (N : ℕ) : ℕ :=
  sorry

/-- The maximum number of nuts that need to be moved for any N -/
def max_nuts_to_move (d : Distribution) : ℕ :=
  sorry

theorem chichikov_guarantee :
  ∀ d : Distribution, valid_distribution d →
  ∃ N : ℕ, 1 ≤ N ∧ N ≤ 1001 ∧ nuts_to_move d N ≥ 71 ∧
  ∀ M : ℕ, M > 71 → ∃ d' : Distribution, valid_distribution d' ∧
  ∀ N' : ℕ, 1 ≤ N' ∧ N' ≤ 1001 → nuts_to_move d' N' < M :=
sorry

end NUMINAMATH_CALUDE_chichikov_guarantee_l1828_182842


namespace NUMINAMATH_CALUDE_basis_properties_l1828_182807

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def is_basis (S : Set V) : Prop :=
  Submodule.span ℝ S = ⊤ ∧ LinearIndependent ℝ (fun x => x : S → V)

theorem basis_properties {a b c : V} (h : is_basis {a, b, c}) :
  is_basis {a + b, b + c, c + a} ∧
  ∀ p : V, ∃ x y z : ℝ, p = x • a + y • b + z • c :=
sorry

end NUMINAMATH_CALUDE_basis_properties_l1828_182807


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l1828_182876

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : ℕ) : ℕ := sorry

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (d : ℕ) : ℕ := sorry

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal 101001110010) = 221302 := by sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l1828_182876


namespace NUMINAMATH_CALUDE_labor_union_tree_planting_l1828_182896

theorem labor_union_tree_planting (x : ℕ) : 
  (2 * x + 21 = x * 2 + 21) ∧ 
  (3 * x - 24 = x * 3 - 24) → 
  2 * x + 21 = 3 * x - 24 := by
sorry

end NUMINAMATH_CALUDE_labor_union_tree_planting_l1828_182896


namespace NUMINAMATH_CALUDE_bead_system_eventually_repeats_l1828_182825

-- Define the bead system
structure BeadSystem where
  n : ℕ  -- number of beads
  ω : ℝ  -- angular speed
  direction : Fin n → Bool  -- true for clockwise, false for counterclockwise
  initial_position : Fin n → ℝ  -- initial angular position of each bead

-- Define the state of the system at a given time
def system_state (bs : BeadSystem) (t : ℝ) : Fin bs.n → ℝ :=
  sorry

-- Define what it means for the system to repeat its initial configuration
def repeats_initial_config (bs : BeadSystem) (t : ℝ) : Prop :=
  ∃ (perm : Equiv.Perm (Fin bs.n)),
    ∀ i, system_state bs t (perm i) = bs.initial_position i

-- State the theorem
theorem bead_system_eventually_repeats (bs : BeadSystem) :
  ∃ t > 0, repeats_initial_config bs t :=
sorry

end NUMINAMATH_CALUDE_bead_system_eventually_repeats_l1828_182825


namespace NUMINAMATH_CALUDE_domain_of_g_l1828_182868

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-3) 5

-- Define the function g
def g (x : ℝ) : ℝ := f (x + 1) + f (x - 2)

-- Define the domain of g
def domain_g : Set ℝ := Set.Icc (-1) 4

-- Theorem statement
theorem domain_of_g :
  ∀ x ∈ domain_g, (x + 1 ∈ domain_f ∧ x - 2 ∈ domain_f) ∧
  ∀ x ∉ domain_g, (x + 1 ∉ domain_f ∨ x - 2 ∉ domain_f) :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l1828_182868


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1828_182849

/-- 
Given a rectangle where:
- The long sides are three times the length of the short sides
- One short side is 80 feet long
Prove that the perimeter of the rectangle is 640 feet.
-/
theorem rectangle_perimeter (short_side : ℝ) (h1 : short_side = 80) : 
  2 * short_side + 2 * (3 * short_side) = 640 := by
  sorry

#check rectangle_perimeter

end NUMINAMATH_CALUDE_rectangle_perimeter_l1828_182849


namespace NUMINAMATH_CALUDE_rabbits_after_four_springs_l1828_182894

/-- Calculates the total number of rabbits after four breeding seasons --/
def totalRabbitsAfterFourSprings (initialBreedingRabbits : ℕ) 
  (spring1KittensPerRabbit spring1AdoptionRate : ℚ) (spring1Returns : ℕ)
  (spring2Kittens : ℕ) (spring2AdoptionRate : ℚ) (spring2Returns : ℕ)
  (spring3BreedingRabbits : ℕ) (spring3KittensPerRabbit : ℕ) (spring3AdoptionRate : ℚ) (spring3Returns : ℕ)
  (spring4BreedingRabbits : ℕ) (spring4KittensPerRabbit : ℕ) (spring4AdoptionRate : ℚ) (spring4Returns : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the total number of rabbits after four springs is 242 --/
theorem rabbits_after_four_springs : 
  totalRabbitsAfterFourSprings 10 10 (1/2) 5 60 (2/5) 10 12 8 (3/10) 3 12 6 (1/5) 2 = 242 :=
by sorry

end NUMINAMATH_CALUDE_rabbits_after_four_springs_l1828_182894


namespace NUMINAMATH_CALUDE_absolute_value_theorem_l1828_182870

theorem absolute_value_theorem (x : ℝ) (h : x < 1) : 
  |x - Real.sqrt ((x - 2)^2)| = 2 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_theorem_l1828_182870


namespace NUMINAMATH_CALUDE_equation_solution_l1828_182834

theorem equation_solution : ∃ x : ℚ, (5 + 3.5 * x = 2.1 * x - 25) ∧ (x = -150/7) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1828_182834


namespace NUMINAMATH_CALUDE_parallelogram_sum_l1828_182884

/-- A parallelogram with sides measuring 6y-2, 4x+5, 12y-10, and 8x+1 has x + y = 7/3 -/
theorem parallelogram_sum (x y : ℚ) : 
  (6 * y - 2 : ℚ) = (12 * y - 10 : ℚ) →
  (4 * x + 5 : ℚ) = (8 * x + 1 : ℚ) →
  x + y = 7/3 := by sorry

end NUMINAMATH_CALUDE_parallelogram_sum_l1828_182884


namespace NUMINAMATH_CALUDE_triangle_probability_is_2ln2_minus_1_l1828_182854

-- Define the rod breaking process
def rod_break (total_length : ℝ) : ℝ × ℝ × ℝ :=
  sorry

-- Define the condition for triangle formation
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the probability of forming a triangle
def triangle_probability : ℝ :=
  sorry

-- Theorem statement
theorem triangle_probability_is_2ln2_minus_1 :
  triangle_probability = 2 * Real.log 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_triangle_probability_is_2ln2_minus_1_l1828_182854


namespace NUMINAMATH_CALUDE_gcd_9247_4567_l1828_182882

theorem gcd_9247_4567 : Nat.gcd 9247 4567 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_9247_4567_l1828_182882


namespace NUMINAMATH_CALUDE_f_property_l1828_182880

/-- A cubic function with specific properties -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 7

/-- Theorem stating that if f(-7) = -17, then f(7) = 31 -/
theorem f_property (a b : ℝ) (h : f a b (-7) = -17) : f a b 7 = 31 := by
  sorry

end NUMINAMATH_CALUDE_f_property_l1828_182880


namespace NUMINAMATH_CALUDE_base_conversion_equality_l1828_182852

theorem base_conversion_equality (k : ℕ) : k = 7 ↔ 
  5 * 8^2 + 2 * 8^1 + 4 * 8^0 = 6 * k^2 + 6 * k^1 + 4 * k^0 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l1828_182852


namespace NUMINAMATH_CALUDE_binomial_2024_1_l1828_182857

theorem binomial_2024_1 : Nat.choose 2024 1 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_binomial_2024_1_l1828_182857


namespace NUMINAMATH_CALUDE_abs_sum_minimum_l1828_182866

theorem abs_sum_minimum (x : ℝ) : 
  |x - 4| + |x - 6| ≥ 2 ∧ ∃ y : ℝ, |y - 4| + |y - 6| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_minimum_l1828_182866


namespace NUMINAMATH_CALUDE_ring_toss_earnings_l1828_182817

/-- The ring toss game earnings problem -/
theorem ring_toss_earnings 
  (daily_earnings : ℕ) 
  (num_days : ℕ) 
  (h1 : daily_earnings = 33) 
  (h2 : num_days = 5) : 
  daily_earnings * num_days = 165 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_earnings_l1828_182817


namespace NUMINAMATH_CALUDE_jessica_allowance_l1828_182809

def weekly_allowance : ℝ := 26.67

theorem jessica_allowance (allowance : ℝ) 
  (h1 : 0.45 * allowance + 17 = 29) : 
  allowance = weekly_allowance := by
  sorry

#check jessica_allowance

end NUMINAMATH_CALUDE_jessica_allowance_l1828_182809


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1828_182877

theorem complex_equation_solution (b : ℝ) : 
  (2 - Complex.I) * (4 * Complex.I) = 4 - b * Complex.I → b = -8 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1828_182877


namespace NUMINAMATH_CALUDE_P_union_Q_eq_Q_l1828_182816

-- Define the sets P and Q
def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | x^2 - x > 0}

-- State the theorem
theorem P_union_Q_eq_Q : P ∪ Q = Q := by
  sorry

end NUMINAMATH_CALUDE_P_union_Q_eq_Q_l1828_182816


namespace NUMINAMATH_CALUDE_relationship_between_exponents_l1828_182848

theorem relationship_between_exponents 
  (a c e f : ℝ) 
  (x y z w : ℝ) 
  (h1 : a^(2*x) = c^(3*y)) 
  (h2 : a^(2*x) = e) 
  (h3 : c^(3*y) = e) 
  (h4 : c^(4*z) = a^(3*w)) 
  (h5 : c^(4*z) = f) 
  (h6 : a^(3*w) = f) 
  (h7 : a ≠ 0) 
  (h8 : c ≠ 0) 
  (h9 : e > 0) 
  (h10 : f > 0) : 
  2*w*z = x*y := by
sorry

end NUMINAMATH_CALUDE_relationship_between_exponents_l1828_182848


namespace NUMINAMATH_CALUDE_quadratic_max_min_l1828_182832

def f (x : ℝ) := x^2 - 4*x + 2

theorem quadratic_max_min :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2 : ℝ) 5, f x ≤ max ∧ min ≤ f x) ∧
    (∃ x₁ ∈ Set.Icc (-2 : ℝ) 5, f x₁ = max) ∧
    (∃ x₂ ∈ Set.Icc (-2 : ℝ) 5, f x₂ = min) ∧
    max = 14 ∧ min = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_min_l1828_182832


namespace NUMINAMATH_CALUDE_concert_audience_fraction_l1828_182811

theorem concert_audience_fraction (total_audience : ℕ) 
  (second_band_fraction : ℚ) (h1 : total_audience = 150) 
  (h2 : second_band_fraction = 2/3) : 
  1 - second_band_fraction = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_concert_audience_fraction_l1828_182811


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1828_182897

/-- A quadratic equation x^2 + 2x - c = 0 has two equal real roots if and only if c = -1 -/
theorem equal_roots_quadratic (c : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x - c = 0 ∧ (∀ y : ℝ, y^2 + 2*y - c = 0 → y = x)) ↔ c = -1 :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1828_182897


namespace NUMINAMATH_CALUDE_path_count_theorem_l1828_182851

/-- The number of paths from (0,0) to (4,3) on a 5x4 grid with exactly 7 steps -/
def number_of_paths : ℕ := 35

/-- The width of the grid -/
def grid_width : ℕ := 5

/-- The height of the grid -/
def grid_height : ℕ := 4

/-- The total number of steps in each path -/
def total_steps : ℕ := 7

/-- The number of steps to the right in each path -/
def right_steps : ℕ := 4

/-- The number of steps up in each path -/
def up_steps : ℕ := 3

theorem path_count_theorem :
  number_of_paths = Nat.choose total_steps up_steps :=
by sorry

end NUMINAMATH_CALUDE_path_count_theorem_l1828_182851


namespace NUMINAMATH_CALUDE_initial_amount_spent_l1828_182847

theorem initial_amount_spent (total_sets : ℕ) (twenty_dollar_sets : ℕ) (price_per_set : ℕ) :
  total_sets = 250 →
  twenty_dollar_sets = 178 →
  price_per_set = 20 →
  (twenty_dollar_sets * price_per_set : ℕ) = 3560 :=
by sorry

end NUMINAMATH_CALUDE_initial_amount_spent_l1828_182847


namespace NUMINAMATH_CALUDE_integer_solutions_of_inequalities_l1828_182860

theorem integer_solutions_of_inequalities :
  let S := { x : ℤ | (4 * (1 + x) : ℚ) / 3 - 1 ≤ (5 + x : ℚ) / 2 ∧
                     (x : ℚ) - 5 ≤ (3 / 2) * ((3 * x : ℚ) - 2) }
  S = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_inequalities_l1828_182860


namespace NUMINAMATH_CALUDE_luncheon_cost_theorem_l1828_182815

/-- Cost of a luncheon item -/
structure LuncheonItem where
  sandwich : ℚ
  coffee : ℚ
  pie : ℚ

/-- Calculate the total cost of a luncheon -/
def luncheonCost (item : LuncheonItem) (s c p : ℕ) : ℚ :=
  s * item.sandwich + c * item.coffee + p * item.pie

theorem luncheon_cost_theorem (item : LuncheonItem) : 
  luncheonCost item 2 5 1 = 3 ∧
  luncheonCost item 5 8 1 = 27/5 ∧
  luncheonCost item 3 4 1 = 18/5 →
  luncheonCost item 2 2 1 = 13/5 := by
sorry

#eval (13 : ℚ) / 5  -- Expected output: 2.6

end NUMINAMATH_CALUDE_luncheon_cost_theorem_l1828_182815


namespace NUMINAMATH_CALUDE_inequality_proof_l1828_182858

theorem inequality_proof (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a + b + c) * (a^2 + b^2 + c^2) ≤ 3 * (a^3 + b^3 + c^3) ∧
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1828_182858


namespace NUMINAMATH_CALUDE_special_circle_equation_l1828_182839

/-- A circle with center on y = x, passing through origin, and chord of length 2 on x-axis -/
structure SpecialCircle where
  center : ℝ × ℝ
  center_on_line : center.2 = center.1
  passes_origin : (center.1 ^ 2 + center.2 ^ 2) = 2 * center.1 ^ 2
  chord_length : ∃ x : ℝ, (x - center.1) ^ 2 + center.2 ^ 2 = 2 * center.1 ^ 2 ∧ 
                           ((x - 1) - center.1) ^ 2 + center.2 ^ 2 = 2 * center.1 ^ 2

theorem special_circle_equation (c : SpecialCircle) : 
  (∀ x y : ℝ, (x - 1) ^ 2 + (y - 1) ^ 2 = 2) ∨ 
  (∀ x y : ℝ, (x + 1) ^ 2 + (y + 1) ^ 2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_special_circle_equation_l1828_182839


namespace NUMINAMATH_CALUDE_solution_set_l1828_182823

theorem solution_set (x : ℝ) : 4 ≤ x / (2 * x - 5) ∧ x / (2 * x - 5) < 7 ↔ x ∈ Set.Ioc (5/2) (20/7) :=
  sorry

end NUMINAMATH_CALUDE_solution_set_l1828_182823


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_l1828_182895

-- Define the total number of students
def total_students : ℕ := 5

-- Define the number of male students
def male_students : ℕ := 3

-- Define the number of female students
def female_students : ℕ := 2

-- Define the number of students to be selected
def selected_students : ℕ := 2

-- Define the event "at least one male student is selected"
def at_least_one_male (selected : Finset (Fin total_students)) : Prop :=
  ∃ s ∈ selected, s.val < male_students

-- Define the event "all female students are selected"
def all_females (selected : Finset (Fin total_students)) : Prop :=
  ∀ s ∈ selected, s.val ≥ male_students

-- Theorem statement
theorem events_mutually_exclusive :
  ∀ selected : Finset (Fin total_students),
  selected.card = selected_students →
  ¬(at_least_one_male selected ∧ all_females selected) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_l1828_182895


namespace NUMINAMATH_CALUDE_book_arrangement_count_l1828_182885

def num_books : ℕ := 7
def num_identical_books : ℕ := 3

theorem book_arrangement_count : 
  (num_books.factorial) / (num_identical_books.factorial) = 840 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l1828_182885


namespace NUMINAMATH_CALUDE_blood_expiration_theorem_l1828_182800

/-- Represents a date and time -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Calculates the expiration date and time for a blood donation -/
def calculateExpirationDateTime (donationDateTime : DateTime) : DateTime :=
  sorry

/-- The number of seconds in a day -/
def secondsPerDay : ℕ := 86400

/-- The expiration time in seconds for a unit of blood -/
def bloodExpirationSeconds : ℕ := Nat.factorial 9

/-- Theorem stating that a blood donation made at 8 AM on January 15th 
    will expire on January 19th at 4:48 AM -/
theorem blood_expiration_theorem 
  (donationDateTime : DateTime)
  (h1 : donationDateTime.year = 2023)
  (h2 : donationDateTime.month = 1)
  (h3 : donationDateTime.day = 15)
  (h4 : donationDateTime.hour = 8)
  (h5 : donationDateTime.minute = 0) :
  let expirationDateTime := calculateExpirationDateTime donationDateTime
  expirationDateTime.year = 2023 ∧
  expirationDateTime.month = 1 ∧
  expirationDateTime.day = 19 ∧
  expirationDateTime.hour = 4 ∧
  expirationDateTime.minute = 48 :=
sorry

end NUMINAMATH_CALUDE_blood_expiration_theorem_l1828_182800


namespace NUMINAMATH_CALUDE_tan_240_degrees_l1828_182822

theorem tan_240_degrees : Real.tan (240 * Real.pi / 180) = Real.sqrt 3 := by
  sorry

#check tan_240_degrees

end NUMINAMATH_CALUDE_tan_240_degrees_l1828_182822


namespace NUMINAMATH_CALUDE_marc_total_spent_l1828_182803

/-- The total amount Marc spent on his purchases -/
def total_spent (model_car_price model_car_quantity paint_price paint_quantity
                 paintbrush_price paintbrush_quantity : ℕ) : ℕ :=
  model_car_price * model_car_quantity +
  paint_price * paint_quantity +
  paintbrush_price * paintbrush_quantity

/-- Theorem stating that Marc spent $160 in total -/
theorem marc_total_spent :
  total_spent 20 5 10 5 2 5 = 160 := by
  sorry

end NUMINAMATH_CALUDE_marc_total_spent_l1828_182803


namespace NUMINAMATH_CALUDE_project_hours_theorem_l1828_182810

theorem project_hours_theorem (kate_hours mark_hours pat_hours : ℕ) : 
  pat_hours = 2 * kate_hours →
  pat_hours = mark_hours / 3 →
  mark_hours = kate_hours + 100 →
  kate_hours + pat_hours + mark_hours = 180 := by
sorry

end NUMINAMATH_CALUDE_project_hours_theorem_l1828_182810


namespace NUMINAMATH_CALUDE_officer_selection_l1828_182889

theorem officer_selection (n m k l : ℕ) (hn : n = 20) (hm : m = 8) (hk : k = 10) (hl : l = 3) :
  Nat.choose n m - (Nat.choose k m + Nat.choose k 1 * Nat.choose (n - k) (m - 1) + Nat.choose k 2 * Nat.choose (n - k) (m - 2)) = 115275 :=
by sorry

end NUMINAMATH_CALUDE_officer_selection_l1828_182889


namespace NUMINAMATH_CALUDE_liars_guessing_game_theorem_l1828_182873

/-- The liar's guessing game -/
structure LiarsGuessingGame where
  k : ℕ+  -- The number of consecutive answers where at least one must be truthful
  n : ℕ+  -- The maximum size of the final guessing set

/-- A winning strategy for player B -/
def has_winning_strategy (game : LiarsGuessingGame) : Prop :=
  ∀ N : ℕ+, ∃ (strategy : ℕ+ → Finset ℕ+), 
    (∀ x : ℕ+, x ≤ N → x ∈ strategy N) ∧
    (Finset.card (strategy N) ≤ game.n)

/-- Main theorem about the liar's guessing game -/
theorem liars_guessing_game_theorem (game : LiarsGuessingGame) :
  (game.n ≥ 2^(game.k : ℕ) → has_winning_strategy game) ∧
  (∃ k : ℕ+, ∃ n : ℕ+, n ≥ (1.99 : ℝ)^(k : ℕ) ∧ 
    ¬(has_winning_strategy ⟨k, n⟩)) := by
  sorry

end NUMINAMATH_CALUDE_liars_guessing_game_theorem_l1828_182873


namespace NUMINAMATH_CALUDE_two_tangent_circles_l1828_182865

/-- The parabola y² = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (2, 0)

/-- The directrix of the parabola -/
def directrix : ℝ → ℝ := λ x => -2

/-- The point M -/
def point_M : ℝ × ℝ := (3, 3)

/-- A circle passing through two points and tangent to a line -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through_focus : dist center focus = radius
  passes_through_M : dist center point_M = radius
  tangent_to_directrix : abs (center.2 - directrix center.1) = radius

/-- The main theorem -/
theorem two_tangent_circles : 
  ∃! (circles : Finset TangentCircle), circles.card = 2 := by sorry

end NUMINAMATH_CALUDE_two_tangent_circles_l1828_182865


namespace NUMINAMATH_CALUDE_inequality_proof_l1828_182806

theorem inequality_proof (a b c : ℝ) 
  (h : a + b + c + a*b + b*c + a*c + a*b*c ≥ 7) :
  Real.sqrt (a^2 + b^2 + 2) + Real.sqrt (b^2 + c^2 + 2) + Real.sqrt (c^2 + a^2 + 2) ≥ 6 := by
sorry


end NUMINAMATH_CALUDE_inequality_proof_l1828_182806


namespace NUMINAMATH_CALUDE_back_wheel_perimeter_l1828_182836

/-- Given a front wheel with perimeter 30 that revolves 240 times, and a back wheel that
    revolves 360 times to cover the same distance, the perimeter of the back wheel is 20. -/
theorem back_wheel_perimeter (front_perimeter : ℝ) (front_revolutions : ℝ) 
  (back_revolutions : ℝ) (back_perimeter : ℝ) : 
  front_perimeter = 30 →
  front_revolutions = 240 →
  back_revolutions = 360 →
  front_perimeter * front_revolutions = back_perimeter * back_revolutions →
  back_perimeter = 20 := by
  sorry

end NUMINAMATH_CALUDE_back_wheel_perimeter_l1828_182836


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1828_182863

theorem contrapositive_equivalence :
  (∀ a b : ℝ, (a + b = 3 → a^2 + b^2 ≥ 4)) ↔
  (∀ a b : ℝ, (a^2 + b^2 < 4 → a + b ≠ 3)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1828_182863


namespace NUMINAMATH_CALUDE_inscribed_circle_triangle_perimeter_l1828_182856

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of XT, where T is the tangency point on XY -/
  xt : ℝ
  /-- The length of TY, where T is the tangency point on XY -/
  ty : ℝ

/-- Calculate the perimeter of a triangle with an inscribed circle -/
def perimeter (t : InscribedCircleTriangle) : ℝ :=
  sorry

theorem inscribed_circle_triangle_perimeter
  (t : InscribedCircleTriangle)
  (h_r : t.r = 24)
  (h_xt : t.xt = 26)
  (h_ty : t.ty = 31) :
  perimeter t = 345 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_triangle_perimeter_l1828_182856


namespace NUMINAMATH_CALUDE_max_surface_area_l1828_182874

/-- A 3D structure made of unit cubes -/
structure CubeStructure where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculate the surface area of a CubeStructure -/
def surface_area (s : CubeStructure) : ℕ :=
  2 * (s.width * s.length + s.width * s.height + s.length * s.height)

/-- The specific cube structure from the problem -/
def problem_structure : CubeStructure :=
  { width := 2, length := 4, height := 2 }

theorem max_surface_area :
  surface_area problem_structure = 48 :=
sorry

end NUMINAMATH_CALUDE_max_surface_area_l1828_182874


namespace NUMINAMATH_CALUDE_roses_cut_l1828_182818

theorem roses_cut (initial_roses final_roses : ℕ) (h1 : initial_roses = 6) (h2 : final_roses = 16) :
  final_roses - initial_roses = 10 := by
  sorry

end NUMINAMATH_CALUDE_roses_cut_l1828_182818


namespace NUMINAMATH_CALUDE_sinks_per_house_l1828_182808

/-- Given that a carpenter bought 266 sinks to cover 44 houses,
    prove that the number of sinks needed for each house is 6. -/
theorem sinks_per_house (total_sinks : ℕ) (num_houses : ℕ) 
  (h1 : total_sinks = 266) (h2 : num_houses = 44) :
  total_sinks / num_houses = 6 := by
  sorry

#check sinks_per_house

end NUMINAMATH_CALUDE_sinks_per_house_l1828_182808


namespace NUMINAMATH_CALUDE_cents_ratio_randi_to_peter_l1828_182835

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The total cents Ray has -/
def ray_total_cents : ℕ := 175

/-- The cents Ray gives to Peter -/
def cents_to_peter : ℕ := 30

/-- The number of extra nickels Randi has compared to Peter -/
def extra_nickels_randi : ℕ := 6

/-- Theorem stating the ratio of cents given to Randi vs Peter -/
theorem cents_ratio_randi_to_peter :
  let peter_nickels := cents_to_peter / nickel_value
  let randi_nickels := peter_nickels + extra_nickels_randi
  let cents_to_randi := randi_nickels * nickel_value
  (cents_to_randi : ℚ) / cents_to_peter = 2 := by sorry

end NUMINAMATH_CALUDE_cents_ratio_randi_to_peter_l1828_182835


namespace NUMINAMATH_CALUDE_boys_in_second_grade_l1828_182869

/-- The number of students in the 3rd grade -/
def third_grade : ℕ := 19

/-- The number of students in the 4th grade -/
def fourth_grade : ℕ := 2 * third_grade

/-- The number of girls in the 2nd grade -/
def second_grade_girls : ℕ := 19

/-- The total number of students across all three grades -/
def total_students : ℕ := 86

/-- The number of boys in the 2nd grade -/
def second_grade_boys : ℕ := total_students - fourth_grade - third_grade - second_grade_girls

theorem boys_in_second_grade : second_grade_boys = 10 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_second_grade_l1828_182869


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l1828_182845

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  base : ℕ
  leg : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := t.base + 2 * t.leg

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * Real.sqrt ((t.leg : ℝ) ^ 2 - ((t.base : ℝ) / 2) ^ 2) / 2

/-- Theorem: The minimum possible value of the common perimeter of two noncongruent
    integer-sided isosceles triangles with the same perimeter, same area, and base
    lengths in the ratio 8:7 is 586 -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    8 * t2.base = 7 * t1.base ∧
    perimeter t1 = 586 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      8 * s2.base = 7 * s1.base →
      perimeter s1 ≥ 586) :=
by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l1828_182845


namespace NUMINAMATH_CALUDE_distinct_pattern_count_is_17_l1828_182801

/-- Represents a 3x3 grid pattern with exactly 3 shaded squares -/
def Pattern := Fin 9 → Bool

/-- Two patterns are rotationally equivalent if one can be obtained from the other by rotation -/
def RotationallyEquivalent (p1 p2 : Pattern) : Prop := sorry

/-- Count of distinct patterns under rotational equivalence -/
def DistinctPatternCount : ℕ := sorry

theorem distinct_pattern_count_is_17 : DistinctPatternCount = 17 := by sorry

end NUMINAMATH_CALUDE_distinct_pattern_count_is_17_l1828_182801


namespace NUMINAMATH_CALUDE_book_pages_l1828_182814

/-- The number of days Lex read the book -/
def days : ℕ := 12

/-- The number of pages Lex read per day -/
def pages_per_day : ℕ := 20

/-- The total number of pages in the book -/
def total_pages : ℕ := days * pages_per_day

theorem book_pages : total_pages = 240 := by sorry

end NUMINAMATH_CALUDE_book_pages_l1828_182814


namespace NUMINAMATH_CALUDE_amelias_apples_l1828_182843

theorem amelias_apples (george_oranges : ℕ) (george_apples_diff : ℕ) (amelia_oranges_diff : ℕ) (total_fruits : ℕ) :
  george_oranges = 45 →
  george_apples_diff = 5 →
  amelia_oranges_diff = 18 →
  total_fruits = 107 →
  ∃ (amelia_apples : ℕ),
    total_fruits = george_oranges + (george_oranges - amelia_oranges_diff) + (amelia_apples + george_apples_diff) + amelia_apples ∧
    amelia_apples = 15 :=
by sorry

end NUMINAMATH_CALUDE_amelias_apples_l1828_182843


namespace NUMINAMATH_CALUDE_units_digit_of_m_cubed_plus_two_to_m_l1828_182813

theorem units_digit_of_m_cubed_plus_two_to_m (m : ℕ) : 
  m = 2021^2 + 2^2021 → (m^3 + 2^m) % 10 = 5 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_m_cubed_plus_two_to_m_l1828_182813


namespace NUMINAMATH_CALUDE_monotonically_decreasing_x_ln_x_l1828_182821

/-- The function f(x) = x ln x is monotonically decreasing on the interval (0, 1/e) -/
theorem monotonically_decreasing_x_ln_x :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1 / Real.exp 1 →
  x₁ * Real.log x₁ > x₂ * Real.log x₂ := by
sorry

/-- The domain of f(x) = x ln x is (0, +∞) -/
def domain_x_ln_x : Set ℝ := {x : ℝ | x > 0}

end NUMINAMATH_CALUDE_monotonically_decreasing_x_ln_x_l1828_182821


namespace NUMINAMATH_CALUDE_total_dolls_l1828_182841

def sister_dolls : ℕ := 8
def hannah_multiplier : ℕ := 5

theorem total_dolls : sister_dolls + hannah_multiplier * sister_dolls = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_dolls_l1828_182841


namespace NUMINAMATH_CALUDE_triangle_side_length_l1828_182830

/-- In a triangle XYZ, if ∠Z = 30°, ∠Y = 60°, and XZ = 12 units, then XY = 24 units. -/
theorem triangle_side_length (X Y Z : ℝ × ℝ) :
  let angle (A B C : ℝ × ℝ) : ℝ := sorry
  let distance (A B : ℝ × ℝ) : ℝ := sorry
  angle Z X Y = π / 6 →  -- 30°
  angle X Y Z = π / 3 →  -- 60°
  distance X Z = 12 →
  distance X Y = 24 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1828_182830


namespace NUMINAMATH_CALUDE_no_solution_implies_a_le_8_l1828_182887

theorem no_solution_implies_a_le_8 (a : ℝ) :
  (∀ x : ℝ, ¬(|x - 5| + |x + 3| < a)) → a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_le_8_l1828_182887


namespace NUMINAMATH_CALUDE_ecosystem_probability_l1828_182859

theorem ecosystem_probability : ∀ (n : ℕ) (p q r : ℚ),
  n = 7 →
  p = 1 / 5 →
  q = 1 / 10 →
  r = 17 / 20 →
  p + q + r = 1 →
  (Nat.choose n 4 : ℚ) * p^4 * r^3 = 34391 / 1000000 :=
by sorry

end NUMINAMATH_CALUDE_ecosystem_probability_l1828_182859


namespace NUMINAMATH_CALUDE_subtracted_amount_l1828_182890

theorem subtracted_amount (number : ℝ) (result : ℝ) (amount : ℝ) : 
  number = 85 → 
  result = 23 → 
  0.4 * number - amount = result →
  amount = 11 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_amount_l1828_182890


namespace NUMINAMATH_CALUDE_range_of_a_l1828_182837

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a-1)*x + 1 > 0) → (-1 < a ∧ a < 3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1828_182837


namespace NUMINAMATH_CALUDE_glasses_in_five_hours_l1828_182881

/-- The number of glasses of water consumed in a given time period -/
def glasses_consumed (rate_minutes : ℕ) (time_hours : ℕ) : ℕ :=
  (time_hours * 60) / rate_minutes

/-- Theorem: Given a rate of 1 glass every 20 minutes, 
    the number of glasses consumed in 5 hours is 15 -/
theorem glasses_in_five_hours : glasses_consumed 20 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_glasses_in_five_hours_l1828_182881


namespace NUMINAMATH_CALUDE_coconut_grove_yield_l1828_182855

theorem coconut_grove_yield (yield_group1 yield_group2 yield_group3 : ℕ) : 
  yield_group1 = 60 →
  yield_group2 = 120 →
  (3 * yield_group1 + 2 * yield_group2 + yield_group3) / 6 = 100 →
  yield_group3 = 180 := by
sorry

end NUMINAMATH_CALUDE_coconut_grove_yield_l1828_182855
