import Mathlib

namespace NUMINAMATH_CALUDE_library_books_problem_l3458_345880

theorem library_books_problem (initial_books : ℕ) : 
  initial_books - 120 + 35 - 15 = 150 → initial_books = 250 := by
  sorry

end NUMINAMATH_CALUDE_library_books_problem_l3458_345880


namespace NUMINAMATH_CALUDE_tangent_product_simplification_l3458_345829

theorem tangent_product_simplification :
  (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_product_simplification_l3458_345829


namespace NUMINAMATH_CALUDE_june_rainfall_l3458_345822

def rainfall_march : ℝ := 3.79
def rainfall_april : ℝ := 4.5
def rainfall_may : ℝ := 3.95
def rainfall_july : ℝ := 4.67
def average_rainfall : ℝ := 4
def num_months : ℕ := 5

theorem june_rainfall :
  let total_rainfall := average_rainfall * num_months
  let known_rainfall := rainfall_march + rainfall_april + rainfall_may + rainfall_july
  let june_rainfall := total_rainfall - known_rainfall
  june_rainfall = 3.09 := by sorry

end NUMINAMATH_CALUDE_june_rainfall_l3458_345822


namespace NUMINAMATH_CALUDE_power_division_19_l3458_345844

theorem power_division_19 : (19 : ℕ)^12 / (19 : ℕ)^5 = 893871739 := by sorry

end NUMINAMATH_CALUDE_power_division_19_l3458_345844


namespace NUMINAMATH_CALUDE_problem_solution_l3458_345888

theorem problem_solution :
  -- 1. Contrapositive statement
  (¬ (∀ a b : ℝ, (a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ 
    (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0))) ∧
  
  -- 2. Sufficient but not necessary condition
  ((∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ∧ 
   (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1)) ∧
  
  -- 3. False conjunction does not imply both propositions are false
  (¬ (∀ P Q : Prop, ¬(P ∧ Q) → (¬P ∧ ¬Q))) ∧
  
  -- 4. Correct negation of existential statement
  (¬ (∀ x : ℝ, ¬(x^2 + x + 1 < 0)) ↔ 
    (∃ x : ℝ, x^2 + x + 1 ≥ 0)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3458_345888


namespace NUMINAMATH_CALUDE_bridgets_score_l3458_345837

theorem bridgets_score (total_students : ℕ) (students_before : ℕ) (avg_before : ℚ) (avg_after : ℚ) 
  (h1 : total_students = 18)
  (h2 : students_before = 17)
  (h3 : avg_before = 76)
  (h4 : avg_after = 78) :
  (total_students : ℚ) * avg_after - (students_before : ℚ) * avg_before = 112 := by
  sorry

end NUMINAMATH_CALUDE_bridgets_score_l3458_345837


namespace NUMINAMATH_CALUDE_no_x_term_condition_l3458_345870

theorem no_x_term_condition (a : ℝ) : 
  (∀ x, (-2*x + a)*(x - 1) = -2*x^2 - a) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_no_x_term_condition_l3458_345870


namespace NUMINAMATH_CALUDE_root_product_expression_l3458_345893

theorem root_product_expression (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 - 2*p*α + 3 = 0) →
  (β^2 - 2*p*β + 3 = 0) →
  (γ^2 - 3*q*γ + 4 = 0) →
  (δ^2 - 3*q*δ + 4 = 0) →
  (α - γ) * (β - δ) * (α + δ) * (β + γ) = 4 * (2*p - 3*q)^2 := by sorry

end NUMINAMATH_CALUDE_root_product_expression_l3458_345893


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3458_345856

theorem simplify_fraction_product : 8 * (15 / 9) * (-45 / 40) = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3458_345856


namespace NUMINAMATH_CALUDE_cone_volume_l3458_345832

/-- The volume of a cone with given slant height and central angle of lateral surface --/
theorem cone_volume (slant_height : ℝ) (central_angle : ℝ) : 
  slant_height = 4 →
  central_angle = (2 * Real.pi) / 3 →
  ∃ (volume : ℝ), volume = (128 * Real.sqrt 2 / 81) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l3458_345832


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3458_345840

theorem cubic_equation_solution (x : ℝ) (h : x^3 + 1/x^3 = 110) : x^2 + 1/x^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3458_345840


namespace NUMINAMATH_CALUDE_knicks_knacks_knocks_equivalence_l3458_345850

theorem knicks_knacks_knocks_equivalence 
  (h1 : (8 : ℚ) * knicks = (3 : ℚ) * knacks)
  (h2 : (4 : ℚ) * knacks = (5 : ℚ) * knocks) :
  (64 : ℚ) * knicks = (30 : ℚ) * knocks := by
  sorry

end NUMINAMATH_CALUDE_knicks_knacks_knocks_equivalence_l3458_345850


namespace NUMINAMATH_CALUDE_gerald_remaining_money_l3458_345860

/-- Represents the cost of items and currency conversions --/
structure Costs where
  meat_pie : ℕ
  sausage_roll : ℕ
  farthings_per_pfennig : ℕ
  pfennigs_per_groat : ℕ
  groats_per_florin : ℕ

/-- Represents Gerald's initial money --/
structure GeraldMoney where
  farthings : ℕ
  groats : ℕ
  florins : ℕ

/-- Calculates the remaining pfennigs after purchase --/
def remaining_pfennigs (c : Costs) (m : GeraldMoney) : ℕ :=
  let total_pfennigs := 
    m.farthings / c.farthings_per_pfennig +
    m.groats * c.pfennigs_per_groat +
    m.florins * c.groats_per_florin * c.pfennigs_per_groat
  total_pfennigs - (c.meat_pie + c.sausage_roll)

/-- Theorem stating Gerald's remaining pfennigs --/
theorem gerald_remaining_money (c : Costs) (m : GeraldMoney) 
  (h1 : c.meat_pie = 120)
  (h2 : c.sausage_roll = 75)
  (h3 : m.farthings = 54)
  (h4 : m.groats = 8)
  (h5 : m.florins = 17)
  (h6 : c.farthings_per_pfennig = 6)
  (h7 : c.pfennigs_per_groat = 4)
  (h8 : c.groats_per_florin = 10) :
  remaining_pfennigs c m = 526 := by
  sorry

end NUMINAMATH_CALUDE_gerald_remaining_money_l3458_345860


namespace NUMINAMATH_CALUDE_percentage_problem_l3458_345849

theorem percentage_problem (x : ℝ) (p : ℝ) : 
  (0.5 * x = 200) → (p * x = 160) → p = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3458_345849


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3458_345884

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)

/-- The main theorem stating that any function satisfying the functional equation
    must be of the form f(x) = kx for some constant k -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3458_345884


namespace NUMINAMATH_CALUDE_remainder_98_102_div_12_l3458_345816

theorem remainder_98_102_div_12 : (98 * 102) % 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_98_102_div_12_l3458_345816


namespace NUMINAMATH_CALUDE_electricity_scientific_notation_equality_l3458_345841

/-- The amount of electricity generated by a wind power station per day -/
def electricity_per_day : ℝ := 74850000

/-- The scientific notation representation of the electricity generated per day -/
def scientific_notation : ℝ := 7.485 * (10^7)

/-- Theorem stating that the electricity_per_day is equal to its scientific notation representation -/
theorem electricity_scientific_notation_equality :
  electricity_per_day = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_electricity_scientific_notation_equality_l3458_345841


namespace NUMINAMATH_CALUDE_smallest_four_digit_with_product_512_l3458_345865

def is_four_digit (n : ℕ) : Prop := n ≥ 1000 ∧ n < 10000

def digit_product (n : ℕ) : ℕ :=
  (n / 1000) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

theorem smallest_four_digit_with_product_512 :
  ∀ n : ℕ, is_four_digit n → digit_product n = 512 → n ≥ 1888 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_with_product_512_l3458_345865


namespace NUMINAMATH_CALUDE_problem_solution_l3458_345818

theorem problem_solution (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  3 * (x^4 + 2*y^2) / 9 = 113/3 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3458_345818


namespace NUMINAMATH_CALUDE_handshake_arrangements_mod_1000_l3458_345853

/-- The number of ways 10 people can shake hands, where each person shakes hands with exactly two others -/
def handshake_arrangements : ℕ := sorry

/-- Theorem stating that the number of handshake arrangements is congruent to 688 modulo 1000 -/
theorem handshake_arrangements_mod_1000 : 
  handshake_arrangements ≡ 688 [ZMOD 1000] := by sorry

end NUMINAMATH_CALUDE_handshake_arrangements_mod_1000_l3458_345853


namespace NUMINAMATH_CALUDE_double_inequality_l3458_345826

theorem double_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (0 < 1 / (x + y + z + 1) - 1 / ((x + 1) * (y + 1) * (z + 1))) ∧
  (1 / (x + y + z + 1) - 1 / ((x + 1) * (y + 1) * (z + 1)) ≤ 1 / 8) ∧
  (1 / (x + y + z + 1) - 1 / ((x + 1) * (y + 1) * (z + 1)) = 1 / 8 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_double_inequality_l3458_345826


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3458_345807

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3458_345807


namespace NUMINAMATH_CALUDE_west_movement_l3458_345811

-- Define a type for direction
inductive Direction
| East
| West

-- Define a function to represent movement
def movement (dir : Direction) (distance : ℤ) : ℤ :=
  match dir with
  | Direction.East => distance
  | Direction.West => -distance

-- State the theorem
theorem west_movement :
  (movement Direction.East 50 = 50) →
  (∀ (d : Direction) (x : ℤ), movement d x = -movement (match d with
    | Direction.East => Direction.West
    | Direction.West => Direction.East) x) →
  (movement Direction.West 60 = -60) :=
by
  sorry

end NUMINAMATH_CALUDE_west_movement_l3458_345811


namespace NUMINAMATH_CALUDE_trapezoid_vector_range_l3458_345868

/-- Right trapezoid ABCD with moving point P -/
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  P : ℝ × ℝ
  h : A.2 = D.2  -- AB ⟂ AD
  i : D.1 - A.1 = 1  -- AD = 1
  j : C.1 - D.1 = 1  -- DC = 1
  k : B.2 - A.2 = 3  -- AB = 3
  l : (P.1 - C.1)^2 + (P.2 - C.2)^2 ≤ 1  -- P is within or on the circle centered at C with radius 1

def vector_decomposition (t : Trapezoid) (α β : ℝ) : Prop :=
  t.P.1 - t.A.1 = α * (t.D.1 - t.A.1) + β * (t.B.1 - t.A.1) ∧
  t.P.2 - t.A.2 = α * (t.D.2 - t.A.2) + β * (t.B.2 - t.A.2)

theorem trapezoid_vector_range (t : Trapezoid) :
  ∃ (α β : ℝ), vector_decomposition t α β ∧ 
  (∀ (γ δ : ℝ), vector_decomposition t γ δ → 1 < γ + δ ∧ γ + δ < 5/3) :=
sorry

end NUMINAMATH_CALUDE_trapezoid_vector_range_l3458_345868


namespace NUMINAMATH_CALUDE_cubic_root_implies_p_value_l3458_345824

theorem cubic_root_implies_p_value : ∀ p : ℝ, (3 : ℝ)^3 + p * 3 - 18 = 0 → p = -3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_implies_p_value_l3458_345824


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l3458_345882

theorem complex_sum_theorem (x : ℂ) (h1 : x^7 = 1) (h2 : x ≠ 1) :
  (x^2)/(x-1) + (x^4)/(x^2-1) + (x^6)/(x^3-1) + (x^8)/(x^4-1) + (x^10)/(x^5-1) + (x^12)/(x^6-1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l3458_345882


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l3458_345801

theorem initial_markup_percentage (C : ℝ) (M : ℝ) : 
  (C * (1 + M) * 1.25 * 0.92 = C * 1.38) → M = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_initial_markup_percentage_l3458_345801


namespace NUMINAMATH_CALUDE_initial_condition_recurrence_relation_diamonds_in_25th_figure_l3458_345872

/-- The number of diamonds in the n-th figure of the sequence -/
def num_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 2 * n^2 + 2 * n - 3

/-- The sequence starts with one diamond in the first figure -/
theorem initial_condition : num_diamonds 1 = 1 := by sorry

/-- The recurrence relation for n ≥ 2 -/
theorem recurrence_relation (n : ℕ) (h : n ≥ 2) :
  num_diamonds n = num_diamonds (n-1) + 4*n := by sorry

/-- The main theorem: The 25th figure contains 1297 diamonds -/
theorem diamonds_in_25th_figure : num_diamonds 25 = 1297 := by sorry

end NUMINAMATH_CALUDE_initial_condition_recurrence_relation_diamonds_in_25th_figure_l3458_345872


namespace NUMINAMATH_CALUDE_train_passing_jogger_train_passes_jogger_in_34_seconds_l3458_345848

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger (jogger_speed : Real) (train_speed : Real) 
  (initial_distance : Real) (train_length : Real) : Real :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- Proof that the train passes the jogger in 34 seconds under given conditions -/
theorem train_passes_jogger_in_34_seconds :
  train_passing_jogger 9 45 240 100 = 34 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_train_passes_jogger_in_34_seconds_l3458_345848


namespace NUMINAMATH_CALUDE_zoo_animals_l3458_345808

/-- The number of sea lions at the zoo -/
def sea_lions : ℕ := 42

/-- The number of penguins at the zoo -/
def penguins : ℕ := sea_lions + 84

/-- The number of flamingos at the zoo -/
def flamingos : ℕ := penguins + 42

theorem zoo_animals :
  (4 : ℚ) * sea_lions = 11 * sea_lions - 7 * 84 ∧
  7 * penguins = 11 * sea_lions + 7 * 42 ∧
  4 * flamingos = 7 * penguins + 4 * 42 :=
by sorry

#check zoo_animals

end NUMINAMATH_CALUDE_zoo_animals_l3458_345808


namespace NUMINAMATH_CALUDE_two_questions_determine_number_l3458_345819

theorem two_questions_determine_number : 
  ∃ (q₁ q₂ : ℕ → ℕ → ℕ), 
    (∀ m : ℕ, m ≥ 2 → q₁ m ≥ 2) ∧ 
    (∀ m : ℕ, m ≥ 2 → q₂ m ≥ 2) ∧ 
    (∀ V : ℕ, 1 ≤ V ∧ V ≤ 100 → 
      ∀ V' : ℕ, 1 ≤ V' ∧ V' ≤ 100 → 
        (V / q₁ V = V' / q₁ V' ∧ V / q₂ V = V' / q₂ V') → V = V') :=
sorry

end NUMINAMATH_CALUDE_two_questions_determine_number_l3458_345819


namespace NUMINAMATH_CALUDE_intersection_complement_equals_l3458_345883

def U : Finset Int := {-1, 0, 1, 2, 3, 4}
def A : Finset Int := {2, 3}
def B : Finset Int := {1, 2, 3, 4} \ A

theorem intersection_complement_equals : B ∩ (U \ A) = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_l3458_345883


namespace NUMINAMATH_CALUDE_problem_solution_l3458_345896

theorem problem_solution (x y : ℚ) 
  (eq1 : 102 * x - 5 * y = 25) 
  (eq2 : 3 * y - x = 10) : 
  10 - x = 2885 / 301 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3458_345896


namespace NUMINAMATH_CALUDE_tray_pieces_count_l3458_345828

def tray_length : ℕ := 24
def tray_width : ℕ := 20
def piece_length : ℕ := 3
def piece_width : ℕ := 2

theorem tray_pieces_count : 
  (tray_length * tray_width) / (piece_length * piece_width) = 80 :=
sorry

end NUMINAMATH_CALUDE_tray_pieces_count_l3458_345828


namespace NUMINAMATH_CALUDE_linoleum_cut_theorem_l3458_345842

/-- Represents a square on the linoleum piece -/
inductive Square
| White
| Black

/-- Represents the modified 8x8 grid with two additional white squares -/
def ModifiedGrid := Array (Array Square)

/-- Represents a cut on the grid -/
structure Cut where
  start_row : Nat
  start_col : Nat
  end_row : Nat
  end_col : Nat

/-- Represents a transformation (rotation and translation) -/
structure Transform where
  rotation : Nat  -- 0, 1, 2, or 3 for 0, 90, 180, 270 degrees
  translation_row : Int
  translation_col : Int

/-- Checks if a grid is a proper 8x8 chessboard -/
def is_proper_chessboard (grid : Array (Array Square)) : Bool :=
  sorry

/-- Applies a cut to the grid, returning two pieces -/
def apply_cut (grid : ModifiedGrid) (cut : Cut) : (ModifiedGrid × ModifiedGrid) :=
  sorry

/-- Applies a transformation to a grid piece -/
def apply_transform (piece : ModifiedGrid) (transform : Transform) : ModifiedGrid :=
  sorry

/-- Combines two grid pieces -/
def combine_pieces (piece1 piece2 : ModifiedGrid) : ModifiedGrid :=
  sorry

theorem linoleum_cut_theorem (original_grid : ModifiedGrid) :
  ∃ (cut : Cut) (transform : Transform),
    let (piece1, piece2) := apply_cut original_grid cut
    let transformed_piece := apply_transform piece1 transform
    let result := combine_pieces transformed_piece piece2
    is_proper_chessboard result :=
  sorry

end NUMINAMATH_CALUDE_linoleum_cut_theorem_l3458_345842


namespace NUMINAMATH_CALUDE_total_area_is_68_l3458_345843

/-- Represents the dimensions of a rectangle -/
structure RectDimensions where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def rectangleArea (rect : RectDimensions) : ℕ :=
  rect.width * rect.height

/-- The dimensions of the four rectangles in the figure -/
def rect1 : RectDimensions := ⟨5, 7⟩
def rect2 : RectDimensions := ⟨3, 3⟩
def rect3 : RectDimensions := ⟨4, 1⟩
def rect4 : RectDimensions := ⟨5, 4⟩

/-- Theorem: The total area of the composite shape is 68 square units -/
theorem total_area_is_68 : 
  rectangleArea rect1 + rectangleArea rect2 + rectangleArea rect3 + rectangleArea rect4 = 68 := by
  sorry

end NUMINAMATH_CALUDE_total_area_is_68_l3458_345843


namespace NUMINAMATH_CALUDE_min_sum_given_product_l3458_345858

theorem min_sum_given_product (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + 3*b = a*b) :
  a + b ≥ 5 + 2*Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_min_sum_given_product_l3458_345858


namespace NUMINAMATH_CALUDE_largest_tray_size_l3458_345877

theorem largest_tray_size (tim_sweets peter_sweets : ℕ) 
  (h1 : tim_sweets = 36) 
  (h2 : peter_sweets = 44) : 
  Nat.gcd tim_sweets peter_sweets = 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_tray_size_l3458_345877


namespace NUMINAMATH_CALUDE_picking_black_is_random_event_l3458_345800

/-- Represents a ball in the box -/
inductive Ball
| White
| Black

/-- Represents the box containing the balls -/
structure Box where
  white_balls : ℕ
  black_balls : ℕ

/-- Defines what a random event is -/
def is_random_event (box : Box) (pick : Ball → Prop) : Prop :=
  (∃ b : Ball, pick b) ∧ 
  (∃ b : Ball, ¬ pick b) ∧ 
  (box.white_balls + box.black_balls > 0)

/-- The main theorem to prove -/
theorem picking_black_is_random_event (box : Box) 
  (h1 : box.white_balls = 1) 
  (h2 : box.black_balls = 200) : 
  is_random_event box (λ b => b = Ball.Black) := by
  sorry


end NUMINAMATH_CALUDE_picking_black_is_random_event_l3458_345800


namespace NUMINAMATH_CALUDE_playground_transfer_l3458_345887

theorem playground_transfer (x : ℤ) : 
  (54 + x = 2 * (48 - x)) ↔ 
  (54 + x = 2 * (48 - x) ∧ 
   54 + x > 0 ∧ 
   48 - x > 0) :=
by sorry

end NUMINAMATH_CALUDE_playground_transfer_l3458_345887


namespace NUMINAMATH_CALUDE_sodium_reduction_is_one_third_l3458_345838

def sodium_reduction_fraction (salt_teaspoons : ℕ) (parmesan_oz : ℕ) 
  (salt_sodium_per_tsp : ℕ) (parmesan_sodium_per_oz : ℕ) 
  (parmesan_reduction_oz : ℕ) : ℚ :=
  let original_sodium := salt_teaspoons * salt_sodium_per_tsp + parmesan_oz * parmesan_sodium_per_oz
  let reduced_sodium := salt_teaspoons * salt_sodium_per_tsp + (parmesan_oz - parmesan_reduction_oz) * parmesan_sodium_per_oz
  (original_sodium - reduced_sodium : ℚ) / original_sodium

theorem sodium_reduction_is_one_third :
  sodium_reduction_fraction 2 8 50 25 4 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sodium_reduction_is_one_third_l3458_345838


namespace NUMINAMATH_CALUDE_expression_change_l3458_345869

theorem expression_change (x a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := λ t ↦ t^2 - 3
  (f (x + a) - f x = 2*a*x + a^2) ∧ (f (x - a) - f x = -2*a*x + a^2) :=
sorry

end NUMINAMATH_CALUDE_expression_change_l3458_345869


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l3458_345809

def z : ℂ := Complex.I + Complex.I^6

theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l3458_345809


namespace NUMINAMATH_CALUDE_investment_amount_is_14400_l3458_345874

/-- Represents the investment scenario --/
structure Investment where
  face_value : ℕ
  premium_percentage : ℕ
  dividend_percentage : ℕ
  total_dividend : ℕ

/-- Calculates the amount invested given the investment parameters --/
def amount_invested (i : Investment) : ℕ :=
  let share_price := i.face_value + i.face_value * i.premium_percentage / 100
  let dividend_per_share := i.face_value * i.dividend_percentage / 100
  let num_shares := i.total_dividend / dividend_per_share
  num_shares * share_price

/-- Theorem stating that the amount invested is 14400 given the specific conditions --/
theorem investment_amount_is_14400 :
  ∀ i : Investment,
    i.face_value = 100 ∧
    i.premium_percentage = 20 ∧
    i.dividend_percentage = 5 ∧
    i.total_dividend = 600 →
    amount_invested i = 14400 :=
by
  sorry

end NUMINAMATH_CALUDE_investment_amount_is_14400_l3458_345874


namespace NUMINAMATH_CALUDE_duck_race_charity_amount_l3458_345885

/-- The amount of money raised for charity in the annual rubber duck race -/
def charity_money_raised (regular_price : ℝ) (large_price : ℝ) (regular_sold : ℕ) (large_sold : ℕ) : ℝ :=
  regular_price * (regular_sold : ℝ) + large_price * (large_sold : ℝ)

/-- Theorem stating the amount of money raised for charity in the given scenario -/
theorem duck_race_charity_amount :
  charity_money_raised 3 5 221 185 = 1588 :=
by
  sorry

end NUMINAMATH_CALUDE_duck_race_charity_amount_l3458_345885


namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_by_11_l3458_345817

theorem smallest_addition_for_divisibility_by_11 (n : ℕ) (h : n = 8261955) :
  ∃ k : ℕ, k > 0 ∧ (n + k) % 11 = 0 ∧ ∀ m : ℕ, m > 0 → (n + m) % 11 = 0 → k ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_by_11_l3458_345817


namespace NUMINAMATH_CALUDE_area_divisibility_l3458_345855

/-- A point with integer coordinates -/
structure IntegerPoint where
  x : ℤ
  y : ℤ

/-- A convex polygon with vertices on a circle -/
structure ConvexPolygonOnCircle where
  vertices : List IntegerPoint
  is_convex : sorry
  on_circle : sorry

/-- The statement of the theorem -/
theorem area_divisibility
  (P : ConvexPolygonOnCircle)
  (n : ℕ)
  (n_odd : Odd n)
  (side_length_squared_div : ∃ (side_length : ℕ), (side_length ^ 2) % n = 0) :
  ∃ (area : ℕ), (2 * area) % n = 0 := by
  sorry


end NUMINAMATH_CALUDE_area_divisibility_l3458_345855


namespace NUMINAMATH_CALUDE_system_solution_l3458_345879

theorem system_solution (a b : ℚ) 
  (eq1 : 2 * a + 3 * b = 18)
  (eq2 : 4 * a + 5 * b = 31) :
  2 * a + b = 8 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3458_345879


namespace NUMINAMATH_CALUDE_sin_double_angle_minus_pi_half_l3458_345861

/-- Given an angle α in the Cartesian coordinate system with the specified properties,
    prove that sin(2α - π/2) = -1/2 -/
theorem sin_double_angle_minus_pi_half (α : ℝ) : 
  (∃ (x y : ℝ), x = Real.sqrt 3 ∧ y = -1 ∧ 
   x * Real.cos α = x ∧ x * Real.sin α = y) →
  Real.sin (2 * α - π / 2) = -1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_double_angle_minus_pi_half_l3458_345861


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3458_345863

theorem simplify_trig_expression (θ : Real) (h : θ ∈ Set.Icc (5 * Real.pi / 4) (3 * Real.pi / 2)) :
  Real.sqrt (1 - Real.sin (2 * θ)) - Real.sqrt (1 + Real.sin (2 * θ)) = 2 * Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3458_345863


namespace NUMINAMATH_CALUDE_divisibility_condition_l3458_345892

def M (n : ℤ) : Finset ℤ := {n, n + 1, n + 2, n + 3, n + 4}

def S (n : ℤ) : ℤ := (M n).sum (fun x => x^2)

def P (n : ℤ) : ℤ := (M n).prod (fun x => x^2)

theorem divisibility_condition (n : ℤ) : S n ∣ P n ↔ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3458_345892


namespace NUMINAMATH_CALUDE_polynomial_coefficient_bound_l3458_345820

def M : Set (ℝ → ℝ) :=
  {P | ∃ a b c d : ℝ, ∀ x, P x = a * x^3 + b * x^2 + c * x + d ∧ ∀ x ∈ Set.Icc (-1) 1, |P x| ≤ 1}

theorem polynomial_coefficient_bound :
  ∃ k : ℝ, k = 4 ∧ (∀ P ∈ M, ∃ a b c d : ℝ, (∀ x, P x = a * x^3 + b * x^2 + c * x + d) ∧ |a| ≤ k) ∧
  ∀ k' : ℝ, (∀ P ∈ M, ∃ a b c d : ℝ, (∀ x, P x = a * x^3 + b * x^2 + c * x + d) ∧ |a| ≤ k') → k' ≥ k :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_bound_l3458_345820


namespace NUMINAMATH_CALUDE_value_of_x_l3458_345876

theorem value_of_x : ∃ x : ℝ, 
  x * 0.48 * 2.50 / (0.12 * 0.09 * 0.5) = 800.0000000000001 ∧ 
  abs (x - 3.6) < 0.0000000000001 :=
by sorry

end NUMINAMATH_CALUDE_value_of_x_l3458_345876


namespace NUMINAMATH_CALUDE_sum_of_squares_l3458_345839

theorem sum_of_squares (a b c : ℝ) 
  (h1 : a * b + b * c + a * c = 131)
  (h2 : a + b + c = 21) : 
  a^2 + b^2 + c^2 = 179 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3458_345839


namespace NUMINAMATH_CALUDE_second_shop_amount_calculation_l3458_345857

/-- The amount paid for books from the second shop -/
def second_shop_amount (books_shop1 books_shop2 : ℕ) (amount_shop1 avg_price : ℚ) : ℚ :=
  (books_shop1 + books_shop2 : ℚ) * avg_price - amount_shop1

/-- Theorem stating the amount paid for books from the second shop -/
theorem second_shop_amount_calculation :
  second_shop_amount 27 20 581 25 = 594 := by
  sorry

end NUMINAMATH_CALUDE_second_shop_amount_calculation_l3458_345857


namespace NUMINAMATH_CALUDE_base5_addition_l3458_345871

/-- Addition of two numbers in base 5 --/
def base5_add (a b : ℕ) : ℕ := sorry

/-- Conversion from base 10 to base 5 --/
def to_base5 (n : ℕ) : ℕ := sorry

/-- Conversion from base 5 to base 10 --/
def from_base5 (n : ℕ) : ℕ := sorry

theorem base5_addition : base5_add 14 132 = 101 := by sorry

end NUMINAMATH_CALUDE_base5_addition_l3458_345871


namespace NUMINAMATH_CALUDE_odd_function_value_l3458_345898

-- Define an odd function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem odd_function_value :
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  (∀ x < 0, f x = x^3 + x + 1) →  -- f(x) = x^3 + x + 1 for x < 0
  f 2 = 9 := by sorry

end NUMINAMATH_CALUDE_odd_function_value_l3458_345898


namespace NUMINAMATH_CALUDE_complex_magnitude_l3458_345851

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 3 - 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3458_345851


namespace NUMINAMATH_CALUDE_smallest_consecutive_sequence_sum_l3458_345873

theorem smallest_consecutive_sequence_sum (B : ℤ) : B = 1011 ↔ 
  (∀ k < B, ¬∃ n : ℕ+, (n : ℤ) * (2 * k + n - 1) = 2023 ∧ n > 1) ∧
  (∃ n : ℕ+, (n : ℤ) * (2 * B + n - 1) = 2023 ∧ n > 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_consecutive_sequence_sum_l3458_345873


namespace NUMINAMATH_CALUDE_cylinder_volume_equals_54_sqrt3_over_sqrt_pi_l3458_345875

/-- Given a cube with side length 3 and a cylinder with the same surface area as the cube,
    where the cylinder's height equals its diameter, prove that the volume of the cylinder
    is 54 * sqrt(3) / sqrt(π). -/
theorem cylinder_volume_equals_54_sqrt3_over_sqrt_pi
  (cube_side : ℝ)
  (cylinder_radius : ℝ)
  (cylinder_height : ℝ)
  (h1 : cube_side = 3)
  (h2 : 6 * cube_side^2 = 2 * π * cylinder_radius^2 + 2 * π * cylinder_radius * cylinder_height)
  (h3 : cylinder_height = 2 * cylinder_radius) :
  π * cylinder_radius^2 * cylinder_height = 54 * Real.sqrt 3 / Real.sqrt π :=
by sorry


end NUMINAMATH_CALUDE_cylinder_volume_equals_54_sqrt3_over_sqrt_pi_l3458_345875


namespace NUMINAMATH_CALUDE_metal_sheet_weight_l3458_345846

/-- Represents a square piece of metal sheet -/
structure MetalSquare where
  side : ℝ
  weight : ℝ

/-- Given conditions of the problem -/
def problem_conditions (s1 s2 : MetalSquare) : Prop :=
  s1.side = 4 ∧ s1.weight = 16 ∧ s2.side = 6

/-- Theorem statement -/
theorem metal_sheet_weight (s1 s2 : MetalSquare) :
  problem_conditions s1 s2 → s2.weight = 36 := by
  sorry

end NUMINAMATH_CALUDE_metal_sheet_weight_l3458_345846


namespace NUMINAMATH_CALUDE_pyramid_surface_area_l3458_345812

/-- The total surface area of a pyramid with a regular hexagonal base -/
theorem pyramid_surface_area (a : ℝ) (h : a > 0) :
  let base_area := (3 * a^2 * Real.sqrt 3) / 2
  let perp_edge_length := a
  let side_triangle_area := a^2 / 2
  let side_triangle_area2 := a^2
  let side_triangle_area3 := (a^2 * Real.sqrt 7) / 4
  base_area + 2 * side_triangle_area + 2 * side_triangle_area2 + 2 * side_triangle_area3 =
    (a^2 * (6 + 3 * Real.sqrt 3 + Real.sqrt 7)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_surface_area_l3458_345812


namespace NUMINAMATH_CALUDE_smaller_circle_radius_l3458_345862

theorem smaller_circle_radius (R : ℝ) (h : R = 12) : 
  ∃ (r : ℝ), r = 3 * Real.sqrt 3 ∧
  r > 0 ∧
  r < R ∧
  (∃ (A B C D E F G : ℝ × ℝ),
    -- A is the center of the left circle
    -- B is on the right circle
    -- C is the center of the right circle
    -- D is the center of the smaller circle
    -- E, F, G are points of tangency

    -- The centers are R apart
    Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = R ∧

    -- AB is a diameter of the right circle
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2*R ∧

    -- D is r away from E, F, and G
    Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) = r ∧
    Real.sqrt ((D.1 - F.1)^2 + (D.2 - F.2)^2) = r ∧
    Real.sqrt ((D.1 - G.1)^2 + (D.2 - G.2)^2) = r ∧

    -- A is R+r away from F
    Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = R + r ∧

    -- C is R-r away from G
    Real.sqrt ((C.1 - G.1)^2 + (C.2 - G.2)^2) = R - r ∧

    -- E is on AB
    (E.2 - A.2) / (E.1 - A.1) = (B.2 - A.2) / (B.1 - A.1)
  ) := by
sorry

end NUMINAMATH_CALUDE_smaller_circle_radius_l3458_345862


namespace NUMINAMATH_CALUDE_polygon_sides_l3458_345845

theorem polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  (exterior_angle = 36) → (n * exterior_angle = 360) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3458_345845


namespace NUMINAMATH_CALUDE_multiplier_value_l3458_345830

theorem multiplier_value (p q : ℕ) (x : ℚ) 
  (h1 : p > 1)
  (h2 : q > 1)
  (h3 : x * (p + 1) = 25 * (q + 1))
  (h4 : p + q ≥ 40)
  (h5 : ∀ p' q' : ℕ, p' > 1 → q' > 1 → x * (p' + 1) = 25 * (q' + 1) → p' + q' < p + q → False) :
  x = 325 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_value_l3458_345830


namespace NUMINAMATH_CALUDE_baker_leftover_cupcakes_l3458_345852

/-- Represents the cupcake distribution problem --/
def cupcake_distribution (total_cupcakes nutty_cupcakes gluten_free_cupcakes num_children : ℕ)
  (num_nut_allergic num_gluten_only : ℕ) : ℕ :=
  let regular_cupcakes := total_cupcakes - nutty_cupcakes - gluten_free_cupcakes
  let nutty_per_child := nutty_cupcakes / (num_children - num_nut_allergic)
  let nutty_distributed := nutty_per_child * (num_children - num_nut_allergic)
  let regular_per_child := regular_cupcakes / num_children
  let regular_distributed := regular_per_child * num_children
  let leftover_nutty := nutty_cupcakes - nutty_distributed
  let leftover_regular := regular_cupcakes - regular_distributed
  leftover_nutty + leftover_regular

/-- Theorem stating that given the specific conditions, Ms. Baker will have 5 cupcakes left over --/
theorem baker_leftover_cupcakes :
  cupcake_distribution 84 18 25 7 2 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_baker_leftover_cupcakes_l3458_345852


namespace NUMINAMATH_CALUDE_smallest_n_for_solutions_greater_than_negative_one_l3458_345890

theorem smallest_n_for_solutions_greater_than_negative_one :
  ∀ (n : ℤ), (∀ (x : ℝ), 
    x^3 - (5*n - 9)*x^2 + (6*n^2 - 31*n - 106)*x - 6*(n - 8)*(n + 2) = 0 
    → x > -1) 
  ↔ n ≥ 8 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_solutions_greater_than_negative_one_l3458_345890


namespace NUMINAMATH_CALUDE_three_numbers_sum_l3458_345836

theorem three_numbers_sum (A B C : ℤ) : 
  A + B + C = 180 ∧ B = 3*C - 2 ∧ A = 2*C + 8 → A = 66 ∧ B = 85 ∧ C = 29 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l3458_345836


namespace NUMINAMATH_CALUDE_find_x_l3458_345847

theorem find_x : ∃ (x : ℕ+), 
  let n : ℤ := (x : ℤ)^2 + 3*(x : ℤ) + 20
  let d : ℤ := 3*(x : ℤ) + 4
  n = d * (x : ℤ) + 8 → x = 2 := by
sorry

end NUMINAMATH_CALUDE_find_x_l3458_345847


namespace NUMINAMATH_CALUDE_commission_is_25_l3458_345894

/-- Represents the sales data for a salesman selling security systems --/
structure SalesData where
  second_street_sales : Nat
  fourth_street_sales : Nat
  total_commission : Nat

/-- Calculates the total number of security systems sold --/
def total_sales (data : SalesData) : Nat :=
  data.second_street_sales + (data.second_street_sales / 2) + data.fourth_street_sales

/-- Calculates the commission per security system --/
def commission_per_system (data : SalesData) : Nat :=
  data.total_commission / (total_sales data)

/-- Theorem stating that given the sales conditions, the commission per system is $25 --/
theorem commission_is_25 (data : SalesData) 
  (h1 : data.second_street_sales = 4)
  (h2 : data.fourth_street_sales = 1)
  (h3 : data.total_commission = 175) :
  commission_per_system data = 25 := by
  sorry

#eval commission_per_system { second_street_sales := 4, fourth_street_sales := 1, total_commission := 175 }

end NUMINAMATH_CALUDE_commission_is_25_l3458_345894


namespace NUMINAMATH_CALUDE_product_of_polynomials_l3458_345854

/-- Given two polynomials A(d) and B(d) whose product is C(d), prove that k + m = -4 --/
theorem product_of_polynomials (k m : ℚ) : 
  (∀ d : ℚ, (5*d^2 - 2*d + k) * (4*d^2 + m*d - 9) = 20*d^4 - 28*d^3 + 13*d^2 - m*d - 18) → 
  k + m = -4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_polynomials_l3458_345854


namespace NUMINAMATH_CALUDE_f_monotonicity_and_intersection_l3458_345835

/-- The function f(x) = x^3 - x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem f_monotonicity_and_intersection (a : ℝ) :
  (∀ x : ℝ, a ≥ 1/3 → Monotone (f a)) ∧
  (∃ x y : ℝ, x = 1 ∧ y = a + 1 ∧ f a x = y ∧ f' a x * x = y) ∧
  (∃ x y : ℝ, x = -1 ∧ y = -a - 1 ∧ f a x = y ∧ f' a x * x = y) := by
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_intersection_l3458_345835


namespace NUMINAMATH_CALUDE_total_books_eq_sum_l3458_345804

/-- The number of different books in the 'crazy silly school' series -/
def total_books : ℕ := sorry

/-- The number of different movies in the 'crazy silly school' series -/
def total_movies : ℕ := 10

/-- The number of books you have read -/
def books_read : ℕ := 12

/-- The number of movies you have watched -/
def movies_watched : ℕ := 56

/-- The number of books you still have to read -/
def books_to_read : ℕ := 10

/-- Theorem: The total number of books in the series is equal to the sum of books read and books yet to read -/
theorem total_books_eq_sum : total_books = books_read + books_to_read := by sorry

end NUMINAMATH_CALUDE_total_books_eq_sum_l3458_345804


namespace NUMINAMATH_CALUDE_quadratic_sequence_exists_l3458_345813

def is_quadratic_sequence (a : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ i : ℕ, i ≤ n → |a i - a (i-1)| = (i : ℤ)^2

theorem quadratic_sequence_exists (h k : ℤ) : 
  ∃ (n : ℕ) (a : ℕ → ℤ), a 0 = h ∧ a n = k ∧ is_quadratic_sequence a n :=
sorry

end NUMINAMATH_CALUDE_quadratic_sequence_exists_l3458_345813


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3458_345810

/-- The length of the major axis of an ellipse formed by the intersection of a plane and a right circular cylinder --/
def major_axis_length (cylinder_radius : ℝ) (percentage_longer : ℝ) : ℝ :=
  2 * cylinder_radius * (1 + percentage_longer)

/-- Theorem: The length of the major axis of the ellipse is 6.4 --/
theorem ellipse_major_axis_length :
  major_axis_length 2 0.6 = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3458_345810


namespace NUMINAMATH_CALUDE_at_most_one_integer_root_l3458_345889

theorem at_most_one_integer_root (n : ℤ) :
  ∃! (k : ℤ), k^4 - 1993*k^3 + (1993 + n)*k^2 - 11*k + n = 0 :=
by sorry

end NUMINAMATH_CALUDE_at_most_one_integer_root_l3458_345889


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l3458_345886

theorem cube_sum_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^3 + b^3 = 2) :
  a + b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l3458_345886


namespace NUMINAMATH_CALUDE_total_coins_is_660_l3458_345802

/-- The number of coins Jayden received -/
def jayden_coins : ℕ := 300

/-- The additional coins Jason received compared to Jayden -/
def jason_extra_coins : ℕ := 60

/-- The total number of coins given to both boys -/
def total_coins : ℕ := jayden_coins + (jayden_coins + jason_extra_coins)

/-- Theorem stating that the total number of coins given to both boys is 660 -/
theorem total_coins_is_660 : total_coins = 660 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_is_660_l3458_345802


namespace NUMINAMATH_CALUDE_log_base_2_derivative_l3458_345866

theorem log_base_2_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_log_base_2_derivative_l3458_345866


namespace NUMINAMATH_CALUDE_solution_difference_l3458_345881

theorem solution_difference : ∃ p q : ℝ, 
  (p - 4) * (p + 4) = 24 * p - 96 ∧ 
  (q - 4) * (q + 4) = 24 * q - 96 ∧ 
  p ≠ q ∧ 
  p > q ∧ 
  p - q = 16 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l3458_345881


namespace NUMINAMATH_CALUDE_arccos_difference_equals_negative_pi_sixth_l3458_345864

theorem arccos_difference_equals_negative_pi_sixth : 
  Real.arccos ((Real.sqrt 6 + 1) / (2 * Real.sqrt 3)) - Real.arccos (Real.sqrt (2/3)) = -π/6 := by
  sorry

end NUMINAMATH_CALUDE_arccos_difference_equals_negative_pi_sixth_l3458_345864


namespace NUMINAMATH_CALUDE_irreducible_fraction_l3458_345825

theorem irreducible_fraction (n : ℕ) : 
  (Nat.gcd (21 * n + 4) (14 * n + 1) = 1) ↔ (n % 5 ≠ 1) := by sorry

end NUMINAMATH_CALUDE_irreducible_fraction_l3458_345825


namespace NUMINAMATH_CALUDE_function_inequality_l3458_345859

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 * Real.log x - a * x + (a + 3) / x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 2 * Real.exp x - 4 * x + 2 * a

theorem function_inequality (a : ℝ) (h₁ : a ≥ 0) :
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (1/2 : ℝ) 2 ∧ x₂ ∈ Set.Icc (1/2 : ℝ) 2 ∧ f a x₁ > g a x₂) →
  a ∈ Set.Icc 1 4 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3458_345859


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l3458_345823

def b (n : ℕ) : ℕ := (n^2).factorial + n

theorem max_gcd_consecutive_terms : 
  ∃ (k : ℕ), k ≥ 1 ∧ Nat.gcd (b k) (b (k+1)) = 2 ∧ 
  ∀ (n : ℕ), n ≥ 1 → Nat.gcd (b n) (b (n+1)) ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l3458_345823


namespace NUMINAMATH_CALUDE_chinese_dream_probability_l3458_345831

/-- The number of character cards -/
def num_cards : Nat := 3

/-- The total number of possible arrangements -/
def total_arrangements : Nat := Nat.factorial num_cards

/-- The number of arrangements forming the desired phrase -/
def desired_arrangements : Nat := 1

/-- The probability of forming the desired phrase -/
def probability : Rat := desired_arrangements / total_arrangements

theorem chinese_dream_probability :
  probability = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_chinese_dream_probability_l3458_345831


namespace NUMINAMATH_CALUDE_ninth_minus_eighth_difference_l3458_345834

/-- The side length of the nth square in the sequence -/
def side_length (n : ℕ) : ℕ := 3 + 2 * (n - 1)

/-- The number of tiles in the nth square -/
def tile_count (n : ℕ) : ℕ := (side_length n) ^ 2

theorem ninth_minus_eighth_difference : tile_count 9 - tile_count 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_ninth_minus_eighth_difference_l3458_345834


namespace NUMINAMATH_CALUDE_percentage_difference_l3458_345878

theorem percentage_difference : (0.80 * 45) - ((4 : ℚ) / 5 * 25) = 16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3458_345878


namespace NUMINAMATH_CALUDE_sqrt_product_equals_sqrt_of_product_l3458_345827

theorem sqrt_product_equals_sqrt_of_product :
  Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_sqrt_of_product_l3458_345827


namespace NUMINAMATH_CALUDE_lewis_items_found_l3458_345895

theorem lewis_items_found (tanya_items samantha_items lewis_items : ℕ) : 
  tanya_items = 4 →
  samantha_items = 4 * tanya_items →
  lewis_items = samantha_items + 4 →
  lewis_items = 20 := by
  sorry

end NUMINAMATH_CALUDE_lewis_items_found_l3458_345895


namespace NUMINAMATH_CALUDE_restaurant_menu_fraction_l3458_345897

/-- Given a restaurant menu with vegan dishes and dietary restrictions, 
    calculate the fraction of suitable dishes. -/
theorem restaurant_menu_fraction (total_dishes : ℕ) 
  (vegan_dishes : ℕ) (restricted_vegan_dishes : ℕ) : 
  vegan_dishes = (3 : ℕ) * total_dishes / 10 →
  vegan_dishes = 9 →
  restricted_vegan_dishes = 7 →
  (vegan_dishes - restricted_vegan_dishes : ℚ) / total_dishes = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_menu_fraction_l3458_345897


namespace NUMINAMATH_CALUDE_prob_one_defective_is_half_l3458_345815

/-- Represents the total number of items -/
def total_items : ℕ := 4

/-- Represents the number of genuine items -/
def genuine_items : ℕ := 3

/-- Represents the number of defective items -/
def defective_items : ℕ := 1

/-- Represents the number of items selected -/
def items_selected : ℕ := 2

/-- Calculates the number of ways to select k items from n items -/
def combinations (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the probability of selecting exactly one defective item -/
def prob_one_defective : ℚ :=
  (combinations defective_items 1 * combinations genuine_items 1) /
  combinations total_items items_selected

theorem prob_one_defective_is_half :
  prob_one_defective = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_prob_one_defective_is_half_l3458_345815


namespace NUMINAMATH_CALUDE_triangle_translation_l3458_345806

-- Define a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a triangle using three points
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

-- Define a translation function
def translate (p : Point) (dx dy : ℝ) : Point :=
  { x := p.x + dx, y := p.y + dy }

-- Define the theorem
theorem triangle_translation :
  let ABC := Triangle.mk
    (Point.mk (-5) 0)
    (Point.mk 4 0)
    (Point.mk 2 5)
  let E := translate ABC.A 2 (-1)
  let F := translate ABC.B 2 (-1)
  let G := translate ABC.C 2 (-1)
  let EFG := Triangle.mk E F G
  E = Point.mk (-3) (-1) ∧
  F = Point.mk 6 (-1) ∧
  G = Point.mk 4 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_translation_l3458_345806


namespace NUMINAMATH_CALUDE_product_digit_permutation_l3458_345867

theorem product_digit_permutation :
  ∃ (x : ℕ) (A B C D : ℕ),
    x * (x + 1) = 1000 * A + 100 * B + 10 * C + D ∧
    (x - 3) * (x - 2) = 1000 * C + 100 * A + 10 * B + D ∧
    (x - 30) * (x - 29) = 1000 * B + 100 * C + 10 * A + D ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
    x = 91 ∧ A = 8 ∧ B = 3 ∧ C = 7 ∧ D = 2 :=
by sorry

end NUMINAMATH_CALUDE_product_digit_permutation_l3458_345867


namespace NUMINAMATH_CALUDE_cube_volume_from_paper_l3458_345891

theorem cube_volume_from_paper (paper_length paper_width : ℝ) 
  (h1 : paper_length = 48)
  (h2 : paper_width = 72)
  (h3 : 1 = 12) : -- 1 foot = 12 inches
  let paper_area := paper_length * paper_width
  let cube_face_area := paper_area / 6
  let cube_side_length := Real.sqrt cube_face_area
  let cube_side_length_feet := cube_side_length / 12
  cube_side_length_feet ^ 3 = 8 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_paper_l3458_345891


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l3458_345899

/-- The volume of a rectangular prism with face areas √3, √5, and √15 is √15 -/
theorem rectangular_prism_volume (x y z : ℝ) 
  (h1 : x * y = Real.sqrt 3)
  (h2 : x * z = Real.sqrt 5)
  (h3 : y * z = Real.sqrt 15) :
  x * y * z = Real.sqrt 15 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l3458_345899


namespace NUMINAMATH_CALUDE_gcd_90_150_l3458_345805

theorem gcd_90_150 : Nat.gcd 90 150 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcd_90_150_l3458_345805


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l3458_345821

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p, p < k → Nat.Prime p → ¬(p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 529) ∧
  (has_no_prime_factors_less_than 529 20) ∧
  (∀ m : ℕ, m < 529 → ¬(is_composite m ∧ has_no_prime_factors_less_than m 20)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l3458_345821


namespace NUMINAMATH_CALUDE_optimal_selling_price_l3458_345833

/-- Represents the problem of finding the optimal selling price -/
def OptimalSellingPrice (purchase_price initial_price initial_volume : ℝ)
                        (volume_decrease_rate : ℝ) (target_profit : ℝ) : Prop :=
  let price_increase (x : ℝ) := initial_price + x
  let volume (x : ℝ) := initial_volume - volume_decrease_rate * x
  let profit (x : ℝ) := (price_increase x - purchase_price) * volume x
  ∃ x : ℝ, profit x = target_profit ∧ (price_increase x = 60 ∨ price_increase x = 80)

/-- The main theorem stating the optimal selling price -/
theorem optimal_selling_price :
  OptimalSellingPrice 40 50 500 10 8000 := by
  sorry

#check optimal_selling_price

end NUMINAMATH_CALUDE_optimal_selling_price_l3458_345833


namespace NUMINAMATH_CALUDE_sine_monotonicity_implies_omega_range_l3458_345803

open Real

theorem sine_monotonicity_implies_omega_range 
  (f : ℝ → ℝ) (ω : ℝ) (h_pos : ω > 0) :
  (∀ x ∈ Set.Ioo (π/2) π, 
    ∀ y ∈ Set.Ioo (π/2) π, 
    x < y → f x < f y) →
  (∀ x, f x = 2 * sin (ω * x + π/6)) →
  0 < ω ∧ ω ≤ 1/3 := by
sorry

end NUMINAMATH_CALUDE_sine_monotonicity_implies_omega_range_l3458_345803


namespace NUMINAMATH_CALUDE_rational_solution_exists_l3458_345814

theorem rational_solution_exists : ∃ (a b : ℚ), (a ≠ 0) ∧ (a + b ≠ 0) ∧ ((a + b) / a + a / (a + b) = b) := by
  sorry

end NUMINAMATH_CALUDE_rational_solution_exists_l3458_345814
