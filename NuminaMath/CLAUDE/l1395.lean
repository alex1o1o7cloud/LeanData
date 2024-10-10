import Mathlib

namespace virginia_eggs_remaining_l1395_139563

/-- Given Virginia starts with 96 eggs and Amy takes 3 eggs away, 
    prove that Virginia ends up with 93 eggs. -/
theorem virginia_eggs_remaining : 
  let initial_eggs : ℕ := 96
  let eggs_taken : ℕ := 3
  initial_eggs - eggs_taken = 93 := by sorry

end virginia_eggs_remaining_l1395_139563


namespace triangle_area_range_line_equation_l1395_139596

/-- Ellipse C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 / 6 + y^2 / 4 = 1

/-- Circle C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 = 2

/-- Point on ellipse C₁ -/
def point_on_C₁ (P : ℝ × ℝ) : Prop := C₁ P.1 P.2

/-- Line passing through (-1, 0) -/
def line_through_M (l : ℝ → ℝ) : Prop := l 0 = -1

/-- Intersection points of line l with C₁ and C₂ -/
def intersection_points (l : ℝ → ℝ) (A B C D : ℝ × ℝ) : Prop :=
  point_on_C₁ A ∧ point_on_C₁ D ∧ C₂ B.1 B.2 ∧ C₂ C.1 C.2 ∧
  A.2 > B.2 ∧ B.2 > C.2 ∧ C.2 > D.2 ∧
  (∀ y, l y = A.1 ↔ y = A.2) ∧ (∀ y, l y = B.1 ↔ y = B.2) ∧
  (∀ y, l y = C.1 ↔ y = C.2) ∧ (∀ y, l y = D.1 ↔ y = D.2)

/-- Theorem 1: Range of triangle area -/
theorem triangle_area_range :
  ∀ P : ℝ × ℝ, point_on_C₁ P →
  ∃ S : ℝ, 1 ≤ S ∧ S ≤ Real.sqrt 2 ∧
  (∃ Q : ℝ × ℝ, C₂ Q.1 Q.2 ∧ S = (1/2) * Real.sqrt ((P.1^2 + P.2^2) * 2 - (P.1 * Q.1 + P.2 * Q.2)^2)) :=
sorry

/-- Theorem 2: Equation of line l -/
theorem line_equation :
  ∀ l : ℝ → ℝ, line_through_M l →
  (∃ A B C D : ℝ × ℝ, intersection_points l A B C D ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2) →
  (∀ y, l y = -1) :=
sorry

end triangle_area_range_line_equation_l1395_139596


namespace decagon_diagonals_l1395_139510

/-- The number of sides in a decagon -/
def decagon_sides : ℕ := 10

/-- Formula for calculating the number of diagonals in a polygon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem stating that the number of diagonals in a regular decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end decagon_diagonals_l1395_139510


namespace slope_range_theorem_l1395_139531

-- Define a line by its slope and a point it passes through
def Line (k : ℝ) (x₀ y₀ : ℝ) :=
  {(x, y) : ℝ × ℝ | y - y₀ = k * (x - x₀)}

-- Define the translation of a line
def translate (L : Set (ℝ × ℝ)) (dx dy : ℝ) :=
  {(x, y) : ℝ × ℝ | (x - dx, y - dy) ∈ L}

-- Define the fourth quadrant
def fourthQuadrant := {(x, y) : ℝ × ℝ | x > 0 ∧ y < 0}

theorem slope_range_theorem (k : ℝ) :
  let l := Line k 1 (-1)
  let m := translate l 3 (-2)
  (∀ p ∈ m, p ∉ fourthQuadrant) → 0 ≤ k ∧ k ≤ 1/4 := by
  sorry

end slope_range_theorem_l1395_139531


namespace solution_implies_a_value_l1395_139540

theorem solution_implies_a_value (a : ℝ) : 
  (∃ x : ℝ, x = 4 ∧ a * x - 3 = 4 * x + 1) → a = 5 := by
sorry

end solution_implies_a_value_l1395_139540


namespace monotone_decreasing_implies_a_range_a_range_l1395_139593

/-- A function f(x) = x^3 - ax that is monotonically decreasing on (-1/2, 0) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

/-- The property of f being monotonically decreasing on (-1/2, 0) -/
def is_monotone_decreasing (a : ℝ) : Prop :=
  ∀ x y, -1/2 < x ∧ x < y ∧ y < 0 → f a x > f a y

/-- The theorem stating that if f is monotonically decreasing on (-1/2, 0), then a ≥ 3/4 -/
theorem monotone_decreasing_implies_a_range (a : ℝ) :
  is_monotone_decreasing a → a ≥ 3/4 := by sorry

/-- The main theorem proving the range of a -/
theorem a_range : 
  {a : ℝ | is_monotone_decreasing a} = {a : ℝ | a ≥ 3/4} := by sorry

end monotone_decreasing_implies_a_range_a_range_l1395_139593


namespace ways_to_put_five_balls_three_boxes_l1395_139592

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def ways_to_put_balls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem ways_to_put_five_balls_three_boxes : ways_to_put_balls 5 3 = 243 := by
  sorry

end ways_to_put_five_balls_three_boxes_l1395_139592


namespace simplify_square_roots_l1395_139524

theorem simplify_square_roots : 
  (Real.sqrt 507 / Real.sqrt 48) - (Real.sqrt 175 / Real.sqrt 112) = 2 := by
  sorry

end simplify_square_roots_l1395_139524


namespace investment_rate_proof_l1395_139575

/-- Proves that given an investment scenario, the unknown rate is 1% -/
theorem investment_rate_proof (total_investment : ℝ) (amount_at_10_percent : ℝ) (total_interest : ℝ)
  (h1 : total_investment = 31000)
  (h2 : amount_at_10_percent = 12000)
  (h3 : total_interest = 1390)
  : (total_interest - 0.1 * amount_at_10_percent) / (total_investment - amount_at_10_percent) = 0.01 := by
  sorry

end investment_rate_proof_l1395_139575


namespace min_value_sum_fractions_l1395_139530

theorem min_value_sum_fractions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ (3 / 2) ∧
  (a / (b + c) + b / (c + a) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c) := by
  sorry

end min_value_sum_fractions_l1395_139530


namespace jerrys_breakfast_calories_l1395_139582

/-- Given Jerry's breakfast composition and total calories, prove the calories per pancake. -/
theorem jerrys_breakfast_calories (pancakes : ℕ) (bacon_strips : ℕ) (bacon_calories : ℕ) 
  (cereal_calories : ℕ) (total_calories : ℕ) (calories_per_pancake : ℕ) :
  pancakes = 6 →
  bacon_strips = 2 →
  bacon_calories = 100 →
  cereal_calories = 200 →
  total_calories = 1120 →
  total_calories = pancakes * calories_per_pancake + bacon_strips * bacon_calories + cereal_calories →
  calories_per_pancake = 120 := by
  sorry

end jerrys_breakfast_calories_l1395_139582


namespace x_equals_cos_alpha_l1395_139515

/-- Given two squares with side length 1/2 inclined at an angle 2α, 
    x is the length of the line segment connecting the midpoints of 
    the non-intersecting sides of the squares -/
def x (α : Real) : Real :=
  sorry

theorem x_equals_cos_alpha (α : Real) : x α = Real.cos α := by
  sorry

end x_equals_cos_alpha_l1395_139515


namespace probability_all_white_or_all_black_l1395_139554

def white_balls : ℕ := 7
def black_balls : ℕ := 8
def total_balls : ℕ := white_balls + black_balls
def drawn_balls : ℕ := 5

theorem probability_all_white_or_all_black :
  (Nat.choose white_balls drawn_balls + Nat.choose black_balls drawn_balls) / Nat.choose total_balls drawn_balls = 77 / 3003 :=
by sorry

end probability_all_white_or_all_black_l1395_139554


namespace min_value_n_over_m_l1395_139508

theorem min_value_n_over_m (m n : ℝ) :
  (∀ x : ℝ, Real.exp x - m * x + n - 1 ≥ 0) →
  (∃ k : ℝ, k = n / m ∧ k ≥ 0 ∧ ∀ j : ℝ, (∀ x : ℝ, Real.exp x - m * x + j * m - 1 ≥ 0) → j ≥ k) :=
by sorry

end min_value_n_over_m_l1395_139508


namespace transformation_correct_l1395_139535

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]
def mirror_scale_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, -2]
def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -2; 2, 0]

theorem transformation_correct :
  mirror_scale_matrix * rotation_matrix = transformation_matrix :=
by sorry

end transformation_correct_l1395_139535


namespace white_area_is_40_l1395_139589

/-- Represents a rectangular bar with given width and height -/
structure Bar where
  width : ℕ
  height : ℕ

/-- Represents a letter composed of rectangular bars -/
structure Letter where
  bars : List Bar

def sign_width : ℕ := 18
def sign_height : ℕ := 6

def letter_F : Letter := ⟨[{width := 4, height := 1}, {width := 4, height := 1}, {width := 1, height := 6}]⟩
def letter_O : Letter := ⟨[{width := 1, height := 6}, {width := 1, height := 6}, {width := 4, height := 1}, {width := 4, height := 1}]⟩
def letter_D : Letter := ⟨[{width := 1, height := 6}, {width := 4, height := 1}, {width := 1, height := 4}]⟩

def word : List Letter := [letter_F, letter_O, letter_O, letter_D]

def total_sign_area : ℕ := sign_width * sign_height

def letter_area (l : Letter) : ℕ :=
  l.bars.map (fun b => b.width * b.height) |> List.sum

def total_black_area : ℕ :=
  word.map letter_area |> List.sum

theorem white_area_is_40 : total_sign_area - total_black_area = 40 := by
  sorry

end white_area_is_40_l1395_139589


namespace hyperbola_triangle_area_l1395_139573

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 6 = 1 ∧ a > 0

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) (a : ℝ) : Prop :=
  hyperbola a F₁.1 F₁.2 ∧ hyperbola a F₂.1 F₂.2

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) (a : ℝ) : Prop :=
  hyperbola a A.1 A.2 ∧ hyperbola a B.1 B.2

-- Define the distance condition
def distance_condition (A F₁ : ℝ × ℝ) (a : ℝ) : Prop :=
  Real.sqrt ((A.1 - F₁.1)^2 + (A.2 - F₁.2)^2) = 2 * a

-- Define the angle condition
def angle_condition (F₁ A F₂ : ℝ × ℝ) : Prop :=
  Real.arccos (
    ((F₁.1 - A.1) * (F₂.1 - A.1) + (F₁.2 - A.2) * (F₂.2 - A.2)) /
    (Real.sqrt ((F₁.1 - A.1)^2 + (F₁.2 - A.2)^2) * Real.sqrt ((F₂.1 - A.1)^2 + (F₂.2 - A.2)^2))
  ) = 2 * Real.pi / 3

-- State the theorem
theorem hyperbola_triangle_area
  (a : ℝ)
  (F₁ F₂ A B : ℝ × ℝ) :
  foci F₁ F₂ a →
  intersection_points A B a →
  distance_condition A F₁ a →
  angle_condition F₁ A F₂ →
  Real.sqrt 3 * (Real.sqrt ((F₁.1 - B.1)^2 + (F₁.2 - B.2)^2) * Real.sqrt ((F₂.1 - B.1)^2 + (F₂.2 - B.2)^2) * Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2)) / 4 = 6 * Real.sqrt 3 :=
sorry

end hyperbola_triangle_area_l1395_139573


namespace feet_in_garden_l1395_139586

theorem feet_in_garden (num_dogs num_ducks : ℕ) (dog_feet duck_feet : ℕ) :
  num_dogs = 6 → num_ducks = 2 → dog_feet = 4 → duck_feet = 2 →
  num_dogs * dog_feet + num_ducks * duck_feet = 28 := by
sorry

end feet_in_garden_l1395_139586


namespace contrapositive_equivalence_l1395_139598

theorem contrapositive_equivalence (x : ℝ) :
  (¬(-1 < x ∧ x < 0) ∨ x^2 < 1) ↔ (x^2 ≥ 1 → (x ≥ 0 ∨ x ≤ -1)) :=
by sorry

end contrapositive_equivalence_l1395_139598


namespace subtracted_number_l1395_139502

theorem subtracted_number (x : ℤ) : 88 - x = 54 → x = 34 := by
  sorry

end subtracted_number_l1395_139502


namespace x_fourth_plus_inverse_x_fourth_l1395_139541

theorem x_fourth_plus_inverse_x_fourth (x : ℝ) (h : x + 1/x = 8) : x^4 + 1/x^4 = 3842 := by
  sorry

end x_fourth_plus_inverse_x_fourth_l1395_139541


namespace sum_gcf_lcm_8_12_l1395_139512

theorem sum_gcf_lcm_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end sum_gcf_lcm_8_12_l1395_139512


namespace inequality_solution_set_max_m_value_l1395_139513

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2|
def g (m : ℝ) (x : ℝ) : ℝ := -|x + 3| + m

-- Theorem for the solution set of the inequality
theorem inequality_solution_set (a : ℝ) :
  (∀ x, f x + a - 1 > 0 ↔ 
    (a = 1 ∧ x ≠ 2) ∨
    (a > 1) ∨
    (a < 1 ∧ (x < a + 1 ∨ x > 3 - a))) :=
sorry

-- Theorem for the maximum value of m
theorem max_m_value :
  ∀ m, (∀ x, f x > g m x) ↔ m < 5 :=
sorry

end inequality_solution_set_max_m_value_l1395_139513


namespace smallest_gcd_bc_l1395_139591

theorem smallest_gcd_bc (a b c : ℕ+) (h1 : Nat.gcd a b = 960) (h2 : Nat.gcd a c = 324) :
  ∃ (d : ℕ), d = Nat.gcd b c ∧ d = 12 ∧ ∀ (e : ℕ), e = Nat.gcd b c → e ≥ d :=
by sorry

end smallest_gcd_bc_l1395_139591


namespace compare_values_l1395_139555

theorem compare_values : 0.5^(1/10) > 0.4^(1/10) ∧ 0.4^(1/10) > Real.log 0.1 / Real.log 4 := by
  sorry

end compare_values_l1395_139555


namespace product_of_arithmetic_sequences_l1395_139511

/-- An arithmetic sequence -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The product sequence of two arithmetic sequences -/
def product_seq (a b : ℕ → ℝ) (c : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, c n = a n * b n

theorem product_of_arithmetic_sequences
  (a b : ℕ → ℝ) (c : ℕ → ℝ)
  (ha : arithmetic_seq a)
  (hb : arithmetic_seq b)
  (hc : product_seq a b c)
  (h1 : c 1 = 1440)
  (h2 : c 2 = 1716)
  (h3 : c 3 = 1848) :
  c 8 = 348 := by
  sorry

end product_of_arithmetic_sequences_l1395_139511


namespace axis_of_symmetry_l1395_139534

-- Define a function f with the given property
def f : ℝ → ℝ := sorry

-- State the condition that f(x) = f(4-x) for all x
axiom f_symmetry (x : ℝ) : f x = f (4 - x)

-- Define what it means for a line to be an axis of symmetry
def is_axis_of_symmetry (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

-- Theorem statement
theorem axis_of_symmetry :
  is_axis_of_symmetry 2 :=
sorry

end axis_of_symmetry_l1395_139534


namespace isabel_camera_pictures_l1395_139545

/-- Represents the number of pictures in Isabel's photo upload scenario -/
structure IsabelPictures where
  phone : ℕ
  camera : ℕ
  albums : ℕ
  pics_per_album : ℕ

/-- The theorem stating the number of pictures Isabel uploaded from her camera -/
theorem isabel_camera_pictures (p : IsabelPictures) 
  (h1 : p.phone = 2)
  (h2 : p.albums = 3)
  (h3 : p.pics_per_album = 2)
  (h4 : p.albums * p.pics_per_album = p.phone + p.camera) :
  p.camera = 4 := by
  sorry

#check isabel_camera_pictures

end isabel_camera_pictures_l1395_139545


namespace custom_chess_pieces_l1395_139595

theorem custom_chess_pieces (num_players : Nat) (std_pieces_per_player : Nat)
  (missing_queens : Nat) (missing_knights : Nat) (missing_pawns : Nat)
  (h1 : num_players = 3)
  (h2 : std_pieces_per_player = 16)
  (h3 : missing_queens = 2)
  (h4 : missing_knights = 5)
  (h5 : missing_pawns = 8) :
  let total_missing := missing_queens + missing_knights + missing_pawns
  let total_original := num_players * std_pieces_per_player
  let pieces_per_player := (total_original - total_missing) / num_players
  (pieces_per_player = 11) ∧ (total_original - total_missing = 33) := by
  sorry

end custom_chess_pieces_l1395_139595


namespace jennifer_book_fraction_l1395_139528

theorem jennifer_book_fraction (total : ℚ) (sandwich_fraction : ℚ) (museum_fraction : ℚ) (leftover : ℚ) :
  total = 90 →
  sandwich_fraction = 1 / 5 →
  museum_fraction = 1 / 6 →
  leftover = 12 →
  let spent := total - leftover
  let sandwich_cost := total * sandwich_fraction
  let museum_cost := total * museum_fraction
  let book_cost := spent - sandwich_cost - museum_cost
  book_cost / total = 1 / 2 := by sorry

end jennifer_book_fraction_l1395_139528


namespace sum_of_two_numbers_l1395_139597

theorem sum_of_two_numbers (smaller larger : ℕ) : 
  smaller = 31 → larger = 3 * smaller → smaller + larger = 124 := by
  sorry

end sum_of_two_numbers_l1395_139597


namespace power_expression_equality_l1395_139581

theorem power_expression_equality (c d : ℝ) 
  (h1 : (80 : ℝ) ^ c = 4)
  (h2 : (80 : ℝ) ^ d = 5) :
  (16 : ℝ) ^ ((1 - c - d) / (2 * (1 - d))) = 4 := by
  sorry

end power_expression_equality_l1395_139581


namespace largest_four_digit_congruent_to_15_mod_25_l1395_139587

theorem largest_four_digit_congruent_to_15_mod_25 : ∃ (n : ℕ), 
  n ≤ 9990 ∧ 
  1000 ≤ n ∧ 
  n < 10000 ∧ 
  n ≡ 15 [MOD 25] ∧
  ∀ (m : ℕ), (1000 ≤ m ∧ m < 10000 ∧ m ≡ 15 [MOD 25]) → m ≤ n :=
by sorry

end largest_four_digit_congruent_to_15_mod_25_l1395_139587


namespace larger_number_of_product_56_sum_15_l1395_139578

theorem larger_number_of_product_56_sum_15 (x y : ℕ) : 
  x * y = 56 → x + y = 15 → max x y = 8 := by
  sorry

end larger_number_of_product_56_sum_15_l1395_139578


namespace triangle_inequality_l1395_139580

theorem triangle_inequality (a b ma mb : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : ma > 0) (h4 : mb > 0) (h5 : a > b) :
  a * ma = b * mb →
  a^2010 + ma^2010 ≥ b^2010 + mb^2010 := by
sorry

end triangle_inequality_l1395_139580


namespace hyperbola_equation_l1395_139504

/-- Given a hyperbola and a parabola satisfying certain conditions, 
    prove that the hyperbola has a specific equation. -/
theorem hyperbola_equation 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (asymptote : b/a = Real.sqrt 3) 
  (focus_on_directrix : a^2 + b^2 = 36) : 
  a^2 = 9 ∧ b^2 = 27 :=
sorry

end hyperbola_equation_l1395_139504


namespace cantor_set_max_operation_l1395_139565

theorem cantor_set_max_operation : 
  ∃ n : ℕ, (∀ k : ℕ, k > n → (2/3 : ℝ)^(k-1) * (1/3) < 1/60) ∧ 
           (2/3 : ℝ)^(n-1) * (1/3) ≥ 1/60 ∧ 
           n = 8 :=
sorry

end cantor_set_max_operation_l1395_139565


namespace log_less_than_zero_implies_x_between_zero_and_one_l1395_139560

theorem log_less_than_zero_implies_x_between_zero_and_one (x : ℝ) :
  (∃ (y : ℝ), y = Real.log x ∧ y < 0) → 0 < x ∧ x < 1 := by
  sorry

end log_less_than_zero_implies_x_between_zero_and_one_l1395_139560


namespace intersection_trajectory_l1395_139532

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the endpoints of the major axis
def majorAxisEndpoints (A₁ A₂ : ℝ × ℝ) : Prop :=
  A₁ = (-3, 0) ∧ A₂ = (3, 0)

-- Define a chord perpendicular to the major axis
def perpendicularChord (P₁ P₂ : ℝ × ℝ) : Prop :=
  ellipse P₁.1 P₁.2 ∧ ellipse P₂.1 P₂.2 ∧ P₁.1 = P₂.1 ∧ P₁.2 = -P₂.2

-- Define the intersection point of A₁P₁ and A₂P₂
def intersectionPoint (Q : ℝ × ℝ) (A₁ A₂ P₁ P₂ : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ,
    Q = (1 - t₁) • A₁ + t₁ • P₁ ∧
    Q = (1 - t₂) • A₂ + t₂ • P₂

-- The theorem to be proved
theorem intersection_trajectory
  (A₁ A₂ P₁ P₂ Q : ℝ × ℝ)
  (h₁ : majorAxisEndpoints A₁ A₂)
  (h₂ : perpendicularChord P₁ P₂)
  (h₃ : intersectionPoint Q A₁ A₂ P₁ P₂) :
  Q.1^2 / 9 - Q.2^2 / 4 = 1 := by
  sorry

end intersection_trajectory_l1395_139532


namespace triangle_properties_l1395_139503

/-- Given a triangle ABC with the following properties:
  * The area of the triangle is 3√15
  * b - c = 2, where b and c are sides of the triangle
  * cos A = -1/4, where A is an angle of the triangle
This theorem proves specific values for a, sin C, and cos(2A + π/6) -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (h_area : (1/2) * b * c * Real.sin A = 3 * Real.sqrt 15)
  (h_sides : b - c = 2)
  (h_cos_A : Real.cos A = -1/4) :
  a = 8 ∧ 
  Real.sin C = Real.sqrt 15 / 8 ∧
  Real.cos (2 * A + π/6) = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16 := by
  sorry

end triangle_properties_l1395_139503


namespace log_equation_solution_l1395_139544

theorem log_equation_solution (x : ℝ) :
  Real.log x / Real.log 8 = 1.75 → x = 32 * Real.sqrt (Real.sqrt 2) := by
  sorry

end log_equation_solution_l1395_139544


namespace min_value_expression_l1395_139546

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  (1 / (2 * a)) + (1 / (2 * b)) + (8 / (a + b)) ≥ 4 := by
  sorry

end min_value_expression_l1395_139546


namespace fraction_sum_difference_l1395_139562

theorem fraction_sum_difference (p q r s : ℚ) 
  (h1 : p / q = 4 / 5) 
  (h2 : r / s = 3 / 7) : 
  (18 / 7) + ((2 * q - p) / (2 * q + p)) - ((3 * s + r) / (3 * s - r)) = 5 / 3 := by
  sorry

end fraction_sum_difference_l1395_139562


namespace picture_distribution_l1395_139509

theorem picture_distribution (total : ℕ) (main_album : ℕ) (other_albums : ℕ) 
  (h1 : total = 33) 
  (h2 : main_album = 27) 
  (h3 : other_albums = 3) :
  (total - main_album) / other_albums = 2 :=
by
  sorry

end picture_distribution_l1395_139509


namespace quartic_equation_roots_l1395_139599

theorem quartic_equation_roots : 
  let f (x : ℝ) := 4*x^4 - 28*x^3 + 53*x^2 - 28*x + 4
  ∀ x : ℝ, f x = 0 ↔ x = 4 ∨ x = 2 ∨ x = (1/4 : ℝ) ∨ x = (1/2 : ℝ) := by
  sorry

end quartic_equation_roots_l1395_139599


namespace range_of_m_when_a_is_one_range_of_a_for_sufficient_condition_l1395_139558

-- Define propositions p and q
def p (m a : ℝ) : Prop := m^2 - 7*a*m + 12*a^2 < 0 ∧ a > 0

def q (m : ℝ) : Prop := ∃ (x y : ℝ), x^2/(m-1) + y^2/(6-m) = 1 ∧ 1 < m ∧ m < 6

-- Theorem for part 1
theorem range_of_m_when_a_is_one :
  ∀ m : ℝ, (p m 1 ∧ q m) → (3 < m ∧ m < 7/2) :=
sorry

-- Theorem for part 2
theorem range_of_a_for_sufficient_condition :
  (∀ m a : ℝ, ¬(q m) → ¬(p m a)) ∧ (∃ m a : ℝ, ¬(p m a) ∧ q m) →
  (∀ a : ℝ, 1/3 ≤ a ∧ a ≤ 7/8) :=
sorry

end range_of_m_when_a_is_one_range_of_a_for_sufficient_condition_l1395_139558


namespace divisibility_by_five_l1395_139517

theorem divisibility_by_five (a b : ℕ) (h : 5 ∣ (a * b)) : 5 ∣ a ∨ 5 ∣ b := by
  sorry

end divisibility_by_five_l1395_139517


namespace adam_tickets_left_l1395_139522

/-- The number of tickets Adam had left after riding the ferris wheel -/
def tickets_left (initial_tickets : ℕ) (ticket_cost : ℕ) (spent_on_ride : ℕ) : ℕ :=
  initial_tickets - (spent_on_ride / ticket_cost)

/-- Theorem stating that Adam had 4 tickets left after riding the ferris wheel -/
theorem adam_tickets_left :
  tickets_left 13 9 81 = 4 := by
  sorry

#eval tickets_left 13 9 81

end adam_tickets_left_l1395_139522


namespace activity_ranking_l1395_139564

def fishing_popularity : ℚ := 13/36
def hiking_popularity : ℚ := 8/27
def painting_popularity : ℚ := 7/18

theorem activity_ranking :
  painting_popularity > fishing_popularity ∧
  fishing_popularity > hiking_popularity := by
  sorry

end activity_ranking_l1395_139564


namespace remainder_after_adding_2040_l1395_139548

theorem remainder_after_adding_2040 (n : ℤ) (h : n % 8 = 3) : (n + 2040) % 8 = 3 := by
  sorry

end remainder_after_adding_2040_l1395_139548


namespace sum_of_digits_of_9N_l1395_139527

/-- A function that checks if each digit of a natural number is strictly greater than the digit to its left -/
def is_strictly_increasing_digits (n : ℕ) : Prop :=
  ∀ i j, i < j → (n.digits 10).get i < (n.digits 10).get j

/-- A function that calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Theorem: For any natural number N where each digit is strictly greater than the digit to its left,
    the sum of the digits of 9N is equal to 9 -/
theorem sum_of_digits_of_9N (N : ℕ) (h : is_strictly_increasing_digits N) :
  sum_of_digits (9 * N) = 9 :=
by sorry

end sum_of_digits_of_9N_l1395_139527


namespace polygonal_number_theorem_l1395_139553

/-- The n-th k-sided polygonal number -/
def N (n k : ℕ) : ℚ :=
  (k - 2) / 2 * n^2 + (4 - k) / 2 * n

/-- Theorem stating the formula for the n-th k-sided polygonal number and the value of N(8,12) -/
theorem polygonal_number_theorem (n k : ℕ) (h1 : k ≥ 3) (h2 : n ≥ 1) : 
  N n k = (k - 2) / 2 * n^2 + (4 - k) / 2 * n ∧ N 8 12 = 288 := by
  sorry

end polygonal_number_theorem_l1395_139553


namespace original_ratio_l1395_139561

theorem original_ratio (x y : ℝ) (h1 : y = 40) (h2 : (x + 10) / (y + 10) = 4/5) :
  x / y = 3 / 4 := by
  sorry

end original_ratio_l1395_139561


namespace circle_circumference_from_chord_l1395_139547

/-- Given a circular path with 8 evenly spaced trees, where the direct distance
    between two trees separated by 3 intervals is 100 feet, the total
    circumference of the circle is 175 feet. -/
theorem circle_circumference_from_chord (n : ℕ) (d : ℝ) (h1 : n = 8) (h2 : d = 100) :
  let interval := d / 4
  let circumference := interval * 7
  circumference = 175 := by
sorry

end circle_circumference_from_chord_l1395_139547


namespace chord_length_l1395_139572

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length (a b c : ℝ) (r : ℝ) (h1 : r > 0) :
  let line := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}
  let d := |c| / Real.sqrt (a^2 + b^2)
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  (∀ p ∈ line ∩ circle, True) →
  chord_length = Real.sqrt 14 :=
by sorry

end chord_length_l1395_139572


namespace arithmetic_geometric_sequence_properties_l1395_139538

-- Define the arithmetic-geometric sequence and its properties
def arithmetic_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n : ℕ, a (n + 1) = r * a n

-- Define S_n as the sum of the first n terms of a_n
def S (a : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => S a n + a (n + 1)

-- Define T_n as the sum of the first n terms of S_n
def T (S : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => T S n + S (n + 1)

theorem arithmetic_geometric_sequence_properties
  (a : ℕ → ℝ)
  (h_ag : arithmetic_geometric_sequence a)
  (h_S3 : S a 3 = 7)
  (h_S6 : S a 6 = 63) :
  (∀ n : ℕ, a n = 2^(n - 1)) ∧
  (∀ n : ℕ, S a n = 2^n - 1) ∧
  (∀ n : ℕ, T (S a) n = 2^(n + 1) - n - 2) := by
  sorry

end arithmetic_geometric_sequence_properties_l1395_139538


namespace smallest_among_given_numbers_l1395_139556

theorem smallest_among_given_numbers :
  let numbers : List ℚ := [-6/7, 2, 0, -1]
  ∀ x ∈ numbers, -1 ≤ x :=
by sorry

end smallest_among_given_numbers_l1395_139556


namespace folded_square_FG_length_l1395_139583

/-- A folded square sheet of paper with side length 1 -/
structure FoldedSquare where
  /-- The point where corners B and D meet after folding -/
  E : ℝ × ℝ
  /-- The point F on side AB -/
  F : ℝ × ℝ
  /-- The point G on side AD -/
  G : ℝ × ℝ
  /-- E lies on the diagonal AC -/
  E_on_diagonal : E.1 = E.2
  /-- F is on side AB -/
  F_on_AB : F.2 = 0 ∧ 0 ≤ F.1 ∧ F.1 ≤ 1
  /-- G is on side AD -/
  G_on_AD : G.1 = 0 ∧ 0 ≤ G.2 ∧ G.2 ≤ 1

/-- The theorem stating that the length of FG in a folded unit square is 2√2 - 2 -/
theorem folded_square_FG_length (s : FoldedSquare) : 
  Real.sqrt ((s.F.1 - s.G.1)^2 + (s.F.2 - s.G.2)^2) = 2 * Real.sqrt 2 - 2 := by
  sorry

end folded_square_FG_length_l1395_139583


namespace carrie_work_duration_l1395_139519

def hourly_wage : ℝ := 8
def weekly_hours : ℝ := 35
def bike_cost : ℝ := 400
def money_left : ℝ := 720

theorem carrie_work_duration :
  (money_left + bike_cost) / (hourly_wage * weekly_hours) = 4 := by
  sorry

end carrie_work_duration_l1395_139519


namespace sqrt_plus_one_iff_ax_plus_x_over_x_minus_one_l1395_139537

theorem sqrt_plus_one_iff_ax_plus_x_over_x_minus_one 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt a + 1 > b ↔ ∀ x > 1, a * x + x / (x - 1) > b := by
sorry

end sqrt_plus_one_iff_ax_plus_x_over_x_minus_one_l1395_139537


namespace work_duration_l1395_139579

/-- Given that x does a work in 20 days and x and y together do the same work in 40/3 days,
    prove that y does the work in 40 days. -/
theorem work_duration (x y : ℝ) (h1 : x = 20) (h2 : 1 / x + 1 / y = 3 / 40) : y = 40 := by
  sorry

end work_duration_l1395_139579


namespace motorcycle_license_combinations_l1395_139574

def letter_choices : ℕ := 3
def digit_choices : ℕ := 10
def license_length : ℕ := 4

theorem motorcycle_license_combinations : 
  letter_choices * digit_choices ^ license_length = 30000 := by
  sorry

end motorcycle_license_combinations_l1395_139574


namespace min_max_sum_l1395_139549

theorem min_max_sum (a b c d e : ℕ+) (h : a + b + c + d + e = 2020) :
  (max (a + b) (max (a + d) (max (b + e) (c + d)))) ≥ 1011 :=
sorry

end min_max_sum_l1395_139549


namespace perfect_cube_base9_last_digit_l1395_139506

/-- Represents a number in base 9 of the form ab4c -/
structure Base9Number where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : a ≠ 0
  h2 : c ≤ 8

/-- Converts a Base9Number to its decimal representation -/
def toDecimal (n : Base9Number) : ℕ :=
  729 * n.a + 81 * n.b + 36 + n.c

/-- Predicate to check if a natural number is a perfect cube -/
def isPerfectCube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem perfect_cube_base9_last_digit 
  (n : Base9Number) 
  (h : isPerfectCube (toDecimal n)) : 
  n.c = 1 ∨ n.c = 8 := by
  sorry

end perfect_cube_base9_last_digit_l1395_139506


namespace negative_reciprocal_equality_l1395_139543

theorem negative_reciprocal_equality (a b : ℝ) : 
  (-1 / a = 8) → (-1 / (-b) = 8) → a = b := by sorry

end negative_reciprocal_equality_l1395_139543


namespace unrestricted_x_l1395_139567

theorem unrestricted_x (x y z w : ℤ) 
  (h1 : (x + 2) / (y - 1) < -(z + 3) / (w - 2))
  (h2 : (y - 1) * (w - 2) ≠ 0) :
  ∃ (x_pos x_neg x_zero : ℤ), 
    (x_pos > 0 ∧ (x_pos + 2) / (y - 1) < -(z + 3) / (w - 2)) ∧
    (x_neg < 0 ∧ (x_neg + 2) / (y - 1) < -(z + 3) / (w - 2)) ∧
    (x_zero = 0 ∧ (x_zero + 2) / (y - 1) < -(z + 3) / (w - 2)) :=
by sorry

end unrestricted_x_l1395_139567


namespace investment_total_l1395_139520

/-- Represents the investment scenario with two parts at different interest rates -/
structure Investment where
  total : ℝ
  part1 : ℝ
  part2 : ℝ
  rate1 : ℝ
  rate2 : ℝ
  total_interest : ℝ

/-- The investment satisfies the given conditions -/
def valid_investment (i : Investment) : Prop :=
  i.total = i.part1 + i.part2 ∧
  i.part1 = 2800 ∧
  i.rate1 = 0.03 ∧
  i.rate2 = 0.05 ∧
  i.total_interest = 144 ∧
  i.part1 * i.rate1 + i.part2 * i.rate2 = i.total_interest

/-- Theorem: Given the conditions, the total amount divided is 4000 -/
theorem investment_total (i : Investment) (h : valid_investment i) : i.total = 4000 := by
  sorry

end investment_total_l1395_139520


namespace x_plus_y_fifth_power_l1395_139500

theorem x_plus_y_fifth_power (x y : ℝ) 
  (sum_eq : x + y = 3)
  (frac_eq : 1 / (x + y^2) + 1 / (x^2 + y) = 1 / 2) :
  x^5 + y^5 = 243 := by
  sorry

end x_plus_y_fifth_power_l1395_139500


namespace coefficient_x3y5_in_expansion_l1395_139523

theorem coefficient_x3y5_in_expansion : 
  (Finset.range 9).sum (fun k => 
    if k = 3 then (Nat.choose 8 k : ℕ) 
    else 0) = 56 := by sorry

end coefficient_x3y5_in_expansion_l1395_139523


namespace xy_squared_l1395_139588

theorem xy_squared (x y : ℝ) (h1 : x + y = 20) (h2 : 2*x + y = 27) : (x + y)^2 = 400 := by
  sorry

end xy_squared_l1395_139588


namespace regular_polygon_reciprocal_sum_l1395_139529

/-- Given a regular polygon with n sides, where the reciprocal of the side length
    equals the sum of reciprocals of two specific diagonals, prove that n = 7. -/
theorem regular_polygon_reciprocal_sum (n : ℕ) (R : ℝ) (h_n : n ≥ 3) :
  (1 : ℝ) / (2 * R * Real.sin (π / n)) =
    1 / (2 * R * Real.sin (2 * π / n)) + 1 / (2 * R * Real.sin (3 * π / n)) →
  n = 7 := by
  sorry

end regular_polygon_reciprocal_sum_l1395_139529


namespace distance_between_points_l1395_139577

theorem distance_between_points : ∃ d : ℝ, 
  let A : ℝ × ℝ := (13, 5)
  let B : ℝ × ℝ := (5, -10)
  d = ((A.1 - B.1)^2 + (A.2 - B.2)^2).sqrt ∧ d = 17 := by
sorry

end distance_between_points_l1395_139577


namespace swimmer_speed_in_still_water_l1395_139542

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  man : ℝ  -- Speed of the man in still water
  stream : ℝ  -- Speed of the stream

/-- Calculates the effective speed when swimming downstream. -/
def downstream_speed (s : SwimmerSpeed) : ℝ := s.man + s.stream

/-- Calculates the effective speed when swimming upstream. -/
def upstream_speed (s : SwimmerSpeed) : ℝ := s.man - s.stream

/-- Theorem stating that given the conditions of the problem, the man's speed in still water is 12 km/h. -/
theorem swimmer_speed_in_still_water :
  ∀ (s : SwimmerSpeed),
    54 = downstream_speed s * 3 →
    18 = upstream_speed s * 3 →
    s.man = 12 := by
  sorry


end swimmer_speed_in_still_water_l1395_139542


namespace number_equation_l1395_139505

theorem number_equation (x : ℚ) : (x + 20 / 90) * 90 = 4520 ↔ x = 50 := by
  sorry

end number_equation_l1395_139505


namespace find_a_value_l1395_139516

theorem find_a_value (x y a : ℝ) 
  (h1 : x / (2 * y) = 3 / 2)
  (h2 : (a * x + 6 * y) / (x - 2 * y) = 27) :
  a = 7 := by
  sorry

end find_a_value_l1395_139516


namespace sqrt_product_sqrt_two_times_sqrt_three_eq_sqrt_six_l1395_139525

theorem sqrt_product (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b := by
  sorry

theorem sqrt_two_times_sqrt_three_eq_sqrt_six : 
  Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end sqrt_product_sqrt_two_times_sqrt_three_eq_sqrt_six_l1395_139525


namespace complex_equation_l1395_139584

theorem complex_equation (z : ℂ) (h : Complex.abs z = 1 + 3*I - z) :
  ((1 + I)^2 * (3 + 4*I)^2) / (2 * z) = 3 + 4*I :=
by sorry

end complex_equation_l1395_139584


namespace harolds_money_l1395_139594

theorem harolds_money (x : ℚ) : 
  (x / 2 + 5) +  -- Ticket and candies
  ((x / 2 - 5) / 2 + 10) +  -- Newspaper
  (((x / 2 - 5) / 2 - 10) / 2) +  -- Bus fare
  15 +  -- Beggar
  5  -- Remaining money
  = x  -- Total initial money
  → x = 210 := by sorry

end harolds_money_l1395_139594


namespace total_amount_l1395_139507

/-- Represents the division of money among three people -/
structure MoneyDivision where
  x : ℝ  -- X's share
  y : ℝ  -- Y's share
  z : ℝ  -- Z's share

/-- The conditions of the money division problem -/
def problem_conditions (d : MoneyDivision) : Prop :=
  d.y = 0.75 * d.x ∧ 
  d.z = (2/3) * d.x ∧ 
  d.y = 48

/-- The theorem stating the total amount -/
theorem total_amount (d : MoneyDivision) 
  (h : problem_conditions d) : d.x + d.y + d.z = 154.67 := by
  sorry

#check total_amount

end total_amount_l1395_139507


namespace f_composition_half_l1395_139568

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x|

-- State the theorem
theorem f_composition_half : f (f (1/2)) = 1 := by
  sorry

end f_composition_half_l1395_139568


namespace lighthouse_lights_sum_l1395_139550

theorem lighthouse_lights_sum : 
  let n : ℕ := 7
  let a₁ : ℕ := 1
  let q : ℕ := 2
  let sum := (a₁ * (1 - q^n)) / (1 - q)
  sum = 127 := by
sorry

end lighthouse_lights_sum_l1395_139550


namespace unique_solution_exists_l1395_139557

theorem unique_solution_exists : 
  ∃! (x y z : ℝ), x + y = 3 ∧ x * y - z^3 = 0 ∧ x = 1.5 ∧ y = 1.5 ∧ z = 0 := by
  sorry

end unique_solution_exists_l1395_139557


namespace sufficient_not_necessary_l1395_139551

/-- The line y = kx + 1 and the parabola y^2 = 4x have only one common point -/
def has_one_common_point (k : ℝ) : Prop :=
  ∃! x y, y = k * x + 1 ∧ y^2 = 4 * x

theorem sufficient_not_necessary :
  (∀ k, k = 0 → has_one_common_point k) ∧
  (∃ k, k ≠ 0 ∧ has_one_common_point k) :=
sorry

end sufficient_not_necessary_l1395_139551


namespace at_most_one_greater_than_one_l1395_139559

theorem at_most_one_greater_than_one (x y : ℝ) (h : x + y < 2) :
  ¬(x > 1 ∧ y > 1) := by
  sorry

end at_most_one_greater_than_one_l1395_139559


namespace minimum_value_problem_l1395_139526

theorem minimum_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 6) : 
  (9 / x) + (25 / y) + (49 / z) ≥ 37.5 ∧ 
  ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ x' + y' + z' = 6 ∧ 
    (9 / x') + (25 / y') + (49 / z') = 37.5 := by
  sorry

end minimum_value_problem_l1395_139526


namespace auditorium_seating_l1395_139552

/-- The number of ways to seat people in an auditorium with the given conditions -/
def seatingArrangements (totalPeople : ℕ) (rowSeats : ℕ) : ℕ :=
  Nat.choose totalPeople rowSeats * 2^(totalPeople - 2)

/-- Theorem stating the number of seating arrangements for the given problem -/
theorem auditorium_seating :
  seatingArrangements 100 50 = Nat.choose 100 50 * 2^98 := by
  sorry

end auditorium_seating_l1395_139552


namespace sequence_bounds_l1395_139569

theorem sequence_bounds (n : ℕ+) (a : ℕ → ℚ) 
  (h0 : a 0 = 1/2)
  (h1 : ∀ k, k < n → a (k + 1) = a k + (1/n) * (a k)^2) :
  1 - 1/n < a n ∧ a n < 1 := by
sorry

end sequence_bounds_l1395_139569


namespace storage_tub_cost_l1395_139590

/-- The cost of storage tubs problem -/
theorem storage_tub_cost (total_cost : ℕ) (num_large : ℕ) (num_small : ℕ) (small_cost : ℕ) :
  total_cost = 48 →
  num_large = 3 →
  num_small = 6 →
  small_cost = 5 →
  ∃ (large_cost : ℕ), num_large * large_cost + num_small * small_cost = total_cost ∧ large_cost = 6 :=
by sorry

end storage_tub_cost_l1395_139590


namespace circle_properties_l1395_139585

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4

-- State the theorem
theorem circle_properties :
  -- The circle passes through the origin
  circle_equation 0 0 ∧
  -- The center is on the negative half of the x-axis
  ∃ (a : ℝ), a < 0 ∧ circle_equation a 0 ∧
  -- The radius is 2
  ∀ (x y : ℝ), circle_equation x y → (x + 2)^2 + y^2 = 4 := by sorry

end circle_properties_l1395_139585


namespace perpendicular_lines_from_quadratic_roots_l1395_139576

theorem perpendicular_lines_from_quadratic_roots (b : ℝ) :
  ∀ k₁ k₂ : ℝ, (k₁^2 + b*k₁ - 1 = 0) → (k₂^2 + b*k₂ - 1 = 0) → k₁ * k₂ = -1 :=
by sorry

end perpendicular_lines_from_quadratic_roots_l1395_139576


namespace chef_apples_l1395_139533

theorem chef_apples (apples_left apples_used : ℕ) 
  (h1 : apples_left = 2) 
  (h2 : apples_used = 41) : 
  apples_left + apples_used = 43 := by
  sorry

end chef_apples_l1395_139533


namespace video_game_lives_l1395_139514

/-- The total number of lives for a group of friends in a video game -/
def totalLives (numFriends : ℕ) (livesPerFriend : ℕ) : ℕ :=
  numFriends * livesPerFriend

/-- Theorem: Given 15 friends, each with 25 lives, the total number of lives is 375 -/
theorem video_game_lives : totalLives 15 25 = 375 := by
  sorry

end video_game_lives_l1395_139514


namespace potato_cooking_time_l1395_139536

theorem potato_cooking_time (total_potatoes : ℕ) (cooked_potatoes : ℕ) (remaining_time : ℕ) :
  total_potatoes = 15 →
  cooked_potatoes = 8 →
  remaining_time = 63 →
  (remaining_time / (total_potatoes - cooked_potatoes) : ℚ) = 9 := by
  sorry

end potato_cooking_time_l1395_139536


namespace dave_tshirts_l1395_139518

def white_packs : ℕ := 3
def blue_packs : ℕ := 2
def red_packs : ℕ := 4
def green_packs : ℕ := 1

def white_per_pack : ℕ := 6
def blue_per_pack : ℕ := 4
def red_per_pack : ℕ := 5
def green_per_pack : ℕ := 3

def total_tshirts : ℕ := 
  white_packs * white_per_pack + 
  blue_packs * blue_per_pack + 
  red_packs * red_per_pack + 
  green_packs * green_per_pack

theorem dave_tshirts : total_tshirts = 49 := by
  sorry

end dave_tshirts_l1395_139518


namespace ramon_twice_loui_in_twenty_years_loui_age_is_23_l1395_139521

/-- Ramon's current age -/
def ramon_current_age : ℕ := 26

/-- Loui's current age -/
def loui_current_age : ℕ := 23

/-- In twenty years, Ramon will be twice as old as Loui today -/
theorem ramon_twice_loui_in_twenty_years :
  ramon_current_age + 20 = 2 * loui_current_age := by sorry

theorem loui_age_is_23 : loui_current_age = 23 := by sorry

end ramon_twice_loui_in_twenty_years_loui_age_is_23_l1395_139521


namespace minimum_tents_l1395_139539

theorem minimum_tents (Y : ℕ) : (∃ X : ℕ, 
  X > 0 ∧ 
  10 * (X - 1) < (3 : ℚ) / 2 * Y ∧ (3 : ℚ) / 2 * Y < 10 * X ∧
  10 * (X + 2) < (8 : ℚ) / 5 * Y ∧ (8 : ℚ) / 5 * Y < 10 * (X + 3)) →
  Y ≥ 213 :=
by sorry

end minimum_tents_l1395_139539


namespace lawn_mowing_problem_l1395_139571

theorem lawn_mowing_problem (initial_people : ℕ) (initial_hours : ℕ) (target_hours : ℕ) :
  initial_people = 8 →
  initial_hours = 5 →
  target_hours = 3 →
  ∃ (additional_people : ℕ),
    additional_people = 6 ∧
    (initial_people + additional_people) * target_hours = initial_people * initial_hours :=
by sorry

end lawn_mowing_problem_l1395_139571


namespace gum_distribution_l1395_139570

theorem gum_distribution (num_cousins : Nat) (gum_per_cousin : Nat) : 
  num_cousins = 4 → gum_per_cousin = 5 → num_cousins * gum_per_cousin = 20 := by
  sorry

end gum_distribution_l1395_139570


namespace count_polynomials_l1395_139566

-- Define a function to check if an expression is a polynomial
def isPolynomial (expr : String) : Bool :=
  match expr with
  | "-7" => true
  | "m" => true
  | "x^3y^2" => true
  | "1/a" => false
  | "2x+3y" => true
  | _ => false

-- Define the list of expressions
def expressions : List String := ["-7", "m", "x^3y^2", "1/a", "2x+3y"]

-- Theorem to prove
theorem count_polynomials :
  (expressions.filter isPolynomial).length = 4 := by sorry

end count_polynomials_l1395_139566


namespace starting_lineup_selection_l1395_139501

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The total number of players in the team -/
def total_players : ℕ := 16

/-- The number of quadruplets -/
def num_quadruplets : ℕ := 4

/-- The number of starters to be selected -/
def num_starters : ℕ := 7

/-- The number of quadruplets that must be in the starting lineup -/
def quadruplets_in_lineup : ℕ := 3

/-- The number of ways to select the starting lineup -/
def num_ways : ℕ := 
  binomial num_quadruplets quadruplets_in_lineup * 
  binomial (total_players - num_quadruplets) (num_starters - quadruplets_in_lineup)

theorem starting_lineup_selection :
  num_ways = 1980 := by sorry

end starting_lineup_selection_l1395_139501
