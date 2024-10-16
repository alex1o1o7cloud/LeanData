import Mathlib

namespace NUMINAMATH_CALUDE_candy_bars_purchased_l3545_354520

theorem candy_bars_purchased (total_cost : ℕ) (price_per_bar : ℕ) (h1 : total_cost = 6) (h2 : price_per_bar = 3) :
  total_cost / price_per_bar = 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_bars_purchased_l3545_354520


namespace NUMINAMATH_CALUDE_arithmetic_mean_special_set_l3545_354549

theorem arithmetic_mean_special_set (n : ℕ) (h : n > 1) :
  let set := List.replicate (n - 1) 1 ++ [1 + 1 / n]
  (set.sum / n : ℚ) = 1 + 1 / n^2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_special_set_l3545_354549


namespace NUMINAMATH_CALUDE_tenthDrawnNumber_l3545_354573

/-- Represents the systematic sampling problem -/
def systematicSampling (totalStudents : Nat) (sampleSize : Nat) (firstDrawn : Nat) (nthDraw : Nat) : Nat :=
  let interval := totalStudents / sampleSize
  firstDrawn + interval * (nthDraw - 1)

/-- Theorem stating the 10th drawn number in the given systematic sampling scenario -/
theorem tenthDrawnNumber :
  systematicSampling 1000 50 15 10 = 195 := by
  sorry

end NUMINAMATH_CALUDE_tenthDrawnNumber_l3545_354573


namespace NUMINAMATH_CALUDE_smallest_n_for_g_equals_seven_g_of_eight_equals_seven_l3545_354507

def g (n : ℕ) : ℕ :=
  if n % 2 = 1 then n^2 + 1 else n/2 + 3

theorem smallest_n_for_g_equals_seven :
  ∀ n : ℕ, n > 0 → g n = 7 → n ≥ 8 :=
by sorry

theorem g_of_eight_equals_seven : g 8 = 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_g_equals_seven_g_of_eight_equals_seven_l3545_354507


namespace NUMINAMATH_CALUDE_line_connecting_circle_centers_l3545_354576

/-- The equation of the line connecting the centers of two circles -/
theorem line_connecting_circle_centers 
  (circle1 : ∀ x y : ℝ, x^2 + y^2 - 4*x + 6*y = 0)
  (circle2 : ∀ x y : ℝ, x^2 + y^2 - 6*x = 0) :
  ∃ (x y : ℝ), 3*x - y - 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_connecting_circle_centers_l3545_354576


namespace NUMINAMATH_CALUDE_inequality_proof_l3545_354551

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 0.5) :
  (x * y^2) / (x^3 + 1) + (y * z^2) / (y^3 + 1) + (z * x^2) / (z^3 + 1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3545_354551


namespace NUMINAMATH_CALUDE_square_side_length_l3545_354591

theorem square_side_length (s : ℝ) (h : s > 0) :
  s^2 = 6 * (4 * s) → s = 24 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3545_354591


namespace NUMINAMATH_CALUDE_pictures_per_album_l3545_354536

theorem pictures_per_album (total_pictures : ℕ) (num_albums : ℕ) 
  (h1 : total_pictures = 480) (h2 : num_albums = 24) :
  total_pictures / num_albums = 20 := by
  sorry

end NUMINAMATH_CALUDE_pictures_per_album_l3545_354536


namespace NUMINAMATH_CALUDE_election_percentage_l3545_354557

theorem election_percentage (total_votes : ℕ) (winning_margin : ℕ) (winning_percentage : ℚ) : 
  total_votes = 7520 →
  winning_margin = 1504 →
  winning_percentage = 60 →
  (winning_percentage / 100) * total_votes - (total_votes - (winning_percentage / 100) * total_votes) = winning_margin :=
by sorry

end NUMINAMATH_CALUDE_election_percentage_l3545_354557


namespace NUMINAMATH_CALUDE_sin_cos_identity_l3545_354534

theorem sin_cos_identity (α : Real) (h : Real.sin α + 2 * Real.cos α = 0) :
  2 * Real.sin α * Real.cos α - Real.cos α ^ 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l3545_354534


namespace NUMINAMATH_CALUDE_total_seashells_l3545_354597

def seashells_day1 : ℕ := 5
def seashells_day2 : ℕ := 7
def seashells_day3 (x y : ℕ) : ℕ := 2 * (x + y)

theorem total_seashells : 
  seashells_day1 + seashells_day2 + seashells_day3 seashells_day1 seashells_day2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l3545_354597


namespace NUMINAMATH_CALUDE_grid_cut_into_L_shapes_l3545_354540

/-- An L-shaped piece is a shape formed by three squares in an L configuration -/
def LShape : Type := Unit

/-- A grid is a collection of squares arranged in rows and columns -/
def Grid (m n : ℕ) : Type := Fin m → Fin n → Bool

/-- A function that checks if a grid can be cut into L-shaped pieces -/
def can_be_cut_into_L_shapes (g : Grid m n) : Prop := sorry

/-- Main theorem: Any (3n+1) × (3n+1) grid with one square removed can be cut into L-shaped pieces -/
theorem grid_cut_into_L_shapes (n : ℕ) (h : n > 0) :
  ∀ (g : Grid (3*n+1) (3*n+1)), (∃ (i j : Fin (3*n+1)), ¬g i j) →
  can_be_cut_into_L_shapes g :=
sorry

end NUMINAMATH_CALUDE_grid_cut_into_L_shapes_l3545_354540


namespace NUMINAMATH_CALUDE_erdos_mordell_two_points_l3545_354552

/-- The Erdős–Mordell inequality for two points -/
theorem erdos_mordell_two_points
  (a b c : ℝ)
  (a₁ b₁ c₁ : ℝ)
  (a₂ b₂ c₂ : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (ha₁ : 0 ≤ a₁) (hb₁ : 0 ≤ b₁) (hc₁ : 0 ≤ c₁)
  (ha₂ : 0 ≤ a₂) (hb₂ : 0 ≤ b₂) (hc₂ : 0 ≤ c₂)
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) :
  a * a₁ * a₂ + b * b₁ * b₂ + c * c₁ * c₂ ≥ a * b * c :=
by sorry

end NUMINAMATH_CALUDE_erdos_mordell_two_points_l3545_354552


namespace NUMINAMATH_CALUDE_sum_of_constants_l3545_354527

theorem sum_of_constants (a b : ℝ) : 
  (∀ x y : ℝ, y = a + b / (x^2 + 1)) →
  (3 = a + b / (1^2 + 1)) →
  (2 = a + b / (0^2 + 1)) →
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_constants_l3545_354527


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l3545_354594

theorem weight_loss_challenge (initial_weight : ℝ) (x : ℝ) : 
  x > 0 →
  (initial_weight * (1 - x / 100 + 2 / 100)) / initial_weight = 1 - 11.26 / 100 →
  x = 13.26 :=
by sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l3545_354594


namespace NUMINAMATH_CALUDE_number_of_girls_in_class_correct_number_of_girls_l3545_354572

theorem number_of_girls_in_class (num_boys : ℕ) (group_size : ℕ) (num_groups : ℕ) : ℕ :=
  let total_members := group_size * num_groups
  let num_girls := total_members - num_boys
  num_girls

theorem correct_number_of_girls :
  number_of_girls_in_class 9 3 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_in_class_correct_number_of_girls_l3545_354572


namespace NUMINAMATH_CALUDE_no_perfect_power_triple_l3545_354531

theorem no_perfect_power_triple (n r : ℕ) (hn : n ≥ 1) (hr : r ≥ 2) :
  ¬∃ m : ℤ, (n : ℤ) * (n + 1) * (n + 2) = m ^ r :=
sorry

end NUMINAMATH_CALUDE_no_perfect_power_triple_l3545_354531


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3545_354515

theorem quadratic_root_problem (m : ℝ) :
  (∃ x : ℝ, 3 * x^2 - m * x - 3 = 0 ∧ x = 1) →
  (∃ y : ℝ, 3 * y^2 - m * y - 3 = 0 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3545_354515


namespace NUMINAMATH_CALUDE_elise_puzzle_cost_l3545_354523

def puzzle_cost (initial_money savings comic_cost final_money : ℕ) : ℕ :=
  initial_money + savings - comic_cost - final_money

theorem elise_puzzle_cost : puzzle_cost 8 13 2 1 = 18 := by
  sorry

end NUMINAMATH_CALUDE_elise_puzzle_cost_l3545_354523


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_and_thirty_l3545_354564

theorem greatest_integer_with_gcf_five_and_thirty : ∃ n : ℕ, 
  n < 200 ∧ 
  n > 185 ∧
  Nat.gcd n 30 = 5 → False ∧ 
  Nat.gcd 185 30 = 5 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_and_thirty_l3545_354564


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l3545_354521

theorem simplify_sqrt_sum : 
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l3545_354521


namespace NUMINAMATH_CALUDE_all_vints_are_xaffs_l3545_354525

-- Define the types
variable (Zibb Xaff Yurn Worb Vint : Type)

-- Define the conditions
variable (h1 : Zibb → Xaff)
variable (h2 : Yurn → Xaff)
variable (h3 : Worb → Zibb)
variable (h4 : Yurn → Worb)
variable (h5 : Worb → Vint)
variable (h6 : Vint → Yurn)

-- Theorem to prove
theorem all_vints_are_xaffs : Vint → Xaff := by sorry

end NUMINAMATH_CALUDE_all_vints_are_xaffs_l3545_354525


namespace NUMINAMATH_CALUDE_pizza_segment_length_squared_l3545_354585

theorem pizza_segment_length_squared (diameter : ℝ) (num_pieces : ℕ) (m : ℝ) : 
  diameter = 18 →
  num_pieces = 4 →
  m = 2 * (diameter / 2) * Real.sin (π / (2 * num_pieces)) →
  m^2 = 162 := by sorry

end NUMINAMATH_CALUDE_pizza_segment_length_squared_l3545_354585


namespace NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l3545_354535

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem smallest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → 26 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l3545_354535


namespace NUMINAMATH_CALUDE_sculpture_surface_area_l3545_354505

/-- Represents the structure of the sculpture --/
structure Sculpture :=
  (num_cubes : ℕ)
  (edge_length : ℝ)
  (top_layer : ℕ)
  (middle_layer : ℕ)
  (bottom_layer : ℕ)

/-- Calculates the exposed surface area of the sculpture --/
def exposed_surface_area (s : Sculpture) : ℝ :=
  let top_area := s.top_layer * (5 * s.edge_length^2 + s.edge_length^2)
  let middle_area := s.middle_layer * s.edge_length^2 + 8 * s.edge_length^2
  let bottom_area := s.bottom_layer * s.edge_length^2
  top_area + middle_area + bottom_area

/-- The main theorem to be proved --/
theorem sculpture_surface_area :
  ∀ s : Sculpture,
    s.num_cubes = 14 ∧
    s.edge_length = 1 ∧
    s.top_layer = 1 ∧
    s.middle_layer = 4 ∧
    s.bottom_layer = 9 →
    exposed_surface_area s = 33 := by
  sorry


end NUMINAMATH_CALUDE_sculpture_surface_area_l3545_354505


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l3545_354592

theorem real_part_of_complex_fraction : 
  let i : ℂ := Complex.I
  Complex.re ((1 - i) / ((1 + i)^2)) = -1/2 := by sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l3545_354592


namespace NUMINAMATH_CALUDE_tetrahedron_inscribed_circle_centers_intersection_l3545_354548

structure Tetrahedron where
  A : Point
  B : Point
  C : Point
  D : Point

def inscribedCircleCenter (p q r : Point) : Point := sorry

def intersect (a b c d : Point) : Prop := sorry

theorem tetrahedron_inscribed_circle_centers_intersection 
  (ABCD : Tetrahedron) 
  (E : Point) 
  (F : Point) 
  (h1 : E = inscribedCircleCenter ABCD.B ABCD.C ABCD.D) 
  (h2 : F = inscribedCircleCenter ABCD.A ABCD.C ABCD.D) 
  (h3 : intersect ABCD.A E ABCD.B F) :
  ∃ (G H : Point), 
    G = inscribedCircleCenter ABCD.A ABCD.B ABCD.D ∧ 
    H = inscribedCircleCenter ABCD.A ABCD.B ABCD.C ∧ 
    intersect ABCD.C G ABCD.D H :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_inscribed_circle_centers_intersection_l3545_354548


namespace NUMINAMATH_CALUDE_volunteer_arrangements_l3545_354554

def num_applicants : ℕ := 5
def num_selected : ℕ := 3
def num_events : ℕ := 3

def permutations (n k : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - k)

theorem volunteer_arrangements : 
  permutations num_applicants num_selected - 
  permutations (num_applicants - 1) (num_selected - 1) = 48 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_arrangements_l3545_354554


namespace NUMINAMATH_CALUDE_extra_workers_for_road_project_l3545_354530

/-- Represents the road construction project -/
structure RoadProject where
  totalLength : ℝ
  totalDays : ℕ
  initialWorkers : ℕ
  completedLength : ℝ
  completedDays : ℕ

/-- Calculates the number of extra workers needed to complete the project on time -/
def extraWorkersNeeded (project : RoadProject) : ℕ :=
  sorry

/-- Theorem stating that for the given project parameters, approximately 53 extra workers are needed -/
theorem extra_workers_for_road_project :
  let project : RoadProject := {
    totalLength := 15,
    totalDays := 300,
    initialWorkers := 35,
    completedLength := 2.5,
    completedDays := 100
  }
  ∃ n : ℕ, n ≥ 53 ∧ n ≤ 54 ∧ extraWorkersNeeded project = n :=
sorry

end NUMINAMATH_CALUDE_extra_workers_for_road_project_l3545_354530


namespace NUMINAMATH_CALUDE_find_n_l3545_354566

-- Define the polynomial
def p (x y : ℝ) := (x^2 - y)^7

-- Define the fourth term of the expansion
def fourth_term (x y : ℝ) := -35 * x^8 * y^3

-- Define the fifth term of the expansion
def fifth_term (x y : ℝ) := 35 * x^6 * y^4

-- Theorem statement
theorem find_n (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m * n = 7)
  (h4 : fourth_term m n = fifth_term m n) : n = (49 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_find_n_l3545_354566


namespace NUMINAMATH_CALUDE_theresa_crayons_l3545_354560

/-- Theresa's initial number of crayons -/
def theresa_initial : ℕ := sorry

/-- Theresa's number of crayons after sharing -/
def theresa_after : ℕ := 19

/-- Janice's initial number of crayons -/
def janice_initial : ℕ := 12

/-- Number of crayons Janice shares with Nancy -/
def janice_shares : ℕ := 13

theorem theresa_crayons : theresa_initial = theresa_after := by sorry

end NUMINAMATH_CALUDE_theresa_crayons_l3545_354560


namespace NUMINAMATH_CALUDE_total_dozens_shipped_l3545_354512

-- Define the number of boxes shipped last week
def boxes_last_week : ℕ := 10

-- Define the total number of pomelos shipped last week
def total_pomelos_last_week : ℕ := 240

-- Define the number of boxes shipped this week
def boxes_this_week : ℕ := 20

-- Theorem to prove
theorem total_dozens_shipped : ℕ := by
  -- The proof goes here
  sorry

-- Goal: prove that total_dozens_shipped = 60
example : total_dozens_shipped = 60 := by sorry

end NUMINAMATH_CALUDE_total_dozens_shipped_l3545_354512


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_circumference_l3545_354501

/-- The circumference of the largest circle inscribed in a square -/
theorem largest_inscribed_circle_circumference (s : ℝ) (h : s = 12) :
  2 * s * Real.pi = 24 * Real.pi := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_circumference_l3545_354501


namespace NUMINAMATH_CALUDE_whatsapp_messages_l3545_354553

/-- The number of messages sent in a Whatsapp group over four days -/
def total_messages (monday tuesday wednesday thursday : ℕ) : ℕ :=
  monday + tuesday + wednesday + thursday

/-- Theorem: Given the conditions of the Whatsapp group messages, 
    the total number of messages over four days is 2000 -/
theorem whatsapp_messages : 
  ∀ (monday tuesday wednesday thursday : ℕ),
    monday = 300 →
    tuesday = 200 →
    wednesday = tuesday + 300 →
    thursday = 2 * wednesday →
    total_messages monday tuesday wednesday thursday = 2000 :=
by
  sorry


end NUMINAMATH_CALUDE_whatsapp_messages_l3545_354553


namespace NUMINAMATH_CALUDE_consecutive_numbers_multiple_l3545_354578

theorem consecutive_numbers_multiple (m : ℝ) : 
  m * 4.2 = 2 * (4.2 + 4) + 2 * (4.2 + 2) + 9 → m = 9 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_multiple_l3545_354578


namespace NUMINAMATH_CALUDE_star_one_one_eq_neg_eleven_l3545_354558

/-- A custom binary operation on real numbers -/
noncomputable def star (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y

/-- Theorem stating that given the conditions, 1 * 1 = -11 -/
theorem star_one_one_eq_neg_eleven 
  (a b : ℝ) 
  (h1 : star a b 3 5 = 15) 
  (h2 : star a b 4 7 = 28) : 
  star a b 1 1 = -11 := by
  sorry

#check star_one_one_eq_neg_eleven

end NUMINAMATH_CALUDE_star_one_one_eq_neg_eleven_l3545_354558


namespace NUMINAMATH_CALUDE_circle_center_sum_l3545_354541

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 = 10*x - 4*y + 14

/-- The center of a circle given by its equation -/
def CircleCenter (x y : ℝ) : Prop :=
  CircleEquation x y ∧ ∀ a b : ℝ, CircleEquation a b → (a - x)^2 + (b - y)^2 ≤ (x - x)^2 + (y - y)^2

theorem circle_center_sum :
  ∀ x y : ℝ, CircleCenter x y → x + y = 3 := by sorry

end NUMINAMATH_CALUDE_circle_center_sum_l3545_354541


namespace NUMINAMATH_CALUDE_lcm_14_21_45_l3545_354583

theorem lcm_14_21_45 : Nat.lcm 14 (Nat.lcm 21 45) = 630 := by sorry

end NUMINAMATH_CALUDE_lcm_14_21_45_l3545_354583


namespace NUMINAMATH_CALUDE_lemon_heads_distribution_l3545_354561

theorem lemon_heads_distribution (total : Nat) (friends : Nat) (each : Nat) : 
  total = 72 → friends = 6 → total / friends = each → each = 12 := by sorry

end NUMINAMATH_CALUDE_lemon_heads_distribution_l3545_354561


namespace NUMINAMATH_CALUDE_simplify_expression_l3545_354562

theorem simplify_expression :
  2 - (1 / (2 + Real.sqrt 5)) + (1 / (2 - Real.sqrt 5)) = 2 * (1 - Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3545_354562


namespace NUMINAMATH_CALUDE_line_passes_through_point_l3545_354590

/-- The line equation y = 2x - 1 passes through the point (0, -1) -/
theorem line_passes_through_point :
  let f : ℝ → ℝ := λ x => 2 * x - 1
  f 0 = -1 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l3545_354590


namespace NUMINAMATH_CALUDE_moon_permutations_l3545_354589

def word_length : ℕ := 4
def repeated_letter_count : ℕ := 2

theorem moon_permutations :
  (word_length.factorial) / (repeated_letter_count.factorial) = 12 := by
  sorry

end NUMINAMATH_CALUDE_moon_permutations_l3545_354589


namespace NUMINAMATH_CALUDE_square_with_semicircles_area_ratio_l3545_354556

/-- The ratio of areas for a square with semicircular arcs -/
theorem square_with_semicircles_area_ratio :
  let square_side : ℝ := 6
  let square_area : ℝ := square_side ^ 2
  let semicircle_radius : ℝ := square_side / 2
  let semicircle_area : ℝ := π * semicircle_radius ^ 2 / 2
  let new_figure_area : ℝ := square_area + 4 * semicircle_area
  new_figure_area / square_area = 1 + π / 2 :=
by sorry

end NUMINAMATH_CALUDE_square_with_semicircles_area_ratio_l3545_354556


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3545_354580

/-- The cubic function f(x) with specific properties -/
def f (x : ℝ) : ℝ := 2*x^3 - 9*x^2 + 12*x - 4

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 6*x^2 - 18*x + 12

theorem cubic_function_properties :
  (f 0 = -4) ∧ 
  (∀ x, f' 0 * x - (f x - f 0) - 4 = 0) ∧
  (f 2 = 0) ∧ 
  (f' 2 = 0) ∧
  (∀ x, x < 1 ∨ x > 2 → f' x > 0) :=
sorry


end NUMINAMATH_CALUDE_cubic_function_properties_l3545_354580


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3545_354532

/-- Prove that x = 2 and y = 1 are the real solutions to the complex equation (4 + 2i)x + (5 - 3i)y = 13 + i -/
theorem complex_equation_solution :
  ∃ (x y : ℝ), (Complex.mk 4 2 * x + Complex.mk 5 (-3) * y = Complex.mk 13 1) ∧ x = 2 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3545_354532


namespace NUMINAMATH_CALUDE_x_value_l3545_354538

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ := 2 * a - b

-- Theorem statement
theorem x_value :
  ∃ x : ℚ, triangle x (triangle 1 3) = 2 ∧ x = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_x_value_l3545_354538


namespace NUMINAMATH_CALUDE_congruence_system_solutions_l3545_354599

theorem congruence_system_solutions (a b c : ℤ) : 
  ∃ (s : Finset ℤ), 
    (∀ x ∈ s, x ≥ 0 ∧ x < 2000 ∧ 
      x % 14 = a % 14 ∧ 
      x % 15 = b % 15 ∧ 
      x % 16 = c % 16) ∧
    s.card = 3 :=
by sorry

end NUMINAMATH_CALUDE_congruence_system_solutions_l3545_354599


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3545_354514

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

theorem arithmetic_sequence_formula 
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 1 + a 2 + a 3 = 12) 
  (h_prod : a 1 * a 2 * a 3 = 48) :
  ∀ n : ℕ, a n = 2 * n := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3545_354514


namespace NUMINAMATH_CALUDE_inverse_square_theorem_l3545_354598

-- Define the relationship between x and y
def inverse_square_relation (x y : ℝ) : Prop := ∃ k : ℝ, x = k / (y * y)

-- Define the theorem
theorem inverse_square_theorem :
  ∀ x y : ℝ,
  inverse_square_relation x y →
  (9 : ℝ) * (9 : ℝ) * (0.1111111111111111 : ℝ) = (3 : ℝ) * (3 : ℝ) * (1 : ℝ) →
  (x = (1 : ℝ) → y = (3 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_inverse_square_theorem_l3545_354598


namespace NUMINAMATH_CALUDE_positive_sqrt_1024_l3545_354569

theorem positive_sqrt_1024 : Real.sqrt 1024 = 32 := by sorry

end NUMINAMATH_CALUDE_positive_sqrt_1024_l3545_354569


namespace NUMINAMATH_CALUDE_coefficient_expansion_l3545_354544

theorem coefficient_expansion (a : ℝ) : 
  (Nat.choose 5 3) * a^3 = 80 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_expansion_l3545_354544


namespace NUMINAMATH_CALUDE_polynomial_value_at_two_l3545_354504

theorem polynomial_value_at_two :
  let f : ℝ → ℝ := fun x ↦ x^2 - 3*x + 2
  f 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_at_two_l3545_354504


namespace NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l3545_354587

def P (n : ℕ) : ℕ := sorry

theorem unique_n_satisfying_conditions : 
  ∃! n : ℕ, n > 1 ∧ 
    P n = n - 8 ∧ 
    P (n + 60) = n + 52 :=
sorry

end NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l3545_354587


namespace NUMINAMATH_CALUDE_subset_condition_l3545_354567

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem subset_condition (a : ℝ) : B a ⊆ A ↔ a = 0 ∨ a = 1/3 ∨ a = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l3545_354567


namespace NUMINAMATH_CALUDE_least_subtraction_l3545_354522

theorem least_subtraction (n : ℕ) : n = 10 ↔ 
  (∀ m : ℕ, m < n → ¬(
    (2590 - n) % 9 = 6 ∧ 
    (2590 - n) % 11 = 6 ∧ 
    (2590 - n) % 13 = 6
  )) ∧
  (2590 - n) % 9 = 6 ∧ 
  (2590 - n) % 11 = 6 ∧ 
  (2590 - n) % 13 = 6 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_l3545_354522


namespace NUMINAMATH_CALUDE_intersection_A_B_m3_union_A_B_eq_A_l3545_354529

-- Define sets A and B
def A : Set ℝ := {x | |x - 1| < 2}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 1 < 0}

-- Theorem 1: Intersection of A and B when m = 3
theorem intersection_A_B_m3 : A ∩ B 3 = {x | 2 < x ∧ x < 3} := by sorry

-- Theorem 2: Condition for A ∪ B = A
theorem union_A_B_eq_A (m : ℝ) : A ∪ B m = A ↔ 0 ≤ m ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_m3_union_A_B_eq_A_l3545_354529


namespace NUMINAMATH_CALUDE_coin_flip_probability_l3545_354574

/-- Represents the outcome of a coin flip -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the set of five coins -/
structure CoinSet :=
  (penny : CoinOutcome)
  (nickel : CoinOutcome)
  (dime : CoinOutcome)
  (quarter : CoinOutcome)
  (half_dollar : CoinOutcome)

/-- The total number of possible outcomes when flipping five coins -/
def total_outcomes : Nat := 32

/-- Predicate for the desired outcome (penny, nickel, and half dollar are heads) -/
def desired_outcome (cs : CoinSet) : Prop :=
  cs.penny = CoinOutcome.Heads ∧
  cs.nickel = CoinOutcome.Heads ∧
  cs.half_dollar = CoinOutcome.Heads

/-- The number of outcomes satisfying the desired condition -/
def successful_outcomes : Nat := 4

/-- The probability of the desired outcome -/
def probability : ℚ := 1 / 8

theorem coin_flip_probability :
  (successful_outcomes : ℚ) / total_outcomes = probability :=
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l3545_354574


namespace NUMINAMATH_CALUDE_cute_six_digit_integers_l3545_354565

def is_permutation (n : ℕ) (digits : List ℕ) : Prop :=
  digits.length = 6 ∧ digits.toFinset = Finset.range 6

def first_k_digits_divisible (digits : List ℕ) : Prop :=
  ∀ k : ℕ, k ≤ 6 → k ∣ (digits.take k).foldl (λ acc d => acc * 10 + d) 0

def is_cute (digits : List ℕ) : Prop :=
  is_permutation 6 digits ∧ first_k_digits_divisible digits

theorem cute_six_digit_integers :
  ∃! (s : Finset (List ℕ)), s.card = 2 ∧ ∀ digits, digits ∈ s ↔ is_cute digits :=
sorry

end NUMINAMATH_CALUDE_cute_six_digit_integers_l3545_354565


namespace NUMINAMATH_CALUDE_matchmaking_theorem_l3545_354542

-- Define a bipartite graph
def BipartiteGraph (α : Type) := (α → Bool) → α → α → Prop

-- Define a matching in a bipartite graph
def Matching (α : Type) (G : BipartiteGraph α) (M : α → α → Prop) :=
  ∀ x y z, M x y → M x z → y = z

-- Define a perfect matching for a subset
def PerfectMatchingForSubset (α : Type) (G : BipartiteGraph α) (S : Set α) (M : α → α → Prop) :=
  Matching α G M ∧ ∀ x ∈ S, ∃ y, M x y

-- Main theorem
theorem matchmaking_theorem (α : Type) (G : BipartiteGraph α) 
  (B W : Set α) (B1 : Set α) (W2 : Set α) 
  (hB1 : B1 ⊆ B) (hW2 : W2 ⊆ W)
  (M1 : α → α → Prop) (M2 : α → α → Prop)
  (hM1 : PerfectMatchingForSubset α G B1 M1)
  (hM2 : PerfectMatchingForSubset α G W2 M2) :
  ∃ M : α → α → Prop, 
    Matching α G M ∧ 
    (∀ x y, M1 x y → M x y) ∧ 
    (∀ x y, M2 x y → M x y) :=
sorry

end NUMINAMATH_CALUDE_matchmaking_theorem_l3545_354542


namespace NUMINAMATH_CALUDE_jack_afternoon_emails_l3545_354588

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 4

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 8

/-- The total number of emails Jack received in the afternoon and evening -/
def afternoon_evening_emails : ℕ := 13

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := afternoon_evening_emails - evening_emails

theorem jack_afternoon_emails :
  afternoon_emails = 5 := by sorry

end NUMINAMATH_CALUDE_jack_afternoon_emails_l3545_354588


namespace NUMINAMATH_CALUDE_min_games_for_condition_l3545_354511

/-- The number of teams in the tournament -/
def num_teams : ℕ := 16

/-- The total number of possible games in a round-robin tournament -/
def total_games : ℕ := num_teams * (num_teams - 1) / 2

/-- The maximum number of non-played games such that no three teams are mutually non-played -/
def max_non_played_games : ℕ := (num_teams / 2) ^ 2

/-- The minimum number of games that must be played to satisfy the condition -/
def min_games_played : ℕ := total_games - max_non_played_games

theorem min_games_for_condition : min_games_played = 56 := by sorry

end NUMINAMATH_CALUDE_min_games_for_condition_l3545_354511


namespace NUMINAMATH_CALUDE_only_zero_and_198_satisfy_l3545_354595

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for numbers equal to 11 times the sum of their digits -/
def is_eleven_times_sum_of_digits (n : ℕ) : Prop :=
  n = 11 * sum_of_digits n

theorem only_zero_and_198_satisfy :
  ∀ n : ℕ, is_eleven_times_sum_of_digits n ↔ n = 0 ∨ n = 198 := by sorry

end NUMINAMATH_CALUDE_only_zero_and_198_satisfy_l3545_354595


namespace NUMINAMATH_CALUDE_terminal_sides_theorem_l3545_354539

/-- Given an angle θ in degrees, returns true if the terminal side of 7θ coincides with the terminal side of θ -/
def terminal_sides_coincide (θ : ℝ) : Prop :=
  ∃ k : ℤ, 7 * θ = θ + k * 360

/-- The set of angles whose terminal sides coincide with their 7θ counterparts -/
def coinciding_angles : Set ℝ := {0, 60, 120, 180, 240, 300}

theorem terminal_sides_theorem (θ : ℝ) :
  0 ≤ θ ∧ θ < 360 ∧ terminal_sides_coincide θ → θ ∈ coinciding_angles := by
  sorry

end NUMINAMATH_CALUDE_terminal_sides_theorem_l3545_354539


namespace NUMINAMATH_CALUDE_max_regions_correct_l3545_354596

/-- The maximum number of regions into which n circles can divide the plane -/
def max_regions (n : ℕ) : ℕ := n^2 - n + 2

/-- Theorem stating that max_regions gives the maximum number of regions -/
theorem max_regions_correct (n : ℕ) :
  max_regions n = n^2 - n + 2 :=
by sorry

end NUMINAMATH_CALUDE_max_regions_correct_l3545_354596


namespace NUMINAMATH_CALUDE_checker_center_on_boundary_l3545_354528

/-- Represents a circular checker on a checkerboard -/
structure Checker where
  center : ℝ × ℝ
  radius : ℝ
  is_on_board : Bool
  covers_equal_areas : Bool

/-- Represents a checkerboard -/
structure Checkerboard where
  size : ℕ
  square_size : ℝ

/-- Checks if a point is on a boundary or junction of squares -/
def is_on_boundary_or_junction (board : Checkerboard) (point : ℝ × ℝ) : Prop :=
  ∃ (n m : ℕ), (n ≤ board.size ∧ m ≤ board.size) ∧
    (point.1 = n * board.square_size ∨ point.2 = m * board.square_size)

/-- Main theorem -/
theorem checker_center_on_boundary (board : Checkerboard) (c : Checker) :
    c.is_on_board = true → c.covers_equal_areas = true →
    is_on_boundary_or_junction board c.center :=
  sorry


end NUMINAMATH_CALUDE_checker_center_on_boundary_l3545_354528


namespace NUMINAMATH_CALUDE_r₂_bound_bound_is_tight_l3545_354545

-- Define the function f
def f (r₂ r₃ : ℝ) (x : ℝ) : ℝ := x^2 - r₂ * x + r₃

-- Define the sequence g
def g (r₂ r₃ : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => f r₂ r₃ (g r₂ r₃ n)

-- Define the conditions on the sequence
def sequence_conditions (r₂ r₃ : ℝ) : Prop :=
  (∀ i ≤ 2011, g r₂ r₃ (2*i) < g r₂ r₃ (2*i + 1) ∧ g r₂ r₃ (2*i + 1) > g r₂ r₃ (2*i + 2)) ∧
  (∃ j : ℕ, ∀ i > j, g r₂ r₃ (i + 1) > g r₂ r₃ i) ∧
  (∀ M : ℝ, ∃ N : ℕ, g r₂ r₃ N > M)

theorem r₂_bound (r₂ r₃ : ℝ) (h : sequence_conditions r₂ r₃) : |r₂| > 2 :=
  sorry

theorem bound_is_tight : ∀ ε > 0, ∃ r₂ r₃ : ℝ, sequence_conditions r₂ r₃ ∧ |r₂| < 2 + ε :=
  sorry

end NUMINAMATH_CALUDE_r₂_bound_bound_is_tight_l3545_354545


namespace NUMINAMATH_CALUDE_coupon_savings_difference_l3545_354508

-- Define the coupon savings functions
def couponA (price : ℝ) : ℝ := 0.18 * price
def couponB : ℝ := 35
def couponC (price : ℝ) : ℝ := 0.20 * (price - 120)

-- Define the conditions for Coupon A to be at least as good as B and C
def couponABestCondition (price : ℝ) : Prop :=
  couponA price ≥ couponB ∧ couponA price ≥ couponC price

-- Define the range of prices where Coupon A is the best
def priceRange : Set ℝ := {price | price > 120 ∧ couponABestCondition price}

-- Theorem statement
theorem coupon_savings_difference :
  ∃ (x y : ℝ), x ∈ priceRange ∧ y ∈ priceRange ∧
  (∀ p ∈ priceRange, x ≤ p ∧ p ≤ y) ∧
  y - x = 1005.56 :=
sorry

end NUMINAMATH_CALUDE_coupon_savings_difference_l3545_354508


namespace NUMINAMATH_CALUDE_solve_equation_l3545_354526

-- Define the new operation
def star_op (a b : ℝ) : ℝ := 3 * a - 2 * b^2

-- Theorem statement
theorem solve_equation (a : ℝ) (h : star_op a 4 = 10) : a = 14 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3545_354526


namespace NUMINAMATH_CALUDE_number_of_sweaters_l3545_354563

def washing_machine_capacity : ℕ := 7
def number_of_shirts : ℕ := 2
def number_of_loads : ℕ := 5

theorem number_of_sweaters : 
  (washing_machine_capacity * number_of_loads) - number_of_shirts = 33 := by
  sorry

end NUMINAMATH_CALUDE_number_of_sweaters_l3545_354563


namespace NUMINAMATH_CALUDE_expression_value_l3545_354543

theorem expression_value (a b : ℤ) (h1 : a = 4) (h2 : b = -5) : 
  -a^2 - b^2 + a*b + b = -66 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3545_354543


namespace NUMINAMATH_CALUDE_red_ball_packs_l3545_354593

theorem red_ball_packs (total_balls : ℕ) (yellow_packs green_packs : ℕ) (balls_per_pack : ℕ) :
  total_balls = 399 →
  yellow_packs = 10 →
  green_packs = 8 →
  balls_per_pack = 19 →
  ∃ red_packs : ℕ, red_packs = 3 ∧ 
    total_balls = (red_packs + yellow_packs + green_packs) * balls_per_pack :=
by sorry

end NUMINAMATH_CALUDE_red_ball_packs_l3545_354593


namespace NUMINAMATH_CALUDE_final_sum_after_transformation_l3545_354510

theorem final_sum_after_transformation (x y T : ℝ) (h : x + y = T) :
  3 * (x + 5) + 3 * (y + 5) = 3 * T + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_transformation_l3545_354510


namespace NUMINAMATH_CALUDE_travel_distance_l3545_354547

theorem travel_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 75 → time = 4 → distance = speed * time → distance = 300 := by
  sorry

end NUMINAMATH_CALUDE_travel_distance_l3545_354547


namespace NUMINAMATH_CALUDE_rectangular_solid_on_sphere_l3545_354519

theorem rectangular_solid_on_sphere (x : ℝ) : 
  let surface_area : ℝ := 18 * Real.pi
  let radius : ℝ := Real.sqrt (surface_area / (4 * Real.pi))
  3^2 + 2^2 + x^2 = 4 * radius^2 → x = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_on_sphere_l3545_354519


namespace NUMINAMATH_CALUDE_fraction_simplification_l3545_354571

theorem fraction_simplification : (3 * 4) / 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3545_354571


namespace NUMINAMATH_CALUDE_paul_juice_bottles_l3545_354550

/-- 
Given that Donald drinks 3 more than twice the number of juice bottles Paul drinks in one day,
and Donald drinks 9 bottles of juice per day, prove that Paul drinks 3 bottles of juice per day.
-/
theorem paul_juice_bottles (paul_bottles : ℕ) (donald_bottles : ℕ) : 
  donald_bottles = 2 * paul_bottles + 3 →
  donald_bottles = 9 →
  paul_bottles = 3 := by
  sorry

end NUMINAMATH_CALUDE_paul_juice_bottles_l3545_354550


namespace NUMINAMATH_CALUDE_angle_measure_with_special_supplement_complement_l3545_354577

-- Define the angle measure in degrees
def angle_measure : ℝ → Prop :=
  λ x => x > 0 ∧ x < 180

-- Define the supplement of an angle
def supplement (x : ℝ) : ℝ := 180 - x

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Theorem statement
theorem angle_measure_with_special_supplement_complement :
  ∀ x : ℝ, angle_measure x → supplement x = 4 * complement x → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_with_special_supplement_complement_l3545_354577


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l3545_354546

/-- Represents a systematic sample of bottles -/
structure SystematicSample where
  total : Nat
  sample_size : Nat
  start : Nat
  step : Nat

/-- Generates the sample numbers for a systematic sample -/
def generate_sample (s : SystematicSample) : List Nat :=
  List.range s.sample_size |>.map (fun i => s.start + i * s.step)

/-- Theorem: The systematic sample for 60 bottles with 6 selections starts at 3 with step 10 -/
theorem systematic_sample_theorem :
  let s : SystematicSample := ⟨60, 6, 3, 10⟩
  generate_sample s = [3, 13, 23, 33, 43, 53] := by sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l3545_354546


namespace NUMINAMATH_CALUDE_pudding_distribution_l3545_354517

theorem pudding_distribution (pudding_cups : ℕ) (students : ℕ) 
  (h1 : pudding_cups = 4752) (h2 : students = 3019) : 
  let additional_cups := (students * ((pudding_cups + students - 1) / students)) - pudding_cups
  additional_cups = 1286 := by
sorry

end NUMINAMATH_CALUDE_pudding_distribution_l3545_354517


namespace NUMINAMATH_CALUDE_unique_root_condition_l3545_354524

/-- The characteristic equation of a thermal energy process -/
def characteristic_equation (x t : ℝ) : Prop := x^3 - 3*x = t

/-- The condition for a unique root -/
def has_unique_root (t : ℝ) : Prop :=
  ∃! x, characteristic_equation x t

/-- The main theorem about the uniqueness and magnitude of the root -/
theorem unique_root_condition (t : ℝ) :
  has_unique_root t ↔ abs t > 2 ∧ ∀ x, characteristic_equation x t → abs x > 2 :=
sorry

end NUMINAMATH_CALUDE_unique_root_condition_l3545_354524


namespace NUMINAMATH_CALUDE_quadratic_root_product_l3545_354586

theorem quadratic_root_product (p q : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 : ℂ) + Complex.I ∈ {z : ℂ | z ^ 2 + p * z + q = 0} →
  p * q = -20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_product_l3545_354586


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l3545_354579

theorem arctan_equation_solution :
  ∃ y : ℝ, 2 * Real.arctan (1/5) + 2 * Real.arctan (1/25) + Real.arctan (1/y) = π/4 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l3545_354579


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l3545_354570

/-- Represents a triangle with side lengths a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle satisfies the triangle inequality. -/
def Triangle.isValid (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Checks if a triangle is isosceles. -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Calculates the perimeter of a triangle. -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Checks if two triangles are similar. -/
def areSimilar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t2.a = k * t1.a ∧ t2.b = k * t1.b ∧ t2.c = k * t1.c

theorem similar_triangle_perimeter 
  (t1 t2 : Triangle) 
  (h1 : t1.isValid)
  (h2 : t2.isValid)
  (h3 : t1.isIsosceles)
  (h4 : t2.isIsosceles)
  (h5 : areSimilar t1 t2)
  (h6 : t1.a = 8 ∧ t1.b = 24 ∧ t1.c = 24)
  (h7 : t2.a = 40) : 
  t2.perimeter = 280 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l3545_354570


namespace NUMINAMATH_CALUDE_fraction_product_exponents_l3545_354537

theorem fraction_product_exponents : (3 / 4 : ℚ)^5 * (4 / 3 : ℚ)^2 = 8 / 19 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_exponents_l3545_354537


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_necessary_not_sufficient_condition_l3545_354581

-- Statement B
theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x > 1 ∧ y > 1 → x + y > 2) ∧
  ¬(x + y > 2 → x > 1 ∧ y > 1) :=
sorry

-- Statement C
theorem necessary_not_sufficient_condition (a b : ℝ) :
  (a > b ∧ b > 0 → 1 / a < 1 / b) ∧
  ¬(1 / a < 1 / b → a > b ∧ b > 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_necessary_not_sufficient_condition_l3545_354581


namespace NUMINAMATH_CALUDE_jennifers_spending_l3545_354509

theorem jennifers_spending (initial_amount : ℚ) : 
  initial_amount / 5 + initial_amount / 6 + initial_amount / 2 + 20 = initial_amount →
  initial_amount = 150 := by
  sorry

end NUMINAMATH_CALUDE_jennifers_spending_l3545_354509


namespace NUMINAMATH_CALUDE_expansion_terms_count_l3545_354513

def factor1 : ℕ := 3
def factor2 : ℕ := 4
def factor3 : ℕ := 5

theorem expansion_terms_count : factor1 * factor2 * factor3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l3545_354513


namespace NUMINAMATH_CALUDE_floor_sqrt_sum_eq_floor_sqrt_product_l3545_354555

theorem floor_sqrt_sum_eq_floor_sqrt_product (n : ℕ+) :
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_sum_eq_floor_sqrt_product_l3545_354555


namespace NUMINAMATH_CALUDE_floor_expression_equals_eight_l3545_354584

def n : ℕ := 2024

theorem floor_expression_equals_eight :
  ⌊(2025^3 : ℚ) / (2023 * 2024) - (2023^3 : ℚ) / (2024 * 2025)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_expression_equals_eight_l3545_354584


namespace NUMINAMATH_CALUDE_percentage_comparison_l3545_354533

theorem percentage_comparison (p q : ℝ) (h : p = 1.5 * q) :
  (q / p - 1) * 100 = -100/3 ∧ (p / q - 1) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_comparison_l3545_354533


namespace NUMINAMATH_CALUDE_equation_property_l3545_354500

theorem equation_property (a b : ℝ) : 3 * a = 3 * b → a = b := by
  sorry

end NUMINAMATH_CALUDE_equation_property_l3545_354500


namespace NUMINAMATH_CALUDE_derivative_log2_l3545_354516

theorem derivative_log2 (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_log2_l3545_354516


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3545_354506

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_sum : a 1 + a 2 = -1)
  (h_diff : a 1 - a 3 = -3) :
  a 4 = -8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3545_354506


namespace NUMINAMATH_CALUDE_budget_allocation_l3545_354575

theorem budget_allocation (home_electronics food_additives gm_microorganisms industrial_lubricants : ℝ)
  (basic_astrophysics_degrees : ℝ) :
  home_electronics = 24 →
  food_additives = 15 →
  gm_microorganisms = 19 →
  industrial_lubricants = 8 →
  basic_astrophysics_degrees = 72 →
  let basic_astrophysics := (basic_astrophysics_degrees / 360) * 100
  let total_known := home_electronics + food_additives + gm_microorganisms + industrial_lubricants + basic_astrophysics
  let microphotonics := 100 - total_known
  microphotonics = 14 := by
  sorry

end NUMINAMATH_CALUDE_budget_allocation_l3545_354575


namespace NUMINAMATH_CALUDE_remainder_theorem_l3545_354503

theorem remainder_theorem : (43^43 + 43) % 44 = 42 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3545_354503


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3545_354582

/-- 
For a quadratic equation x^2 - mx - 1 = 0 to have two roots, 
one greater than 2 and the other less than 2, m must be in the range (3/2, +∞).
-/
theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x y : ℝ, x < 2 ∧ y > 2 ∧ x^2 - m*x - 1 = 0 ∧ y^2 - m*y - 1 = 0) ↔ 
  m > 3/2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3545_354582


namespace NUMINAMATH_CALUDE_solution_set_f_geq_2_min_value_f_f_equals_one_condition_l3545_354568

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for part I
theorem solution_set_f_geq_2 :
  {x : ℝ | f (x + 2) ≥ 2} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 1/2} := by sorry

-- Theorem for part II
theorem min_value_f :
  ∀ x : ℝ, f x ≥ 1 := by sorry

-- Theorem for the condition when f(x) = 1
theorem f_equals_one_condition (x : ℝ) :
  f x = 1 ↔ 1 ≤ x ∧ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_2_min_value_f_f_equals_one_condition_l3545_354568


namespace NUMINAMATH_CALUDE_quadratic_from_means_l3545_354518

theorem quadratic_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 6)
  (h_geometric : Real.sqrt (a * b) = 10) :
  ∀ x : ℝ, x^2 - 12*x + 100 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_from_means_l3545_354518


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3545_354559

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := fun x : ℝ => a^(x + 1) - 1
  f (-1) = 0 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3545_354559


namespace NUMINAMATH_CALUDE_rental_company_fixed_amount_l3545_354502

/-- The fixed amount charged by the first rental company -/
def F : ℝ := 41.95

/-- The per-mile rate charged by the first rental company -/
def rate1 : ℝ := 0.29

/-- The fixed amount charged by City Rentals -/
def fixed2 : ℝ := 38.95

/-- The per-mile rate charged by City Rentals -/
def rate2 : ℝ := 0.31

/-- The number of miles driven -/
def miles : ℝ := 150.0

theorem rental_company_fixed_amount :
  F + rate1 * miles = fixed2 + rate2 * miles :=
sorry

end NUMINAMATH_CALUDE_rental_company_fixed_amount_l3545_354502
