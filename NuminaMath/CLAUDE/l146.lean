import Mathlib

namespace cable_length_l146_14670

/-- The length of a curve defined by the intersection of a plane and a sphere -/
theorem cable_length (x y z : ℝ) : 
  x + y + z = 10 → 
  x * y + y * z + x * z = -22 → 
  (4 * Real.pi * Real.sqrt (83 / 3)) = 
    (2 * Real.pi * Real.sqrt (144 - (10 ^ 2) / 3)) := by
  sorry

end cable_length_l146_14670


namespace intersection_of_M_and_N_l146_14685

def M : Set ℕ := {0, 1}

def N : Set ℕ := {y | ∃ x ∈ M, y = x^2 + 1}

theorem intersection_of_M_and_N : M ∩ N = {1} := by
  sorry

end intersection_of_M_and_N_l146_14685


namespace niles_collection_l146_14609

/-- The total amount collected by Niles from the book club -/
def total_collected (num_members : ℕ) (snack_fee : ℕ) (num_hardcover : ℕ) (hardcover_price : ℕ) (num_paperback : ℕ) (paperback_price : ℕ) : ℕ :=
  num_members * (snack_fee + num_hardcover * hardcover_price + num_paperback * paperback_price)

/-- Theorem stating the total amount collected by Niles -/
theorem niles_collection : total_collected 6 150 6 30 6 12 = 2412 := by
  sorry

end niles_collection_l146_14609


namespace factorization_xy_squared_minus_x_l146_14605

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end factorization_xy_squared_minus_x_l146_14605


namespace adam_quarters_l146_14627

/-- The number of quarters Adam spent at the arcade -/
def quarters_spent : ℕ := 9

/-- The number of quarters Adam had left over -/
def quarters_left : ℕ := 79

/-- The initial number of quarters Adam had -/
def initial_quarters : ℕ := quarters_spent + quarters_left

theorem adam_quarters : initial_quarters = 88 := by
  sorry

end adam_quarters_l146_14627


namespace carolyn_initial_marbles_l146_14644

/-- The number of marbles Carolyn shared with Diana -/
def marbles_shared : ℕ := 42

/-- The number of marbles Carolyn had left after sharing -/
def marbles_left : ℕ := 5

/-- The number of oranges Carolyn started with (not used in the proof, but mentioned in the problem) -/
def initial_oranges : ℕ := 6

/-- Carolyn's initial number of marbles -/
def initial_marbles : ℕ := marbles_shared + marbles_left

theorem carolyn_initial_marbles :
  initial_marbles = 47 :=
by sorry

end carolyn_initial_marbles_l146_14644


namespace ceiling_abs_negative_l146_14632

theorem ceiling_abs_negative : ⌈|(-52.7 : ℝ)|⌉ = 53 := by sorry

end ceiling_abs_negative_l146_14632


namespace pure_imaginary_fraction_l146_14646

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (2 * a + I) / (1 - 2 * I) = b * I) → a = 1 := by
  sorry

end pure_imaginary_fraction_l146_14646


namespace product_and_multiply_l146_14647

theorem product_and_multiply : (3.6 * 0.25) * 0.4 = 0.36 := by
  sorry

end product_and_multiply_l146_14647


namespace equation_solution_l146_14694

theorem equation_solution : 
  ∃ x : ℝ, (Real.sqrt (2 + Real.sqrt (3 + Real.sqrt x)) = (2 + Real.sqrt x) ^ (1/3)) ∧ 
  (x = ((((1 + Real.sqrt 17) / 2) ^ 3 - 2) ^ 2)) := by
  sorry

end equation_solution_l146_14694


namespace birthday_money_ratio_l146_14614

theorem birthday_money_ratio : 
  let aunt_money : ℚ := 75
  let grandfather_money : ℚ := 150
  let bank_money : ℚ := 45
  let total_money := aunt_money + grandfather_money
  (bank_money / total_money) = 1 / 5 := by
  sorry

end birthday_money_ratio_l146_14614


namespace smallest_a_value_l146_14659

theorem smallest_a_value (a b : ℕ) (r₁ r₂ r₃ : ℕ+) : 
  (∀ x : ℝ, x^3 - a*x^2 + b*x - 2310 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
  r₁ * r₂ * r₃ = 2310 →
  a = r₁ + r₂ + r₃ →
  28 ≤ a :=
sorry

end smallest_a_value_l146_14659


namespace charity_fundraising_l146_14678

theorem charity_fundraising (people : ℕ) (total_amount : ℕ) (amount_per_person : ℕ) :
  people = 8 →
  total_amount = 3000 →
  amount_per_person * people = total_amount →
  amount_per_person = 375 := by
  sorry

end charity_fundraising_l146_14678


namespace tortoise_wins_l146_14654

-- Define the race distance
def race_distance : ℝ := 100

-- Define the animals
inductive Animal
| tortoise
| hare

-- Define the speed function for each animal
def speed (a : Animal) (t : ℝ) : ℝ :=
  match a with
  | Animal.tortoise => sorry -- Increasing speed function
  | Animal.hare => sorry -- Piecewise function for hare's speed

-- Define the position function for each animal
def position (a : Animal) (t : ℝ) : ℝ :=
  sorry -- Integral of speed function

-- Define the finish time for each animal
def finish_time (a : Animal) : ℝ :=
  sorry -- Time when position equals race_distance

-- Theorem stating the tortoise wins
theorem tortoise_wins :
  finish_time Animal.tortoise < finish_time Animal.hare :=
sorry


end tortoise_wins_l146_14654


namespace math_problems_l146_14698

theorem math_problems :
  (∀ a b : ℝ, a < b ∧ b < 0 → a^2 > a*b ∧ a*b > b^2) ∧
  (∀ a b c d : ℝ, c > d ∧ a > b → a - d > b - c) ∧
  (∀ a b c : ℝ, b < a ∧ a < 0 ∧ c < 0 → c/a > c/b) ∧
  (∀ a b c : ℝ, a > 0 ∧ b > c ∧ c > 0 → (c+a)/(b+a) > c/b) :=
by sorry

end math_problems_l146_14698


namespace parallel_iff_perpendicular_iff_l146_14621

-- Define the lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := x + m * y + 6 = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + 3 * m * y + 2 * m = 0

-- Define parallel lines
def parallel (m : ℝ) : Prop := 
  ∀ x y z w, l1 m x y ∧ l2 m z w → (x - z) * (m - 2) = m * (y - w)

-- Define perpendicular lines
def perpendicular (m : ℝ) : Prop := 
  ∀ x y z w, l1 m x y ∧ l2 m z w → (x - z) * (z - x) + m * (y - w) * (w - y) = 0

-- Theorem for parallel lines
theorem parallel_iff : 
  ∀ m : ℝ, parallel m ↔ m = 0 ∨ m = 5 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_iff : 
  ∀ m : ℝ, perpendicular m ↔ m = -1 ∨ m = 2/3 :=
sorry

end parallel_iff_perpendicular_iff_l146_14621


namespace square_EFGH_product_l146_14612

/-- A point on a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A square on the grid -/
structure GridSquare where
  E : GridPoint
  F : GridPoint
  G : GridPoint
  H : GridPoint

/-- The side length of a square given two of its corners -/
def sideLength (p1 p2 : GridPoint) : ℤ :=
  max (abs (p1.x - p2.x)) (abs (p1.y - p2.y))

/-- The area of a square -/
def area (s : GridSquare) : ℤ :=
  (sideLength s.E s.F) ^ 2

/-- The perimeter of a square -/
def perimeter (s : GridSquare) : ℤ :=
  4 * (sideLength s.E s.F)

theorem square_EFGH_product :
  ∃ (s : GridSquare),
    s.E = ⟨1, 5⟩ ∧
    s.F = ⟨5, 5⟩ ∧
    s.G = ⟨5, 1⟩ ∧
    s.H = ⟨1, 1⟩ ∧
    (area s * perimeter s = 256) := by
  sorry

end square_EFGH_product_l146_14612


namespace triangle_perimeter_with_inscribed_circles_l146_14620

/-- The perimeter of an equilateral triangle inscribing three circles -/
theorem triangle_perimeter_with_inscribed_circles (r : ℝ) :
  r > 0 →
  let side_length := 4 * r + 4 * r * Real.sqrt 3
  3 * side_length = 12 * r * Real.sqrt 3 + 48 * r :=
by sorry

end triangle_perimeter_with_inscribed_circles_l146_14620


namespace train_speed_l146_14680

/-- Calculates the speed of a train passing through a tunnel -/
theorem train_speed (train_length : ℝ) (tunnel_length : ℝ) (time_minutes : ℝ) :
  train_length = 1 →
  tunnel_length = 70 →
  time_minutes = 6 →
  (train_length + tunnel_length) / (time_minutes / 60) = 710 := by
  sorry

end train_speed_l146_14680


namespace complete_square_factorize_l146_14606

-- Problem 1: Complete the square
theorem complete_square (x p : ℝ) : x^2 + 2*p*x + 1 = (x + p)^2 + (1 - p^2) := by sorry

-- Problem 2: Factorization
theorem factorize (a b : ℝ) : a^2 - b^2 + 4*a + 2*b + 3 = (a + b + 1)*(a - b + 3) := by sorry

end complete_square_factorize_l146_14606


namespace triangle_existence_l146_14691

/-- A set of points in space -/
structure PointSet where
  n : ℕ
  points : Finset (Fin (2 * n))
  segments : Finset (Fin (2 * n) × Fin (2 * n))
  n_gt_one : n > 1
  segment_count : segments.card ≥ n^2 + 1

/-- A triangle in a point set -/
def Triangle (ps : PointSet) : Prop :=
  ∃ a b c, a ∈ ps.points ∧ b ∈ ps.points ∧ c ∈ ps.points ∧
    (a, b) ∈ ps.segments ∧ (b, c) ∈ ps.segments ∧ (c, a) ∈ ps.segments

/-- Theorem: If a point set satisfies the conditions, then it contains a triangle -/
theorem triangle_existence (ps : PointSet) : Triangle ps := by
  sorry

end triangle_existence_l146_14691


namespace circumcircle_radius_of_intersecting_circles_l146_14677

/-- Given two circles with radii R and r that touch a common line and intersect each other,
    the radius ρ of the circumcircle of the triangle formed by their two points of tangency
    and one point of intersection is equal to √(R * r). -/
theorem circumcircle_radius_of_intersecting_circles (R r : ℝ) (hR : R > 0) (hr : r > 0) :
  ∃ (ρ : ℝ), ρ > 0 ∧ ρ * ρ = R * r := by sorry

end circumcircle_radius_of_intersecting_circles_l146_14677


namespace strawberry_picking_l146_14616

theorem strawberry_picking (basket_capacity : ℕ) (picked_ratio : ℚ) : 
  basket_capacity = 60 → 
  picked_ratio = 4/5 → 
  (basket_capacity / picked_ratio : ℚ) * 5 = 75 := by
sorry

end strawberry_picking_l146_14616


namespace olivia_chocolate_sales_l146_14611

/-- The amount of money Olivia made selling chocolate bars -/
def olivia_money (total_bars : ℕ) (unsold_bars : ℕ) (price_per_bar : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

theorem olivia_chocolate_sales : olivia_money 7 4 3 = 9 := by
  sorry

end olivia_chocolate_sales_l146_14611


namespace reciprocal_of_negative_one_twenty_fourth_l146_14645

theorem reciprocal_of_negative_one_twenty_fourth :
  ((-1 / 24)⁻¹ : ℚ) = -24 := by sorry

end reciprocal_of_negative_one_twenty_fourth_l146_14645


namespace polynomial_simplification_l146_14631

theorem polynomial_simplification (x : ℝ) :
  (x^3 + 4*x^2 - 7*x + 11) + (-4*x^4 - x^3 + x^2 + 7*x + 3) + (3*x^4 - 2*x^3 + 5*x - 1) =
  -x^4 - 2*x^3 + 5*x^2 + 5*x + 13 := by
  sorry

end polynomial_simplification_l146_14631


namespace practice_schedule_l146_14640

theorem practice_schedule (trumpet flute piano : ℕ) 
  (h_trumpet : trumpet = 11)
  (h_flute : flute = 3)
  (h_piano : piano = 7) :
  Nat.lcm trumpet (Nat.lcm flute piano) = 231 := by
  sorry

end practice_schedule_l146_14640


namespace tangent_line_properties_l146_14665

open Real

theorem tangent_line_properties (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₁ ≠ 1) :
  (∃ (k b : ℝ), 
    (∀ x, k * x + b = (1 / x₁) * x - 1 + log x₁) ∧
    (∀ x, k * x + b = exp x₂ * x + exp x₂ * (1 - x₂))) →
  (x₁ * exp x₂ = 1 ∧ (x₁ + 1) / (x₁ - 1) + x₂ = 0) :=
by sorry

end tangent_line_properties_l146_14665


namespace goshawk_eurasian_nature_reserve_birds_l146_14613

theorem goshawk_eurasian_nature_reserve_birds (B : ℝ) (h : B > 0) :
  let hawks := 0.30 * B
  let non_hawks := B - hawks
  let paddyfield_warblers := 0.40 * non_hawks
  let other_birds := 0.35 * B
  let kingfishers := B - hawks - paddyfield_warblers - other_birds
  kingfishers / paddyfield_warblers = 0.25
:= by sorry

end goshawk_eurasian_nature_reserve_birds_l146_14613


namespace no_articles_in_general_context_l146_14633

/-- Represents the possible article choices for a noun in a sentence -/
inductive Article
  | Definite   -- represents "the"
  | Indefinite -- represents "a" or "an"
  | None       -- represents no article

/-- Represents the context of a sentence -/
inductive Context
  | General
  | Specific

/-- Represents a noun in the sentence -/
inductive Noun
  | College
  | Prison

/-- Determines the correct article for a noun given the context -/
def correctArticle (context : Context) (noun : Noun) : Article :=
  match context, noun with
  | Context.General, _ => Article.None
  | Context.Specific, _ => Article.Definite

/-- The main theorem stating that in a general context, 
    both "college" and "prison" should have no article -/
theorem no_articles_in_general_context : 
  ∀ (context : Context),
    context = Context.General →
    correctArticle context Noun.College = Article.None ∧
    correctArticle context Noun.Prison = Article.None :=
by sorry

end no_articles_in_general_context_l146_14633


namespace debate_team_boys_l146_14657

theorem debate_team_boys (total : ℕ) (girls : ℕ) (groups : ℕ) :
  total % 9 = 0 →
  total / 9 = groups →
  girls = 46 →
  groups = 8 →
  total - girls = 26 :=
by sorry

end debate_team_boys_l146_14657


namespace field_length_width_ratio_l146_14624

theorem field_length_width_ratio :
  ∀ (w : ℝ),
    w > 0 →
    24 > 0 →
    ∃ (k : ℕ), 24 = k * w →
    36 = (1/8) * (24 * w) →
    24 / w = 2 := by
  sorry

end field_length_width_ratio_l146_14624


namespace accuracy_of_0_598_l146_14623

/-- Represents the place value of a digit in a decimal number. -/
inductive PlaceValue
  | Ones
  | Tenths
  | Hundredths
  | Thousandths
  | TenThousandths
  deriving Repr

/-- Determines the place value of accuracy for a given decimal number. -/
def placeOfAccuracy (n : Float) : PlaceValue :=
  match n.toString.split (· = '.') with
  | [_, fractional] =>
    match fractional.length with
    | 1 => PlaceValue.Tenths
    | 2 => PlaceValue.Hundredths
    | 3 => PlaceValue.Thousandths
    | _ => PlaceValue.TenThousandths
  | _ => PlaceValue.Ones

/-- Theorem: The approximate number 0.598 is accurate to the thousandths place. -/
theorem accuracy_of_0_598 :
  placeOfAccuracy 0.598 = PlaceValue.Thousandths := by
  sorry

end accuracy_of_0_598_l146_14623


namespace birds_on_fence_l146_14671

theorem birds_on_fence (initial_birds : ℝ) (birds_flown_away : ℝ) :
  initial_birds = 12.0 →
  birds_flown_away = 8.0 →
  initial_birds - birds_flown_away = 4.0 := by
  sorry

end birds_on_fence_l146_14671


namespace trout_catch_total_l146_14650

theorem trout_catch_total (people : ℕ) (individual_share : ℕ) (h1 : people = 2) (h2 : individual_share = 9) :
  people * individual_share = 18 := by
  sorry

end trout_catch_total_l146_14650


namespace fraction_division_eval_l146_14697

theorem fraction_division_eval : (7 / 3) / (8 / 15) = 35 / 8 := by
  sorry

end fraction_division_eval_l146_14697


namespace point_2_4_is_D_l146_14600

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the diagram points
def F : Point2D := ⟨5, 5⟩
def D : Point2D := ⟨2, 4⟩

-- Theorem statement
theorem point_2_4_is_D : 
  ∃ (p : Point2D), p.x = 2 ∧ p.y = 4 ∧ p = D :=
sorry

end point_2_4_is_D_l146_14600


namespace third_fourth_product_l146_14618

def arithmetic_sequence (a : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => arithmetic_sequence a d n + d

theorem third_fourth_product (a : ℝ) (d : ℝ) :
  arithmetic_sequence a d 5 = 17 ∧ d = 2 →
  (arithmetic_sequence a d 2) * (arithmetic_sequence a d 3) = 143 := by
sorry

end third_fourth_product_l146_14618


namespace intersection_of_M_and_N_l146_14610

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log x}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x^2)}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end intersection_of_M_and_N_l146_14610


namespace function_symmetry_l146_14655

/-- Given a function f(x) = ax^4 + b*cos(x) - x, if f(-3) = 7, then f(3) = 1 -/
theorem function_symmetry (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^4 + b * Real.cos x - x
  f (-3) = 7 → f 3 = 1 := by
  sorry

end function_symmetry_l146_14655


namespace ellipse_properties_l146_14653

/-- Ellipse C: x^2/4 + y^2 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Circle: x^2 + y^2 = 1 -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Line passing through F2 -/
def line_through_F2 (x y : ℝ) (m : ℝ) : Prop := y = m * x

/-- Line 2mx - 2y - 2m + 1 = 0 -/
def intersecting_line (x y : ℝ) (m : ℝ) : Prop := 2*m*x - 2*y - 2*m + 1 = 0

/-- Theorem stating the properties of the ellipse -/
theorem ellipse_properties :
  ∃ (F1 F2 : ℝ × ℝ),
    (∀ x y : ℝ, ellipse_C x y →
      (∃ A B : ℝ × ℝ, 
        line_through_F2 A.1 A.2 (F2.2 / F2.1) ∧
        line_through_F2 B.1 B.2 (F2.2 / F2.1) ∧
        ellipse_C A.1 A.2 ∧
        ellipse_C B.1 B.2 ∧
        (Real.sqrt ((A.1 - F1.1)^2 + (A.2 - F1.2)^2) +
         Real.sqrt ((B.1 - F1.1)^2 + (B.2 - F1.2)^2) +
         Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8))) ∧
    (∀ m : ℝ, ∃ x y : ℝ, ellipse_C x y ∧ intersecting_line x y m) ∧
    (∀ P Q : ℝ × ℝ, 
      ellipse_C P.1 P.2 →
      unit_circle Q.1 Q.2 →
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ 3) ∧
    (∃ P Q : ℝ × ℝ,
      ellipse_C P.1 P.2 ∧
      unit_circle Q.1 Q.2 ∧
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 3) :=
by sorry

end ellipse_properties_l146_14653


namespace ducks_joined_l146_14622

theorem ducks_joined (initial_ducks final_ducks : ℕ) (h : final_ducks ≥ initial_ducks) :
  final_ducks - initial_ducks = final_ducks - initial_ducks :=
by sorry

end ducks_joined_l146_14622


namespace friendly_seq_uniqueness_l146_14663

/-- A sequence of strictly increasing natural numbers -/
def IncreasingSeq := ℕ → ℕ

/-- Two sequences are friendly if every natural number is represented exactly once as their product -/
def Friendly (a b : IncreasingSeq) : Prop :=
  ∀ n : ℕ, ∃! (i j : ℕ), n = a i * b j

/-- The theorem stating that one friendly sequence uniquely determines the other -/
theorem friendly_seq_uniqueness (a b c : IncreasingSeq) :
  Friendly a b → Friendly a c → b = c := by sorry

end friendly_seq_uniqueness_l146_14663


namespace distance_difference_l146_14651

-- Define the distances
def mart_to_home : ℕ := 800
def home_to_academy : ℕ := 1300  -- 1 km + 300 m = 1000 m + 300 m = 1300 m
def academy_to_restaurant : ℕ := 1700

-- Theorem to prove
theorem distance_difference :
  (mart_to_home + home_to_academy) - academy_to_restaurant = 400 := by
  sorry

end distance_difference_l146_14651


namespace beth_friends_count_l146_14689

theorem beth_friends_count (initial_packs : ℝ) (additional_packs : ℝ) (final_packs : ℝ) :
  initial_packs = 4 →
  additional_packs = 6 →
  final_packs = 6.4 →
  ∃ (num_friends : ℝ),
    num_friends > 0 ∧
    final_packs = additional_packs + initial_packs / num_friends ∧
    num_friends = 10 := by
  sorry

end beth_friends_count_l146_14689


namespace opposite_of_fraction_l146_14681

theorem opposite_of_fraction (n : ℕ) (h : n ≠ 0) : 
  (-(1 : ℚ) / n) = -((1 : ℚ) / n) := by sorry

end opposite_of_fraction_l146_14681


namespace factor_expression_l146_14666

theorem factor_expression (b : ℝ) : 275 * b^2 + 55 * b = 55 * b * (5 * b + 1) := by
  sorry

end factor_expression_l146_14666


namespace linear_function_passes_through_point_l146_14673

/-- A linear function f(x) = kx + k - 1 passes through the point (-1, -1) for any real k. -/
theorem linear_function_passes_through_point
  (k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ k * x + k - 1
  f (-1) = -1 := by
  sorry

end linear_function_passes_through_point_l146_14673


namespace inscribed_squares_inequality_l146_14626

/-- Given a triangle ABC with semiperimeter s and area F, and squares with side lengths x, y, and z
    inscribed such that:
    - Square with side x has two vertices on BC
    - Square with side y has two vertices on AC
    - Square with side z has two vertices on AB
    The sum of the reciprocals of their side lengths is less than or equal to s(2+√3)/(2F) -/
theorem inscribed_squares_inequality (s F x y z : ℝ) (h_pos : s > 0 ∧ F > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0) :
  1/x + 1/y + 1/z ≤ s * (2 + Real.sqrt 3) / (2 * F) := by
  sorry


end inscribed_squares_inequality_l146_14626


namespace score_not_above_average_l146_14656

structure ClassData where
  participants : ℕ
  mean : ℝ
  median : ℝ
  mode : ℝ
  variance : ℝ
  excellenceRate : ℝ

def class901 : ClassData :=
  { participants := 40
  , mean := 75
  , median := 78
  , mode := 77
  , variance := 158
  , excellenceRate := 0.2 }

def class902 : ClassData :=
  { participants := 45
  , mean := 75
  , median := 76
  , mode := 74
  , variance := 122
  , excellenceRate := 0.2 }

theorem score_not_above_average (score : ℝ) :
  score = 77 → ¬(score > class902.mean) := by
  sorry

end score_not_above_average_l146_14656


namespace probability_is_one_third_l146_14661

/-- The set of digits used to form the number -/
def digits : Finset Nat := {2, 4, 6, 7}

/-- A function to check if a number is odd -/
def isOdd (n : Nat) : Bool := n % 2 = 1

/-- A function to check if a number is not a multiple of 3 -/
def notMultipleOf3 (n : Nat) : Bool := n % 3 ≠ 0

/-- The set of all four-digit numbers that can be formed using the given digits -/
def allNumbers : Finset Nat := sorry

/-- The set of favorable numbers (odd with hundreds digit not multiple of 3) -/
def favorableNumbers : Finset Nat := sorry

/-- The probability of forming a favorable number -/
def probability : Rat := (Finset.card favorableNumbers : Rat) / (Finset.card allNumbers : Rat)

theorem probability_is_one_third :
  probability = 1 / 3 := by sorry

end probability_is_one_third_l146_14661


namespace min_value_implies_a_eq_one_exp_log_sin_positive_l146_14664

noncomputable section

variable (x : ℝ)
variable (a : ℝ)

def f (x : ℝ) : ℝ := Real.log x + a / x - 1

theorem min_value_implies_a_eq_one (h : ∀ x > 0, f x ≥ 0) (h' : ∃ x > 0, f x = 0) : a = 1 :=
sorry

theorem exp_log_sin_positive : ∀ x > 0, Real.exp x + (Real.log x - 1) * Real.sin x > 0 :=
sorry

end min_value_implies_a_eq_one_exp_log_sin_positive_l146_14664


namespace max_leftover_stickers_l146_14630

theorem max_leftover_stickers (y : ℕ+) : 
  ∃ (q r : ℕ), y = 12 * q + r ∧ r < 12 ∧ r ≤ 11 ∧ 
  ∀ (q' r' : ℕ), y = 12 * q' + r' ∧ r' < 12 → r' ≤ r :=
by sorry

end max_leftover_stickers_l146_14630


namespace boyden_family_children_l146_14628

theorem boyden_family_children (adult_ticket_cost child_ticket_cost total_cost : ℕ) 
  (num_adults : ℕ) (h1 : adult_ticket_cost = child_ticket_cost + 6)
  (h2 : total_cost = 77) (h3 : adult_ticket_cost = 19) (h4 : num_adults = 2) :
  ∃ (num_children : ℕ), 
    num_children * child_ticket_cost + num_adults * adult_ticket_cost = total_cost ∧ 
    num_children = 3 := by
  sorry

end boyden_family_children_l146_14628


namespace local_minimum_condition_l146_14699

/-- The function f(x) = x^3 + (x-a)^2 has a local minimum at x = 2 if and only if a = 8 -/
theorem local_minimum_condition (a : ℝ) : 
  (∃ δ > 0, ∀ x ∈ Set.Ioo (2 - δ) (2 + δ), 
    x^3 + (x - a)^2 ≥ 2^3 + (2 - a)^2) ↔ a = 8 := by
  sorry


end local_minimum_condition_l146_14699


namespace inverse_mod_89_l146_14601

theorem inverse_mod_89 (h : (16⁻¹ : ZMod 89) = 28) : (256⁻¹ : ZMod 89) = 56 := by
  sorry

end inverse_mod_89_l146_14601


namespace least_subtraction_for_divisibility_least_subtraction_for_1000_l146_14690

theorem least_subtraction_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (k : Nat), k ≤ d - 1 ∧ (n - k) % d = 0 ∧
  ∀ (m : Nat), m < k → (n - m) % d ≠ 0 :=
by sorry

theorem least_subtraction_for_1000 :
  ∃ (k : Nat), k = 398 ∧ 
  (427398 - k) % 1000 = 0 ∧
  ∀ (m : Nat), m < k → (427398 - m) % 1000 ≠ 0 :=
by sorry

end least_subtraction_for_divisibility_least_subtraction_for_1000_l146_14690


namespace a_explicit_formula_l146_14668

def a : ℕ → ℤ
  | 0 => -1
  | 1 => -3
  | 2 => -5
  | 3 => 5
  | (n + 4) => 8 * a (n + 3) - 22 * a (n + 2) + 24 * a (n + 1) - 9 * a n

theorem a_explicit_formula (n : ℕ) :
  a n = 2 + n - 3^(n + 1) + n * 3^n :=
by sorry

end a_explicit_formula_l146_14668


namespace multiply_57_47_l146_14638

theorem multiply_57_47 : 57 * 47 = 2820 := by
  sorry

end multiply_57_47_l146_14638


namespace badge_exchange_l146_14672

theorem badge_exchange (x : ℕ) : 
  -- Vasya initially had 5 more badges than Tolya
  let vasya_initial := x + 5
  -- Vasya exchanged 24% of his badges for 20% of Tolya's badges
  let vasya_final := vasya_initial - (24 * vasya_initial) / 100 + (20 * x) / 100
  let tolya_final := x - (20 * x) / 100 + (24 * vasya_initial) / 100
  -- After the exchange, Vasya had one badge less than Tolya
  vasya_final + 1 = tolya_final →
  -- Prove that Tolya initially had 45 badges and Vasya initially had 50 badges
  x = 45 ∧ vasya_initial = 50 := by
sorry

end badge_exchange_l146_14672


namespace city_partition_theorem_l146_14629

/-- A directed graph where each vertex has outdegree 2 -/
structure CityGraph (V : Type) :=
  (edges : V → V → Prop)
  (outdegree_two : ∀ v : V, ∃ u w : V, u ≠ w ∧ edges v u ∧ edges v w ∧ ∀ x : V, edges v x → (x = u ∨ x = w))

/-- A partition of the vertices into 1014 sets -/
def ValidPartition (V : Type) (G : CityGraph V) :=
  ∃ (f : V → Fin 1014),
    (∀ v w : V, G.edges v w → f v ≠ f w) ∧
    (∀ i j : Fin 1014, i ≠ j →
      (∀ v w : V, f v = i ∧ f w = j → G.edges v w) ∨
      (∀ v w : V, f v = i ∧ f w = j → G.edges w v))

/-- The main theorem: every CityGraph has a ValidPartition -/
theorem city_partition_theorem (V : Type) (G : CityGraph V) :
  ValidPartition V G :=
sorry

end city_partition_theorem_l146_14629


namespace smallest_student_count_l146_14687

/-- Represents the number of students in each grade --/
structure StudentCounts where
  grade9 : ℕ
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Checks if the given student counts satisfy the required ratios --/
def satisfiesRatios (counts : StudentCounts) : Prop :=
  3 * counts.grade10 = 2 * counts.grade12 ∧
  7 * counts.grade11 = 4 * counts.grade12 ∧
  5 * counts.grade9 = 3 * counts.grade12

/-- Calculates the total number of students --/
def totalStudents (counts : StudentCounts) : ℕ :=
  counts.grade9 + counts.grade10 + counts.grade11 + counts.grade12

/-- Theorem stating the smallest possible number of students --/
theorem smallest_student_count :
  ∃ (counts : StudentCounts),
    satisfiesRatios counts ∧
    totalStudents counts = 298 ∧
    (∀ (other : StudentCounts),
      satisfiesRatios other → totalStudents other ≥ 298) :=
  sorry

end smallest_student_count_l146_14687


namespace binomial_15_4_l146_14648

theorem binomial_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end binomial_15_4_l146_14648


namespace sum_base3_equals_10200_l146_14679

/-- Converts a base 3 number represented as a list of digits to its decimal equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun digit acc => acc * 3 + digit) 0

/-- Represents a number in base 3 -/
structure Base3 where
  digits : List Nat
  valid : ∀ d ∈ digits, d < 3

/-- Addition of Base3 numbers -/
def addBase3 (a b : Base3) : Base3 :=
  sorry

theorem sum_base3_equals_10200 :
  let a := Base3.mk [1] (by simp)
  let b := Base3.mk [2, 1] (by simp)
  let c := Base3.mk [2, 1, 2] (by simp)
  let d := Base3.mk [1, 2, 1, 2] (by simp)
  let result := Base3.mk [0, 0, 2, 0, 1] (by simp)
  addBase3 (addBase3 (addBase3 a b) c) d = result :=
sorry

end sum_base3_equals_10200_l146_14679


namespace value_of_a_minus_b_l146_14662

theorem value_of_a_minus_b (a b : ℚ) 
  (eq1 : 2025 * a + 2030 * b = 2035)
  (eq2 : 2027 * a + 2032 * b = 2037) : 
  a - b = -3 := by
sorry

end value_of_a_minus_b_l146_14662


namespace vector_collinearity_l146_14660

theorem vector_collinearity (k : ℝ) : 
  let a : Fin 2 → ℝ := ![Real.sqrt 3, 1]
  let b : Fin 2 → ℝ := ![0, -1]
  let c : Fin 2 → ℝ := ![k, Real.sqrt 3]
  (∃ (t : ℝ), a + 2 • b = t • c) → k = -3 := by
sorry

end vector_collinearity_l146_14660


namespace connie_red_markers_l146_14602

theorem connie_red_markers (total_markers blue_markers : ℕ) 
  (h1 : total_markers = 3343)
  (h2 : blue_markers = 1028) :
  total_markers - blue_markers = 2315 :=
by
  sorry

end connie_red_markers_l146_14602


namespace inequality_and_equality_condition_l146_14604

theorem inequality_and_equality_condition 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) 
  (ha₁ : a₁ > 0) (ha₂ : a₂ > 0) (ha₃ : a₃ > 0) 
  (hb₁ : b₁ > 0) (hb₂ : b₂ > 0) (hb₃ : b₃ > 0) : 
  (a₁ * b₂ + a₁ * b₃ + a₂ * b₁ + a₂ * b₃ + a₃ * b₁ + a₃ * b₂)^2 ≥ 
    4 * (a₁ * a₂ + a₂ * a₃ + a₃ * a₁) * (b₁ * b₂ + b₂ * b₃ + b₃ * b₁) ∧ 
  ((a₁ * b₂ + a₁ * b₃ + a₂ * b₁ + a₂ * b₃ + a₃ * b₁ + a₃ * b₂)^2 = 
    4 * (a₁ * a₂ + a₂ * a₃ + a₃ * a₁) * (b₁ * b₂ + b₂ * b₃ + b₃ * b₁) ↔ 
    a₁ * b₂ = a₂ * b₁ ∧ a₂ * b₃ = a₃ * b₂ ∧ a₃ * b₁ = a₁ * b₃) :=
by sorry

end inequality_and_equality_condition_l146_14604


namespace parallel_vectors_m_value_l146_14639

/-- Two 2D vectors are parallel if the cross product of their components is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (m, -1)
  let b : ℝ × ℝ := (1, m + 2)
  parallel a b → m = -1 :=
by
  sorry

end parallel_vectors_m_value_l146_14639


namespace finite_prime_triples_l146_14693

theorem finite_prime_triples (k : ℕ) :
  Set.Finite {triple : ℕ × ℕ × ℕ | 
    let (p, q, r) := triple
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    (q * r - k) % p = 0 ∧
    (p * r - k) % q = 0 ∧
    (p * q - k) % r = 0} :=
by sorry

end finite_prime_triples_l146_14693


namespace y_intercept_range_l146_14676

-- Define the points A and B
def A : ℝ × ℝ := (-1, -2)
def B : ℝ × ℝ := (2, 3)

-- Define the line l: x + y - c = 0
def line_l (c : ℝ) (x y : ℝ) : Prop := x + y - c = 0

-- Define what it means for a point to be on the line
def point_on_line (p : ℝ × ℝ) (c : ℝ) : Prop :=
  line_l c p.1 p.2

-- Define what it means for a line to intersect a segment
def intersects_segment (c : ℝ) : Prop :=
  ∃ t : ℝ, t ∈ (Set.Icc 0 1) ∧
    point_on_line ((1 - t) • A.1 + t • B.1, (1 - t) • A.2 + t • B.2) c

-- State the theorem
theorem y_intercept_range :
  ∀ c : ℝ, intersects_segment c → c ∈ Set.Icc (-3) 5 :=
sorry

end y_intercept_range_l146_14676


namespace jack_king_ace_probability_l146_14669

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the event of drawing three specific cards in order -/
def draw_three_cards (d : Deck) (first second third : Fin 52) : ℚ :=
  (4 : ℚ) / 52 * (4 : ℚ) / 51 * (4 : ℚ) / 50

/-- The probability of drawing a Jack, then a King, then an Ace from a standard deck without replacement -/
theorem jack_king_ace_probability (d : Deck) :
  ∃ (j k a : Fin 52), draw_three_cards d j k a = 16 / 33150 :=
sorry

end jack_king_ace_probability_l146_14669


namespace stability_comparison_l146_14615

/-- Represents an athlete's performance in a series of tests -/
structure AthletePerformance where
  average_score : ℝ
  variance : ℝ

/-- Defines stability of performance based on variance -/
def more_stable (a b : AthletePerformance) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two athletes with the same average score,
    the one with lower variance has more stable performance -/
theorem stability_comparison 
  (athlete_A athlete_B : AthletePerformance)
  (h_same_average : athlete_A.average_score = athlete_B.average_score)
  (h_A_variance : athlete_A.variance = 1.2)
  (h_B_variance : athlete_B.variance = 1) :
  more_stable athlete_B athlete_A :=
sorry

end stability_comparison_l146_14615


namespace loan_duration_l146_14674

/-- Proves that the first part of a loan was lent for 8 years given specific conditions -/
theorem loan_duration (total sum : ℕ) (second_part : ℕ) (first_rate second_rate : ℚ) (second_duration : ℕ) : 
  total = 2730 →
  second_part = 1680 →
  first_rate = 3 / 100 →
  second_rate = 5 / 100 →
  second_duration = 3 →
  ∃ (first_duration : ℕ), 
    (total - second_part) * first_rate * first_duration = second_part * second_rate * second_duration ∧
    first_duration = 8 :=
by sorry

end loan_duration_l146_14674


namespace chime_2500_date_l146_14642

/-- Represents a date with year, month, and day -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Represents a time with hour and minute -/
structure Time where
  hour : ℕ
  minute : ℕ

/-- Calculates the number of chimes from a given start time to midnight -/
def chimesToMidnight (startTime : Time) : ℕ :=
  sorry

/-- Calculates the number of chimes in a full day -/
def chimesPerDay : ℕ :=
  sorry

/-- Calculates the date of the nth chime given a start date and time -/
def dateOfNthChime (n : ℕ) (startDate : Date) (startTime : Time) : Date :=
  sorry

/-- Theorem stating that the 2500th chime occurs on January 21, 2023 -/
theorem chime_2500_date :
  let startDate := Date.mk 2023 1 1
  let startTime := Time.mk 14 30
  dateOfNthChime 2500 startDate startTime = Date.mk 2023 1 21 :=
sorry

end chime_2500_date_l146_14642


namespace alcohol_mixture_proof_l146_14683

/-- Proves that mixing 175 gallons of 15% alcohol solution with 75 gallons of 35% alcohol solution 
    results in 250 gallons of 21% alcohol solution. -/
theorem alcohol_mixture_proof :
  let solution_1_volume : ℝ := 175
  let solution_1_concentration : ℝ := 0.15
  let solution_2_volume : ℝ := 75
  let solution_2_concentration : ℝ := 0.35
  let total_volume : ℝ := 250
  let final_concentration : ℝ := 0.21
  (solution_1_volume + solution_2_volume = total_volume) ∧
  (solution_1_volume * solution_1_concentration + solution_2_volume * solution_2_concentration = 
   total_volume * final_concentration) :=
by sorry


end alcohol_mixture_proof_l146_14683


namespace complement_intersection_theorem_l146_14695

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}
def M : Finset ℕ := {1, 4}
def N : Finset ℕ := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {2, 3} := by sorry

end complement_intersection_theorem_l146_14695


namespace pythagorean_triple_for_eleven_l146_14634

theorem pythagorean_triple_for_eleven : ∃ b c : ℕ, 11^2 + b^2 = c^2 ∧ c = 61 := by
  sorry

end pythagorean_triple_for_eleven_l146_14634


namespace complex_sum_powers_l146_14643

theorem complex_sum_powers (w : ℂ) (hw : w^2 - w + 1 = 0) : 
  w^101 + w^102 + w^103 + w^104 + w^105 = 4*w - 1 := by
sorry

end complex_sum_powers_l146_14643


namespace stratified_sampling_survey_l146_14637

theorem stratified_sampling_survey (teachers : ℕ) (male_students : ℕ) (female_students : ℕ) 
  (sample_female : ℕ) (sample_size : ℕ) : 
  teachers = 200 → 
  male_students = 1200 → 
  female_students = 1000 → 
  sample_female = 80 → 
  sample_size * (female_students / (teachers + male_students + female_students)) = sample_female → 
  sample_size = 192 := by
sorry

end stratified_sampling_survey_l146_14637


namespace intersection_complement_equality_l146_14636

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

theorem intersection_complement_equality : B ∩ (U \ A) = {3, 4} := by
  sorry

end intersection_complement_equality_l146_14636


namespace wall_bricks_l146_14652

/-- Represents the number of bricks in the wall -/
def num_bricks : ℕ := 288

/-- Time taken by the first bricklayer to build the wall alone -/
def time1 : ℕ := 8

/-- Time taken by the second bricklayer to build the wall alone -/
def time2 : ℕ := 12

/-- Reduction in combined output when working together -/
def output_reduction : ℕ := 12

/-- Time taken by both bricklayers working together -/
def combined_time : ℕ := 6

theorem wall_bricks :
  (combined_time : ℚ) * ((num_bricks / time1 : ℚ) + (num_bricks / time2 : ℚ) - output_reduction) = num_bricks := by
  sorry

#check wall_bricks

end wall_bricks_l146_14652


namespace geometric_sequence_property_l146_14617

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence, if a_7 · a_19 = 8, then a_3 · a_23 = 8 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geom : geometric_sequence a) 
    (h_prod : a 7 * a 19 = 8) : 
  a 3 * a 23 = 8 := by
  sorry

end geometric_sequence_property_l146_14617


namespace empty_seats_l146_14692

theorem empty_seats (children : ℕ) (adults : ℕ) (total_seats : ℕ) : 
  children = 52 → adults = 29 → total_seats = 95 → 
  total_seats - (children + adults) = 14 := by
  sorry

end empty_seats_l146_14692


namespace loan_duration_l146_14675

/-- Proves that the first part of a loan is lent for 8 years given specific conditions -/
theorem loan_duration (total_sum interest_rate1 interest_rate2 duration2 : ℚ) 
  (second_part : ℚ) : 
  total_sum = 2743 →
  second_part = 1688 →
  interest_rate1 = 3/100 →
  interest_rate2 = 5/100 →
  duration2 = 3 →
  let first_part := total_sum - second_part
  let duration1 := (second_part * interest_rate2 * duration2) / (first_part * interest_rate1)
  duration1 = 8 := by
  sorry

end loan_duration_l146_14675


namespace value_of_x_l146_14696

theorem value_of_x (x y z d e f : ℝ) 
  (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0)
  (h1 : x * y / (x + 2 * y) = d)
  (h2 : x * z / (2 * x + z) = e)
  (h3 : y * z / (y + 2 * z) = f) :
  x = 3 * d * e * f / (d * e - 2 * d * f + e * f) := by
  sorry

end value_of_x_l146_14696


namespace fixed_monthly_charge_l146_14684

-- Define the fixed monthly charge for internet service
def F : ℝ := sorry

-- Define the charge for calls in January
def C : ℝ := sorry

-- Define the total bill for January
def january_bill : ℝ := 50

-- Define the total bill for February
def february_bill : ℝ := 76

-- Theorem to prove the fixed monthly charge for internet service
theorem fixed_monthly_charge :
  (F + C = january_bill) →
  (F + 2 * C = february_bill) →
  F = 24 := by sorry

end fixed_monthly_charge_l146_14684


namespace equal_ratios_sum_ratio_l146_14635

theorem equal_ratios_sum_ratio (x y z : ℚ) : 
  x / 2 = y / 3 ∧ y / 3 = z / 4 → (x + y + z) / (2 * z) = 9 / 8 := by
  sorry

end equal_ratios_sum_ratio_l146_14635


namespace cube_roots_of_unity_powers_l146_14641

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- Definition of x -/
noncomputable def x : ℂ := (-1 + i * Real.sqrt 3) / 2

/-- Definition of y -/
noncomputable def y : ℂ := (-1 - i * Real.sqrt 3) / 2

/-- Main theorem -/
theorem cube_roots_of_unity_powers :
  (x ^ 5 + y ^ 5 = -2) ∧
  (x ^ 7 + y ^ 7 = 2) ∧
  (x ^ 9 + y ^ 9 = -2) ∧
  (x ^ 11 + y ^ 11 = 2) ∧
  (x ^ 13 + y ^ 13 = -2) :=
by sorry

end cube_roots_of_unity_powers_l146_14641


namespace ninth_row_sum_l146_14686

/-- Yang Hui's Triangle (Pascal's Triangle) -/
def yangHuiTriangle (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

/-- Sum of elements in a row of Yang Hui's Triangle -/
def rowSum (n : ℕ) : ℕ :=
  (List.range (n + 1)).map (yangHuiTriangle n) |>.sum

/-- Theorem: The sum of all numbers in the 9th row of Yang Hui's Triangle is 2^8 -/
theorem ninth_row_sum : rowSum 8 = 2^8 := by
  sorry

end ninth_row_sum_l146_14686


namespace line_y_intercept_l146_14649

/-- Given a line passing through points (3,2), (1,k), and (-4,1), 
    prove that its y-intercept is 11/7 -/
theorem line_y_intercept (k : ℚ) : 
  (∃ m b : ℚ, (3 : ℚ) * m + b = 2 ∧ 1 * m + b = k ∧ (-4 : ℚ) * m + b = 1) → 
  (∃ m b : ℚ, (3 : ℚ) * m + b = 2 ∧ 1 * m + b = k ∧ (-4 : ℚ) * m + b = 1 ∧ b = 11/7) :=
by sorry

end line_y_intercept_l146_14649


namespace bridge_length_l146_14603

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 170 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 205 := by
  sorry

end bridge_length_l146_14603


namespace box_number_problem_l146_14625

theorem box_number_problem (a b c d e : ℕ) 
  (sum_all : a + b + c + d + e = 35)
  (sum_first_three : a + b + c = 22)
  (sum_last_three : c + d + e = 25)
  (first_box : a = 3)
  (last_box : e = 4) :
  b * d = 63 := by
  sorry

end box_number_problem_l146_14625


namespace circle_diameter_from_area_l146_14667

/-- Given a circle with area 225π cm², its diameter is 30 cm. -/
theorem circle_diameter_from_area : 
  ∀ (r : ℝ), r > 0 → π * r^2 = 225 * π → 2 * r = 30 := by
  sorry

end circle_diameter_from_area_l146_14667


namespace delta_value_l146_14682

theorem delta_value : ∀ Δ : ℤ, 4 * (-3) = Δ - 1 → Δ = -11 := by
  sorry

end delta_value_l146_14682


namespace tree_growth_relation_l146_14658

/-- The height of a tree after a number of months -/
def tree_height (initial_height growth_rate : ℝ) (months : ℝ) : ℝ :=
  initial_height + growth_rate * months

/-- Theorem: The height of the tree after x months is 80 + 2x -/
theorem tree_growth_relation (x : ℝ) :
  tree_height 80 2 x = 80 + 2 * x := by sorry

end tree_growth_relation_l146_14658


namespace no_perfect_square_exists_l146_14608

theorem no_perfect_square_exists (a : ℕ) : 
  (∃ k : ℕ, ((a^2 - 3)^3 + 1)^a - 1 = k^2) → False ∧
  (∃ k : ℕ, ((a^2 - 3)^3 + 1)^(a+1) - 1 = k^2) → False :=
by sorry

end no_perfect_square_exists_l146_14608


namespace tax_discount_commute_mathville_problem_l146_14688

/-- Proves that the order of applying tax and discount doesn't affect the final price --/
theorem tax_discount_commute (price : ℝ) (tax_rate discount_rate : ℝ) 
  (h_tax : 0 ≤ tax_rate) (h_discount : 0 ≤ discount_rate) (h_price : 0 < price) :
  price * (1 + tax_rate) * (1 - discount_rate) = price * (1 - discount_rate) * (1 + tax_rate) := by
  sorry

/-- Calculates Bob's method: tax first, then discount --/
def bob_method (price tax_rate discount_rate : ℝ) : ℝ :=
  price * (1 + tax_rate) * (1 - discount_rate)

/-- Calculates Alice's method: discount first, then tax --/
def alice_method (price tax_rate discount_rate : ℝ) : ℝ :=
  price * (1 - discount_rate) * (1 + tax_rate)

theorem mathville_problem (price : ℝ) (tax_rate discount_rate : ℝ) 
  (h_tax : tax_rate = 0.08) (h_discount : discount_rate = 0.25) (h_price : price = 120) :
  bob_method price tax_rate discount_rate - alice_method price tax_rate discount_rate = 0 := by
  sorry

end tax_discount_commute_mathville_problem_l146_14688


namespace min_investment_amount_l146_14619

/-- Represents an investment plan with two interest rates -/
structure InvestmentPlan where
  amount_at_7_percent : ℝ
  amount_at_12_percent : ℝ

/-- Calculates the total interest earned from an investment plan -/
def total_interest (plan : InvestmentPlan) : ℝ :=
  0.07 * plan.amount_at_7_percent + 0.12 * plan.amount_at_12_percent

/-- Calculates the total investment amount -/
def total_investment (plan : InvestmentPlan) : ℝ :=
  plan.amount_at_7_percent + plan.amount_at_12_percent

/-- Theorem: The minimum total investment amount is $25,000 -/
theorem min_investment_amount :
  ∀ (plan : InvestmentPlan),
    plan.amount_at_7_percent ≤ 11000 →
    total_interest plan ≥ 2450 →
    total_investment plan ≥ 25000 :=
by sorry

end min_investment_amount_l146_14619


namespace central_angle_doubles_when_radius_halves_l146_14607

theorem central_angle_doubles_when_radius_halves (r l α β : ℝ) (h1 : r > 0) (h2 : l > 0) (h3 : α > 0) :
  α = l / r →
  β = l / (r / 2) →
  β = 2 * α := by
sorry

end central_angle_doubles_when_radius_halves_l146_14607
