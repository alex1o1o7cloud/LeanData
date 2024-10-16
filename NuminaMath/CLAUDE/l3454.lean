import Mathlib

namespace NUMINAMATH_CALUDE_pants_bought_with_tshirts_l3454_345482

/-- Given the price relationships of pants and t-shirts, prove that 1 pant was bought with 6 t-shirts -/
theorem pants_bought_with_tshirts (x : ℚ) :
  (∃ (p t : ℚ), p > 0 ∧ t > 0 ∧ 
    x * p + 6 * t = 750 ∧
    p + 12 * t = 750 ∧
    8 * t = 400) →
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_pants_bought_with_tshirts_l3454_345482


namespace NUMINAMATH_CALUDE_bob_probability_after_two_turns_l3454_345470

/-- Represents the player who has the ball -/
inductive Player : Type
| Alice : Player
| Bob : Player

/-- The probability of keeping the ball for each player -/
def keep_prob (p : Player) : ℚ :=
  match p with
  | Player.Alice => 2/3
  | Player.Bob => 3/4

/-- The probability of tossing the ball for each player -/
def toss_prob (p : Player) : ℚ :=
  1 - keep_prob p

/-- The probability that Bob has the ball after two turns, given he starts with it -/
def bob_has_ball_after_two_turns : ℚ :=
  keep_prob Player.Bob * keep_prob Player.Bob +
  keep_prob Player.Bob * toss_prob Player.Bob * keep_prob Player.Alice +
  toss_prob Player.Bob * toss_prob Player.Alice

theorem bob_probability_after_two_turns :
  bob_has_ball_after_two_turns = 37/48 := by
  sorry

end NUMINAMATH_CALUDE_bob_probability_after_two_turns_l3454_345470


namespace NUMINAMATH_CALUDE_caroline_lassi_production_l3454_345494

/-- Given that Caroline can make 15 lassis out of 3 mangoes, 
    prove that she can make 75 lassis out of 15 mangoes. -/
theorem caroline_lassi_production :
  (∃ (lassis_per_3_mangoes : ℕ), lassis_per_3_mangoes = 15) →
  (∃ (lassis_per_15_mangoes : ℕ), lassis_per_15_mangoes = 75) :=
by
  sorry

end NUMINAMATH_CALUDE_caroline_lassi_production_l3454_345494


namespace NUMINAMATH_CALUDE_fraction_addition_l3454_345429

theorem fraction_addition (c : ℝ) : (6 + 5 * c) / 5 + 3 = (21 + 5 * c) / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3454_345429


namespace NUMINAMATH_CALUDE_blue_faces_proportion_l3454_345475

/-- Given a cube of side length n, prove that if one-third of the faces of its unit cubes are blue, then n = 3 -/
theorem blue_faces_proportion (n : ℕ) : n ≥ 1 →
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → n = 3 := by sorry

end NUMINAMATH_CALUDE_blue_faces_proportion_l3454_345475


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l3454_345478

/-- Represents a systematic sample of students -/
structure SystematicSample where
  totalStudents : Nat
  sampleSize : Nat
  sampleNumbers : List Nat

/-- Checks if a given sample is a valid systematic sample -/
def isValidSystematicSample (sample : SystematicSample) : Prop :=
  sample.totalStudents = 20 ∧
  sample.sampleSize = 4 ∧
  sample.sampleNumbers = [5, 10, 15, 20]

/-- Theorem stating that the given sample is the correct systematic sample -/
theorem correct_systematic_sample :
  ∃ (sample : SystematicSample), isValidSystematicSample sample :=
sorry

end NUMINAMATH_CALUDE_correct_systematic_sample_l3454_345478


namespace NUMINAMATH_CALUDE_favorite_fruit_pears_l3454_345467

theorem favorite_fruit_pears (total students_oranges students_apples students_strawberries : ℕ) 
  (h1 : total = 450)
  (h2 : students_oranges = 70)
  (h3 : students_apples = 147)
  (h4 : students_strawberries = 113) :
  total - (students_oranges + students_apples + students_strawberries) = 120 := by
  sorry

end NUMINAMATH_CALUDE_favorite_fruit_pears_l3454_345467


namespace NUMINAMATH_CALUDE_max_value_theorem_l3454_345413

theorem max_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x^2 + y^2 + z^2 = 1) :
  2 * x * y * Real.sqrt 2 + 6 * y * z ≤ Real.sqrt 11 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3454_345413


namespace NUMINAMATH_CALUDE_candy_distribution_bijection_l3454_345416

/-- The candy distribution function -/
def f (x n : ℕ) : ℕ := (x * (x + 1) / 2) % n

/-- Theorem: The candy distribution function is a bijection iff n is a power of 2 -/
theorem candy_distribution_bijection (n : ℕ) :
  Function.Bijective (λ x => f x n) ↔ ∃ a : ℕ, n = 2^a :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_bijection_l3454_345416


namespace NUMINAMATH_CALUDE_book_ratio_proof_l3454_345449

theorem book_ratio_proof (total_books : ℕ) (hardcover_nonfiction : ℕ) :
  total_books = 280 →
  hardcover_nonfiction = 55 →
  ∃ (paperback_fiction paperback_nonfiction : ℕ),
    paperback_fiction + paperback_nonfiction + hardcover_nonfiction = total_books ∧
    paperback_nonfiction = hardcover_nonfiction + 20 ∧
    paperback_fiction = 2 * paperback_nonfiction :=
by
  sorry


end NUMINAMATH_CALUDE_book_ratio_proof_l3454_345449


namespace NUMINAMATH_CALUDE_cornelia_triple_kilee_age_l3454_345484

/-- The number of years in the future when Cornelia will be three times as old as Kilee -/
def future_years : ℕ := 10

/-- Kilee's current age -/
def kilee_age : ℕ := 20

/-- Cornelia's current age -/
def cornelia_age : ℕ := 80

/-- Theorem stating that in 'future_years' years, Cornelia will be three times as old as Kilee -/
theorem cornelia_triple_kilee_age :
  cornelia_age + future_years = 3 * (kilee_age + future_years) :=
by sorry

end NUMINAMATH_CALUDE_cornelia_triple_kilee_age_l3454_345484


namespace NUMINAMATH_CALUDE_draw_three_with_red_standard_deck_l3454_345425

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (black_suits : Nat)
  (red_suits : Nat)

/-- Calculate the number of ways to draw three cards with at least one red card -/
def draw_three_with_red (d : Deck) : Nat :=
  d.total_cards * (d.total_cards - 1) * (d.total_cards - 2) - 
  (d.black_suits * d.cards_per_suit) * (d.black_suits * d.cards_per_suit - 1) * (d.black_suits * d.cards_per_suit - 2)

/-- Theorem: The number of ways to draw three cards with at least one red from a standard deck is 117000 -/
theorem draw_three_with_red_standard_deck :
  let standard_deck : Deck := {
    total_cards := 52,
    suits := 4,
    cards_per_suit := 13,
    black_suits := 2,
    red_suits := 2
  }
  draw_three_with_red standard_deck = 117000 := by
  sorry

end NUMINAMATH_CALUDE_draw_three_with_red_standard_deck_l3454_345425


namespace NUMINAMATH_CALUDE_fraction_simplification_l3454_345428

theorem fraction_simplification (y : ℚ) (h : y = 77) : (7 * y + 77) / 77 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3454_345428


namespace NUMINAMATH_CALUDE_unique_three_digit_divisible_by_11_l3454_345439

theorem unique_three_digit_divisible_by_11 : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  n % 10 = 2 ∧          -- units digit is 2
  n / 100 = 7 ∧         -- hundreds digit is 7
  n % 11 = 0 ∧          -- divisible by 11
  n = 792 := by          -- the number is 792
sorry


end NUMINAMATH_CALUDE_unique_three_digit_divisible_by_11_l3454_345439


namespace NUMINAMATH_CALUDE_no_solution_exists_l3454_345473

theorem no_solution_exists : ¬∃ x : ℝ, x^2 * 1 * 3 - x * 1 * 3^2 = 6 := by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3454_345473


namespace NUMINAMATH_CALUDE_tax_percentage_calculation_l3454_345408

theorem tax_percentage_calculation (initial_bars : ℕ) (remaining_bars : ℕ) : 
  initial_bars = 60 →
  remaining_bars = 27 →
  ∃ (tax_percentage : ℚ),
    tax_percentage = 10 ∧
    remaining_bars = (initial_bars * (1 - tax_percentage / 100) / 2).floor :=
by sorry

end NUMINAMATH_CALUDE_tax_percentage_calculation_l3454_345408


namespace NUMINAMATH_CALUDE_escalator_length_is_200_l3454_345447

/-- The length of an escalator given its speed, a person's walking speed, and the time taken to cover the entire length. -/
def escalator_length (escalator_speed : ℝ) (person_speed : ℝ) (time_taken : ℝ) : ℝ :=
  (escalator_speed + person_speed) * time_taken

/-- Theorem stating that the length of the escalator is 200 feet -/
theorem escalator_length_is_200 :
  escalator_length 15 5 10 = 200 := by
  sorry

end NUMINAMATH_CALUDE_escalator_length_is_200_l3454_345447


namespace NUMINAMATH_CALUDE_opposite_sqrt_81_l3454_345479

theorem opposite_sqrt_81 : -(Real.sqrt (Real.sqrt 81)) = -9 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sqrt_81_l3454_345479


namespace NUMINAMATH_CALUDE_solution_set_equality_l3454_345438

theorem solution_set_equality : Set ℝ := by
  have h : Set ℝ := {x | (x - 1)^2 < 1}
  have g : Set ℝ := Set.Ioo 0 2
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l3454_345438


namespace NUMINAMATH_CALUDE_fixed_point_symmetric_coordinates_l3454_345480

/-- Given a line that always passes through a fixed point P and P is symmetric about x + y = 0, prove P's coordinates -/
theorem fixed_point_symmetric_coordinates :
  ∀ (k : ℝ), 
  (∃ (P : ℝ × ℝ), ∀ (x y : ℝ), k * x - y + k - 2 = 0 → (x, y) = P) →
  (∃ (P' : ℝ × ℝ), 
    (P'.1 + P'.2 = 0) ∧ 
    (P'.1 - P.1)^2 + (P'.2 - P.2)^2 = 2 * ((P.1 + P.2) / 2)^2) →
  P = (2, 1) :=
by sorry


end NUMINAMATH_CALUDE_fixed_point_symmetric_coordinates_l3454_345480


namespace NUMINAMATH_CALUDE_ethanol_percentage_in_fuel_B_l3454_345456

/-- Proves that the percentage of ethanol in fuel B is 16% given the problem conditions -/
theorem ethanol_percentage_in_fuel_B (tank_capacity : ℝ) (ethanol_A : ℝ) (total_ethanol : ℝ) (fuel_A_volume : ℝ) : 
  tank_capacity = 200 →
  ethanol_A = 0.12 →
  total_ethanol = 28 →
  fuel_A_volume = 99.99999999999999 →
  (total_ethanol - ethanol_A * fuel_A_volume) / (tank_capacity - fuel_A_volume) = 0.16 := by
sorry

#eval (28 - 0.12 * 99.99999999999999) / (200 - 99.99999999999999)

end NUMINAMATH_CALUDE_ethanol_percentage_in_fuel_B_l3454_345456


namespace NUMINAMATH_CALUDE_probability_two_pairs_l3454_345493

def total_socks : ℕ := 10
def drawn_socks : ℕ := 4
def distinct_pairs : ℕ := 5

theorem probability_two_pairs : 
  (Nat.choose distinct_pairs 2) / (Nat.choose total_socks drawn_socks) = 1 / 21 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_pairs_l3454_345493


namespace NUMINAMATH_CALUDE_power_sum_equality_l3454_345465

theorem power_sum_equality : (-2)^48 + 3^(4^3 + 5^2 - 7^2) = 2^48 + 3^40 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3454_345465


namespace NUMINAMATH_CALUDE_arcsin_cos_arcsin_plus_arccos_sin_arccos_l3454_345452

theorem arcsin_cos_arcsin_plus_arccos_sin_arccos (x : ℝ) : 
  Real.arcsin (Real.cos (Real.arcsin x)) + Real.arccos (Real.sin (Real.arccos x)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_cos_arcsin_plus_arccos_sin_arccos_l3454_345452


namespace NUMINAMATH_CALUDE_car_travel_time_difference_l3454_345448

/-- Proves that the time difference between two cars traveling 150 miles is 2 hours,
    given their speeds differ by 10 mph and one car's speed is 22.83882181415011 mph. -/
theorem car_travel_time_difference 
  (distance : ℝ) 
  (speed_R : ℝ) 
  (speed_P : ℝ) : 
  distance = 150 →
  speed_R = 22.83882181415011 →
  speed_P = speed_R + 10 →
  distance / speed_R - distance / speed_P = 2 := by
sorry

end NUMINAMATH_CALUDE_car_travel_time_difference_l3454_345448


namespace NUMINAMATH_CALUDE_part_one_part_two_l3454_345446

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2| - |2*x - a|

-- Part I
theorem part_one :
  {x : ℝ | f 3 x > 0} = {x : ℝ | 1/3 < x ∧ x < 5} :=
sorry

-- Part II
theorem part_two :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 0 → f a x < 3) → a < 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3454_345446


namespace NUMINAMATH_CALUDE_median_line_property_l3454_345411

-- Define the plane α
variable (α : Plane)

-- Define points A, B, and C
variable (A B C : Point)

-- Define the property of being non-collinear
def NonCollinear (A B C : Point) : Prop := sorry

-- Define the property of a point being outside a plane
def OutsidePlane (P : Point) (π : Plane) : Prop := sorry

-- Define the property of a point being equidistant from a plane
def EquidistantFromPlane (P : Point) (π : Plane) : Prop := sorry

-- Define a median line of a triangle
def MedianLine (M : Line) (A B C : Point) : Prop := sorry

-- Define the property of a line being parallel to a plane
def ParallelToPlane (L : Line) (π : Plane) : Prop := sorry

-- Define the property of a line lying within a plane
def LiesWithinPlane (L : Line) (π : Plane) : Prop := sorry

-- The theorem statement
theorem median_line_property 
  (h1 : NonCollinear A B C)
  (h2 : OutsidePlane A α ∧ OutsidePlane B α ∧ OutsidePlane C α)
  (h3 : EquidistantFromPlane A α ∧ EquidistantFromPlane B α ∧ EquidistantFromPlane C α) :
  ∃ (M : Line), MedianLine M A B C ∧ (ParallelToPlane M α ∨ LiesWithinPlane M α) :=
sorry

end NUMINAMATH_CALUDE_median_line_property_l3454_345411


namespace NUMINAMATH_CALUDE_quadratic_distinct_integer_roots_l3454_345414

theorem quadratic_distinct_integer_roots (a : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ 2 * x^2 - a * x + 2 * a = 0 ∧ 2 * y^2 - a * y + 2 * a = 0) ↔ 
  (a = -2 ∨ a = 18) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_integer_roots_l3454_345414


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3454_345418

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- A sequence is arithmetic if the difference between consecutive terms is constant -/
def is_arithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Points lie on the line y = x + 1 -/
def points_on_line (a : Sequence) : Prop :=
  ∀ n : ℕ+, a n = n + 1

theorem sufficient_not_necessary :
  (∀ a : Sequence, points_on_line a → is_arithmetic a) ∧
  (∃ a : Sequence, is_arithmetic a ∧ ¬points_on_line a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3454_345418


namespace NUMINAMATH_CALUDE_u_2002_equals_2_l3454_345486

def f (x : ℕ) : ℕ :=
  match x with
  | 1 => 4
  | 2 => 1
  | 3 => 3
  | 4 => 5
  | 5 => 2
  | _ => 0  -- Default case for completeness

def u : ℕ → ℕ
  | 0 => 4
  | n + 1 => f (u n)

theorem u_2002_equals_2 : u 2002 = 2 := by
  sorry

end NUMINAMATH_CALUDE_u_2002_equals_2_l3454_345486


namespace NUMINAMATH_CALUDE_exact_power_pair_l3454_345437

theorem exact_power_pair : 
  ∀ (a b : ℕ), 
  (∀ (n : ℕ), ∃ (c : ℕ), a^n + b^n = c^(n+1)) → 
  (a = 2 ∧ b = 2) := by
sorry

end NUMINAMATH_CALUDE_exact_power_pair_l3454_345437


namespace NUMINAMATH_CALUDE_initial_working_hours_l3454_345441

/-- Given the following conditions:
  - 75 men initially working
  - Initial depth dug: 50 meters
  - New depth to dig: 70 meters
  - New working hours: 6 hours/day
  - 65 extra men added
Prove that the initial working hours H satisfy the equation:
  75 * H * 50 = (75 + 65) * 6 * 70
-/
theorem initial_working_hours (H : ℝ) : 75 * H * 50 = (75 + 65) * 6 * 70 := by
  sorry

end NUMINAMATH_CALUDE_initial_working_hours_l3454_345441


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l3454_345403

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * (x + 1)^2 + 4 - a

-- State the theorem
theorem quadratic_function_inequality (a x₁ x₂ : ℝ) 
  (ha : 0 < a ∧ a < 3) 
  (hx : x₁ > x₂) 
  (hsum : x₁ + x₂ = 1 - a) : 
  f a x₁ > f a x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l3454_345403


namespace NUMINAMATH_CALUDE_football_team_yardage_l3454_345444

/-- A football team's yardage problem -/
theorem football_team_yardage (L : ℤ) : 
  ((-L : ℤ) + 13 = 8) → L = 5 := by
  sorry

end NUMINAMATH_CALUDE_football_team_yardage_l3454_345444


namespace NUMINAMATH_CALUDE_some_number_value_l3454_345422

theorem some_number_value : ∃ n : ℤ, (481 + 426) * n - 4 * 481 * 426 = 3025 ∧ n = 906 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3454_345422


namespace NUMINAMATH_CALUDE_journey_time_equation_l3454_345427

theorem journey_time_equation (x : ℝ) (h : x > 0) : 
  let distance : ℝ := 50
  let taxi_speed : ℝ := x + 15
  let bus_speed : ℝ := x
  let taxi_time : ℝ := distance / taxi_speed
  let bus_time : ℝ := distance / bus_speed
  taxi_time = 2/3 * bus_time → distance / taxi_speed = 2/3 * (distance / bus_speed) :=
by sorry

end NUMINAMATH_CALUDE_journey_time_equation_l3454_345427


namespace NUMINAMATH_CALUDE_plane_binary_trees_eq_triangulations_l3454_345492

/-- A plane binary tree -/
structure PlaneBinaryTree where
  vertices : Set Nat
  edges : Set (Nat × Nat)
  root : Nat
  leaves : Set Nat

/-- A triangulation of a polygon -/
structure Triangulation where
  vertices : Set Nat
  diagonals : Set (Nat × Nat)

/-- The number of different plane binary trees with one root and n leaves -/
def num_plane_binary_trees (n : Nat) : Nat :=
  sorry

/-- The number of triangulations of an (n+1)-gon -/
def num_triangulations (n : Nat) : Nat :=
  sorry

/-- Theorem stating the equality between the number of plane binary trees and triangulations -/
theorem plane_binary_trees_eq_triangulations (n : Nat) :
  num_plane_binary_trees n = num_triangulations n :=
  sorry

end NUMINAMATH_CALUDE_plane_binary_trees_eq_triangulations_l3454_345492


namespace NUMINAMATH_CALUDE_ellipse_intersection_ratio_l3454_345476

/-- First ellipse -/
def ellipse1 (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Second ellipse -/
def ellipse2 (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

/-- Point P on the first ellipse -/
def P : ℝ × ℝ := sorry

/-- Point O is the origin -/
def O : ℝ × ℝ := (0, 0)

/-- Q is the intersection of ray PO with the second ellipse -/
noncomputable def Q : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem ellipse_intersection_ratio :
  ellipse1 P.1 P.2 →
  ellipse2 Q.1 Q.2 →
  distance P Q / distance O P = 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_ratio_l3454_345476


namespace NUMINAMATH_CALUDE_special_ellipse_eccentricity_special_ellipse_equation_l3454_345405

/-- An ellipse with the given properties -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_foci : F₁.1 < F₂.1 -- F₁ is left focus, F₂ is right focus
  h_ellipse : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ ({A, B} : Set (ℝ × ℝ))
  h_line : A.2 - F₁.2 = A.1 - F₁.1 ∧ B.2 - F₁.2 = B.1 - F₁.1 -- Line through F₁ with slope 1
  h_arithmetic : ∃ (d : ℝ), dist A F₂ + d = dist A B ∧ dist A B + d = dist B F₂
  h_circle : ∃ (r : ℝ), dist A (-2, 0) = r ∧ dist B (-2, 0) = r

/-- The eccentricity of the special ellipse is √2/2 -/
theorem special_ellipse_eccentricity (E : SpecialEllipse) : 
  (E.a^2 - E.b^2) / E.a^2 = 1/2 := by sorry

/-- The equation of the special ellipse is x²/72 + y²/36 = 1 -/
theorem special_ellipse_equation (E : SpecialEllipse) : 
  E.a^2 = 72 ∧ E.b^2 = 36 := by sorry

end NUMINAMATH_CALUDE_special_ellipse_eccentricity_special_ellipse_equation_l3454_345405


namespace NUMINAMATH_CALUDE_area_ratio_for_specific_trapezoid_l3454_345495

/-- Represents a trapezoid with extended legs -/
structure ExtendedTrapezoid where
  PQ : ℝ  -- Length of base PQ
  RS : ℝ  -- Length of base RS
  -- Assume other necessary properties of a trapezoid

/-- The ratio of the area of triangle TPQ to the area of trapezoid PQRS -/
def area_ratio (t : ExtendedTrapezoid) : ℚ :=
  100 / 341

/-- Theorem stating the area ratio for the given trapezoid -/
theorem area_ratio_for_specific_trapezoid :
  ∃ t : ExtendedTrapezoid, t.PQ = 10 ∧ t.RS = 21 ∧ area_ratio t = 100 / 341 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_for_specific_trapezoid_l3454_345495


namespace NUMINAMATH_CALUDE_pens_pencils_cost_l3454_345443

def total_spent : ℝ := 32
def backpack_cost : ℝ := 15
def notebook_cost : ℝ := 3
def num_notebooks : ℕ := 5

def cost_pens_pencils : ℝ := total_spent - (backpack_cost + notebook_cost * num_notebooks)

theorem pens_pencils_cost (h : cost_pens_pencils = 2) : 
  cost_pens_pencils / 2 = 1 := by sorry

end NUMINAMATH_CALUDE_pens_pencils_cost_l3454_345443


namespace NUMINAMATH_CALUDE_symmetric_lines_line_symmetry_l3454_345431

/-- Given two lines in a plane and a point, this theorem states that these lines are symmetric with respect to the given point. -/
theorem symmetric_lines (l₁ l₂ : ℝ → ℝ) (M : ℝ × ℝ) : Prop :=
  ∀ x y : ℝ, l₁ y = x → l₂ (4 - y) = 6 - x

/-- The main theorem proving that y = 3x - 17 is symmetric to y = 3x + 3 with respect to (3, 2) -/
theorem line_symmetry : 
  symmetric_lines (λ y => (y + 17) / 3) (λ y => (y - 3) / 3) (3, 2) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_lines_line_symmetry_l3454_345431


namespace NUMINAMATH_CALUDE_carpet_length_proof_l3454_345488

theorem carpet_length_proof (length width diagonal : ℝ) : 
  length > 0 ∧ width > 0 ∧
  length * width = 60 ∧
  diagonal + length = 5 * width ∧
  diagonal^2 = length^2 + width^2 →
  length = 2 * Real.sqrt 30 :=
by sorry

end NUMINAMATH_CALUDE_carpet_length_proof_l3454_345488


namespace NUMINAMATH_CALUDE_tommy_initial_balloons_l3454_345455

/-- The number of balloons Tommy's mom gave him -/
def balloons_given : ℝ := 34.5

/-- The total number of balloons Tommy had after receiving more -/
def total_balloons : ℝ := 60.75

/-- The number of balloons Tommy had initially -/
def initial_balloons : ℝ := total_balloons - balloons_given

theorem tommy_initial_balloons :
  initial_balloons = 26.25 := by sorry

end NUMINAMATH_CALUDE_tommy_initial_balloons_l3454_345455


namespace NUMINAMATH_CALUDE_chimney_bricks_l3454_345472

/-- Represents the time (in hours) it takes Brenda to build the chimney alone -/
def brenda_time : ℝ := 8

/-- Represents the time (in hours) it takes Bob to build the chimney alone -/
def bob_time : ℝ := 12

/-- Represents the decrease in productivity (in bricks per hour) when working together -/
def productivity_decrease : ℝ := 15

/-- Represents the time (in hours) it takes Brenda and Bob to build the chimney together -/
def joint_time : ℝ := 6

/-- Theorem stating that the number of bricks in the chimney is 360 -/
theorem chimney_bricks : ℝ := by
  sorry

end NUMINAMATH_CALUDE_chimney_bricks_l3454_345472


namespace NUMINAMATH_CALUDE_student_money_proof_l3454_345489

/-- The amount of money (in rubles) the student has after buying 11 pens -/
def remaining_after_11 : ℝ := 8

/-- The additional amount (in rubles) needed to buy 15 pens -/
def additional_for_15 : ℝ := 12.24

/-- The cost of one pen in rubles -/
noncomputable def pen_cost : ℝ :=
  (additional_for_15 + remaining_after_11) / (15 - 11)

/-- The initial amount of money the student had in rubles -/
noncomputable def initial_amount : ℝ :=
  11 * pen_cost + remaining_after_11

theorem student_money_proof :
  initial_amount = 63.66 := by sorry

end NUMINAMATH_CALUDE_student_money_proof_l3454_345489


namespace NUMINAMATH_CALUDE_inscribed_square_area_ratio_l3454_345485

/-- The ratio of areas between an inscribed square and a larger square -/
theorem inscribed_square_area_ratio :
  let large_square_side : ℝ := 4
  let inscribed_square_horizontal_offset : ℝ := 1.5
  let inscribed_square_vertical_offset : ℝ := 4/3
  let inscribed_square_side : ℝ := large_square_side - 2 * inscribed_square_horizontal_offset
  let large_square_area : ℝ := large_square_side ^ 2
  let inscribed_square_area : ℝ := inscribed_square_side ^ 2
  inscribed_square_area / large_square_area = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_ratio_l3454_345485


namespace NUMINAMATH_CALUDE_early_arrival_speed_l3454_345454

/-- Represents the travel scenario for Mrs. Early --/
structure TravelScenario where
  speed : ℝ
  timeDifference : ℝ  -- in hours, positive for early, negative for late

/-- Calculates the required speed to arrive exactly on time --/
def exactTimeSpeed (scenario1 scenario2 : TravelScenario) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the correct speed for Mrs. Early to arrive on time --/
theorem early_arrival_speed : 
  let scenario1 : TravelScenario := { speed := 50, timeDifference := -1/15 }
  let scenario2 : TravelScenario := { speed := 70, timeDifference := 1/12 }
  let requiredSpeed := exactTimeSpeed scenario1 scenario2
  57 < requiredSpeed ∧ requiredSpeed < 58 := by
  sorry

end NUMINAMATH_CALUDE_early_arrival_speed_l3454_345454


namespace NUMINAMATH_CALUDE_fencing_match_prob_increase_correct_l3454_345481

def fencing_match_prob_increase 
  (k l : ℕ) 
  (hk : k < 15) 
  (hl : l < 15) 
  (p : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 1) : ℝ :=
  Nat.choose (k + l) k * p^k * (1 - p)^(l + 1)

theorem fencing_match_prob_increase_correct 
  (k l : ℕ) 
  (hk : k < 15) 
  (hl : l < 15) 
  (p : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 1) :
  fencing_match_prob_increase k l hk hl p hp = 
    Nat.choose (k + l) k * p^k * (1 - p)^(l + 1) := by
  sorry

end NUMINAMATH_CALUDE_fencing_match_prob_increase_correct_l3454_345481


namespace NUMINAMATH_CALUDE_soap_box_dimension_proof_l3454_345426

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ := d.length * d.width * d.height

theorem soap_box_dimension_proof 
  (carton : BoxDimensions)
  (soap : BoxDimensions)
  (h_carton : carton = ⟨25, 48, 60⟩)
  (h_soap : soap = ⟨8, soap.width, 5⟩)
  (h_max_boxes : (300 : ℝ) * boxVolume soap ≤ boxVolume carton) :
  soap.width ≤ 6 := by
sorry

end NUMINAMATH_CALUDE_soap_box_dimension_proof_l3454_345426


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_even_integers_l3454_345459

/-- The sum of the first n positive even integers -/
def sumFirstNEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

/-- The sum of five consecutive even integers starting from k -/
def sumFiveConsecutiveEvenIntegers (k : ℕ) : ℕ := 5 * k - 10

theorem largest_of_five_consecutive_even_integers :
  ∃ k : ℕ, 
    sumFirstNEvenIntegers 30 = sumFiveConsecutiveEvenIntegers k ∧ 
    k = 190 := by
  sorry

#eval sumFirstNEvenIntegers 30  -- Should output 930
#eval sumFiveConsecutiveEvenIntegers 190  -- Should also output 930

end NUMINAMATH_CALUDE_largest_of_five_consecutive_even_integers_l3454_345459


namespace NUMINAMATH_CALUDE_smallest_cuboid_face_area_l3454_345412

/-- Given a cuboid with integer volume and face areas 7, 27, and L, 
    prove that the smallest possible integer value for L is 21 -/
theorem smallest_cuboid_face_area (a b c : ℕ+) (L : ℕ) : 
  (a * b : ℕ) = 7 →
  (a * c : ℕ) = 27 →
  (b * c : ℕ) = L →
  (∃ (v : ℕ), v = a * b * c) →
  L ≥ 21 ∧ 
  (∀ L' : ℕ, L' ≥ 21 → ∃ (a' b' c' : ℕ+), 
    (a' * b' : ℕ) = 7 ∧ 
    (a' * c' : ℕ) = 27 ∧ 
    (b' * c' : ℕ) = L') :=
by sorry

#check smallest_cuboid_face_area

end NUMINAMATH_CALUDE_smallest_cuboid_face_area_l3454_345412


namespace NUMINAMATH_CALUDE_alf3_weight_l3454_345404

/-- The molecular weight of a compound -/
def molecularWeight (alWeight fWeight : ℝ) : ℝ := alWeight + 3 * fWeight

/-- The total weight of a given number of moles of a compound -/
def totalWeight (molWeight : ℝ) (moles : ℝ) : ℝ := molWeight * moles

/-- Theorem stating the total weight of 10 moles of aluminum fluoride -/
theorem alf3_weight : 
  let alWeight : ℝ := 26.98
  let fWeight : ℝ := 19.00
  let moles : ℝ := 10
  totalWeight (molecularWeight alWeight fWeight) moles = 839.8 := by
sorry

end NUMINAMATH_CALUDE_alf3_weight_l3454_345404


namespace NUMINAMATH_CALUDE_solve_for_k_l3454_345464

-- Define the polynomials
def p (x y k : ℝ) : ℝ := x^3 - 2*k*x*y
def q (x y : ℝ) : ℝ := y^2 + 4*x*y

-- Define the condition that the difference doesn't contain xy term
def no_xy_term (k : ℝ) : Prop :=
  ∀ x y, ∃ a b c, p x y k - q x y = a*x^3 + b*y^2 + c

-- State the theorem
theorem solve_for_k :
  ∃ k : ℝ, no_xy_term k ∧ k = -2 :=
sorry

end NUMINAMATH_CALUDE_solve_for_k_l3454_345464


namespace NUMINAMATH_CALUDE_trig_system_solution_l3454_345457

theorem trig_system_solution (x y : ℝ) 
  (h1 : Real.tan x * Real.tan y = 1/6)
  (h2 : Real.sin x * Real.sin y = 1/(5 * Real.sqrt 2)) :
  Real.cos (x + y) = 1/Real.sqrt 2 ∧ 
  Real.cos (x - y) = 7/(5 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_trig_system_solution_l3454_345457


namespace NUMINAMATH_CALUDE_mindmaster_codes_l3454_345477

/-- The number of available colors in the Mindmaster game -/
def num_colors : ℕ := 8

/-- The number of slots in each code -/
def num_slots : ℕ := 4

/-- The total number of possible secret codes in the Mindmaster game -/
def total_codes : ℕ := num_colors ^ num_slots

/-- Theorem stating that the total number of secret codes is 4096 -/
theorem mindmaster_codes : total_codes = 4096 := by
  sorry

end NUMINAMATH_CALUDE_mindmaster_codes_l3454_345477


namespace NUMINAMATH_CALUDE_product_xy_equals_one_l3454_345462

theorem product_xy_equals_one (x y : ℝ) 
  (h : (Real.sqrt (x^2 + 1) - x + 1) * (Real.sqrt (y^2 + 1) - y + 1) = 2) : 
  x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_equals_one_l3454_345462


namespace NUMINAMATH_CALUDE_tangent_line_at_one_max_value_of_f_l3454_345442

/-- The function f(x) defined as 2a ln x - x^2 --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.log x - x^2

/-- Theorem stating the equation of the tangent line when a = 2 --/
theorem tangent_line_at_one (a : ℝ) (h : a = 2) :
  ∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ 2 * x - y - 3 = 0 :=
sorry

/-- Theorem stating the maximum value of f(x) when a > 0 --/
theorem max_value_of_f (a : ℝ) (h : a > 0) :
  ∃ x_max : ℝ, x_max = Real.sqrt 2 ∧
    ∀ x : ℝ, x > 0 → f a x ≤ f a x_max ∧ f a x_max = Real.log 2 - 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_max_value_of_f_l3454_345442


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l3454_345499

theorem divisibility_by_eleven (m : Nat) : 
  m < 10 → -- m is a single digit
  (742 * 100000 + m * 10000 + 834) % 11 = 0 → -- 742m834 is divisible by 11
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l3454_345499


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l3454_345445

/-- A geometric sequence with sum S_n = k^n + r^m -/
structure GeometricSequence where
  k : ℝ
  r : ℝ
  m : ℤ
  a : ℕ → ℝ
  sum : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  sum_formula : ∀ n, sum n = k^n + r^m

/-- The properties of r and m in the geometric sequence -/
theorem geometric_sequence_properties (seq : GeometricSequence) : 
  seq.r = -1 ∧ Odd seq.m :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l3454_345445


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3454_345461

theorem lcm_from_product_and_hcf (a b : ℕ+) :
  a * b = 84942 ∧ Nat.gcd a b = 33 → Nat.lcm a b = 2574 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3454_345461


namespace NUMINAMATH_CALUDE_prob_different_fruits_l3454_345434

/-- The number of fruit types Joe can choose from -/
def num_fruit_types : ℕ := 4

/-- The number of meals Joe has in a day -/
def num_meals : ℕ := 4

/-- The probability of Joe eating the same fruit for all meals -/
def prob_same_fruit : ℚ := (1 / num_fruit_types) ^ num_meals * num_fruit_types

/-- The probability of Joe eating at least two different kinds of fruit in a day -/
theorem prob_different_fruits : (1 : ℚ) - prob_same_fruit = 63 / 64 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_fruits_l3454_345434


namespace NUMINAMATH_CALUDE_tess_distance_graph_l3454_345407

-- Define the triangular block
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define Tess's position as a function of time
def tessPosition (t : ℝ) (tri : Triangle) : ℝ × ℝ :=
  sorry

-- Define the straight-line distance from A to Tess's position
def distanceFromA (t : ℝ) (tri : Triangle) : ℝ :=
  sorry

-- Define the properties of the distance function
def isRisingThenFalling (f : ℝ → ℝ) : Prop :=
  sorry

def peaksAtB (f : ℝ → ℝ) (tri : Triangle) : Prop :=
  sorry

-- Theorem statement
theorem tess_distance_graph (tri : Triangle) :
  isRisingThenFalling (fun t => distanceFromA t tri) ∧
  peaksAtB (fun t => distanceFromA t tri) tri :=
sorry

end NUMINAMATH_CALUDE_tess_distance_graph_l3454_345407


namespace NUMINAMATH_CALUDE_special_polynomial_sum_l3454_345421

theorem special_polynomial_sum (d₁ d₂ d₃ d₄ e₁ e₂ e₃ e₄ : ℝ) 
  (h : ∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + d₁*x + e₁)*(x^2 + d₂*x + e₂)*(x^2 + d₃*x + e₃)*(x^2 + d₄*x + e₄)) : 
  d₁*e₁ + d₂*e₂ + d₃*e₃ + d₄*e₄ = -1 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_sum_l3454_345421


namespace NUMINAMATH_CALUDE_lg_difference_equals_two_l3454_345487

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_difference_equals_two : lg 25 - lg (1/4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_lg_difference_equals_two_l3454_345487


namespace NUMINAMATH_CALUDE_total_money_proof_l3454_345423

/-- The total amount of money p, q, and r have among themselves -/
def total_amount (r_amount : ℚ) (r_fraction : ℚ) : ℚ :=
  r_amount / r_fraction

theorem total_money_proof (r_amount : ℚ) (h1 : r_amount = 2000) 
  (r_fraction : ℚ) (h2 : r_fraction = 2/3) : 
  total_amount r_amount r_fraction = 5000 := by
  sorry

#check total_money_proof

end NUMINAMATH_CALUDE_total_money_proof_l3454_345423


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l3454_345424

theorem nested_fraction_equality : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = 21 / 55 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l3454_345424


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l3454_345435

noncomputable def f (x : ℝ) : ℝ := Real.exp x / (x + 2)

theorem f_derivative_at_zero : 
  deriv f 0 = 1/4 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l3454_345435


namespace NUMINAMATH_CALUDE_fixed_point_of_function_l3454_345440

/-- The function f(x) = kx - k - a^(x-1) always passes through the point (1, -1) -/
theorem fixed_point_of_function (k : ℝ) (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := fun x => k * x - k - a^(x - 1)
  f 1 = -1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_function_l3454_345440


namespace NUMINAMATH_CALUDE_solve_flower_problem_l3454_345460

def flower_problem (yoojung_flowers namjoon_flowers : ℕ) : Prop :=
  (yoojung_flowers = 32) ∧
  (yoojung_flowers = 4 * namjoon_flowers) ∧
  (yoojung_flowers + namjoon_flowers = 40)

theorem solve_flower_problem :
  ∃ (yoojung_flowers namjoon_flowers : ℕ),
    flower_problem yoojung_flowers namjoon_flowers :=
by
  sorry

end NUMINAMATH_CALUDE_solve_flower_problem_l3454_345460


namespace NUMINAMATH_CALUDE_square_friendly_unique_l3454_345491

def square_friendly (c : ℤ) : Prop :=
  ∀ m : ℤ, ∃ n : ℤ, m^2 + 18*m + c = n^2

theorem square_friendly_unique : 
  (square_friendly 81) ∧ (∀ c : ℤ, square_friendly c → c = 81) :=
sorry

end NUMINAMATH_CALUDE_square_friendly_unique_l3454_345491


namespace NUMINAMATH_CALUDE_broken_crayons_percentage_l3454_345453

theorem broken_crayons_percentage (total : ℕ) (slightly_used : ℕ) :
  total = 120 →
  slightly_used = 56 →
  (total / 3 : ℚ) + slightly_used + (total / 5 : ℚ) = total →
  (total / 5 : ℚ) / total * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_broken_crayons_percentage_l3454_345453


namespace NUMINAMATH_CALUDE_barium_oxide_required_l3454_345471

/-- Represents a chemical substance with its number of moles -/
structure Substance where
  name : String
  moles : ℚ

/-- Represents a chemical reaction with reactants and products -/
structure Reaction where
  reactants : List Substance
  products : List Substance

def barium_oxide_water_reaction : Reaction :=
  { reactants := [
      { name := "BaO", moles := 1 },
      { name := "H2O", moles := 1 }
    ],
    products := [
      { name := "Ba(OH)2", moles := 1 }
    ]
  }

theorem barium_oxide_required (water_moles : ℚ) (barium_hydroxide_moles : ℚ) :
  water_moles = barium_hydroxide_moles →
  (∃ (bao : Substance),
    bao.name = "BaO" ∧
    bao.moles = water_moles ∧
    bao.moles = barium_hydroxide_moles ∧
    (∃ (h2o : Substance) (baoh2 : Substance),
      h2o.name = "H2O" ∧
      h2o.moles = water_moles ∧
      baoh2.name = "Ba(OH)2" ∧
      baoh2.moles = barium_hydroxide_moles ∧
      barium_oxide_water_reaction.reactants = [bao, h2o] ∧
      barium_oxide_water_reaction.products = [baoh2])) :=
by
  sorry

end NUMINAMATH_CALUDE_barium_oxide_required_l3454_345471


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l3454_345410

theorem sqrt_expression_equality (t : ℝ) : 
  Real.sqrt (9 * t^4 + 4 * t^2 + 4 * t) = |t| * Real.sqrt ((3 * t^2 + 2 * t) * (3 * t^2 + 2 * t + 2)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l3454_345410


namespace NUMINAMATH_CALUDE_ratio_value_l3454_345497

theorem ratio_value (a b c d : ℚ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  a * c / (b * d) = 20 := by
sorry

end NUMINAMATH_CALUDE_ratio_value_l3454_345497


namespace NUMINAMATH_CALUDE_water_consumption_proof_l3454_345406

/-- Calculates the total water consumption for horses over a given number of days -/
def total_water_consumption (initial_horses : ℕ) (added_horses : ℕ) (drinking_water : ℕ) (bathing_water : ℕ) (days : ℕ) : ℕ :=
  (initial_horses + added_horses) * (drinking_water + bathing_water) * days

/-- Proves that under given conditions, the total water consumption for 28 days is 1568 liters -/
theorem water_consumption_proof :
  total_water_consumption 3 5 5 2 28 = 1568 := by
  sorry

#eval total_water_consumption 3 5 5 2 28

end NUMINAMATH_CALUDE_water_consumption_proof_l3454_345406


namespace NUMINAMATH_CALUDE_one_third_percent_of_180_l3454_345483

theorem one_third_percent_of_180 : (1 / 3 : ℚ) / 100 * 180 = 0.6 := by sorry

end NUMINAMATH_CALUDE_one_third_percent_of_180_l3454_345483


namespace NUMINAMATH_CALUDE_laura_walk_distance_l3454_345466

/-- Calculates the total distance walked in miles given the number of blocks walked east and north, and the length of each block in miles. -/
def total_distance (blocks_east blocks_north : ℕ) (block_length : ℚ) : ℚ :=
  (blocks_east + blocks_north : ℚ) * block_length

/-- Proves that walking 8 blocks east and 14 blocks north, with each block being 1/4 mile, results in a total distance of 5.5 miles. -/
theorem laura_walk_distance : total_distance 8 14 (1/4) = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_laura_walk_distance_l3454_345466


namespace NUMINAMATH_CALUDE_salary_ratio_proof_l3454_345433

/-- Proves that the ratio of Shyam's monthly salary to Abhinav's monthly salary is 2:1 -/
theorem salary_ratio_proof (ram_salary shyam_salary abhinav_annual_salary : ℕ) : 
  ram_salary = 25600 →
  abhinav_annual_salary = 192000 →
  10 * ram_salary = 8 * shyam_salary →
  ∃ (k : ℕ), shyam_salary = k * (abhinav_annual_salary / 12) →
  shyam_salary / (abhinav_annual_salary / 12) = 2 := by
  sorry

end NUMINAMATH_CALUDE_salary_ratio_proof_l3454_345433


namespace NUMINAMATH_CALUDE_no_function_satisfies_conditions_l3454_345436

theorem no_function_satisfies_conditions : ¬∃ (f : ℝ → ℝ), 
  (∃ (M : ℝ), M > 0 ∧ ∀ (x : ℝ), -M ≤ f x ∧ f x ≤ M) ∧ 
  (f 1 = 1) ∧
  (∀ (x : ℝ), x ≠ 0 → f (x + 1 / x^2) = f x + (f (1 / x))^2) :=
by sorry

end NUMINAMATH_CALUDE_no_function_satisfies_conditions_l3454_345436


namespace NUMINAMATH_CALUDE_smallest_triangle_side_l3454_345468

-- Define the triangle sides
def a : ℕ := 7
def b : ℕ := 11

-- Define the triangle inequality
def is_triangle (x y z : ℕ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

-- Define the property we want to prove
def smallest_side (s : ℕ) : Prop :=
  is_triangle a b s ∧ ∀ t : ℕ, t < s → ¬(is_triangle a b t)

-- The theorem to prove
theorem smallest_triangle_side : smallest_side 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_triangle_side_l3454_345468


namespace NUMINAMATH_CALUDE_triangle_condition_right_triangle_condition_l3454_345432

/-- Given vectors in 2D space -/
def OA : ℝ × ℝ := (3, -4)
def OB : ℝ × ℝ := (6, -3)
def OC (m : ℝ) : ℝ × ℝ := (5 - m, -(4 + m))

/-- Vector subtraction -/
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

/-- Dot product of 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Condition for three points to form a triangle -/
def forms_triangle (m : ℝ) : Prop :=
  let AB := vec_sub OB OA
  let AC := vec_sub (OC m) OA
  let BC := vec_sub (OC m) OB
  AB.1 / AB.2 ≠ AC.1 / AC.2 ∧ AB.1 / AB.2 ≠ BC.1 / BC.2 ∧ AC.1 / AC.2 ≠ BC.1 / BC.2

/-- Theorem: Condition for A, B, and C to form a triangle -/
theorem triangle_condition : ∀ m : ℝ, forms_triangle m ↔ m ≠ -1 := by sorry

/-- Theorem: Condition for ABC to be a right triangle with angle A as the right angle -/
theorem right_triangle_condition : 
  ∀ m : ℝ, dot_product (vec_sub OB OA) (vec_sub (OC m) OA) = 0 ↔ m = 3/2 := by sorry

end NUMINAMATH_CALUDE_triangle_condition_right_triangle_condition_l3454_345432


namespace NUMINAMATH_CALUDE_cricket_game_overs_l3454_345415

/-- Represents the number of overs played initially in a cricket game -/
def initial_overs : ℝ :=
  -- Definition will be provided in the proof
  sorry

/-- The total target score in runs -/
def total_target : ℝ := 282

/-- The initial run rate in runs per over -/
def initial_run_rate : ℝ := 4.6

/-- The required run rate for the remaining overs in runs per over -/
def required_run_rate : ℝ := 5.9

/-- The number of remaining overs -/
def remaining_overs : ℝ := 40

theorem cricket_game_overs :
  initial_overs = 10 ∧
  total_target = initial_run_rate * initial_overs + required_run_rate * remaining_overs :=
by sorry

end NUMINAMATH_CALUDE_cricket_game_overs_l3454_345415


namespace NUMINAMATH_CALUDE_curve_self_intersection_l3454_345469

/-- The x-coordinate of a point on the curve as a function of t -/
def x (t : ℝ) : ℝ := 2 * t^2 - 4

/-- The y-coordinate of a point on the curve as a function of t -/
def y (t : ℝ) : ℝ := t^3 - 6 * t^2 + 11 * t - 6

/-- The theorem stating that the curve intersects itself at (18, -44√11 - 6) -/
theorem curve_self_intersection :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
    x t₁ = x t₂ ∧ 
    y t₁ = y t₂ ∧ 
    x t₁ = 18 ∧ 
    y t₁ = -44 * Real.sqrt 11 - 6 :=
sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l3454_345469


namespace NUMINAMATH_CALUDE_line_equation_l3454_345498

/-- Given a line passing through (a, 0) and cutting a triangular region
    with area T in the first quadrant, prove its equation. -/
theorem line_equation (a T : ℝ) (h₁ : a ≠ 0) (h₂ : T > 0) :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (x = a ∧ y = 0) ∨ (x = 0 ∧ y > 0) →
    (y = m * x + b ↔ a^2 * y + 2 * T * x - 2 * a * T = 0) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l3454_345498


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3454_345430

theorem quadratic_inequality_solution (x : ℝ) :
  (8 * x^2 + 6 * x > 10) ↔ (x < -1 ∨ x > 5/4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3454_345430


namespace NUMINAMATH_CALUDE_problem_1_l3454_345402

theorem problem_1 : (-2)^3 + (1/9)⁻¹ - (3.14 - Real.pi)^0 = 0 := by sorry

end NUMINAMATH_CALUDE_problem_1_l3454_345402


namespace NUMINAMATH_CALUDE_election_votes_l3454_345400

theorem election_votes (candidate_percentage : ℝ) (vote_difference : ℕ) (total_votes : ℕ) : 
  candidate_percentage = 35 / 100 → 
  vote_difference = 2100 →
  total_votes = (vote_difference : ℝ) / (1 - 2 * candidate_percentage) →
  total_votes = 7000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l3454_345400


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3454_345451

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 - 6*x + 5 = 2*x - 8) → 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 6*x₁ + 5 = 2*x₁ - 8 ∧ x₂^2 - 6*x₂ + 5 = 2*x₂ - 8 ∧ x₁ + x₂ = 8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3454_345451


namespace NUMINAMATH_CALUDE_number_calculation_l3454_345496

theorem number_calculation : ∃ x : ℚ, x = 2/15 + 1/5 + 1/2 :=
by sorry

end NUMINAMATH_CALUDE_number_calculation_l3454_345496


namespace NUMINAMATH_CALUDE_number_problem_l3454_345463

theorem number_problem (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 25) : 
  0.40 * N = 300 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3454_345463


namespace NUMINAMATH_CALUDE_moores_law_transistor_growth_l3454_345417

/-- Moore's Law Transistor Growth --/
theorem moores_law_transistor_growth
  (initial_year : Nat)
  (final_year : Nat)
  (initial_transistors : Nat)
  (doubling_period : Nat)
  (h1 : initial_year = 1985)
  (h2 : final_year = 2005)
  (h3 : initial_transistors = 500000)
  (h4 : doubling_period = 2) :
  initial_transistors * 2^((final_year - initial_year) / doubling_period) = 512000000 :=
by sorry

end NUMINAMATH_CALUDE_moores_law_transistor_growth_l3454_345417


namespace NUMINAMATH_CALUDE_golden_ratio_pentagon_l3454_345450

theorem golden_ratio_pentagon (a : ℝ) : 
  a = 2 * Real.cos (72 * π / 180) → 
  (a * Real.cos (18 * π / 180)) / Real.sqrt (2 - a) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_golden_ratio_pentagon_l3454_345450


namespace NUMINAMATH_CALUDE_museum_entrance_cost_l3454_345419

theorem museum_entrance_cost (group_size : ℕ) (ticket_price : ℚ) (tax_rate : ℚ) : 
  group_size = 25 →
  ticket_price = 35.91 →
  tax_rate = 0.05 →
  (group_size : ℚ) * ticket_price * (1 + tax_rate) = 942.64 := by
sorry

end NUMINAMATH_CALUDE_museum_entrance_cost_l3454_345419


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l3454_345474

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = 2^x}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - 2*x)}

-- Define the intersection of M and N
def M_intersect_N : Set ℝ := M ∩ N

-- Define the open-closed interval (0, 1/2]
def open_closed_interval : Set ℝ := {x | 0 < x ∧ x ≤ 1/2}

-- Theorem statement
theorem intersection_equals_interval : M_intersect_N = open_closed_interval := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l3454_345474


namespace NUMINAMATH_CALUDE_friday_sales_l3454_345409

/-- Kim's cupcake sales pattern --/
def cupcake_sales (tuesday_before_discount : ℕ) : ℕ :=
  let tuesday := tuesday_before_discount + (tuesday_before_discount * 5 / 100)
  let monday := tuesday + (tuesday * 50 / 100)
  let wednesday := tuesday * 3 / 2
  let thursday := wednesday - (wednesday * 20 / 100)
  let friday := thursday * 13 / 10
  friday

/-- Theorem: Kim sold 1310 boxes on Friday --/
theorem friday_sales : cupcake_sales 800 = 1310 := by
  sorry

end NUMINAMATH_CALUDE_friday_sales_l3454_345409


namespace NUMINAMATH_CALUDE_function_value_at_negative_pi_fourth_l3454_345420

theorem function_value_at_negative_pi_fourth 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, f x = a * Real.tan x - b * Real.sin x + 1) 
  (h2 : f (π/4) = 7) : 
  f (-π/4) = -5 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_pi_fourth_l3454_345420


namespace NUMINAMATH_CALUDE_twenty_bulb_series_string_possibilities_l3454_345458

/-- Represents a string of decorative lights -/
structure LightString where
  num_bulbs : ℕ
  is_series : Bool

/-- Calculates the number of ways a light string can be non-functioning -/
def non_functioning_possibilities (ls : LightString) : ℕ :=
  if ls.is_series then 2^ls.num_bulbs - 1 else 0

/-- Theorem stating the number of non-functioning possibilities for a specific light string -/
theorem twenty_bulb_series_string_possibilities :
  ∃ (ls : LightString), ls.num_bulbs = 20 ∧ ls.is_series = true ∧ non_functioning_possibilities ls = 2^20 - 1 :=
sorry

end NUMINAMATH_CALUDE_twenty_bulb_series_string_possibilities_l3454_345458


namespace NUMINAMATH_CALUDE_product_of_digits_8056_base_8_l3454_345490

def base_8_representation (n : ℕ) : List ℕ :=
  sorry

theorem product_of_digits_8056_base_8 :
  (base_8_representation 8056).foldl (·*·) 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_digits_8056_base_8_l3454_345490


namespace NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l3454_345401

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A pentagon is a polygon with 5 sides -/
def pentagon_sides : ℕ := 5

/-- Theorem: The sum of the interior angles of a pentagon is 540° -/
theorem sum_interior_angles_pentagon :
  sum_interior_angles pentagon_sides = 540 := by sorry

end NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l3454_345401
