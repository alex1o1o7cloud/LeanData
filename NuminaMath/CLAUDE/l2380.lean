import Mathlib

namespace parallel_lines_m_values_l2380_238036

/-- Two lines are parallel if and only if their slopes are equal and they don't coincide -/
def parallel (m : ℝ) : Prop :=
  (m / 3 = 1 / (m - 2)) ∧ (-5 ≠ -1 / (m - 2))

/-- The theorem states that if the lines are parallel, then m is either 3 or -1 -/
theorem parallel_lines_m_values (m : ℝ) :
  parallel m → m = 3 ∨ m = -1 := by
  sorry

end parallel_lines_m_values_l2380_238036


namespace correlatedRelationships_l2380_238032

-- Define the type for relationships
inductive Relationship
  | GreatTeachersAndStudents
  | SphereVolumeAndRadius
  | AppleYieldAndClimate
  | TreeDiameterAndHeight
  | StudentAndID
  | CrowCawAndOmen

-- Define a function to check if a relationship has correlation
def hasCorrelation (r : Relationship) : Prop :=
  match r with
  | Relationship.GreatTeachersAndStudents => True
  | Relationship.SphereVolumeAndRadius => False
  | Relationship.AppleYieldAndClimate => True
  | Relationship.TreeDiameterAndHeight => True
  | Relationship.StudentAndID => False
  | Relationship.CrowCawAndOmen => False

-- Theorem stating which relationships have correlation
theorem correlatedRelationships :
  (hasCorrelation Relationship.GreatTeachersAndStudents) ∧
  (hasCorrelation Relationship.AppleYieldAndClimate) ∧
  (hasCorrelation Relationship.TreeDiameterAndHeight) ∧
  (¬hasCorrelation Relationship.SphereVolumeAndRadius) ∧
  (¬hasCorrelation Relationship.StudentAndID) ∧
  (¬hasCorrelation Relationship.CrowCawAndOmen) :=
by sorry


end correlatedRelationships_l2380_238032


namespace intersection_distance_implies_omega_l2380_238037

/-- Given a function f(x) = 2sin(ωx + φ) where ω > 0, if the curve y = f(x) intersects
    the line y = √3 and the distance between two adjacent intersection points is π/6,
    then ω = 2 or ω = 10. -/
theorem intersection_distance_implies_omega (ω φ : ℝ) (h1 : ω > 0) :
  (∃ x1 x2 : ℝ, x2 - x1 = π / 6 ∧
    2 * Real.sin (ω * x1 + φ) = Real.sqrt 3 ∧
    2 * Real.sin (ω * x2 + φ) = Real.sqrt 3) →
  ω = 2 ∨ ω = 10 := by
  sorry


end intersection_distance_implies_omega_l2380_238037


namespace smallest_base_for_inequality_l2380_238012

theorem smallest_base_for_inequality (k : ℕ) (h : k = 6) : 
  ∀ b : ℕ, b > 0 → b ≤ 4 ↔ b^16 ≤ 64^k :=
by sorry

end smallest_base_for_inequality_l2380_238012


namespace simplify_exponential_fraction_l2380_238055

theorem simplify_exponential_fraction (n : ℕ) :
  (3^(n+5) - 3 * 3^n) / (3 * 3^(n+4)) = 240 / 81 := by
  sorry

end simplify_exponential_fraction_l2380_238055


namespace line_bisects_circle_l2380_238023

/-- The equation of the circle -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 1 = 0

/-- The equation of the bisecting line -/
def line_eq (x y : ℝ) : Prop := x - y + 1 = 0

/-- A point is the center of the circle if it satisfies the circle's equation 
    and is equidistant from all points on the circle -/
def is_center (cx cy : ℝ) : Prop :=
  circle_eq cx cy ∧ 
  ∀ x y : ℝ, circle_eq x y → (x - cx)^2 + (y - cy)^2 = 4

/-- A line bisects a circle if and only if it passes through the circle's center -/
axiom bisects_iff_passes_through_center (a b c : ℝ) :
  (∀ x y : ℝ, circle_eq x y → a*x + b*y + c = 0) ↔
  (∃ cx cy : ℝ, is_center cx cy ∧ a*cx + b*cy + c = 0)

/-- The main theorem: the line x - y + 1 = 0 bisects the circle -/
theorem line_bisects_circle :
  ∀ x y : ℝ, circle_eq x y → line_eq x y :=
by sorry

end line_bisects_circle_l2380_238023


namespace exists_balanced_partition_l2380_238078

/-- An undirected graph represented by its vertex set and edge relation -/
structure Graph (V : Type) where
  edge : V → V → Prop
  symm : ∀ u v, edge u v → edge v u

/-- The neighborhood of a vertex v in a set S -/
def neighborhood {V : Type} (G : Graph V) (S : Set V) (v : V) : Set V :=
  {u ∈ S | G.edge v u}

/-- A partition of a set into two disjoint subsets -/
structure Partition (V : Type) where
  A : Set V
  B : Set V
  disjoint : A ∩ B = ∅
  complete : A ∪ B = Set.univ

/-- The main theorem statement -/
theorem exists_balanced_partition {V : Type} (G : Graph V) :
  ∃ (P : Partition V), 
    (∀ v ∈ P.A, (neighborhood G P.B v).ncard ≥ (neighborhood G P.A v).ncard) ∧
    (∀ v ∈ P.B, (neighborhood G P.A v).ncard ≥ (neighborhood G P.B v).ncard) := by
  sorry

end exists_balanced_partition_l2380_238078


namespace ice_cube_volume_l2380_238092

theorem ice_cube_volume (original_volume : ℝ) : 
  (original_volume > 0) →
  (original_volume * (1/4) * (1/4) = 0.4) →
  original_volume = 6.4 := by
sorry

end ice_cube_volume_l2380_238092


namespace paint_ornaments_l2380_238018

/-- Represents the problem of painting star-shaped ornaments on tiles --/
theorem paint_ornaments (num_tiles : ℕ) (paint_coverage : ℝ) (tile_side : ℝ) 
  (pentagon_area : ℝ) (triangle_base triangle_height : ℝ) : 
  num_tiles = 20 → 
  paint_coverage = 750 → 
  tile_side = 12 → 
  pentagon_area = 15 → 
  triangle_base = 4 → 
  triangle_height = 6 → 
  (num_tiles * (tile_side^2 - 4*pentagon_area - 2*triangle_base*triangle_height) ≤ paint_coverage) :=
by sorry

end paint_ornaments_l2380_238018


namespace complex_reciprocal_sum_l2380_238051

theorem complex_reciprocal_sum (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by
  sorry

end complex_reciprocal_sum_l2380_238051


namespace sequence_minimum_l2380_238033

/-- Given a sequence {a_n} satisfying the conditions:
    a_1 = p, a_2 = p + 1, and a_{n+2} - 2a_{n+1} + a_n = n - 20,
    where p is a real number and n is a positive integer,
    prove that a_n is minimized when n = 40. -/
theorem sequence_minimum (p : ℝ) : 
  ∃ (a : ℕ → ℝ), 
    (a 1 = p) ∧ 
    (a 2 = p + 1) ∧ 
    (∀ n : ℕ, n ≥ 1 → a (n + 2) - 2 * a (n + 1) + a n = n - 20) ∧
    (∀ n : ℕ, n ≥ 1 → a 40 ≤ a n) :=
by sorry

end sequence_minimum_l2380_238033


namespace system_solutions_l2380_238048

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  x^2 - y = z^2 ∧ y^2 - z = x^2 ∧ z^2 - x = y^2

/-- The set of solutions -/
def solutions : Set (ℝ × ℝ × ℝ) :=
  {(0, 0, 0), (1, 0, -1), (0, -1, 1), (-1, 1, 0)}

/-- Theorem stating that the solutions are correct and complete -/
theorem system_solutions :
  ∀ x y z : ℝ, system x y z ↔ (x, y, z) ∈ solutions := by
  sorry

end system_solutions_l2380_238048


namespace max_cake_pieces_l2380_238067

/-- The size of the large cake in inches -/
def large_cake_size : ℕ := 15

/-- The size of the small cake piece in inches -/
def small_piece_size : ℕ := 5

/-- The maximum number of small pieces that can be cut from the large cake -/
def max_pieces : ℕ := (large_cake_size * large_cake_size) / (small_piece_size * small_piece_size)

theorem max_cake_pieces : max_pieces = 9 := by
  sorry

end max_cake_pieces_l2380_238067


namespace parallelogram_angle_ratio_l2380_238094

-- Define the parallelogram ABCD and point O
variable (A B C D O : Point)

-- Define the property of being a parallelogram
def is_parallelogram (A B C D : Point) : Prop := sorry

-- Define the property of O being the intersection of diagonals
def is_diagonal_intersection (A B C D O : Point) : Prop := sorry

-- Define angle measure
def angle_measure (P Q R : Point) : ℝ := sorry

-- State the theorem
theorem parallelogram_angle_ratio 
  (h_para : is_parallelogram A B C D)
  (h_diag : is_diagonal_intersection A B C D O)
  (h_cab : angle_measure C A B = 3 * angle_measure D B A)
  (h_dbc : angle_measure D B C = 3 * angle_measure D B A)
  (h_acb : ∃ r : ℝ, angle_measure A C B = r * angle_measure A O B) :
  ∃ r : ℝ, angle_measure A C B = r * angle_measure A O B ∧ r = 2 := by sorry

end parallelogram_angle_ratio_l2380_238094


namespace least_five_digit_divisible_by_12_15_18_l2380_238053

theorem least_five_digit_divisible_by_12_15_18 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- 5-digit number
  (n % 12 = 0 ∧ n % 15 = 0 ∧ n % 18 = 0) ∧  -- divisible by 12, 15, and 18
  (∀ m : ℕ, m ≥ 10000 ∧ m < n ∧ m % 12 = 0 ∧ m % 15 = 0 ∧ m % 18 = 0 → false) ∧  -- least such number
  n = 10080 :=  -- the answer
by sorry

end least_five_digit_divisible_by_12_15_18_l2380_238053


namespace dance_off_ratio_l2380_238043

-- Define the dancing times and break time
def john_first_dance : ℕ := 3
def john_break : ℕ := 1
def john_second_dance : ℕ := 5
def combined_dance_time : ℕ := 20

-- Define John's total dancing and resting time
def john_total_time : ℕ := john_first_dance + john_break + john_second_dance

-- Define John's dancing time
def john_dance_time : ℕ := john_first_dance + john_second_dance

-- Define James' dancing time
def james_dance_time : ℕ := combined_dance_time - john_dance_time

-- Define James' additional dancing time
def james_additional_time : ℕ := james_dance_time - john_dance_time

-- Theorem to prove
theorem dance_off_ratio : 
  (james_additional_time : ℚ) / john_total_time = 4 / 9 := by sorry

end dance_off_ratio_l2380_238043


namespace square_of_2m2_plus_n2_l2380_238073

theorem square_of_2m2_plus_n2 (m n : ℤ) :
  ∃ k l : ℤ, (2 * m^2 + n^2)^2 = 2 * k^2 + l^2 := by
sorry

end square_of_2m2_plus_n2_l2380_238073


namespace inequality_proof_l2380_238050

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 + 1/a) * (1 + 1/b) ≥ 8 / (1 + a*b) := by
  sorry

end inequality_proof_l2380_238050


namespace unique_prime_solution_l2380_238038

theorem unique_prime_solution :
  ∃! (p q r : ℕ),
    Prime p ∧ Prime q ∧ Prime r ∧
    p < q ∧ q < r ∧
    25 * p * q + r = 2004 ∧
    ∃ m : ℕ, p * q * r + 1 = m * m ∧
    p = 7 ∧ q = 11 ∧ r = 79 := by
  sorry

end unique_prime_solution_l2380_238038


namespace root_sum_reciprocal_equals_62_l2380_238040

theorem root_sum_reciprocal_equals_62 :
  ∃ (a b : ℝ),
    (∀ x : ℝ, x^3 - 9*x^2 + 9*x = 1 ↔ x = a ∨ x = b ∨ x = 1) ∧
    a > b ∧
    a > 1 ∧
    b < 1 ∧
    a/b + b/a = 62 :=
by
  sorry

end root_sum_reciprocal_equals_62_l2380_238040


namespace soda_cost_calculation_l2380_238076

def restaurant_bill (num_adults num_children : ℕ) (adult_meal_cost child_meal_cost soda_cost : ℚ) : Prop :=
  let total_people := num_adults + num_children
  let meal_cost := (num_adults * adult_meal_cost) + (num_children * child_meal_cost)
  let total_bill := meal_cost + (total_people * soda_cost)
  (num_adults = 6) ∧ 
  (num_children = 2) ∧ 
  (adult_meal_cost = 6) ∧ 
  (child_meal_cost = 4) ∧ 
  (total_bill = 60)

theorem soda_cost_calculation :
  ∃ (soda_cost : ℚ), restaurant_bill 6 2 6 4 soda_cost ∧ soda_cost = 2 := by
  sorry

end soda_cost_calculation_l2380_238076


namespace acute_triangle_cosine_inequality_l2380_238004

theorem acute_triangle_cosine_inequality (A B C : ℝ) 
  (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) :
  (Real.cos A / Real.cos (B - C)) + 
  (Real.cos B / Real.cos (C - A)) + 
  (Real.cos C / Real.cos (A - B)) ≥ 3/2 := by sorry

end acute_triangle_cosine_inequality_l2380_238004


namespace partial_fraction_decomposition_l2380_238093

theorem partial_fraction_decomposition (A B C : ℝ) :
  (∀ x : ℝ, x ≠ -2 ∧ x ≠ 1 → 
    1 / (x^3 - 2*x^2 - 13*x + 10) = A / (x + 2) + B / (x - 1) + C / ((x - 1)^2)) →
  A = 1/9 := by
sorry

end partial_fraction_decomposition_l2380_238093


namespace trousers_final_cost_l2380_238017

def calculate_final_cost (original_price : ℝ) (in_store_discount : ℝ) (additional_promotion : ℝ) (sales_tax : ℝ) (handling_fee : ℝ) : ℝ :=
  let price_after_in_store_discount := original_price * (1 - in_store_discount)
  let price_after_additional_promotion := price_after_in_store_discount * (1 - additional_promotion)
  let price_with_tax := price_after_additional_promotion * (1 + sales_tax)
  price_with_tax + handling_fee

theorem trousers_final_cost :
  calculate_final_cost 100 0.20 0.10 0.05 5 = 80.60 := by
  sorry

end trousers_final_cost_l2380_238017


namespace solution_set_l2380_238041

-- Define the condition
def always_positive (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*a*x + a > 0

-- Define the inequality
def inequality (a t : ℝ) : Prop :=
  a^(2*t + 1) < a^(t^2 + 2*t - 3)

-- State the theorem
theorem solution_set (a : ℝ) (h : always_positive a) :
  {t : ℝ | inequality a t} = {t : ℝ | -2 < t ∧ t < 2} :=
sorry

end solution_set_l2380_238041


namespace seafood_noodles_plates_l2380_238034

/-- Given a chef's banquet with a total of 55 plates, 25 plates of lobster rolls,
    and 14 plates of spicy hot noodles, prove that the number of seafood noodle plates is 16. -/
theorem seafood_noodles_plates (total : ℕ) (lobster : ℕ) (spicy : ℕ) (seafood : ℕ)
  (h1 : total = 55)
  (h2 : lobster = 25)
  (h3 : spicy = 14)
  (h4 : total = lobster + spicy + seafood) :
  seafood = 16 := by
  sorry

end seafood_noodles_plates_l2380_238034


namespace infinitely_many_minimal_points_l2380_238052

/-- Distance function from origin to point (x, y) -/
def distance (x y : ℝ) : ℝ := |x| + |y|

/-- The set of points (x, y) on the line y = x + 1 that minimize the distance from the origin -/
def minimal_distance_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 + 1 ∧ ∀ q : ℝ × ℝ, q.2 = q.1 + 1 → distance p.1 p.2 ≤ distance q.1 q.2}

/-- Theorem stating that there are infinitely many points that minimize the distance -/
theorem infinitely_many_minimal_points : Set.Infinite minimal_distance_points := by
  sorry

end infinitely_many_minimal_points_l2380_238052


namespace third_roll_six_prob_l2380_238022

-- Define the probabilities for each die
def fair_die_prob : ℚ := 1/6
def biased_die_six_prob : ℚ := 2/3
def biased_die_other_prob : ℚ := 1/15

-- Define the probability of choosing each die
def die_choice_prob : ℚ := 1/2

-- Define the event of rolling two sixes
def two_sixes_prob (die_prob : ℚ) : ℚ := die_prob * die_prob

-- Define the total probability of rolling two sixes
def total_two_sixes_prob : ℚ := 
  die_choice_prob * two_sixes_prob fair_die_prob + 
  die_choice_prob * two_sixes_prob biased_die_six_prob

-- Define the conditional probability of choosing each die given two sixes
def fair_die_given_two_sixes : ℚ := 
  (two_sixes_prob fair_die_prob * die_choice_prob) / total_two_sixes_prob

def biased_die_given_two_sixes : ℚ := 
  (two_sixes_prob biased_die_six_prob * die_choice_prob) / total_two_sixes_prob

-- Theorem statement
theorem third_roll_six_prob : 
  fair_die_prob * fair_die_given_two_sixes + 
  biased_die_six_prob * biased_die_given_two_sixes = 65/102 := by
  sorry

end third_roll_six_prob_l2380_238022


namespace minimum_value_and_inequality_l2380_238086

def f (x : ℝ) : ℝ := |x + 3| + |x - 1|

theorem minimum_value_and_inequality (p q r : ℝ) :
  (∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, f x = m) ∧ m = 4) ∧
  (p^2 + 2*q^2 + r^2 = 4 → q*(p + r) ≤ 2) := by
  sorry

end minimum_value_and_inequality_l2380_238086


namespace coin_flip_probability_l2380_238056

def num_coins : ℕ := 6

def all_outcomes : ℕ := 2^num_coins

def favorable_outcomes : ℕ := 2 + 2 * (num_coins.choose 1)

theorem coin_flip_probability : 
  (favorable_outcomes : ℚ) / all_outcomes = 7 / 32 :=
sorry

end coin_flip_probability_l2380_238056


namespace sqrt_sum_equals_6_sqrt_3_l2380_238070

theorem sqrt_sum_equals_6_sqrt_3 :
  Real.sqrt (31 - 12 * Real.sqrt 3) + Real.sqrt (31 + 12 * Real.sqrt 3) = 6 * Real.sqrt 3 := by
  sorry

end sqrt_sum_equals_6_sqrt_3_l2380_238070


namespace angle_C_in_triangle_l2380_238064

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem angle_C_in_triangle (t : Triangle) :
  t.a = Real.sqrt 2 ∧ 
  t.b = Real.sqrt 3 ∧ 
  t.A = 45 * (π / 180) →
  t.C = 75 * (π / 180) ∨ t.C = 15 * (π / 180) := by
  sorry


end angle_C_in_triangle_l2380_238064


namespace furniture_purchase_price_l2380_238025

theorem furniture_purchase_price :
  let marked_price : ℝ := 132
  let discount_rate : ℝ := 0.1
  let profit_rate : ℝ := 0.1
  let purchase_price : ℝ := 108
  marked_price * (1 - discount_rate) - purchase_price = profit_rate * purchase_price :=
by
  sorry

end furniture_purchase_price_l2380_238025


namespace cube_volume_from_surface_area_l2380_238028

theorem cube_volume_from_surface_area : 
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 864 → s^3 = 1728 := by
  sorry

end cube_volume_from_surface_area_l2380_238028


namespace combinatorial_equality_l2380_238045

theorem combinatorial_equality (n : ℕ) : 
  (Nat.choose n 3 = Nat.choose n 5) → n = 8 := by
sorry

end combinatorial_equality_l2380_238045


namespace pineapples_sold_l2380_238075

theorem pineapples_sold (initial : ℕ) (rotten : ℕ) (fresh : ℕ) : 
  initial = 86 → rotten = 9 → fresh = 29 → initial - (fresh + rotten) = 48 := by
sorry

end pineapples_sold_l2380_238075


namespace problem_statement_l2380_238082

theorem problem_statement : (0.125 : ℝ)^2012 * (2^2012)^3 = 1 := by sorry

end problem_statement_l2380_238082


namespace longest_tape_measure_l2380_238065

theorem longest_tape_measure (a b c : ℕ) 
  (ha : a = 315) 
  (hb : b = 458) 
  (hc : c = 1112) : 
  Nat.gcd a (Nat.gcd b c) = 1 := by
  sorry

end longest_tape_measure_l2380_238065


namespace reciprocal_sum_of_quadratic_roots_l2380_238003

theorem reciprocal_sum_of_quadratic_roots : 
  ∀ x₁ x₂ : ℝ, 
  (x₁^2 + x₁ = 5*x₁ + 6) → 
  (x₂^2 + x₂ = 5*x₂ + 6) → 
  x₁ ≠ x₂ →
  (1/x₁ + 1/x₂ = -2/3) :=
by sorry

end reciprocal_sum_of_quadratic_roots_l2380_238003


namespace intersection_complement_equal_l2380_238095

def U : Finset Nat := {1,2,3,4,5}
def M : Finset Nat := {1,4}
def N : Finset Nat := {1,3,5}

theorem intersection_complement_equal : N ∩ (U \ M) = {3,5} := by
  sorry

end intersection_complement_equal_l2380_238095


namespace chipmunk_acorns_l2380_238054

/-- Represents the number of acorns hidden in each hole by an animal -/
structure AcornsPerHole where
  chipmunk : ℕ
  squirrel : ℕ

/-- Represents the number of holes dug by each animal -/
structure HolesDug where
  chipmunk : ℕ
  squirrel : ℕ

/-- The main theorem about the number of acorns hidden by the chipmunk -/
theorem chipmunk_acorns (aph : AcornsPerHole) (h : HolesDug) : 
  aph.chipmunk = 3 → 
  aph.squirrel = 4 → 
  h.chipmunk = h.squirrel + 4 → 
  aph.chipmunk * h.chipmunk = aph.squirrel * h.squirrel → 
  aph.chipmunk * h.chipmunk = 48 := by
  sorry

#check chipmunk_acorns

end chipmunk_acorns_l2380_238054


namespace deposit_withdrawal_ratio_l2380_238068

/-- Prove that the ratio of the deposited amount to the withdrawn amount is 2:1 --/
theorem deposit_withdrawal_ratio (initial_savings withdrawal final_balance : ℚ) 
  (h1 : initial_savings = 230)
  (h2 : withdrawal = 60)
  (h3 : final_balance = 290) : 
  (final_balance - (initial_savings - withdrawal)) / withdrawal = 2 := by
  sorry

end deposit_withdrawal_ratio_l2380_238068


namespace mike_picked_seven_apples_l2380_238029

/-- The number of apples picked by Mike, given the total number of apples and the number picked by Nancy and Keith. -/
def mike_apples (total : ℕ) (nancy : ℕ) (keith : ℕ) : ℕ :=
  total - (nancy + keith)

/-- Theorem stating that Mike picked 7 apples given the problem conditions. -/
theorem mike_picked_seven_apples :
  mike_apples 16 3 6 = 7 := by
  sorry

end mike_picked_seven_apples_l2380_238029


namespace average_position_l2380_238097

def fractions : List ℚ := [1/2, 1/3, 1/4, 1/5, 1/6, 1/7]

theorem average_position :
  let avg := (List.sum fractions) / fractions.length
  avg = 223 / 840 ∧ 1/4 < avg ∧ avg < 1/3 := by
  sorry

end average_position_l2380_238097


namespace pizzeria_sales_l2380_238020

theorem pizzeria_sales (small_price large_price total_revenue small_count : ℕ) 
  (h1 : small_price = 2)
  (h2 : large_price = 8)
  (h3 : total_revenue = 40)
  (h4 : small_count = 8) :
  (total_revenue - small_price * small_count) / large_price = 3 := by
  sorry

end pizzeria_sales_l2380_238020


namespace greatest_multiple_of_5_and_6_less_than_800_l2380_238026

theorem greatest_multiple_of_5_and_6_less_than_800 : 
  ∀ n : ℕ, n < 800 ∧ 5 ∣ n ∧ 6 ∣ n → n ≤ 780 :=
by sorry

end greatest_multiple_of_5_and_6_less_than_800_l2380_238026


namespace min_value_sum_reciprocals_l2380_238099

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_9 : a + b + c + d + e + f = 9) :
  (1/a + 9/b + 25/c + 49/d + 81/e + 121/f) ≥ 286^2/9 := by
  sorry

end min_value_sum_reciprocals_l2380_238099


namespace power_product_rule_l2380_238019

theorem power_product_rule (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 := by sorry

end power_product_rule_l2380_238019


namespace electric_distance_average_costs_annual_mileage_threshold_l2380_238010

-- Define variables and constants
variable (x : ℝ) -- Average charging cost per km for electric vehicle
def fuel_cost_diff : ℝ := 0.6 -- Difference in cost per km between fuel and electric
def charging_cost : ℝ := 300 -- Charging cost for electric vehicle
def refueling_cost : ℝ := 300 -- Refueling cost for fuel vehicle
def distance_ratio : ℝ := 4 -- Ratio of electric vehicle distance to fuel vehicle distance
def other_cost_fuel : ℝ := 4800 -- Other annual costs for fuel vehicle
def other_cost_electric : ℝ := 7800 -- Other annual costs for electric vehicle

-- Theorem statements
theorem electric_distance (hx : x > 0) : 
  (charging_cost : ℝ) / x = 300 / x :=
sorry

theorem average_costs (hx : x > 0) : 
  x = 0.2 ∧ x + fuel_cost_diff = 0.8 :=
sorry

theorem annual_mileage_threshold (y : ℝ) :
  0.2 * y + other_cost_electric < 0.8 * y + other_cost_fuel ↔ y > 5000 :=
sorry

end electric_distance_average_costs_annual_mileage_threshold_l2380_238010


namespace max_product_constrained_l2380_238027

theorem max_product_constrained (x y : ℕ+) (h : 7 * x + 4 * y = 140) :
  x * y ≤ 168 :=
sorry

end max_product_constrained_l2380_238027


namespace train_journey_time_l2380_238000

/-- Proves that given a train moving at 6/7 of its usual speed and arriving 30 minutes late, 
    the usual time for the train to complete the journey is 3 hours. -/
theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) : 
  usual_speed > 0 → usual_time > 0 →
  (6 / 7 * usual_speed) * (usual_time + 1 / 2) = usual_speed * usual_time →
  usual_time = 3 := by
  sorry

end train_journey_time_l2380_238000


namespace marvin_bottle_caps_l2380_238098

theorem marvin_bottle_caps (initial : ℕ) (remaining : ℕ) (taken : ℕ) : 
  initial = 16 → remaining = 10 → taken = initial - remaining → taken = 6 := by
  sorry

end marvin_bottle_caps_l2380_238098


namespace system_solution_l2380_238001

theorem system_solution (x₁ x₂ x₃ x₄ : ℝ) : 
  x₁^2 + x₂^2 + x₃^2 + x₄^2 = 4 ∧
  x₁*x₃ + x₂*x₄ + x₃*x₂ + x₄*x₁ = 0 ∧
  x₁*x₂*x₃ + x₁*x₂*x₄ + x₁*x₃*x₄ + x₂*x₃*x₄ = -2 ∧
  x₁*x₂*x₃*x₄ = -1 →
  (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1 ∧ x₄ = -1) ∨
  (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = -1 ∧ x₄ = 1) ∨
  (x₁ = 1 ∧ x₂ = -1 ∧ x₃ = 1 ∧ x₄ = 1) ∨
  (x₁ = -1 ∧ x₂ = 1 ∧ x₃ = 1 ∧ x₄ = 1) :=
by sorry


end system_solution_l2380_238001


namespace direction_field_properties_l2380_238074

open Real

-- Define the differential equation
def y' (x y : ℝ) : ℝ := x^2 + y^2

-- Theorem statement
theorem direction_field_properties :
  -- 1. Slope at origin is 0
  y' 0 0 = 0 ∧
  -- 2. Slope at (1, 0) is 1
  y' 1 0 = 1 ∧
  -- 3. Slope is 1 for any point on the unit circle
  (∀ x y : ℝ, x^2 + y^2 = 1 → y' x y = 1) ∧
  -- 4. Slope increases as distance from origin increases
  (∀ x1 y1 x2 y2 : ℝ, x1^2 + y1^2 < x2^2 + y2^2 → y' x1 y1 < y' x2 y2) :=
by sorry

end direction_field_properties_l2380_238074


namespace arithmetic_sequence_quadratic_roots_l2380_238058

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

-- Define the quadratic equation
def quadratic_equation (a : ℕ → ℝ) (k : ℕ) (x : ℝ) :=
  a k * x^2 + 2 * a (k + 1) * x + a (k + 2) = 0

-- Main theorem
theorem arithmetic_sequence_quadratic_roots
  (a : ℕ → ℝ) (d : ℝ) (h_d : d ≠ 0)
  (h_a : ∀ n, a n ≠ 0)
  (h_arith : arithmetic_sequence a d) :
  (∀ k, ∃ x, quadratic_equation a k x ∧ x = -1) ∧
  (∃ f : ℕ → ℝ, ∀ n, f (n + 1) - f n = -1/2 ∧
    ∃ x, quadratic_equation a n x ∧ f n = 1 / (x + 1)) :=
sorry

end arithmetic_sequence_quadratic_roots_l2380_238058


namespace alice_walking_time_l2380_238084

/-- Given Bob's walking time and distance, and the relationship between Alice and Bob's walking times and distances, prove that Alice would take 21 minutes to walk 7 miles. -/
theorem alice_walking_time 
  (bob_distance : ℝ) 
  (bob_time : ℝ) 
  (alice_distance : ℝ) 
  (alice_bob_time_ratio : ℝ) 
  (alice_target_distance : ℝ) 
  (h1 : bob_distance = 6) 
  (h2 : bob_time = 36) 
  (h3 : alice_distance = 4) 
  (h4 : alice_bob_time_ratio = 1/3) 
  (h5 : alice_target_distance = 7) : 
  (alice_target_distance / (alice_distance / (alice_bob_time_ratio * bob_time))) = 21 :=
by sorry

end alice_walking_time_l2380_238084


namespace hexagon_angle_sum_l2380_238042

-- Define the angles
def angle_A : ℝ := 35
def angle_B : ℝ := 80
def angle_C : ℝ := 30

-- Define the hexagon
def is_hexagon (x y : ℝ) : Prop :=
  angle_A + angle_B + (360 - x) + 90 + 60 + y = 720

-- Theorem statement
theorem hexagon_angle_sum (x y : ℝ) (h : is_hexagon x y) : x + y = 95 := by
  sorry

end hexagon_angle_sum_l2380_238042


namespace open_box_volume_l2380_238008

theorem open_box_volume
  (sheet_length : ℝ)
  (sheet_width : ℝ)
  (cut_square_side : ℝ)
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 38)
  (h3 : cut_square_side = 8) :
  (sheet_length - 2 * cut_square_side) * (sheet_width - 2 * cut_square_side) * cut_square_side = 5632 :=
by sorry

end open_box_volume_l2380_238008


namespace parallel_vectors_x_value_l2380_238057

/-- Two vectors are parallel if their coordinates are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2*x, -3)
  are_parallel a b → x = -3/4 := by
  sorry

end parallel_vectors_x_value_l2380_238057


namespace circle_tangency_problem_l2380_238072

theorem circle_tangency_problem (r : ℕ) : 
  (0 < r ∧ r < 60 ∧ 120 % r = 0) → 
  (∃ (S : Finset ℕ), S = {x : ℕ | 0 < x ∧ x < 60 ∧ 120 % x = 0} ∧ Finset.card S = 14) :=
by sorry

end circle_tangency_problem_l2380_238072


namespace quadratic_factorization_l2380_238085

/-- Factorization of a quadratic expression -/
theorem quadratic_factorization (a : ℝ) : a^2 - 8*a + 16 = (a - 4)^2 := by
  sorry

end quadratic_factorization_l2380_238085


namespace solution_set_of_f_greater_than_one_l2380_238009

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - 2 * |x - 1|

-- State the theorem
theorem solution_set_of_f_greater_than_one :
  {x : ℝ | f x > 1} = Set.Ioo (2/3) 2 := by sorry

end solution_set_of_f_greater_than_one_l2380_238009


namespace quadratic_function_zeros_range_l2380_238062

theorem quadratic_function_zeros_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 
    (x₁ ∈ Set.Ioo (-2) 0 ∧ x₂ ∈ Set.Ioo 2 3) ∧
    (x₁^2 - 2*x₁ + a = 0 ∧ x₂^2 - 2*x₂ + a = 0)) →
  a ∈ Set.Ioo (-3) 0 :=
by sorry

end quadratic_function_zeros_range_l2380_238062


namespace carrie_vegetable_revenue_l2380_238071

/-- Represents the revenue calculation for Carrie's vegetable sales --/
theorem carrie_vegetable_revenue : 
  let tomatoes := 200
  let carrots := 350
  let eggplants := 120
  let cucumbers := 75
  let tomato_price := 1
  let carrot_price := 1.5
  let eggplant_price := 2.5
  let cucumber_price := 1.75
  let tomato_discount := 0.05
  let carrot_discount_price := 1.25
  let eggplant_free_per := 10
  let cucumber_discount := 0.1
  
  let tomato_revenue := tomatoes * (tomato_price * (1 - tomato_discount))
  let carrot_revenue := carrots * carrot_discount_price
  let eggplant_revenue := (eggplants - (eggplants / eggplant_free_per)) * eggplant_price
  let cucumber_revenue := cucumbers * (cucumber_price * (1 - cucumber_discount))
  
  tomato_revenue + carrot_revenue + eggplant_revenue + cucumber_revenue = 1015.625 := by
  sorry


end carrie_vegetable_revenue_l2380_238071


namespace blueberry_pies_count_l2380_238005

/-- Given a total of 30 pies and a ratio of 2:3:4:1 for apple:blueberry:cherry:peach pies,
    the number of blueberry pies is 9. -/
theorem blueberry_pies_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio peach_ratio : ℕ) :
  total_pies = 30 →
  apple_ratio = 2 →
  blueberry_ratio = 3 →
  cherry_ratio = 4 →
  peach_ratio = 1 →
  blueberry_ratio * (total_pies / (apple_ratio + blueberry_ratio + cherry_ratio + peach_ratio)) = 9 := by
  sorry

end blueberry_pies_count_l2380_238005


namespace sweets_spending_proof_l2380_238035

/-- Calculates the amount spent on sweets given a weekly allowance, junk food spending ratio, and savings amount. -/
def amount_spent_on_sweets (allowance : ℚ) (junk_food_ratio : ℚ) (savings : ℚ) : ℚ :=
  allowance - allowance * junk_food_ratio - savings

/-- Proves that given a weekly allowance of $30, spending 1/3 on junk food, and saving $12, the amount spent on sweets is $8. -/
theorem sweets_spending_proof :
  amount_spent_on_sweets 30 (1/3) 12 = 8 := by
  sorry

#eval amount_spent_on_sweets 30 (1/3) 12

end sweets_spending_proof_l2380_238035


namespace gcd_372_684_l2380_238049

theorem gcd_372_684 : Nat.gcd 372 684 = 12 := by
  sorry

end gcd_372_684_l2380_238049


namespace investment_problem_l2380_238007

theorem investment_problem (total : ℝ) (rate_a rate_b rate_c : ℝ) 
  (h_total : total = 425)
  (h_rate_a : rate_a = 0.05)
  (h_rate_b : rate_b = 0.08)
  (h_rate_c : rate_c = 0.10)
  (h_equal_increase : ∃ (k : ℝ), k > 0 ∧ 
    ∀ (a b c : ℝ), a + b + c = total → 
    rate_a * a = k ∧ rate_b * b = k ∧ rate_c * c = k) :
  ∃ (a b c : ℝ), a + b + c = total ∧ 
    rate_a * a = rate_b * b ∧ rate_b * b = rate_c * c ∧ 
    c = 100 := by
  sorry

end investment_problem_l2380_238007


namespace binomial_expansion_101_2_l2380_238061

theorem binomial_expansion_101_2 : 
  101^3 + 3*(101^2)*2 + 3*101*(2^2) + 2^3 = 1092727 := by
  sorry

end binomial_expansion_101_2_l2380_238061


namespace correct_product_after_reversal_error_l2380_238079

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ := 
  (n % 10) * 10 + (n / 10)

theorem correct_product_after_reversal_error (a b : ℕ) : 
  is_two_digit a → 
  is_two_digit b → 
  reverse_digits a * b = 378 → 
  a * b = 504 := by
  sorry

end correct_product_after_reversal_error_l2380_238079


namespace f_max_min_on_interval_l2380_238059

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- State the theorem
theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max = 5 ∧ min = -15 := by
  sorry

end f_max_min_on_interval_l2380_238059


namespace checkerboard_coverage_uncoverable_boards_l2380_238069

/-- Represents a rectangular checkerboard -/
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Checks if a checkerboard can be completely covered by non-overlapping dominos -/
def can_cover (board : Checkerboard) : Prop :=
  Even (board.rows * board.cols)

/-- Theorem: A checkerboard can be covered iff its total number of squares is even -/
theorem checkerboard_coverage (board : Checkerboard) :
  can_cover board ↔ Even (board.rows * board.cols) :=
sorry

/-- Examples of checkerboards -/
def board1 := Checkerboard.mk 4 6
def board2 := Checkerboard.mk 5 5
def board3 := Checkerboard.mk 4 7
def board4 := Checkerboard.mk 5 6
def board5 := Checkerboard.mk 3 7

/-- Theorem: Specific boards that cannot be covered -/
theorem uncoverable_boards :
  ¬(can_cover board2) ∧ ¬(can_cover board5) :=
sorry

end checkerboard_coverage_uncoverable_boards_l2380_238069


namespace opposite_of_2023_l2380_238077

theorem opposite_of_2023 : 
  -(2023 : ℤ) = -2023 :=
by sorry

end opposite_of_2023_l2380_238077


namespace three_heads_probability_l2380_238021

theorem three_heads_probability (p : ℝ) (h_fair : p = 1 / 2) :
  p * p * p = 1 / 8 := by
  sorry

end three_heads_probability_l2380_238021


namespace not_divisible_by_four_l2380_238091

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => a n * a (n + 1) + 1

theorem not_divisible_by_four : ¬ (4 ∣ a 2008) := by
  sorry

end not_divisible_by_four_l2380_238091


namespace max_sum_pair_contains_96420_l2380_238015

def is_valid_pair (a b : ℕ) : Prop :=
  a ≠ b ∧ 
  a ≥ 10000 ∧ a < 100000 ∧ 
  b ≥ 10000 ∧ b < 100000 ∧
  (∀ d : ℕ, d < 10 → 
    (∃! i : ℕ, i < 5 ∧ (a / 10^i) % 10 = d) ∨
    (∃! i : ℕ, i < 5 ∧ (b / 10^i) % 10 = d))

def is_max_sum_pair (a b : ℕ) : Prop :=
  is_valid_pair a b ∧
  ∀ c d : ℕ, is_valid_pair c d → a + b ≥ c + d

theorem max_sum_pair_contains_96420 :
  ∃ n : ℕ, is_max_sum_pair 96420 n ∨ is_max_sum_pair n 96420 :=
sorry

end max_sum_pair_contains_96420_l2380_238015


namespace investment_problem_solution_l2380_238011

/-- Investment problem with two partners -/
structure InvestmentProblem where
  /-- Ratio of investments for partners p and q -/
  investmentRatio : Rat × Rat
  /-- Ratio of profits for partners p and q -/
  profitRatio : Rat × Rat
  /-- Investment period for partner q in months -/
  qPeriod : ℕ

/-- Solution to the investment problem -/
def solveProblem (prob : InvestmentProblem) : ℚ :=
  let (pInvest, qInvest) := prob.investmentRatio
  let (pProfit, qProfit) := prob.profitRatio
  (qProfit * pInvest * prob.qPeriod) / (pProfit * qInvest)

/-- Theorem stating the solution to the specific problem -/
theorem investment_problem_solution :
  let prob : InvestmentProblem := {
    investmentRatio := (7, 5)
    profitRatio := (7, 10)
    qPeriod := 4
  }
  solveProblem prob = 2 := by sorry


end investment_problem_solution_l2380_238011


namespace shiny_igneous_rocks_l2380_238089

theorem shiny_igneous_rocks (total : ℕ) (sedimentary : ℕ) (igneous : ℕ) :
  total = 270 →
  igneous = sedimentary / 2 →
  total = sedimentary + igneous →
  (igneous / 3 : ℚ) = 30 := by
  sorry

end shiny_igneous_rocks_l2380_238089


namespace triple_sum_arithmetic_sequence_l2380_238081

def arithmetic_sequence (a₁ l n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + i * ((l - a₁) / (n - 1)))

def sum_arithmetic_sequence (a₁ l n : ℕ) : ℕ :=
  (n * (a₁ + l)) / 2

theorem triple_sum_arithmetic_sequence :
  let a₁ := 74
  let l := 107
  let n := 12
  3 * (sum_arithmetic_sequence a₁ l n) = 3258 := by
  sorry

end triple_sum_arithmetic_sequence_l2380_238081


namespace tank_capacity_l2380_238016

/-- 
Given a tank with an unknown capacity, prove that if it's initially filled to 3/4 of its capacity,
and adding 7 gallons fills it to 9/10 of its capacity, then the tank's total capacity is 140/3 gallons.
-/
theorem tank_capacity (tank_capacity : ℝ) : 
  (3 / 4 * tank_capacity + 7 = 9 / 10 * tank_capacity) ↔ 
  (tank_capacity = 140 / 3) := by
sorry

end tank_capacity_l2380_238016


namespace terminal_side_half_angle_l2380_238031

-- Define a function to determine the quadrant of an angle
def quadrant (θ : ℝ) : Set Nat :=
  if 0 < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi / 2 then {1}
  else if Real.pi / 2 < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi then {2}
  else if Real.pi < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < 3 * Real.pi / 2 then {3}
  else {4}

-- Theorem statement
theorem terminal_side_half_angle (α : ℝ) :
  quadrant α = {3} → quadrant (α / 2) = {2} ∨ quadrant (α / 2) = {4} := by
  sorry

end terminal_side_half_angle_l2380_238031


namespace fruit_problem_max_a_l2380_238002

/-- Represents the fruit purchase and sale problem -/
def FruitProblem (totalCost totalWeight cherryPrice cantaloupePrice : ℝ)
  (secondTotalWeight secondMaxCost minProfit : ℝ)
  (cherrySellingPrice cantaloupeSellingPrice : ℝ) :=
  ∀ (a : ℕ),
    let n := (secondMaxCost - 6 * secondTotalWeight) / 29
    (35 * n + 6 * (secondTotalWeight - n) ≤ secondMaxCost) ∧
    (20 * (n - a) + 4 * (secondTotalWeight - n - 2 * a) ≥ minProfit) →
    a ≤ 35

/-- The maximum value of a in the fruit problem is 35 -/
theorem fruit_problem_max_a :
  FruitProblem 9160 560 35 6 300 5280 2120 55 10 :=
sorry

end fruit_problem_max_a_l2380_238002


namespace f_properties_l2380_238080

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x + x * Real.log x

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := a + Real.log x + 1

-- State the theorem
theorem f_properties (a : ℝ) :
  (f_deriv a (Real.exp 1) = 3) →
  (∃ (k : ℤ), k = 3 ∧ 
    (∀ x > 1, f 1 x - ↑k * x + ↑k > 0) ∧
    (∀ k' > ↑k, ∃ x > 1, f 1 x - ↑k' * x + ↑k' ≤ 0)) →
  (a = 1) ∧
  (∀ x ∈ Set.Ioo 0 (Real.exp (-2)), ∀ y ∈ Set.Ioo x (Real.exp (-2)), f 1 y < f 1 x) ∧
  (∀ x ∈ Set.Ioi (Real.exp (-2)), ∀ y ∈ Set.Ioi x, f 1 y > f 1 x) :=
by sorry

end

end f_properties_l2380_238080


namespace dvd_price_ratio_l2380_238046

theorem dvd_price_ratio (mike_price steve_total_price : ℝ) 
  (h1 : mike_price = 5)
  (h2 : steve_total_price = 18)
  (h3 : ∃ (steve_online_price : ℝ), 
    steve_total_price = steve_online_price + 0.8 * steve_online_price) :
  ∃ (steve_online_price : ℝ), 
    steve_online_price / mike_price = 2 := by
  sorry

end dvd_price_ratio_l2380_238046


namespace cookie_cost_calculation_l2380_238044

def cookies_per_dozen : ℕ := 12

-- Define the problem parameters
def total_dozens : ℕ := 6
def selling_price : ℚ := 3/2
def charity_share : ℚ := 45

-- Theorem to prove
theorem cookie_cost_calculation :
  let total_cookies := total_dozens * cookies_per_dozen
  let total_revenue := total_cookies * selling_price
  let total_profit := 2 * charity_share
  let total_cost := total_revenue - total_profit
  (total_cost / total_cookies : ℚ) = 1/4 := by sorry

end cookie_cost_calculation_l2380_238044


namespace ratio_problem_l2380_238030

theorem ratio_problem (x y : ℝ) (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0) :
  (1 / 5 * x) / (1 / 6 * y) = 0.72 := by
sorry

end ratio_problem_l2380_238030


namespace b_bounds_l2380_238047

theorem b_bounds (a : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 1) :
  let b := a^3 + 1 / (1 + a)
  (b ≥ 1 - a + a^2) ∧ (3/4 < b) ∧ (b ≤ 3/2) := by sorry

end b_bounds_l2380_238047


namespace relay_race_distance_per_member_l2380_238014

theorem relay_race_distance_per_member 
  (total_distance : ℕ) 
  (team_size : ℕ) 
  (h1 : total_distance = 150) 
  (h2 : team_size = 5) :
  total_distance / team_size = 30 := by
sorry

end relay_race_distance_per_member_l2380_238014


namespace cube_edge_probability_cube_edge_probability_proof_l2380_238088

/-- The probability of randomly selecting two vertices that form an edge in a cube -/
theorem cube_edge_probability : ℚ :=
let num_vertices : ℕ := 8
let num_edges : ℕ := 12
let total_pairs : ℕ := num_vertices.choose 2
3 / 7

/-- Proof that the probability of randomly selecting two vertices that form an edge in a cube is 3/7 -/
theorem cube_edge_probability_proof :
  cube_edge_probability = 3 / 7 := by
  sorry

end cube_edge_probability_cube_edge_probability_proof_l2380_238088


namespace product_from_gcd_lcm_l2380_238060

theorem product_from_gcd_lcm (a b : ℤ) : 
  Int.gcd a b = 8 → Int.lcm a b = 24 → a * b = 192 := by
  sorry

end product_from_gcd_lcm_l2380_238060


namespace unique_intersection_line_l2380_238024

theorem unique_intersection_line (m b : ℝ) : 
  (∃! k : ℝ, ∃ y₁ y₂ : ℝ, 
    y₁ = k^2 + 4*k + 4 ∧ 
    y₂ = m*k + b ∧ 
    |y₁ - y₂| = 6) →
  (7 = 2*m + b) →
  (m = 8 ∧ b = -9) := by
sorry

end unique_intersection_line_l2380_238024


namespace no_integer_solutions_l2380_238096

theorem no_integer_solutions : ¬∃ (x y : ℤ), (x^7 - 1) / (x - 1) = y^5 - 1 := by
  sorry

end no_integer_solutions_l2380_238096


namespace tourist_contact_probability_l2380_238090

/-- The probability that at least one tourist from the first group can contact at least one tourist from the second group -/
def contact_probability (p : ℝ) : ℝ := 1 - (1 - p)^42

/-- Theorem stating the probability of contact between two groups of tourists -/
theorem tourist_contact_probability 
  (p : ℝ) 
  (h1 : 0 ≤ p) 
  (h2 : p ≤ 1) 
  (group1 : Fin 6 → Type) 
  (group2 : Fin 7 → Type) :
  contact_probability p = 1 - (1 - p)^42 := by
sorry

end tourist_contact_probability_l2380_238090


namespace sandwich_cost_l2380_238063

theorem sandwich_cost (total_cost soda_cost : ℝ) 
  (h1 : total_cost = 10.46)
  (h2 : soda_cost = 0.87) : 
  ∃ sandwich_cost : ℝ, 
    sandwich_cost = 3.49 ∧ 
    2 * sandwich_cost + 4 * soda_cost = total_cost :=
by sorry

end sandwich_cost_l2380_238063


namespace tv_cost_l2380_238083

theorem tv_cost (mixer_cost tv_cost : ℕ) : 
  (2 * mixer_cost + tv_cost = 7000) → 
  (mixer_cost + 2 * tv_cost = 9800) → 
  tv_cost = 4200 := by
sorry

end tv_cost_l2380_238083


namespace fourth_term_is_ten_l2380_238006

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
structure ArithmeticSequence where
  a : ℝ  -- First term
  d : ℝ  -- Common difference

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a + (n - 1) * seq.d

theorem fourth_term_is_ten
  (seq : ArithmeticSequence)
  (h : seq.nthTerm 2 + seq.nthTerm 6 = 20) :
  seq.nthTerm 4 = 10 := by
  sorry

#check fourth_term_is_ten

end fourth_term_is_ten_l2380_238006


namespace right_triangle_smaller_angle_l2380_238087

theorem right_triangle_smaller_angle (a b c : Real) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  c = 90 →           -- One angle is 90° (right angle)
  b = 2 * a →        -- One angle is twice the other
  a = 30 :=          -- The smaller angle is 30°
by sorry

end right_triangle_smaller_angle_l2380_238087


namespace alcohol_concentration_after_mixing_l2380_238039

/-- Represents a vessel with a given capacity and alcohol concentration -/
structure Vessel where
  capacity : ℝ
  alcoholConcentration : ℝ

/-- Calculates the amount of alcohol in a vessel -/
def alcoholAmount (v : Vessel) : ℝ := v.capacity * v.alcoholConcentration

/-- Theorem: The alcohol concentration in vessel D after mixing is 35% -/
theorem alcohol_concentration_after_mixing 
  (vesselA : Vessel)
  (vesselB : Vessel)
  (vesselC : Vessel)
  (vesselD : Vessel)
  (h1 : vesselA.capacity = 5)
  (h2 : vesselA.alcoholConcentration = 0.25)
  (h3 : vesselB.capacity = 12)
  (h4 : vesselB.alcoholConcentration = 0.45)
  (h5 : vesselC.capacity = 7)
  (h6 : vesselC.alcoholConcentration = 0.35)
  (h7 : vesselD.capacity = 26) :
  let totalAlcohol := alcoholAmount vesselA + alcoholAmount vesselB + alcoholAmount vesselC
  let totalVolume := vesselA.capacity + vesselB.capacity + vesselC.capacity + (vesselD.capacity - (vesselA.capacity + vesselB.capacity + vesselC.capacity))
  totalAlcohol / totalVolume = 0.35 := by
    sorry

end alcohol_concentration_after_mixing_l2380_238039


namespace bread_inventory_l2380_238066

/-- The number of loaves of bread sold during the day -/
def loaves_sold : ℕ := 629

/-- The number of loaves of bread delivered in the evening -/
def loaves_delivered : ℕ := 489

/-- The number of loaves of bread at the end of the day -/
def loaves_end : ℕ := 2215

/-- The number of loaves of bread at the start of the day -/
def loaves_start : ℕ := 2355

theorem bread_inventory : loaves_start - loaves_sold + loaves_delivered = loaves_end := by
  sorry

end bread_inventory_l2380_238066


namespace inclination_angle_of_line_l2380_238013

/-- Given a function f(x) = a*sin(x) - b*cos(x) with symmetry axis x = π/4,
    prove that the inclination angle of the line ax - by + c = 0 is 3π/4 -/
theorem inclination_angle_of_line (a b c : ℝ) :
  (∀ x, a * Real.sin (π/4 + x) - b * Real.cos (π/4 + x) = 
        a * Real.sin (π/4 - x) - b * Real.cos (π/4 - x)) →
  Real.arctan (a / b) = 3 * π / 4 :=
by sorry

end inclination_angle_of_line_l2380_238013
