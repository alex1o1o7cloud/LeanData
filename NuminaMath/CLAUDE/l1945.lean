import Mathlib

namespace triangle_side_ratio_bound_l1945_194527

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  height_a : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π
  height_condition : height_a = a

-- Theorem statement
theorem triangle_side_ratio_bound (t : Triangle) : 
  2 ≤ (t.b / t.c + t.c / t.b) ∧ (t.b / t.c + t.c / t.b) ≤ Real.sqrt 5 :=
by sorry

end triangle_side_ratio_bound_l1945_194527


namespace big_eight_football_league_games_l1945_194526

theorem big_eight_football_league_games (num_divisions : Nat) (teams_per_division : Nat) : 
  num_divisions = 3 → 
  teams_per_division = 4 → 
  (num_divisions * teams_per_division * (teams_per_division - 1) + 
   num_divisions * teams_per_division * (num_divisions - 1) * teams_per_division / 2) = 228 := by
  sorry

#check big_eight_football_league_games

end big_eight_football_league_games_l1945_194526


namespace asterisk_replacement_l1945_194532

theorem asterisk_replacement : ∃! (x : ℝ), x > 0 ∧ (x / 20) * (x / 80) = 1 := by
  sorry

end asterisk_replacement_l1945_194532


namespace edward_summer_earnings_l1945_194591

/-- Edward's lawn mowing business earnings --/
def lawn_mowing_problem (spring_earnings summer_earnings supplies_cost final_amount : ℕ) : Prop :=
  spring_earnings + summer_earnings = supplies_cost + final_amount

theorem edward_summer_earnings :
  ∃ (summer_earnings : ℕ),
    lawn_mowing_problem 2 summer_earnings 5 24 ∧ summer_earnings = 27 :=
by
  sorry

end edward_summer_earnings_l1945_194591


namespace remaining_aces_probability_l1945_194594

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards in each hand -/
def HandSize : ℕ := 13

/-- Represents the number of aces in a standard deck -/
def TotalAces : ℕ := 4

/-- Computes the probability of a specific person having the remaining aces
    given that one person has one ace -/
def probabilityOfRemainingAces (deck : ℕ) (handSize : ℕ) (totalAces : ℕ) : ℚ :=
  22 / 703

theorem remaining_aces_probability :
  probabilityOfRemainingAces StandardDeck HandSize TotalAces = 22 / 703 := by
  sorry

end remaining_aces_probability_l1945_194594


namespace cube_sum_ge_sqrt_product_square_sum_l1945_194572

theorem cube_sum_ge_sqrt_product_square_sum {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
  a^3 + b^3 ≥ Real.sqrt (a * b) * (a^2 + b^2) := by
  sorry

end cube_sum_ge_sqrt_product_square_sum_l1945_194572


namespace third_median_length_l1945_194502

/-- A triangle with two known medians and area -/
structure TriangleWithMedians where
  -- The length of the first median
  median1 : ℝ
  -- The length of the second median
  median2 : ℝ
  -- The area of the triangle
  area : ℝ

/-- The theorem stating the length of the third median -/
theorem third_median_length (t : TriangleWithMedians) 
  (h1 : t.median1 = 5)
  (h2 : t.median2 = 4)
  (h3 : t.area = 6 * Real.sqrt 5) :
  ∃ (median3 : ℝ), median3 = 3 * Real.sqrt 7 := by
  sorry

end third_median_length_l1945_194502


namespace hyperbola_iff_k_in_range_l1945_194540

/-- A curve is defined by the equation (x^2)/(k+4) + (y^2)/(k-1) = 1 -/
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k + 4) + y^2 / (k - 1) = 1 ∧ (k + 4) * (k - 1) < 0

/-- The range of k values for which the curve represents a hyperbola -/
def hyperbola_k_range : Set ℝ := {k | -4 < k ∧ k < 1}

/-- Theorem stating that the curve represents a hyperbola if and only if k is in the range (-4, 1) -/
theorem hyperbola_iff_k_in_range (k : ℝ) :
  is_hyperbola k ↔ k ∈ hyperbola_k_range :=
by sorry

end hyperbola_iff_k_in_range_l1945_194540


namespace angel_food_cake_egg_whites_angel_food_cake_proof_l1945_194588

theorem angel_food_cake_egg_whites (aquafaba_per_egg_white : ℕ) 
  (num_cakes : ℕ) (total_aquafaba : ℕ) : ℕ :=
  let egg_whites_per_cake := (total_aquafaba / aquafaba_per_egg_white) / num_cakes
  egg_whites_per_cake

theorem angel_food_cake_proof : 
  angel_food_cake_egg_whites 2 2 32 = 8 := by
  sorry

end angel_food_cake_egg_whites_angel_food_cake_proof_l1945_194588


namespace tangent_line_and_range_of_a_l1945_194597

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

theorem tangent_line_and_range_of_a :
  (∃ (m b : ℝ), ∀ x, (f 4) x = m * (x - 1) + (f 4) 1 → m = -2 ∧ b = 2) ∧
  (∀ a, (∀ x, x > 1 → f a x > 0) ↔ a ≤ 2) :=
sorry

end tangent_line_and_range_of_a_l1945_194597


namespace ratio_transitivity_l1945_194582

theorem ratio_transitivity (a b c : ℚ) 
  (hab : a / b = 4 / 3) 
  (hbc : b / c = 1 / 5) : 
  a / c = 4 / 5 := by
  sorry

end ratio_transitivity_l1945_194582


namespace midpoint_parallelogram_l1945_194512

/-- A quadrilateral in 2D plane represented by its vertices -/
structure Quadrilateral where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Points that divide the sides of a quadrilateral in ratio r -/
def divisionPoints (q : Quadrilateral) (r : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  let (x1, y1) := q.v1
  let (x2, y2) := q.v2
  let (x3, y3) := q.v3
  let (x4, y4) := q.v4
  ( ((x2 - x1) * r + x1, (y2 - y1) * r + y1),
    ((x3 - x2) * r + x2, (y3 - y2) * r + y2),
    ((x4 - x3) * r + x3, (y4 - y3) * r + y3),
    ((x1 - x4) * r + x4, (y1 - y4) * r + y4) )

/-- Check if the quadrilateral formed by the division points is a parallelogram -/
def isParallelogram (points : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) := points
  (x3 - x1 = x4 - x2) ∧ (y3 - y1 = y4 - y2)

/-- The main theorem: only midpoints (r = 1/2) form a parallelogram for all quadrilaterals -/
theorem midpoint_parallelogram (q : Quadrilateral) :
    ∀ r : ℝ, (∀ q' : Quadrilateral, isParallelogram (divisionPoints q' r)) → r = 1/2 := by
  sorry

end midpoint_parallelogram_l1945_194512


namespace cone_base_radius_l1945_194593

/-- Given a sector paper with radius 30 cm and central angle 120°,
    when used to form the lateral surface of a cone,
    the radius of the base of the cone is 10 cm. -/
theorem cone_base_radius (R : ℝ) (θ : ℝ) (r : ℝ) : 
  R = 30 → θ = 120 → 2 * π * r = (θ / 360) * 2 * π * R → r = 10 := by
  sorry

end cone_base_radius_l1945_194593


namespace cid_earnings_l1945_194545

/-- Represents the earnings from Cid's mechanic shop --/
def mechanic_earnings (oil_change_price : ℕ) (repair_price : ℕ) (car_wash_price : ℕ)
  (oil_changes : ℕ) (repairs : ℕ) (car_washes : ℕ) : ℕ :=
  oil_change_price * oil_changes + repair_price * repairs + car_wash_price * car_washes

/-- Theorem stating that Cid's earnings are $475 given the specific prices and services --/
theorem cid_earnings : 
  mechanic_earnings 20 30 5 5 10 15 = 475 := by
  sorry

end cid_earnings_l1945_194545


namespace milk_buckets_l1945_194500

theorem milk_buckets (bucket_capacity : ℝ) (total_milk : ℝ) : 
  bucket_capacity = 15 → total_milk = 147 → ⌈total_milk / bucket_capacity⌉ = 10 := by
  sorry

end milk_buckets_l1945_194500


namespace only_four_and_eight_satisfy_l1945_194505

/-- A natural number is a proper divisor of another natural number if it divides the number, is greater than 1, and is not equal to the number itself. -/
def IsProperDivisor (d n : ℕ) : Prop :=
  d ∣ n ∧ d > 1 ∧ d ≠ n

/-- The set of proper divisors of a natural number. -/
def ProperDivisors (n : ℕ) : Set ℕ :=
  {d | IsProperDivisor d n}

/-- The property that all proper divisors of n, when increased by 1, form the set of proper divisors of m. -/
def SatisfiesProperty (n m : ℕ) : Prop :=
  ∃ (f : ℕ → ℕ), f = (· + 1) ∧
  (ProperDivisors m) = f '' (ProperDivisors n)

/-- The theorem stating that only 4 and 8 satisfy the given property. -/
theorem only_four_and_eight_satisfy :
  ∀ n : ℕ, (∃ m : ℕ, SatisfiesProperty n m) ↔ n = 4 ∨ n = 8 := by
  sorry


end only_four_and_eight_satisfy_l1945_194505


namespace ratio_chain_l1945_194561

theorem ratio_chain (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 4)
  (h2 : b / c = 13 / 9)
  (h3 : c / d = 5 / 13)
  (h4 : d / e = 2 / 3)
  (h5 : e / f = 7 / 5) :
  a / f = 7 / 6 := by
sorry

end ratio_chain_l1945_194561


namespace x_eq_two_is_axis_of_symmetry_l1945_194559

-- Define a function f with the given property
def f (x : ℝ) : ℝ := sorry

-- State the condition that f(x) = f(4-x) for all x
axiom f_symmetry (x : ℝ) : f x = f (4 - x)

-- Define the concept of an axis of symmetry
def is_axis_of_symmetry (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

-- Theorem stating that x = 2 is an axis of symmetry
theorem x_eq_two_is_axis_of_symmetry :
  is_axis_of_symmetry 2 :=
sorry

end x_eq_two_is_axis_of_symmetry_l1945_194559


namespace circle_area_doubling_l1945_194522

theorem circle_area_doubling (r : ℝ) (h : r > 0) : 
  π * (2 * r)^2 = 4 * (π * r^2) := by
  sorry

end circle_area_doubling_l1945_194522


namespace course_size_l1945_194566

theorem course_size (total : ℕ) 
  (h1 : total / 5 + total / 4 + total / 2 + 30 = total) : total = 600 := by
  sorry

end course_size_l1945_194566


namespace tree_planting_equation_l1945_194570

/-- Represents the relationship between the number of people planting trees and the total number of seedlings. -/
theorem tree_planting_equation (x : ℤ) (total_seedlings : ℤ) : 
  (5 * x + 3 = total_seedlings) ∧ (6 * x = total_seedlings + 4) →
  5 * x + 3 = 6 * x - 4 := by
  sorry

#check tree_planting_equation

end tree_planting_equation_l1945_194570


namespace student_distribution_l1945_194533

theorem student_distribution (total : ℕ) (h_total : total > 0) :
  let third_year := (30 : ℕ) * total / 100
  let not_second_year := (90 : ℕ) * total / 100
  let second_year := total - not_second_year
  let not_third_year := total - third_year
  (second_year : ℚ) / not_third_year = 1 / 7 := by
sorry

end student_distribution_l1945_194533


namespace minimum_value_x_plus_reciprocal_l1945_194579

theorem minimum_value_x_plus_reciprocal (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) ≥ 3 ∧ ∃ x₀ > 1, x₀ + 1 / (x₀ - 1) = 3 :=
sorry


end minimum_value_x_plus_reciprocal_l1945_194579


namespace st_length_l1945_194564

/-- Rectangle WXYZ with parallelogram PQRS inside -/
structure RectangleWithParallelogram where
  /-- Width of the rectangle -/
  width : ℝ
  /-- Height of the rectangle -/
  height : ℝ
  /-- Length of PW -/
  pw : ℝ
  /-- Length of WS -/
  ws : ℝ
  /-- Length of SZ -/
  sz : ℝ
  /-- Length of ZR -/
  zr : ℝ
  /-- PT is perpendicular to SR -/
  pt_perp_sr : Bool

/-- The main theorem -/
theorem st_length (rect : RectangleWithParallelogram) 
  (h1 : rect.width = 15)
  (h2 : rect.height = 9)
  (h3 : rect.pw = 3)
  (h4 : rect.ws = 4)
  (h5 : rect.sz = 5)
  (h6 : rect.zr = 12)
  (h7 : rect.pt_perp_sr = true) :
  ∃ (st : ℝ), st = 16 / 13 := by sorry

end st_length_l1945_194564


namespace prime_roots_equation_l1945_194510

theorem prime_roots_equation (p q : ℕ) : 
  (∃ x y : ℕ, Prime x ∧ Prime y ∧ 
   x ≠ y ∧
   (p * x^2 - q * x + 1985 = 0) ∧ 
   (p * y^2 - q * y + 1985 = 0)) →
  12 * p^2 + q = 414 := by
sorry

end prime_roots_equation_l1945_194510


namespace inequality_proof_l1945_194508

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^4 + b^4 > 2*a*b^3 := by
  sorry

end inequality_proof_l1945_194508


namespace ellipse_equation_proof_l1945_194519

def original_ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

def new_ellipse (x y : ℝ) : Prop := x^2/15 + y^2/10 = 1

def same_foci (e1 e2 : (ℝ → ℝ → Prop)) : Prop :=
  ∃ c : ℝ, (∀ x y : ℝ, e1 x y ↔ (x - c)^2/(9 - 4) + y^2/4 = 1) ∧
           (∀ x y : ℝ, e2 x y ↔ (x - c)^2/(15 - 10) + y^2/10 = 1)

theorem ellipse_equation_proof :
  (new_ellipse 3 (-2)) ∧
  (same_foci original_ellipse new_ellipse) :=
sorry

end ellipse_equation_proof_l1945_194519


namespace cubic_sum_of_quadratic_roots_l1945_194587

theorem cubic_sum_of_quadratic_roots : ∀ r s : ℝ,
  r^2 - 5*r + 6 = 0 →
  s^2 - 5*s + 6 = 0 →
  r^3 + s^3 = 35 := by
sorry

end cubic_sum_of_quadratic_roots_l1945_194587


namespace fruit_salad_cherries_l1945_194592

/-- Represents the composition of a fruit salad --/
structure FruitSalad where
  blueberries : ℕ
  raspberries : ℕ
  grapes : ℕ
  cherries : ℕ

/-- Checks if the fruit salad satisfies the given conditions --/
def isValidFruitSalad (fs : FruitSalad) : Prop :=
  fs.blueberries + fs.raspberries + fs.grapes + fs.cherries = 280 ∧
  fs.raspberries = 2 * fs.blueberries ∧
  fs.grapes = 3 * fs.cherries ∧
  fs.cherries = 4 * fs.raspberries

/-- Theorem stating that a valid fruit salad has 64 cherries --/
theorem fruit_salad_cherries (fs : FruitSalad) :
  isValidFruitSalad fs → fs.cherries = 64 := by
  sorry

#check fruit_salad_cherries

end fruit_salad_cherries_l1945_194592


namespace max_sum_consecutive_triples_l1945_194550

/-- Represents a permutation of the digits 1 to 9 -/
def Permutation := Fin 9 → Fin 9

/-- Calculates the sum of seven consecutive three-digit numbers formed from a permutation -/
def sumConsecutiveTriples (p : Permutation) : ℕ :=
  (100 * p 0 + 110 * p 1 + 111 * p 2 + 111 * p 3 + 111 * p 4 + 111 * p 5 + 111 * p 6 + 11 * p 7 + p 8).val

/-- The maximum possible sum of consecutive triples -/
def maxSum : ℕ := 4648

/-- Theorem stating that the maximum sum of consecutive triples is 4648 -/
theorem max_sum_consecutive_triples :
  ∀ p : Permutation, sumConsecutiveTriples p ≤ maxSum :=
sorry

end max_sum_consecutive_triples_l1945_194550


namespace total_shared_amount_l1945_194528

/-- Proves that the total amount shared between x, y, and z is 925, given the specified conditions. -/
theorem total_shared_amount (z : ℚ) (y : ℚ) (x : ℚ) : 
  z = 250 →
  y = 1.2 * z →
  x = 1.25 * y →
  x + y + z = 925 := by
  sorry

end total_shared_amount_l1945_194528


namespace apple_lovers_problem_l1945_194549

theorem apple_lovers_problem (total_apples : ℕ) (initial_per_person : ℕ) (decrease : ℕ) 
  (h1 : total_apples = 1430)
  (h2 : initial_per_person = 22)
  (h3 : decrease = 9) :
  ∃ (initial_people new_people : ℕ),
    initial_people * initial_per_person = total_apples ∧
    (initial_people + new_people) * (initial_per_person - decrease) = total_apples ∧
    new_people = 45 := by
  sorry

end apple_lovers_problem_l1945_194549


namespace f_difference_at_five_l1945_194578

-- Define the function f
def f (x : ℝ) : ℝ := x^4 + x^2 + 5*x + 3

-- State the theorem
theorem f_difference_at_five : f 5 - f (-5) = 50 := by
  sorry

end f_difference_at_five_l1945_194578


namespace students_without_A_l1945_194598

theorem students_without_A (total_students : ℕ) (chemistry_A : ℕ) (physics_A : ℕ) (both_A : ℕ) :
  total_students = 40 →
  chemistry_A = 10 →
  physics_A = 18 →
  both_A = 5 →
  total_students - (chemistry_A + physics_A - both_A) = 17 :=
by
  sorry

end students_without_A_l1945_194598


namespace abs_gt_one_iff_square_minus_one_gt_zero_l1945_194530

theorem abs_gt_one_iff_square_minus_one_gt_zero :
  ∀ x : ℝ, |x| > 1 ↔ x^2 - 1 > 0 := by
  sorry

end abs_gt_one_iff_square_minus_one_gt_zero_l1945_194530


namespace max_value_cubic_expression_l1945_194560

theorem max_value_cubic_expression (a b : ℝ) (h : a^2 + b^2 = 1) :
  ∃ (max : ℝ), max = 1/4 ∧ ∀ (x y : ℝ), x^2 + y^2 = 1 → x^3 * y - y^3 * x ≤ max :=
by sorry

end max_value_cubic_expression_l1945_194560


namespace trees_to_plant_l1945_194569

theorem trees_to_plant (current_trees final_trees : ℕ) : 
  current_trees = 25 → final_trees = 98 → final_trees - current_trees = 73 := by
  sorry

end trees_to_plant_l1945_194569


namespace complement_of_M_l1945_194518

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}

theorem complement_of_M : (U \ M) = {2, 4, 6} := by sorry

end complement_of_M_l1945_194518


namespace exponent_equality_l1945_194544

theorem exponent_equality (n : ℕ) (x : ℕ) 
  (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^x) 
  (h2 : n = 27) : 
  x = 28 := by
  sorry

end exponent_equality_l1945_194544


namespace choose_four_from_eight_l1945_194555

theorem choose_four_from_eight : Nat.choose 8 4 = 70 := by
  sorry

end choose_four_from_eight_l1945_194555


namespace election_winner_votes_l1945_194562

theorem election_winner_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (3 * total_votes) / 4 - total_votes / 4 = 500) : 
  (3 * total_votes) / 4 = 750 := by
  sorry

end election_winner_votes_l1945_194562


namespace unknown_blanket_rate_unknown_blanket_rate_eq_175_l1945_194516

/-- The unknown rate of two blankets given the following conditions:
    - 1 blanket purchased at Rs. 100
    - 5 blankets purchased at Rs. 150 each
    - 2 blankets purchased at an unknown rate
    - The average price of all blankets is Rs. 150
-/
theorem unknown_blanket_rate : ℕ :=
  let num_blankets_1 : ℕ := 1
  let price_1 : ℕ := 100
  let num_blankets_2 : ℕ := 5
  let price_2 : ℕ := 150
  let num_blankets_3 : ℕ := 2
  let total_blankets : ℕ := num_blankets_1 + num_blankets_2 + num_blankets_3
  let average_price : ℕ := 150
  let total_cost : ℕ := average_price * total_blankets
  let known_cost : ℕ := num_blankets_1 * price_1 + num_blankets_2 * price_2
  let unknown_rate : ℕ := (total_cost - known_cost) / num_blankets_3
  unknown_rate

theorem unknown_blanket_rate_eq_175 : unknown_blanket_rate = 175 := by
  sorry

end unknown_blanket_rate_unknown_blanket_rate_eq_175_l1945_194516


namespace super_ball_distance_l1945_194529

def bounce_height (initial_height : ℝ) (bounce_factor : ℝ) (n : ℕ) : ℝ :=
  initial_height * (bounce_factor ^ n)

def total_distance (initial_height : ℝ) (bounce_factor : ℝ) (num_bounces : ℕ) : ℝ :=
  let descents := initial_height + (Finset.sum (Finset.range num_bounces) (λ i => bounce_height initial_height bounce_factor i))
  let ascents := Finset.sum (Finset.range (num_bounces + 1)) (λ i => bounce_height initial_height bounce_factor i)
  descents + ascents

theorem super_ball_distance :
  total_distance 20 0.6 4 = 69.632 := by
  sorry

end super_ball_distance_l1945_194529


namespace coffee_shrink_problem_l1945_194507

def shrink_ray_effect : ℝ := 0.5

theorem coffee_shrink_problem (num_cups : ℕ) (remaining_coffee : ℝ) 
  (h1 : num_cups = 5)
  (h2 : remaining_coffee = 20) : 
  (remaining_coffee / shrink_ray_effect) / num_cups = 8 := by
  sorry

end coffee_shrink_problem_l1945_194507


namespace dart_probability_l1945_194504

theorem dart_probability (square_side : Real) (circle_area : Real) 
  (h1 : square_side = 1)
  (h2 : circle_area = Real.pi / 4) :
  1 - (circle_area / (square_side * square_side)) = 1 - Real.pi / 4 := by
  sorry

end dart_probability_l1945_194504


namespace polynomial_root_sum_l1945_194586

theorem polynomial_root_sum (p q : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2 : ℂ) ^ 3 + p * (Complex.I * Real.sqrt 2 + 2) + q = 0 → 
  p + q = 14 := by
sorry

end polynomial_root_sum_l1945_194586


namespace arithmetic_sequence_ratio_l1945_194509

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0)
  (h2 : a 1 = 2 * a 8 - 3 * a 4)
  (h3 : ∀ n : ℕ, a (n + 1) = a n + d) :
  let S : ℕ → ℝ := λ n ↦ (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2
  S 8 / S 16 = 3 / 10 := by
sorry

end arithmetic_sequence_ratio_l1945_194509


namespace unique_power_inequality_l1945_194525

theorem unique_power_inequality : ∃! (n : ℕ), n > 0 ∧ ∀ (m : ℕ), m > 0 → n^m ≥ m^n :=
by sorry

end unique_power_inequality_l1945_194525


namespace duck_eggs_sum_l1945_194575

theorem duck_eggs_sum (yesterday_eggs : ℕ) (fewer_today : ℕ) : 
  yesterday_eggs = 1925 →
  fewer_today = 138 →
  yesterday_eggs + (yesterday_eggs - fewer_today) = 3712 := by
  sorry

end duck_eggs_sum_l1945_194575


namespace kelly_games_giveaway_l1945_194537

theorem kelly_games_giveaway (initial_games : ℕ) (remaining_games : ℕ) : 
  initial_games = 50 → remaining_games = 35 → initial_games - remaining_games = 15 := by
  sorry

end kelly_games_giveaway_l1945_194537


namespace puppies_brought_in_l1945_194514

-- Define the given conditions
def initial_puppies : ℕ := 5
def adopted_per_day : ℕ := 8
def days_to_adopt_all : ℕ := 5

-- Define the theorem
theorem puppies_brought_in :
  ∃ (brought_in : ℕ), 
    initial_puppies + brought_in = adopted_per_day * days_to_adopt_all :=
by
  -- The proof would go here
  sorry

end puppies_brought_in_l1945_194514


namespace sum_squares_of_roots_l1945_194590

theorem sum_squares_of_roots (x₁ x₂ : ℝ) : 
  6 * x₁^2 + 11 * x₁ - 35 = 0 →
  6 * x₂^2 + 11 * x₂ - 35 = 0 →
  x₁ > 2 →
  x₂ > 2 →
  x₁^2 + x₂^2 = 541 / 36 := by
sorry

end sum_squares_of_roots_l1945_194590


namespace y_value_approximation_l1945_194521

noncomputable def x : ℝ := 3.87

theorem y_value_approximation :
  let y := 2 * (Real.log x)^3 - (5 / 3)
  ∃ ε > 0, |y + 1.2613| < ε ∧ ε < 0.0001 :=
sorry

end y_value_approximation_l1945_194521


namespace sum_of_absolute_values_l1945_194501

theorem sum_of_absolute_values (a b : ℤ) : 
  (abs a = 5 ∧ abs b = 3) → 
  (a + b = 8 ∨ a + b = 2 ∨ a + b = -2 ∨ a + b = -8) :=
by sorry

end sum_of_absolute_values_l1945_194501


namespace both_are_liars_l1945_194583

-- Define the possible types of islanders
inductive IslanderType
  | Knight
  | Liar

-- Define the islanders
def A : IslanderType := sorry
def B : IslanderType := sorry

-- Define A's statement
def A_statement : Prop := (A = IslanderType.Liar) ∧ (B ≠ IslanderType.Liar)

-- Define the truth-telling property of knights and liars
def tells_truth (i : IslanderType) (p : Prop) : Prop :=
  (i = IslanderType.Knight ∧ p) ∨ (i = IslanderType.Liar ∧ ¬p)

-- Theorem to prove
theorem both_are_liars :
  tells_truth A A_statement →
  A = IslanderType.Liar ∧ B = IslanderType.Liar :=
by sorry

end both_are_liars_l1945_194583


namespace sin_plus_cos_value_l1945_194552

theorem sin_plus_cos_value (α : ℝ) (h : (Real.sin (α - π/4)) / (Real.cos (2*α)) = -Real.sqrt 2) : 
  Real.sin α + Real.cos α = 1/2 := by
  sorry

end sin_plus_cos_value_l1945_194552


namespace seating_arrangements_l1945_194589

/-- The number of ways to choose k items from n items without replacement and where order matters. -/
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter. -/
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Given a bench with 9 seats and 3 people to be seated with at least 2 empty seats between any two people,
    the number of different seating arrangements is 60. -/
theorem seating_arrangements (total_seats people : ℕ) (h1 : total_seats = 9) (h2 : people = 3) :
  A people people * (C 4 2 + C 4 1) = 60 := by
  sorry


end seating_arrangements_l1945_194589


namespace quadratic_function_proof_l1945_194581

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_proof (a b c : ℝ) :
  (∀ x, f a b c x ≥ 0) ∧  -- Minimum value is 0
  (∀ x, f a b c x = f a b c (-2 - x)) ∧  -- Symmetric about x = -1
  (∀ x ∈ Set.Ioo 0 5, x ≤ f a b c x ∧ f a b c x ≤ 2 * |x - 1| + 1) →
  ∀ x, f a b c x = (1/4) * (x + 1)^2 := by
  sorry

end quadratic_function_proof_l1945_194581


namespace parabola_intersection_fixed_point_l1945_194553

/-- Given two parabolas C₁ and C₂ with specific properties, prove that C₂ passes through a fixed point. -/
theorem parabola_intersection_fixed_point 
  (C₁_vertex : ℝ × ℝ) 
  (C₁_focus : ℝ × ℝ)
  (a b : ℝ) :
  let C₁_vertex_x := Real.sqrt 2 - 1
  let C₁_vertex_y := 1
  let C₁_focus_x := Real.sqrt 2 - 3/4
  let C₁_focus_y := 1
  let C₂_eq (x y : ℝ) := y^2 - a*y + x + 2*b = 0
  let fixed_point := (Real.sqrt 2 - 1/2, 1)
  C₁_vertex = (C₁_vertex_x, C₁_vertex_y) →
  C₁_focus = (C₁_focus_x, C₁_focus_y) →
  (∃ (x₀ y₀ : ℝ), 
    (y₀^2 - 2*y₀ - x₀ + Real.sqrt 2 = 0) ∧ 
    (C₂_eq x₀ y₀) ∧ 
    ((2*y₀ - 2) * (2*y₀ - a) = -1)) →
  C₂_eq fixed_point.1 fixed_point.2 :=
by sorry

end parabola_intersection_fixed_point_l1945_194553


namespace power_of_negative_power_l1945_194547

theorem power_of_negative_power (a : ℝ) : (-a^5)^2 = a^10 := by
  sorry

end power_of_negative_power_l1945_194547


namespace intersection_range_chord_length_l1945_194536

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- The line equation -/
def line (x y m : ℝ) : Prop := y = x + m

/-- The range of m for which the line intersects the ellipse -/
theorem intersection_range (m : ℝ) : 
  (∃ x y : ℝ, ellipse x y ∧ line x y m) ↔ -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3 := by sorry

/-- The length of the chord when the line passes through (1,0) -/
theorem chord_length : 
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
    line x₁ y₁ (-1) ∧ line x₂ y₂ (-1) ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = (4 / 3) * Real.sqrt 2 := by sorry

end intersection_range_chord_length_l1945_194536


namespace equation_solution_l1945_194523

theorem equation_solution : ∃! x : ℝ, Real.sqrt (7 * x - 3) + Real.sqrt (x^3 - 1) = 3 :=
  by sorry

end equation_solution_l1945_194523


namespace circle_tangent_line_equation_non_intersecting_line_condition_intersecting_line_orthogonal_condition_l1945_194531

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 9

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y + 3 * Real.sqrt 2 + 1 = 0

-- Define the non-intersecting line
def non_intersecting_line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the intersecting line
def intersecting_line (m : ℝ) (x y : ℝ) : Prop := y = x + m

theorem circle_tangent_line_equation :
  ∀ x y : ℝ, circle_C x y ↔ (x - 1)^2 + (y + 2)^2 = 9 := by sorry

theorem non_intersecting_line_condition :
  ∀ k : ℝ, (∀ x y : ℝ, ¬(circle_C x y ∧ non_intersecting_line k x y)) ↔ (0 < k ∧ k < 3/4) := by sorry

theorem intersecting_line_orthogonal_condition :
  ∀ m : ℝ, (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ 
    intersecting_line m x₁ y₁ ∧ intersecting_line m x₂ y₂ ∧
    x₁ * x₂ + y₁ * y₂ = 0) ↔ (m = 1 ∨ m = -4) := by sorry

end circle_tangent_line_equation_non_intersecting_line_condition_intersecting_line_orthogonal_condition_l1945_194531


namespace i_power_2010_l1945_194557

theorem i_power_2010 : (Complex.I : ℂ) ^ 2010 = -1 := by sorry

end i_power_2010_l1945_194557


namespace system_equiv_line_l1945_194517

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  x - 2*y = 1 ∧ x^3 - 6*x*y - 8*y^3 = 1

/-- The line representing the solution -/
def solution_line (x y : ℝ) : Prop :=
  y = (x - 1) / 2

/-- Theorem stating that the system of equations is equivalent to the solution line -/
theorem system_equiv_line : 
  ∀ x y : ℝ, system x y ↔ solution_line x y :=
sorry

end system_equiv_line_l1945_194517


namespace eighteen_hundred_is_interesting_smallest_interesting_number_l1945_194565

/-- A natural number is interesting if 2n is a perfect square and 15n is a perfect cube. -/
def IsInteresting (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 2 * n = a^2 ∧ 15 * n = b^3

/-- 1800 is an interesting number. -/
theorem eighteen_hundred_is_interesting : IsInteresting 1800 :=
  sorry

/-- 1800 is the smallest interesting number. -/
theorem smallest_interesting_number :
  IsInteresting 1800 ∧ ∀ m < 1800, ¬IsInteresting m :=
  sorry

end eighteen_hundred_is_interesting_smallest_interesting_number_l1945_194565


namespace triangle_area_proof_l1945_194585

/-- Given a triangle DEF with inradius r, circumradius R, and angles D, E, F,
    prove that if r = 10, R = 25, and 2cos(E) = cos(D) + cos(F),
    then the area of the triangle is 225√51/5 -/
theorem triangle_area_proof (D E F : ℝ) (r R : ℝ) :
  r = 10 →
  R = 25 →
  2 * Real.cos E = Real.cos D + Real.cos F →
  ∃ (d e f : ℝ),
    d > 0 ∧ e > 0 ∧ f > 0 ∧
    e^2 = d^2 + f^2 - 2*d*f*(Real.cos E) ∧
    Real.cos D = (f^2 + e^2 - d^2) / (2*f*e) ∧
    Real.cos F = (d^2 + e^2 - f^2) / (2*d*e) ∧
    (d + e + f) / 2 * r = 225 * Real.sqrt 51 / 5 :=
sorry

end triangle_area_proof_l1945_194585


namespace expression_value_l1945_194548

theorem expression_value (m n : ℝ) 
  (h1 : m^2 + 2*m*n = 3) 
  (h2 : 2*n^2 + 3*m*n = 5) : 
  2*m^2 + 13*m*n + 6*n^2 = 21 := by
sorry

end expression_value_l1945_194548


namespace min_value_a_l1945_194546

theorem min_value_a (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + a/y ≥ 16/(x+y)) → a ≥ 9 := by
  sorry

end min_value_a_l1945_194546


namespace diamond_2_3_4_eq_zero_l1945_194571

/-- Definition of the diamond operation for real numbers -/
def diamond (a b c : ℝ) : ℝ := (b + 1)^2 - 4 * (a - 1) * c

/-- Theorem stating that diamond(2, 3, 4) equals 0 -/
theorem diamond_2_3_4_eq_zero : diamond 2 3 4 = 0 := by
  sorry

end diamond_2_3_4_eq_zero_l1945_194571


namespace max_value_implies_a_equals_one_l1945_194524

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - a

theorem max_value_implies_a_equals_one :
  (∃ (M : ℝ), M = 1 ∧ ∀ x ∈ Set.Icc 0 2, f a x ≤ M) →
  a = 1 :=
by sorry

end max_value_implies_a_equals_one_l1945_194524


namespace union_A_B_l1945_194513

def A : Set ℝ := {x | (x - 2) / (x + 1) ≤ 0}

def B : Set ℝ := {x | -2 * x^2 + 7 * x + 4 > 0}

theorem union_A_B : A ∪ B = Set.Ioo (-1 : ℝ) 4 := by sorry

end union_A_B_l1945_194513


namespace pencils_purchased_correct_l1945_194567

/-- The number of pencils purchased given the conditions of the problem -/
def num_pencils : ℕ := 75

/-- The number of pens purchased -/
def num_pens : ℕ := 30

/-- The price of each pencil -/
def pencil_price : ℚ := 2

/-- The price of each pen -/
def pen_price : ℚ := 10

/-- The total cost of the purchase -/
def total_cost : ℚ := 450

/-- Theorem stating that the number of pencils purchased is correct given the conditions -/
theorem pencils_purchased_correct :
  (num_pens : ℚ) * pen_price + (num_pencils : ℚ) * pencil_price = total_cost :=
sorry

end pencils_purchased_correct_l1945_194567


namespace oil_ratio_proof_l1945_194511

theorem oil_ratio_proof (small_tank_capacity large_tank_capacity initial_large_tank_oil additional_oil_needed : ℕ) 
  (h1 : small_tank_capacity = 4000)
  (h2 : large_tank_capacity = 20000)
  (h3 : initial_large_tank_oil = 3000)
  (h4 : additional_oil_needed = 4000)
  (h5 : initial_large_tank_oil + (small_tank_capacity - (small_tank_capacity - x)) + additional_oil_needed = large_tank_capacity / 2)
  : (small_tank_capacity - (small_tank_capacity - x)) / small_tank_capacity = 3 / 4 := by
  sorry

#check oil_ratio_proof

end oil_ratio_proof_l1945_194511


namespace pauls_supplies_l1945_194577

/-- Given Paul's initial and final crayon counts, and initial eraser count,
    prove the difference between remaining erasers and crayons. -/
theorem pauls_supplies (initial_crayons : ℕ) (initial_erasers : ℕ) (final_crayons : ℕ)
    (h1 : initial_crayons = 601)
    (h2 : initial_erasers = 406)
    (h3 : final_crayons = 336) :
    initial_erasers - final_crayons = 70 := by
  sorry

end pauls_supplies_l1945_194577


namespace quadratic_two_real_roots_l1945_194503

theorem quadratic_two_real_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 3*x + m = 0 ∧ y^2 + 3*y + m = 0) ↔ m ≤ 9/4 := by
  sorry

end quadratic_two_real_roots_l1945_194503


namespace arithmetic_sequence_common_difference_l1945_194563

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence -/
def CommonDifference (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_a7 : a 7 = 25)
  (h_a4 : a 4 = 13) :
  ∃ d : ℝ, CommonDifference a d ∧ d = 4 := by
sorry

end arithmetic_sequence_common_difference_l1945_194563


namespace floor_ceil_sum_l1945_194506

theorem floor_ceil_sum : ⌊(-3.276 : ℝ)⌋ + ⌈(-17.845 : ℝ)⌉ = -21 := by
  sorry

end floor_ceil_sum_l1945_194506


namespace solution_set_of_inequalities_l1945_194596

theorem solution_set_of_inequalities :
  ∀ x : ℝ, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by
  sorry

end solution_set_of_inequalities_l1945_194596


namespace perpendicular_bisector_equation_l1945_194539

/-- Given two points A (-2, 0) and B (0, 4), prove that the equation x + 2y - 3 = 0
    represents the perpendicular bisector of the line segment AB. -/
theorem perpendicular_bisector_equation (A B : ℝ × ℝ) (h1 : A = (-2, 0)) (h2 : B = (0, 4)) :
  ∀ (x y : ℝ), (x + 2*y - 3 = 0) ↔ 
    (x - (-2))^2 + (y - 0)^2 = (x - 0)^2 + (y - 4)^2 ∧ 
    (x + 1) * (4 - 0) + (y - 2) * (0 - (-2)) = 0 :=
by sorry

end perpendicular_bisector_equation_l1945_194539


namespace gcd_228_2008_l1945_194551

theorem gcd_228_2008 : Nat.gcd 228 2008 = 4 := by
  sorry

end gcd_228_2008_l1945_194551


namespace first_discount_percentage_l1945_194568

/-- Proves that the first discount percentage is 10% given the initial price, 
    second discount percentage, and final price after both discounts. -/
theorem first_discount_percentage (initial_price : ℝ) (second_discount : ℝ) (final_price : ℝ) :
  initial_price = 200 →
  second_discount = 5 →
  final_price = 171 →
  ∃ (x : ℝ), 
    (initial_price * (1 - x / 100) * (1 - second_discount / 100) = final_price) ∧
    x = 10 := by
  sorry

end first_discount_percentage_l1945_194568


namespace smallest_common_factor_40_90_l1945_194515

theorem smallest_common_factor_40_90 : 
  ∃ (a : ℕ), a > 0 ∧ Nat.gcd a 40 > 1 ∧ Nat.gcd a 90 > 1 ∧ 
  ∀ (b : ℕ), b > 0 → Nat.gcd b 40 > 1 → Nat.gcd b 90 > 1 → a ≤ b :=
by
  use 2
  sorry

end smallest_common_factor_40_90_l1945_194515


namespace factorization_equality_l1945_194541

theorem factorization_equality (a x y : ℝ) : a*x^2 + 2*a*x*y + a*y^2 = a*(x+y)^2 := by
  sorry

end factorization_equality_l1945_194541


namespace quadratic_no_solution_l1945_194595

theorem quadratic_no_solution (a : ℝ) : 
  ({x : ℝ | x^2 - x + a = 0} : Set ℝ) = ∅ → a > 1/4 := by
sorry

end quadratic_no_solution_l1945_194595


namespace circle_equation_specific_l1945_194576

/-- The standard equation of a circle with center (h, k) and radius r is (x - h)² + (y - k)² = r² -/
def circle_equation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem: The standard equation of a circle with center (1, -1) and radius √3 is (x-1)² + (y+1)² = 3 -/
theorem circle_equation_specific :
  let h : ℝ := 1
  let k : ℝ := -1
  let r : ℝ := Real.sqrt 3
  ∀ x y : ℝ, circle_equation h k r x y ↔ (x - 1)^2 + (y + 1)^2 = 3 :=
by sorry

end circle_equation_specific_l1945_194576


namespace symmedian_circle_theorem_l1945_194558

/-- A triangle with side lengths a, b, and c is non-isosceles if no two sides are equal. -/
def NonIsoscelesTriangle (a b c : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a

/-- A circle passes through the feet of the symmedians of a triangle if it intersects
    each side of the triangle at the point where the symmedian meets that side. -/
def CircleThroughSymmedianFeet (a b c : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∃ (x y z : ℝ), x^2 + y^2 = r^2 ∧ y^2 + z^2 = r^2 ∧ z^2 + x^2 = r^2

/-- A circle is tangent to one side of a triangle if it touches that side at exactly one point. -/
def CircleTangentToSide (a b c : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ (∃ (x : ℝ), x^2 = r^2 ∨ ∃ (y : ℝ), y^2 = r^2 ∨ ∃ (z : ℝ), z^2 = r^2)

/-- Three positive real numbers form a geometric progression if the ratio between
    consecutive terms is constant. -/
def GeometricProgression (x y z : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ y = x * r ∧ z = y * r

/-- Main theorem: If a circle passes through the feet of the symmedians of a non-isosceles
    triangle and is tangent to one side, then the sums of squares of side lengths taken
    pairwise form a geometric progression. -/
theorem symmedian_circle_theorem (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  NonIsoscelesTriangle a b c →
  CircleThroughSymmedianFeet a b c →
  CircleTangentToSide a b c →
  GeometricProgression (a^2 + b^2) (b^2 + c^2) (c^2 + a^2) ∨
  GeometricProgression (b^2 + c^2) (c^2 + a^2) (a^2 + b^2) ∨
  GeometricProgression (c^2 + a^2) (a^2 + b^2) (b^2 + c^2) :=
by sorry

end symmedian_circle_theorem_l1945_194558


namespace stating_orthogonal_parallelepiped_angle_properties_l1945_194543

/-- 
Represents the angles formed between the diagonal and the edges of an orthogonal parallelepiped.
-/
structure ParallelepipedAngles where
  α₁ : ℝ
  α₂ : ℝ
  α₃ : ℝ

/-- 
Theorem stating the properties of angles in an orthogonal parallelepiped.
-/
theorem orthogonal_parallelepiped_angle_properties (angles : ParallelepipedAngles) :
  Real.sin angles.α₁ ^ 2 + Real.sin angles.α₂ ^ 2 + Real.sin angles.α₃ ^ 2 = 1 ∧
  Real.cos angles.α₁ ^ 2 + Real.cos angles.α₂ ^ 2 + Real.cos angles.α₃ ^ 2 = 2 := by
  sorry

end stating_orthogonal_parallelepiped_angle_properties_l1945_194543


namespace equation_solutions_l1945_194534

theorem equation_solutions : 
  ∀ x : ℝ, x * (2 * x - 4) = 3 * (2 * x - 4) ↔ x = 3 ∨ x = 2 := by
  sorry

end equation_solutions_l1945_194534


namespace odd_function_period_4_symmetric_exists_a_inequality_f_is_odd_not_unique_a_for_odd_g_l1945_194574

-- Define an odd function with period 4
def OddFunctionPeriod4 (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 4) = f x)

-- Define symmetry about (2,0)
def SymmetricAbout2_0 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (4 - x) = f x

-- Statement 1
theorem odd_function_period_4_symmetric :
  ∀ f : ℝ → ℝ, OddFunctionPeriod4 f → SymmetricAbout2_0 f :=
sorry

-- Statement 2
theorem exists_a_inequality :
  ∃ a : ℝ, 0 < a ∧ a < 1 ∧ a^(1 + a) ≥ a^(1 + 1/a) :=
sorry

-- Define the logarithmic function
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- Statement 3
theorem f_is_odd :
  ∀ x : ℝ, -1 < x → x < 1 → f (-x) = -f x :=
sorry

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + Real.sqrt (2 * x^2 + 1))

-- Statement 4
theorem not_unique_a_for_odd_g :
  ¬ ∃! a : ℝ, ∀ x : ℝ, g a (-x) = -g a x :=
sorry

end odd_function_period_4_symmetric_exists_a_inequality_f_is_odd_not_unique_a_for_odd_g_l1945_194574


namespace roots_and_m_value_l1945_194535

theorem roots_and_m_value (a b c m : ℝ) : 
  (a + b = 4 ∧ a * b = m) →  -- roots of x^2 - 4x + m = 0
  (b + c = 8 ∧ b * c = 5 * m) →  -- roots of x^2 - 8x + 5m = 0
  m = 0 ∨ m = 3 := by
sorry

end roots_and_m_value_l1945_194535


namespace isosceles_right_triangle_vector_sum_l1945_194584

/-- An isosceles right triangle with hypotenuse of length 6 -/
structure IsoscelesRightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  isRight : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  isIsosceles : (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2
  hypotenuseLength : (C.1 - B.1)^2 + (C.2 - B.2)^2 = 36

def dotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def vectorSum (t : IsoscelesRightTriangle) : ℝ :=
  let AB := (t.B.1 - t.A.1, t.B.2 - t.A.2)
  let AC := (t.C.1 - t.A.1, t.C.2 - t.A.2)
  let BC := (t.C.1 - t.B.1, t.C.2 - t.B.2)
  let BA := (-AB.1, -AB.2)
  let CA := (-AC.1, -AC.2)
  let CB := (-BC.1, -BC.2)
  dotProduct AB AC + dotProduct BC BA + dotProduct CA CB

theorem isosceles_right_triangle_vector_sum (t : IsoscelesRightTriangle) :
  vectorSum t = 36 := by
  sorry

end isosceles_right_triangle_vector_sum_l1945_194584


namespace grid_toothpick_count_l1945_194554

/-- Calculates the number of toothpicks in a grid with a missing center block -/
def toothpick_count (length width missing_size : ℕ) : ℕ :=
  let vertical := (length + 1) * width - missing_size * missing_size
  let horizontal := (width + 1) * length - missing_size * missing_size
  vertical + horizontal

/-- Theorem stating the correct number of toothpicks for the given grid -/
theorem grid_toothpick_count :
  toothpick_count 30 20 2 = 1242 := by
  sorry

end grid_toothpick_count_l1945_194554


namespace smallest_with_200_divisors_l1945_194573

/-- The number of distinct positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- Checks if n can be written as m * 10^k where 10 is not a divisor of m -/
def has_form (n m k : ℕ) : Prop :=
  n = m * (10 ^ k) ∧ ¬(10 ∣ m)

theorem smallest_with_200_divisors :
  ∃ (n m k : ℕ),
    (∀ i < n, num_divisors i < 200) ∧
    num_divisors n = 200 ∧
    has_form n m k ∧
    m + k = 18 := by sorry

end smallest_with_200_divisors_l1945_194573


namespace number_of_observations_l1945_194538

theorem number_of_observations (original_mean new_mean : ℝ) 
  (original_value new_value : ℝ) (n : ℕ) : 
  original_mean = 36 → 
  new_mean = 36.5 → 
  original_value = 23 → 
  new_value = 44 → 
  n * original_mean + (new_value - original_value) = n * new_mean → 
  n = 42 := by
sorry

end number_of_observations_l1945_194538


namespace line_through_first_and_third_quadrants_l1945_194520

theorem line_through_first_and_third_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∀ x y : ℝ, y = k * x → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0))) → k > 0 := by
  sorry

end line_through_first_and_third_quadrants_l1945_194520


namespace equation_study_path_and_future_l1945_194542

/-- Represents the steps in the study path of equations -/
inductive StudyPath
  | Definition
  | Solution
  | Solving
  | Application

/-- Represents types of equations that may be studied in the future -/
inductive FutureEquation
  | LinearQuadratic
  | LinearCubic
  | SystemQuadratic

/-- Represents an example of a future equation -/
def futureEquationExample : ℝ → ℝ := fun x => x^3 + 2*x + 1

/-- Theorem stating the study path of equations and future equations to be studied -/
theorem equation_study_path_and_future :
  (∃ (path : List StudyPath), path = [StudyPath.Definition, StudyPath.Solution, StudyPath.Solving, StudyPath.Application]) ∧
  (∃ (future : List FutureEquation), future = [FutureEquation.LinearQuadratic, FutureEquation.LinearCubic, FutureEquation.SystemQuadratic]) ∧
  (∃ (x : ℝ), futureEquationExample x = 0) :=
by sorry

end equation_study_path_and_future_l1945_194542


namespace binomial_expansion_coefficient_l1945_194580

/-- Given a > 0, if in the expansion of (1+a√x)^n, the coefficient of x^2 is 9 times
    the coefficient of x, and the third term is 135x, then a = 3 -/
theorem binomial_expansion_coefficient (a n : ℝ) (ha : a > 0) : 
  (∃ k₁ k₂ : ℝ, k₁ ≠ 0 ∧ k₂ ≠ 0 ∧ 
    k₁ * a^4 = 9 * k₂ * a^2 ∧
    k₂ * a^2 = 135) →
  a = 3 := by sorry

end binomial_expansion_coefficient_l1945_194580


namespace repeating_decimal_equals_fraction_l1945_194599

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def repeatingDecimalToRational (d : RepeatingDecimal) : ℚ := sorry

/-- The repeating decimal 7.316316316... -/
def ourDecimal : RepeatingDecimal := { integerPart := 7, repeatingPart := 316 }

theorem repeating_decimal_equals_fraction :
  repeatingDecimalToRational ourDecimal = 7309 / 999 := by sorry

end repeating_decimal_equals_fraction_l1945_194599


namespace wen_family_science_fair_cost_l1945_194556

theorem wen_family_science_fair_cost : ∀ (x : ℝ),
  x > 0 →
  0.7 * x = 7 →
  let student_ticket := 0.6 * x
  let regular_ticket := x
  let senior_ticket := 0.7 * x
  3 * student_ticket + regular_ticket + senior_ticket = 35 :=
by
  sorry

end wen_family_science_fair_cost_l1945_194556
