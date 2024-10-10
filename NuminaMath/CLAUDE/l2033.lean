import Mathlib

namespace butterfly_collection_l2033_203392

theorem butterfly_collection (total : ℕ) (black : ℕ) :
  total = 11 →
  black = 5 →
  ∃ (blue yellow : ℕ),
    blue = 2 * yellow ∧
    blue + yellow + black = total ∧
    blue = 4 := by
  sorry

end butterfly_collection_l2033_203392


namespace store_price_difference_l2033_203336

/-- Calculates the final price after applying a discount percentage to a full price -/
def final_price (full_price : ℚ) (discount_percent : ℚ) : ℚ :=
  full_price * (1 - discount_percent / 100)

/-- Proves that Store A's smartphone is $2 cheaper than Store B's after discounts -/
theorem store_price_difference (store_a_full_price store_b_full_price : ℚ)
  (store_a_discount store_b_discount : ℚ)
  (h1 : store_a_full_price = 125)
  (h2 : store_b_full_price = 130)
  (h3 : store_a_discount = 8)
  (h4 : store_b_discount = 10) :
  final_price store_b_full_price store_b_discount -
  final_price store_a_full_price store_a_discount = 2 := by
sorry

end store_price_difference_l2033_203336


namespace cd_length_sum_l2033_203304

theorem cd_length_sum : 
  let num_cds : ℕ := 3
  let regular_cd_length : ℝ := 1.5
  let long_cd_length : ℝ := 2 * regular_cd_length
  let total_length : ℝ := 2 * regular_cd_length + long_cd_length
  total_length = 6 := by sorry

end cd_length_sum_l2033_203304


namespace solve_for_y_l2033_203383

-- Define the variables
variable (n x y : ℝ)

-- Define the conditions
def condition1 : Prop := (n + 200 + 300 + x) / 4 = 250
def condition2 : Prop := (300 + 150 + n + x + y) / 5 = 200

-- Theorem statement
theorem solve_for_y (h1 : condition1 n x) (h2 : condition2 n x y) : y = 50 := by
  sorry

end solve_for_y_l2033_203383


namespace square_expression_is_perfect_square_l2033_203381

theorem square_expression_is_perfect_square (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  (2 * l - n - k) * (2 * l - n + k) / 2 = (l - n)^2 := by
  sorry

end square_expression_is_perfect_square_l2033_203381


namespace radical_product_equality_l2033_203333

theorem radical_product_equality : Real.sqrt 81 * Real.sqrt 16 * (64 ^ (1/4 : ℝ)) = 72 * Real.sqrt 2 := by
  sorry

end radical_product_equality_l2033_203333


namespace frog_corner_probability_l2033_203376

/-- Represents a position on the 3x3 grid -/
inductive Position
| Center
| Edge
| Corner

/-- Represents the number of hops made -/
def MaxHops : Nat := 4

/-- Probability of reaching a corner from a given position in n hops -/
noncomputable def reachCornerProb (pos : Position) (n : Nat) : Real :=
  sorry

/-- The main theorem to prove -/
theorem frog_corner_probability :
  reachCornerProb Position.Center MaxHops = 25 / 32 := by
  sorry

end frog_corner_probability_l2033_203376


namespace share_of_a_l2033_203332

theorem share_of_a (total : ℝ) (a b c : ℝ) : 
  total = 600 →
  a = (2/3) * (b + c) →
  b = (6/9) * (a + c) →
  a + b + c = total →
  a = 240 := by sorry

end share_of_a_l2033_203332


namespace barbara_candies_l2033_203311

/-- The number of candies Barbara used -/
def candies_used (initial : ℝ) (remaining : ℕ) : ℝ :=
  initial - remaining

theorem barbara_candies : 
  let initial : ℝ := 18.0
  let remaining : ℕ := 9
  candies_used initial remaining = 9 := by
  sorry

end barbara_candies_l2033_203311


namespace total_jump_distance_l2033_203343

/-- The total distance jumped by a grasshopper and a frog -/
def total_jump (grasshopper_jump frog_jump : ℕ) : ℕ :=
  grasshopper_jump + frog_jump

/-- Theorem: The total jump distance is 66 inches -/
theorem total_jump_distance :
  total_jump 31 35 = 66 := by
  sorry

end total_jump_distance_l2033_203343


namespace smaller_square_area_percentage_l2033_203364

/-- Represents a square inscribed in a circle with another smaller square -/
structure InscribedSquares where
  -- Radius of the circle
  r : ℝ
  -- Side length of the larger square
  s : ℝ
  -- Side length of the smaller square
  x : ℝ
  -- The larger square is inscribed in the circle
  h1 : r = s * Real.sqrt 2 / 2
  -- The smaller square has one side coinciding with the larger square
  h2 : x ≤ s
  -- Two vertices of the smaller square are on the circle
  h3 : (s/2 + x)^2 + x^2 = r^2

/-- The theorem stating that the area of the smaller square is 4% of the larger square -/
theorem smaller_square_area_percentage (sq : InscribedSquares) (h : sq.s = 4) :
  (sq.x^2) / (sq.s^2) = 0.04 := by
  sorry

end smaller_square_area_percentage_l2033_203364


namespace apple_banana_equivalence_l2033_203348

theorem apple_banana_equivalence (apple_value banana_value : ℚ) : 
  (3 / 4 * 12 : ℚ) * apple_value = 10 * banana_value →
  (2 / 3 * 9 : ℚ) * apple_value = (20 / 3 : ℚ) * banana_value := by
  sorry

end apple_banana_equivalence_l2033_203348


namespace evaluate_expression_l2033_203316

theorem evaluate_expression : 
  (3^1005 + 4^1006)^2 - (3^1005 - 4^1006)^2 = 16 * 12^1005 := by
  sorry

end evaluate_expression_l2033_203316


namespace f_eval_neg_one_l2033_203363

-- Define the polynomials f and g
def f (p q r : ℝ) (x : ℝ) : ℝ := x^4 + 2*x^3 + q*x^2 + 200*x + r
def g (p : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + 2*x + 20

-- State the theorem
theorem f_eval_neg_one (p q r : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g p x = 0 ∧ g p y = 0 ∧ g p z = 0) →
  (∀ x : ℝ, g p x = 0 → f p q r x = 0) →
  f p q r (-1) = -6319 :=
by sorry

end f_eval_neg_one_l2033_203363


namespace exterior_angle_regular_octagon_l2033_203346

theorem exterior_angle_regular_octagon :
  ∀ (n : ℕ) (interior_angle exterior_angle : ℝ),
    n = 8 →
    interior_angle = (180 * (n - 2 : ℝ)) / n →
    exterior_angle = 180 - interior_angle →
    exterior_angle = 45 := by
  sorry

end exterior_angle_regular_octagon_l2033_203346


namespace max_perfect_squares_l2033_203312

theorem max_perfect_squares (a b : ℕ) (h : a ≠ b) : 
  let products := [a * (a + 2), a * b, a * (b + 2), (a + 2) * b, (a + 2) * (b + 2), b * (b + 2)]
  (∃ (n : ℕ), n * n ∈ products) ∧ 
  ¬(∃ (m n : ℕ) (hm : m * m ∈ products) (hn : n * n ∈ products), m ≠ n) :=
by sorry

end max_perfect_squares_l2033_203312


namespace sixtieth_pair_is_five_seven_l2033_203320

/-- Represents a pair of integers -/
structure IntPair :=
  (first : ℤ)
  (second : ℤ)

/-- Generates the nth pair in the sequence -/
def generateNthPair (n : ℕ) : IntPair :=
  sorry

/-- The sequence of integer pairs as described in the problem -/
def sequencePairs : ℕ → IntPair :=
  generateNthPair

theorem sixtieth_pair_is_five_seven :
  sequencePairs 60 = IntPair.mk 5 7 := by
  sorry

end sixtieth_pair_is_five_seven_l2033_203320


namespace total_plums_picked_l2033_203313

-- Define the number of plums picked by each person
def melanie_plums : ℕ := 4
def dan_plums : ℕ := 9
def sally_plums : ℕ := 3

-- Define Ben's plums as twice the sum of Melanie's and Dan's
def ben_plums : ℕ := 2 * (melanie_plums + dan_plums)

-- Define the number of plums Sally ate
def sally_ate : ℕ := 2

-- Theorem: The total number of plums picked is 40
theorem total_plums_picked : 
  melanie_plums + dan_plums + sally_plums + ben_plums - sally_ate = 40 := by
  sorry

end total_plums_picked_l2033_203313


namespace richmond_victoria_difference_l2033_203323

def richmond_population : ℕ := 3000
def beacon_population : ℕ := 500
def victoria_population : ℕ := 4 * beacon_population

theorem richmond_victoria_difference : 
  richmond_population - victoria_population = 1000 ∧ richmond_population > victoria_population := by
  sorry

end richmond_victoria_difference_l2033_203323


namespace all_squares_similar_l2033_203369

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define similarity for squares
def similar (s1 s2 : Square) : Prop :=
  ∃ k : ℝ, k > 0 ∧ s1.side = k * s2.side

-- Theorem: Any two squares are similar
theorem all_squares_similar (s1 s2 : Square) : similar s1 s2 := by
  sorry


end all_squares_similar_l2033_203369


namespace train_length_equals_distance_traveled_l2033_203321

/-- Calculates the length of a train based on its speed and the time it takes to pass through a tunnel. -/
def train_length (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem stating that the length of a train is equal to the distance it travels while passing through a tunnel. -/
theorem train_length_equals_distance_traveled (speed : ℝ) (time : ℝ) :
  train_length speed time = speed * time :=
by
  sorry

#check train_length_equals_distance_traveled

end train_length_equals_distance_traveled_l2033_203321


namespace picture_books_count_l2033_203317

theorem picture_books_count (total : ℕ) (fiction : ℕ) 
  (h1 : total = 35)
  (h2 : fiction = 5)
  (h3 : total = fiction + (fiction + 4) + (2 * fiction) + picture_books) :
  picture_books = 11 := by
  sorry

end picture_books_count_l2033_203317


namespace range_of_a_l2033_203384

-- Define the propositions P and Q as functions of a
def P (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def Q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (5 - 2*a)^x < (5 - 2*a)^y

-- Define the theorem
theorem range_of_a : 
  (∀ a : ℝ, (P a ∨ Q a) ∧ ¬(P a ∧ Q a)) → 
  {a : ℝ | a ≤ -2} = {a : ℝ | ∀ a' : ℝ, (P a' ∨ Q a') ∧ ¬(P a' ∧ Q a') → a ≤ a'} :=
sorry

end range_of_a_l2033_203384


namespace binomial_variance_4_half_l2033_203306

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ :=
  ξ.n * ξ.p * (1 - ξ.p)

/-- Theorem: The variance of a binomial distribution B(4, 1/2) is 1 -/
theorem binomial_variance_4_half :
  ∀ ξ : BinomialDistribution, ξ.n = 4 ∧ ξ.p = 1/2 → variance ξ = 1 := by
  sorry

end binomial_variance_4_half_l2033_203306


namespace shape_area_is_94_l2033_203307

/-- A shape composed of three rectangles with given dimensions -/
structure Shape where
  rect1_width : ℕ
  rect1_height : ℕ
  rect2_width : ℕ
  rect2_height : ℕ
  rect3_width : ℕ
  rect3_height : ℕ

/-- Calculate the area of a rectangle -/
def rectangle_area (width height : ℕ) : ℕ := width * height

/-- Calculate the total area of the shape -/
def total_area (s : Shape) : ℕ :=
  rectangle_area s.rect1_width s.rect1_height +
  rectangle_area s.rect2_width s.rect2_height +
  rectangle_area s.rect3_width s.rect3_height

/-- The shape described in the problem -/
def problem_shape : Shape :=
  { rect1_width := 7
  , rect1_height := 7
  , rect2_width := 3
  , rect2_height := 5
  , rect3_width := 5
  , rect3_height := 6 }

theorem shape_area_is_94 : total_area problem_shape = 94 := by
  sorry


end shape_area_is_94_l2033_203307


namespace triangle_perimeter_l2033_203303

/-- Given a triangle with inradius 2.5 cm and area 50 cm², its perimeter is 40 cm. -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 2.5 → A = 50 → A = r * (p / 2) → p = 40 := by
  sorry

end triangle_perimeter_l2033_203303


namespace quadratic_one_solution_l2033_203368

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x, 3 * x^2 + m * x + 9 = 0) ↔ m = 6 * Real.sqrt 3 :=
by sorry

end quadratic_one_solution_l2033_203368


namespace part_one_part_two_l2033_203380

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Part 1
theorem part_one (t : Triangle) (h1 : t.a + t.b + t.c = 16) (h2 : t.a = 4) (h3 : t.b = 5) :
  Real.cos t.C = -1/5 := by sorry

-- Part 2
theorem part_two (t : Triangle) (h1 : t.a + t.b + t.c = 16) 
  (h2 : Real.sin t.A + Real.sin t.B = 3 * Real.sin t.C)
  (h3 : 1/2 * t.a * t.b * Real.sin t.C = 18 * Real.sin t.C) :
  t.a = 6 ∧ t.b = 6 := by sorry

end part_one_part_two_l2033_203380


namespace eleven_motorcycles_in_lot_l2033_203355

/-- Represents the number of motorcycles in a parking lot --/
def motorcycles_in_lot (total_wheels car_count : ℕ) : ℕ :=
  (total_wheels - 5 * car_count) / 2

/-- Theorem: Given the conditions in the problem, there are 11 motorcycles in the parking lot --/
theorem eleven_motorcycles_in_lot :
  motorcycles_in_lot 117 19 = 11 := by
  sorry

#eval motorcycles_in_lot 117 19

end eleven_motorcycles_in_lot_l2033_203355


namespace solve_equation_l2033_203361

theorem solve_equation (x : ℝ) : 3 * x = 2 * x + 6 → x = 6 := by
  sorry

end solve_equation_l2033_203361


namespace intersection_point_l2033_203354

/-- The point (3, 2) is the unique solution to the system of equations x + y = 5 and x - y = 1 -/
theorem intersection_point : ∃! p : ℝ × ℝ, p.1 + p.2 = 5 ∧ p.1 - p.2 = 1 ∧ p = (3, 2) := by
  sorry

end intersection_point_l2033_203354


namespace sphere_volume_from_surface_area_l2033_203388

/-- Given a sphere with surface area 256π cm², its volume is 2048/3 π cm³. -/
theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 4 * π * r^2 = 256 * π → (4 / 3) * π * r^3 = (2048 / 3) * π := by
  sorry

end sphere_volume_from_surface_area_l2033_203388


namespace q_is_false_l2033_203393

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q := by
  sorry

end q_is_false_l2033_203393


namespace light_bulb_probabilities_l2033_203389

/-- Represents a light bulb factory -/
inductive Factory
| A
| B

/-- Properties of the light bulb inventory -/
structure LightBulbInventory where
  total : ℕ
  factoryA_fraction : ℝ
  factoryB_fraction : ℝ
  factoryA_firstclass_rate : ℝ
  factoryB_firstclass_rate : ℝ

/-- The specific light bulb inventory in the problem -/
def problem_inventory : LightBulbInventory :=
  { total := 50
  , factoryA_fraction := 0.6
  , factoryB_fraction := 0.4
  , factoryA_firstclass_rate := 0.9
  , factoryB_firstclass_rate := 0.8
  }

/-- The probability of randomly selecting a first-class product from Factory A -/
def prob_firstclass_A (inv : LightBulbInventory) : ℝ :=
  inv.factoryA_fraction * inv.factoryA_firstclass_rate

/-- The expected value of first-class products from Factory A when selecting two light bulbs -/
def expected_firstclass_A_two_selections (inv : LightBulbInventory) : ℝ :=
  2 * prob_firstclass_A inv

/-- Main theorem stating the probabilities for the given inventory -/
theorem light_bulb_probabilities :
  prob_firstclass_A problem_inventory = 0.54 ∧
  expected_firstclass_A_two_selections problem_inventory = 1.08 := by
  sorry

end light_bulb_probabilities_l2033_203389


namespace absolute_value_inequality_l2033_203337

theorem absolute_value_inequality (x : ℝ) :
  |x - 2| + |x + 3| > 7 ↔ x < -4 ∨ x > 3 := by
sorry

end absolute_value_inequality_l2033_203337


namespace absolute_value_inequality_l2033_203360

theorem absolute_value_inequality (x y ε : ℝ) (h_pos : ε > 0) 
  (hx : |x - 2| < ε) (hy : |y - 2| < ε) : |x - y| < 2 * ε := by
  sorry

end absolute_value_inequality_l2033_203360


namespace modified_cube_surface_area_l2033_203373

/-- Represents a cube with given dimensions -/
structure Cube where
  size : ℕ
  deriving Repr

/-- Represents the modified cube structure after tunneling -/
structure ModifiedCube where
  original : Cube
  smallCubeSize : ℕ
  removedCenters : ℕ
  deriving Repr

/-- Calculates the surface area of the modified cube structure -/
def surfaceArea (mc : ModifiedCube) : ℕ :=
  sorry

/-- Theorem stating the surface area of the specific modified cube structure -/
theorem modified_cube_surface_area :
  let originalCube : Cube := { size := 12 }
  let modifiedCube : ModifiedCube := {
    original := originalCube,
    smallCubeSize := 2,
    removedCenters := 6
  }
  surfaceArea modifiedCube = 1824 := by
  sorry

end modified_cube_surface_area_l2033_203373


namespace residue_mod_17_l2033_203322

theorem residue_mod_17 : (245 * 15 - 18 * 8 + 5) % 17 = 0 := by
  sorry

end residue_mod_17_l2033_203322


namespace intersection_distance_l2033_203351

/-- The distance between intersection points of a line and a circle --/
theorem intersection_distance (t : ℝ) : 
  let x : ℝ → ℝ := λ t => -1 + (Real.sqrt 3 / 2) * t
  let y : ℝ → ℝ := λ t => (1 / 2) * t
  let l : ℝ → ℝ × ℝ := λ t => (x t, y t)
  let C : ℝ → ℝ := λ θ => 4 * Real.cos θ
  ∃ P Q : ℝ × ℝ, P ≠ Q ∧ 
    (P.1 - 2)^2 + P.2^2 = 4 ∧
    (Q.1 - 2)^2 + Q.2^2 = 4 ∧
    P.1 - Real.sqrt 3 * P.2 + 1 = 0 ∧
    Q.1 - Real.sqrt 3 * Q.2 + 1 = 0 ∧
    (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 7 :=
by
  sorry

end intersection_distance_l2033_203351


namespace square_last_digits_l2033_203370

theorem square_last_digits :
  (∃ n : ℕ, n^2 ≡ 444 [ZMOD 1000]) ∧
  (∀ k : ℤ, (1000*k + 38)^2 ≡ 444 [ZMOD 1000]) ∧
  (¬ ∃ n : ℤ, n^2 ≡ 4444 [ZMOD 10000]) := by
  sorry

end square_last_digits_l2033_203370


namespace travel_distance_theorem_l2033_203327

/-- The total distance Amoli and Anayet need to travel -/
def total_distance (amoli_speed : ℝ) (amoli_time : ℝ) (anayet_speed : ℝ) (anayet_time : ℝ) (remaining_distance : ℝ) : ℝ :=
  amoli_speed * amoli_time + anayet_speed * anayet_time + remaining_distance

/-- Theorem stating the total distance Amoli and Anayet need to travel -/
theorem travel_distance_theorem :
  let amoli_speed : ℝ := 42
  let amoli_time : ℝ := 3
  let anayet_speed : ℝ := 61
  let anayet_time : ℝ := 2
  let remaining_distance : ℝ := 121
  total_distance amoli_speed amoli_time anayet_speed anayet_time remaining_distance = 369 := by
sorry

end travel_distance_theorem_l2033_203327


namespace lucy_age_l2033_203366

/-- Given the ages of Inez, Zack, Jose, and Lucy, prove Lucy's age --/
theorem lucy_age (inez zack jose lucy : ℕ) 
  (h1 : lucy = jose + 2)
  (h2 : jose + 6 = zack)
  (h3 : zack = inez + 4)
  (h4 : inez = 18) : 
  lucy = 18 := by
  sorry

end lucy_age_l2033_203366


namespace positive_intervals_l2033_203328

-- Define the expression
def f (x : ℝ) : ℝ := (x + 1) * (x - 1) * (x - 2)

-- State the theorem
theorem positive_intervals (x : ℝ) : f x > 0 ↔ x ∈ Set.Ioo (-1) 1 ∪ Set.Ioi 2 :=
sorry

end positive_intervals_l2033_203328


namespace gcd_18_24_l2033_203356

theorem gcd_18_24 : Nat.gcd 18 24 = 6 := by
  sorry

end gcd_18_24_l2033_203356


namespace four_numbers_between_l2033_203301

theorem four_numbers_between :
  ∃ (a b c d : ℝ), 5.45 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < 5.47 := by
  sorry

end four_numbers_between_l2033_203301


namespace medal_distribution_theorem_l2033_203347

/-- The number of ways to distribute medals to students -/
def distribute_medals (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of students -/
def num_students : ℕ := 12

/-- The number of medal types -/
def num_medal_types : ℕ := 3

/-- The number of ways to distribute medals -/
def num_distributions : ℕ := distribute_medals (num_students - num_medal_types) num_medal_types

theorem medal_distribution_theorem : num_distributions = 55 := by
  sorry

end medal_distribution_theorem_l2033_203347


namespace triangle_angle_sum_l2033_203390

theorem triangle_angle_sum (first_angle second_angle third_angle : ℝ) : 
  second_angle = 2 * first_angle →
  third_angle = 15 →
  first_angle = third_angle + 40 →
  first_angle + second_angle = 165 := by
sorry

end triangle_angle_sum_l2033_203390


namespace triangle_median_altitude_equations_l2033_203371

/-- Triangle ABC in the Cartesian coordinate system -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of the median from a vertex to the opposite side -/
def median (t : Triangle) (v : ℝ × ℝ) : Line := sorry

/-- Definition of the altitude from a vertex to the opposite side -/
def altitude (t : Triangle) (v : ℝ × ℝ) : Line := sorry

theorem triangle_median_altitude_equations :
  let t : Triangle := { A := (7, 8), B := (10, 4), C := (2, -4) }
  (median t t.B = { a := 8, b := -1, c := -48 }) ∧
  (altitude t t.B = { a := 1, b := 1, c := -15 }) := by sorry

end triangle_median_altitude_equations_l2033_203371


namespace hyperbola_line_intersection_l2033_203331

/-- A hyperbola with eccentricity √3 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0
  h_ecc : b^2 = 2 * a^2

/-- A line with slope 1 -/
structure Line where
  k : ℝ

/-- Points P and Q on the hyperbola, and R on the y-axis -/
structure Points (h : Hyperbola) (l : Line) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  h_on_hyperbola : 
    (P.1^2 / h.a^2 - P.2^2 / h.b^2 = 1) ∧
    (Q.1^2 / h.a^2 - Q.2^2 / h.b^2 = 1)
  h_on_line : 
    (P.2 = P.1 + l.k) ∧
    (Q.2 = Q.1 + l.k)
  h_R : R = (0, l.k)
  h_dot_product : P.1 * Q.1 + P.2 * Q.2 = -3
  h_vector_ratio : (Q.1 - P.1, Q.2 - P.2) = (4 * (Q.1 - R.1), 4 * (Q.2 - R.2))

theorem hyperbola_line_intersection 
  (h : Hyperbola) (l : Line) (pts : Points h l) :
  (l.k = 1 ∨ l.k = -1) ∧ h.a = 1 := by sorry

#check hyperbola_line_intersection

end hyperbola_line_intersection_l2033_203331


namespace smallest_days_to_triple_l2033_203372

def borrowed_amount : ℝ := 20
def interest_rate : ℝ := 0.12

def amount_owed (days : ℕ) : ℝ :=
  borrowed_amount + borrowed_amount * interest_rate * days

def is_at_least_triple (days : ℕ) : Prop :=
  amount_owed days ≥ 3 * borrowed_amount

theorem smallest_days_to_triple : 
  (∀ d : ℕ, d < 17 → ¬(is_at_least_triple d)) ∧ 
  (is_at_least_triple 17) :=
sorry

end smallest_days_to_triple_l2033_203372


namespace tennis_net_max_cuts_l2033_203357

/-- Represents a grid of squares -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Calculates the total number of edges in a grid -/
def total_edges (g : Grid) : ℕ :=
  (g.rows + 1) * g.cols + (g.cols + 1) * g.rows

/-- Calculates the maximum number of edges that can be cut without disconnecting the grid -/
def max_cuttable_edges (g : Grid) : ℕ :=
  (g.rows - 1) * (g.cols - 1)

/-- Theorem stating that for a 100 × 10 grid, the maximum number of cuttable edges is 891 -/
theorem tennis_net_max_cuts :
  let g : Grid := ⟨10, 100⟩
  max_cuttable_edges g = 891 :=
by sorry

end tennis_net_max_cuts_l2033_203357


namespace a_greater_than_b_l2033_203394

open Real

theorem a_greater_than_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : exp a + 2*a = exp b + 3*b) : a > b := by
  sorry

end a_greater_than_b_l2033_203394


namespace lines_perpendicular_to_parallel_planes_l2033_203342

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- State the theorem
theorem lines_perpendicular_to_parallel_planes 
  (m n : Line) (α β : Plane) :
  m ≠ n →
  α ≠ β →
  parallel m n →
  perpendicular m α →
  perpendicular n β →
  plane_parallel α β :=
sorry

end lines_perpendicular_to_parallel_planes_l2033_203342


namespace right_triangle_with_given_sides_l2033_203310

theorem right_triangle_with_given_sides :
  ∃ (a b c : ℝ), a = 8 ∧ b = 15 ∧ c = Real.sqrt 161 ∧ a^2 + b^2 = c^2 :=
by sorry

end right_triangle_with_given_sides_l2033_203310


namespace dice_collinearity_probability_l2033_203308

def dice_roll := Finset.range 6

def vector_p (m n : ℕ) := (m, n)
def vector_q := (3, 6)

def is_collinear (p q : ℕ × ℕ) : Prop :=
  p.1 * q.2 = p.2 * q.1

def collinear_outcomes : Finset (ℕ × ℕ) :=
  {(1, 2), (2, 4), (3, 6)}

theorem dice_collinearity_probability :
  (collinear_outcomes.card : ℚ) / (dice_roll.card * dice_roll.card : ℚ) = 1 / 12 :=
sorry

end dice_collinearity_probability_l2033_203308


namespace bowTie_equation_solution_l2033_203340

-- Define the bow tie operation
noncomputable def bowTie (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem bowTie_equation_solution (y : ℝ) : bowTie 4 y = 10 → y = 30 := by
  sorry

end bowTie_equation_solution_l2033_203340


namespace min_value_expression_l2033_203300

theorem min_value_expression (x : ℝ) (h : x > 10) : 
  x^2 / (x - 10) ≥ 40 ∧ ∃ x₀ > 10, x₀^2 / (x₀ - 10) = 40 := by
  sorry

end min_value_expression_l2033_203300


namespace symmetric_point_coordinates_l2033_203344

/-- Given two points M and N in a 2D plane, where N is symmetric to M about the y-axis,
    this theorem proves that the coordinates of M with respect to N are (2, 1) when M has coordinates (-2, 1). -/
theorem symmetric_point_coordinates (M N : ℝ × ℝ) :
  M = (-2, 1) →
  N.1 = -M.1 ∧ N.2 = M.2 →
  (M.1 - N.1, M.2 - N.2) = (2, 1) := by
  sorry

end symmetric_point_coordinates_l2033_203344


namespace cubic_inequality_solution_l2033_203335

theorem cubic_inequality_solution (x : ℝ) : 
  x^3 - 12*x^2 + 35*x + 48 < 0 ↔ x ∈ Set.Ioo (-1 : ℝ) 3 ∪ Set.Ioi 16 := by
  sorry

end cubic_inequality_solution_l2033_203335


namespace perpendicular_vector_scalar_l2033_203309

/-- Given plane vectors a and b, if m * a + b is perpendicular to a, then m = 1 -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (m : ℝ) 
  (h1 : a = (-1, 3)) 
  (h2 : b = (4, -2)) 
  (h3 : (m * a.1 + b.1) * a.1 + (m * a.2 + b.2) * a.2 = 0) : 
  m = 1 := by
  sorry

end perpendicular_vector_scalar_l2033_203309


namespace beach_trip_seashells_l2033_203302

/-- Calculates the total number of seashells found during a beach trip -/
def total_seashells (days : ℕ) (shells_per_day : ℕ) : ℕ :=
  days * shells_per_day

theorem beach_trip_seashells :
  let days : ℕ := 5
  let shells_per_day : ℕ := 7
  total_seashells days shells_per_day = 35 := by
  sorry

end beach_trip_seashells_l2033_203302


namespace quadratic_inequality_solution_l2033_203386

-- Define the quadratic function
def f (a b x : ℝ) := x^2 - a*x + b

-- Define the solution set
def solution_set (a b : ℝ) := {x : ℝ | f a b x < 0}

-- State the theorem
theorem quadratic_inequality_solution (a b : ℝ) :
  solution_set a b = {x : ℝ | -1 < x ∧ x < 3} →
  a + b = -1 := by sorry

end quadratic_inequality_solution_l2033_203386


namespace pythagorean_diagonal_l2033_203382

/-- 
For a right triangle with width 2m (where m ≥ 3 and m is a positive integer) 
and the difference between the diagonal and the height being 2, 
the diagonal is equal to m² - 1.
-/
theorem pythagorean_diagonal (m : ℕ) (h : m ≥ 3) : 
  let width : ℕ := 2 * m
  let diagonal : ℕ := m^2 - 1
  let height : ℕ := diagonal - 2
  width^2 + height^2 = diagonal^2 := by sorry

end pythagorean_diagonal_l2033_203382


namespace theater_rows_l2033_203341

/-- Represents the number of rows in the theater. -/
def num_rows : ℕ := sorry

/-- Represents the number of students in the first condition. -/
def students_first_condition : ℕ := 30

/-- Represents the number of students in the second condition. -/
def students_second_condition : ℕ := 26

/-- Represents the minimum number of empty rows in the second condition. -/
def min_empty_rows : ℕ := 3

theorem theater_rows :
  (∀ (seating : Fin students_first_condition → Fin num_rows),
    ∃ (row : Fin num_rows) (s1 s2 : Fin students_first_condition),
      s1 ≠ s2 ∧ seating s1 = seating s2) ∧
  (∀ (seating : Fin students_second_condition → Fin num_rows),
    ∃ (empty_rows : Finset (Fin num_rows)),
      empty_rows.card ≥ min_empty_rows ∧
      ∀ (row : Fin num_rows),
        row ∈ empty_rows ↔ ∀ (s : Fin students_second_condition), seating s ≠ row) →
  num_rows = 29 :=
sorry

end theater_rows_l2033_203341


namespace cubic_factorization_l2033_203378

theorem cubic_factorization (x : ℝ) : x^3 - 9*x = x*(x+3)*(x-3) := by
  sorry

end cubic_factorization_l2033_203378


namespace lecture_arrangements_l2033_203399

theorem lecture_arrangements (n : ℕ) (h : n = 3) : Nat.factorial n = 6 := by
  sorry

end lecture_arrangements_l2033_203399


namespace seven_people_arrangement_l2033_203395

def number_of_people : ℕ := 7
def number_of_special_people : ℕ := 3

def arrangements (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem seven_people_arrangement :
  let regular_people := number_of_people - number_of_special_people
  let gaps := regular_people + 1
  arrangements regular_people regular_people * arrangements gaps number_of_special_people = 1440 :=
sorry

end seven_people_arrangement_l2033_203395


namespace intersection_empty_implies_m_range_l2033_203334

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | x^2 + x + m + 2 = 0}
def B : Set ℝ := {x | x > 0}

-- Theorem statement
theorem intersection_empty_implies_m_range (m : ℝ) :
  A m ∩ B = ∅ → m ≤ -2 := by
  sorry


end intersection_empty_implies_m_range_l2033_203334


namespace union_of_A_and_B_l2033_203325

def A : Set ℤ := {-1, 2, 3, 5}
def B : Set ℤ := {2, 4, 5}

theorem union_of_A_and_B : A ∪ B = {-1, 2, 3, 4, 5} := by
  sorry

end union_of_A_and_B_l2033_203325


namespace sales_solution_l2033_203326

def sales_problem (sales1 sales2 sales3 sales4 desired_average : ℕ) : Prop :=
  let total_months : ℕ := 5
  let known_sales_sum : ℕ := sales1 + sales2 + sales3 + sales4
  let total_required : ℕ := desired_average * total_months
  let fifth_month_sales : ℕ := total_required - known_sales_sum
  fifth_month_sales = 7870

theorem sales_solution :
  sales_problem 5420 5660 6200 6350 6300 :=
by
  sorry

end sales_solution_l2033_203326


namespace largest_angle_in_triangle_l2033_203345

theorem largest_angle_in_triangle (x : ℝ) : 
  x + 50 + 55 = 180 → 
  max x (max 50 55) = 75 :=
by sorry

end largest_angle_in_triangle_l2033_203345


namespace quadratic_form_and_sum_l2033_203339

theorem quadratic_form_and_sum (x : ℝ) : ∃! (a b c : ℝ),
  (6 * x^2 + 48 * x + 300 = a * (x + b)^2 + c) ∧
  (a + b + c = 214) := by
  sorry

end quadratic_form_and_sum_l2033_203339


namespace min_value_triangle_ratio_l2033_203319

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if c - a = 2a * cos B, then the minimum possible value of (3a + c) / b is 2√2. -/
theorem min_value_triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  c - a = 2 * a * Real.cos B →
  ∃ (m : ℝ), m = 2 * Real.sqrt 2 ∧ ∀ (x : ℝ), (3 * a + c) / b ≥ m :=
by sorry

end min_value_triangle_ratio_l2033_203319


namespace roden_fish_count_l2033_203315

/-- The number of gold fish Roden bought -/
def gold_fish : ℕ := 15

/-- The number of blue fish Roden bought -/
def blue_fish : ℕ := 7

/-- The total number of fish Roden bought -/
def total_fish : ℕ := gold_fish + blue_fish

theorem roden_fish_count : total_fish = 22 := by
  sorry

end roden_fish_count_l2033_203315


namespace beth_dive_tanks_l2033_203397

/-- Calculates the number of supplemental tanks needed for a scuba dive. -/
def supplementalTanksNeeded (totalDiveTime primaryTankDuration supplementalTankDuration : ℕ) : ℕ :=
  (totalDiveTime - primaryTankDuration) / supplementalTankDuration

/-- Proves that for the given dive parameters, 6 supplemental tanks are needed. -/
theorem beth_dive_tanks : 
  supplementalTanksNeeded 8 2 1 = 6 := by
  sorry

#eval supplementalTanksNeeded 8 2 1

end beth_dive_tanks_l2033_203397


namespace smallest_dual_palindrome_seventeen_is_dual_palindrome_seventeen_is_smallest_dual_palindrome_l2033_203396

/-- Checks if a natural number is a palindrome in the given base. -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in the given base. -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_palindrome : 
  ∀ k : ℕ, k > 15 → 
    (isPalindrome k 2 ∧ isPalindrome k 4) → 
    k ≥ 17 := by sorry

theorem seventeen_is_dual_palindrome : 
  isPalindrome 17 2 ∧ isPalindrome 17 4 := by sorry

theorem seventeen_is_smallest_dual_palindrome : 
  ∀ k : ℕ, k > 15 → 
    (isPalindrome k 2 ∧ isPalindrome k 4) → 
    k = 17 :=
by sorry

end smallest_dual_palindrome_seventeen_is_dual_palindrome_seventeen_is_smallest_dual_palindrome_l2033_203396


namespace apples_in_good_condition_l2033_203329

-- Define the total number of apples
def total_apples : ℕ := 75

-- Define the percentage of rotten apples
def rotten_percentage : ℚ := 12 / 100

-- Define the number of apples in good condition
def good_apples : ℕ := 66

-- Theorem statement
theorem apples_in_good_condition :
  (total_apples : ℚ) * (1 - rotten_percentage) = good_apples := by
  sorry

end apples_in_good_condition_l2033_203329


namespace stacy_paper_pages_per_day_l2033_203352

/-- Given a paper with a certain number of pages due in a certain number of days,
    calculate the number of pages that need to be written per day to finish on time. -/
def pages_per_day (total_pages : ℕ) (days : ℕ) : ℚ :=
  total_pages / days

/-- Theorem stating that for a 100-page paper due in 5 days,
    the number of pages to be written per day is 20. -/
theorem stacy_paper_pages_per_day :
  pages_per_day 100 5 = 20 := by
  sorry

end stacy_paper_pages_per_day_l2033_203352


namespace three_white_marbles_possible_l2033_203367

/-- Represents the possible operations on the urn --/
inductive Operation
  | op1 : Operation  -- Remove 4 black, add 2 black
  | op2 : Operation  -- Remove 3 black and 1 white, add 1 black
  | op3 : Operation  -- Remove 2 black and 2 white, add 2 white and 1 black
  | op4 : Operation  -- Remove 1 black and 3 white, add 3 white
  | op5 : Operation  -- Remove 4 white, add 2 black and 1 white

/-- Represents the state of the urn --/
structure UrnState :=
  (white : ℕ)
  (black : ℕ)

/-- Applies an operation to the urn state --/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.op1 => ⟨state.white, state.black - 2⟩
  | Operation.op2 => ⟨state.white - 1, state.black - 2⟩
  | Operation.op3 => ⟨state.white, state.black - 1⟩
  | Operation.op4 => ⟨state.white, state.black - 1⟩
  | Operation.op5 => ⟨state.white - 3, state.black + 2⟩

/-- Theorem: It's possible to reach a state with 3 white marbles --/
theorem three_white_marbles_possible :
  ∃ (ops : List Operation), 
    let finalState := ops.foldl applyOperation ⟨150, 150⟩
    finalState.white = 3 := by
  sorry

end three_white_marbles_possible_l2033_203367


namespace unit_digit_of_product_l2033_203365

theorem unit_digit_of_product : (5 + 1) * (5^3 + 1) * (5^6 + 1) * (5^12 + 1) % 10 = 6 := by
  sorry

end unit_digit_of_product_l2033_203365


namespace broker_commission_slump_l2033_203398

theorem broker_commission_slump (initial_rate final_rate : ℝ) 
  (initial_business final_business : ℝ) (h1 : initial_rate = 0.04) 
  (h2 : final_rate = 0.05) (h3 : initial_rate * initial_business = final_rate * final_business) :
  (initial_business - final_business) / initial_business = 0.2 := by
  sorry

end broker_commission_slump_l2033_203398


namespace perfect_square_theorem_l2033_203338

/-- A function that checks if a number is a 3-digit number -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A function that represents the 6-digit number formed by n and its successor -/
def sixDigitNumber (n : ℕ) : ℕ := 1001 * n + 1

/-- The set of valid 3-digit numbers satisfying the condition -/
def validNumbers : Set ℕ := {183, 328, 528, 715}

/-- Theorem stating that the set of 3-digit numbers n such that 1001n + 1 
    is a perfect square is exactly the set {183, 328, 528, 715} -/
theorem perfect_square_theorem :
  ∀ n : ℕ, isThreeDigit n ∧ (∃ m : ℕ, sixDigitNumber n = m ^ 2) ↔ n ∈ validNumbers := by
  sorry

end perfect_square_theorem_l2033_203338


namespace minimum_sum_l2033_203387

theorem minimum_sum (x y z : ℝ) (h : (4 / x) + (2 / y) + (1 / z) = 1) :
  x + 8 * y + 4 * z ≥ 64 ∧
  (x + 8 * y + 4 * z = 64 ↔ x = 16 ∧ y = 4 ∧ z = 4) := by
  sorry

end minimum_sum_l2033_203387


namespace deck_cost_l2033_203374

def rare_cards : ℕ := 19
def uncommon_cards : ℕ := 11
def common_cards : ℕ := 30

def rare_cost : ℚ := 1
def uncommon_cost : ℚ := 0.5
def common_cost : ℚ := 0.25

def total_cost : ℚ := rare_cards * rare_cost + uncommon_cards * uncommon_cost + common_cards * common_cost

theorem deck_cost : total_cost = 32 := by sorry

end deck_cost_l2033_203374


namespace big_toenail_count_l2033_203375

/-- Represents the capacity and contents of a toenail jar -/
structure ToenailJar where
  capacity : ℕ  -- Total capacity in terms of regular toenails
  regularSize : ℕ  -- Size of a regular toenail (set to 1)
  bigSize : ℕ  -- Size of a big toenail
  regularCount : ℕ  -- Number of regular toenails in the jar
  remainingSpace : ℕ  -- Remaining space in terms of regular toenails
  bigCount : ℕ  -- Number of big toenails in the jar

/-- Theorem stating the number of big toenails in the jar -/
theorem big_toenail_count (jar : ToenailJar)
  (h1 : jar.capacity = 100)
  (h2 : jar.bigSize = 2 * jar.regularSize)
  (h3 : jar.regularCount = 40)
  (h4 : jar.remainingSpace = 20)
  : jar.bigCount = 10 := by
  sorry

end big_toenail_count_l2033_203375


namespace probability_product_216_l2033_203349

def standard_die := Finset.range 6

def roll_product (x y z : ℕ) : ℕ := x * y * z

theorem probability_product_216 :
  (Finset.filter (λ (t : ℕ × ℕ × ℕ) => roll_product t.1 t.2.1 t.2.2 = 216) 
    (standard_die.product (standard_die.product standard_die))).card / 
  (standard_die.card ^ 3 : ℚ) = 1 / 216 :=
sorry

end probability_product_216_l2033_203349


namespace eighth_term_is_one_l2033_203358

-- Define the sequence a_n
def a (n : ℕ+) : ℤ := (-1) ^ n.val

-- Theorem statement
theorem eighth_term_is_one : a 8 = 1 := by
  sorry

end eighth_term_is_one_l2033_203358


namespace miles_driven_with_thirty_dollars_l2033_203379

theorem miles_driven_with_thirty_dollars (miles_per_gallon : ℝ) (dollars_per_gallon : ℝ) (budget : ℝ) :
  miles_per_gallon = 40 →
  dollars_per_gallon = 4 →
  budget = 30 →
  (budget / dollars_per_gallon) * miles_per_gallon = 300 :=
by
  sorry

end miles_driven_with_thirty_dollars_l2033_203379


namespace complex_magnitude_product_l2033_203330

theorem complex_magnitude_product : 
  Complex.abs ((3 * Real.sqrt 2 - 3 * Complex.I) * (2 * Real.sqrt 5 + 5 * Complex.I)) = 9 * Real.sqrt 15 := by
  sorry

end complex_magnitude_product_l2033_203330


namespace dog_hare_speed_ratio_challenging_terrain_l2033_203324

/-- Represents the ratio of dog leaps to hare leaps -/
def dogHareLeapRatio : ℚ := 10 / 2

/-- Represents the ratio of dog leap distance to hare leap distance -/
def dogHareDistanceRatio : ℚ := 2 / 1

/-- Represents the reduction factor of dog's leap distance on challenging terrain -/
def dogReductionFactor : ℚ := 3 / 4

/-- Represents the reduction factor of hare's leap distance on challenging terrain -/
def hareReductionFactor : ℚ := 1 / 2

/-- Theorem stating the speed ratio of dog to hare on challenging terrain -/
theorem dog_hare_speed_ratio_challenging_terrain :
  (dogHareDistanceRatio * dogReductionFactor) / hareReductionFactor = 3 / 1 := by
  sorry

end dog_hare_speed_ratio_challenging_terrain_l2033_203324


namespace coefficient_x2y2_in_expansion_expansion_coefficient_is_18_l2033_203359

theorem coefficient_x2y2_in_expansion : ℕ → Prop :=
  fun n => (Finset.sum (Finset.range 4) fun i =>
    (Finset.sum (Finset.range 5) fun j =>
      if i + j = 4 then
        (Nat.choose 3 i) * (Nat.choose 4 j)
      else
        0)) = n

theorem expansion_coefficient_is_18 : coefficient_x2y2_in_expansion 18 := by
  sorry

end coefficient_x2y2_in_expansion_expansion_coefficient_is_18_l2033_203359


namespace no_linear_term_implies_k_equals_four_l2033_203362

theorem no_linear_term_implies_k_equals_four (k : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x + k) * (x - 4) = a * x^2 + b) → k = 4 := by
  sorry

end no_linear_term_implies_k_equals_four_l2033_203362


namespace flat_terrain_distance_l2033_203391

theorem flat_terrain_distance (total_time : ℚ) (total_distance : ℕ) 
  (speed_uphill speed_flat speed_downhill : ℚ) 
  (h_total_time : total_time = 29 / 15)
  (h_total_distance : total_distance = 9)
  (h_speed_uphill : speed_uphill = 4)
  (h_speed_flat : speed_flat = 5)
  (h_speed_downhill : speed_downhill = 6) :
  ∃ (x y : ℕ), 
    x + y ≤ total_distance ∧
    x / speed_uphill + y / speed_flat + (total_distance - x - y) / speed_downhill = total_time ∧
    y = 3 := by
  sorry

end flat_terrain_distance_l2033_203391


namespace open_box_volume_l2033_203318

/-- The volume of an open box constructed from a rectangular sheet --/
def boxVolume (sheetLength sheetWidth x : ℝ) : ℝ :=
  (sheetLength - 2*x) * (sheetWidth - 2*x) * x

theorem open_box_volume :
  ∀ x : ℝ, 1 ≤ x → x ≤ 3 →
  boxVolume 16 12 x = 4*x^3 - 56*x^2 + 192*x :=
by sorry

end open_box_volume_l2033_203318


namespace girls_only_wind_count_l2033_203350

/-- Represents the number of students in different categories of the school bands -/
structure BandParticipation where
  wind_boys : ℕ
  wind_girls : ℕ
  string_boys : ℕ
  string_girls : ℕ
  total_students : ℕ
  boys_in_both : ℕ

/-- Calculates the number of girls participating only in the wind band -/
def girls_only_wind (bp : BandParticipation) : ℕ :=
  bp.wind_girls - (bp.total_students - (bp.wind_boys + bp.wind_girls + bp.string_boys + bp.string_girls - bp.boys_in_both) - bp.boys_in_both)

/-- The main theorem stating that given the specific band participation numbers, 
    the number of girls participating only in the wind band is 10 -/
theorem girls_only_wind_count : 
  let bp : BandParticipation := {
    wind_boys := 100,
    wind_girls := 80,
    string_boys := 80,
    string_girls := 100,
    total_students := 230,
    boys_in_both := 60
  }
  girls_only_wind bp = 10 := by sorry

end girls_only_wind_count_l2033_203350


namespace system_solution_l2033_203305

theorem system_solution (x y : ℝ) : 
  (3 * x^2 + 9 * x + 3 * y + 2 = 0 ∧ 3 * x + y + 4 = 0) ↔ 
  (y = -4 + Real.sqrt 30 ∨ y = -4 - Real.sqrt 30) := by
sorry

end system_solution_l2033_203305


namespace num_valid_schedules_is_336_l2033_203314

/-- Represents the days of the week excluding Saturday -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

/-- Represents the teachers -/
inductive Teacher
| Math
| English
| Other1
| Other2
| Other3
| Other4

/-- A schedule is a function from Teacher to Day -/
def Schedule := Teacher → Day

/-- Predicate to check if a schedule is valid -/
def validSchedule (s : Schedule) : Prop :=
  s Teacher.Math ≠ Day.Monday ∧
  s Teacher.Math ≠ Day.Wednesday ∧
  s Teacher.English ≠ Day.Tuesday ∧
  s Teacher.English ≠ Day.Thursday ∧
  (∀ t1 t2 : Teacher, t1 ≠ t2 → s t1 ≠ s t2)

/-- The number of valid schedules -/
def numValidSchedules : ℕ := sorry

theorem num_valid_schedules_is_336 : numValidSchedules = 336 := by sorry

end num_valid_schedules_is_336_l2033_203314


namespace book_purchase_problem_l2033_203377

theorem book_purchase_problem (total_volumes : ℕ) (paperback_price hardcover_price : ℚ) 
  (discount : ℚ) (total_cost : ℚ) :
  total_volumes = 12 ∧ 
  paperback_price = 16 ∧ 
  hardcover_price = 27 ∧ 
  discount = 6 ∧ 
  total_cost = 278 →
  ∃ (h : ℕ), 
    h = 8 ∧ 
    h ≤ total_volumes ∧ 
    (h > 5 → hardcover_price * h + paperback_price * (total_volumes - h) - discount = total_cost) ∧
    (h ≤ 5 → hardcover_price * h + paperback_price * (total_volumes - h) = total_cost) :=
by sorry

end book_purchase_problem_l2033_203377


namespace inscribed_trapezoid_theorem_l2033_203385

/-- An isosceles trapezoid inscribed in a circle -/
structure InscribedTrapezoid where
  r : ℝ  -- radius of the circle
  a : ℝ  -- half of the shorter base
  c : ℝ  -- half of the longer base
  h : 0 < r ∧ 0 < a ∧ 0 < c ∧ a < c  -- conditions for a valid trapezoid

/-- Theorem: For an isosceles trapezoid inscribed in a circle, r^2 = ac -/
theorem inscribed_trapezoid_theorem (t : InscribedTrapezoid) : t.r^2 = t.a * t.c := by
  sorry

end inscribed_trapezoid_theorem_l2033_203385


namespace dot_movement_l2033_203353

-- Define a square
structure Square where
  side : ℝ
  center : ℝ × ℝ

-- Define a point on the square
structure Point where
  x : ℝ
  y : ℝ

-- Define the operations
def fold_diagonal (s : Square) (p : Point) : Point :=
  sorry

def rotate_90_clockwise (s : Square) (p : Point) : Point :=
  sorry

def unfold (s : Square) (p : Point) : Point :=
  sorry

-- Define the initial and final positions
def top_right (s : Square) : Point :=
  sorry

def top_center (s : Square) : Point :=
  sorry

-- Theorem statement
theorem dot_movement (s : Square) :
  let initial_pos := top_right s
  let folded_pos := fold_diagonal s initial_pos
  let rotated_pos := rotate_90_clockwise s folded_pos
  let final_pos := unfold s rotated_pos
  final_pos = top_center s :=
sorry

end dot_movement_l2033_203353
