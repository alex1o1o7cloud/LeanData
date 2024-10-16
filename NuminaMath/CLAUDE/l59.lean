import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_side_difference_l59_5911

theorem rectangle_side_difference (p d : ℝ) (h_positive : p > 0 ∧ d > 0) :
  ∃ (x y : ℝ), 
    x > 0 ∧ y > 0 ∧
    x = 2 * y ∧
    2 * (x + y) = p ∧
    x^2 + y^2 = d^2 ∧
    x - y = p / 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_difference_l59_5911


namespace NUMINAMATH_CALUDE_pet_store_combinations_l59_5972

def num_puppies : ℕ := 12
def num_kittens : ℕ := 10
def num_hamsters : ℕ := 5
def num_birds : ℕ := 3
def num_people : ℕ := 4

def ways_to_choose_pets : ℕ := num_puppies * num_kittens * num_hamsters * num_birds

def permutations_of_choices : ℕ := Nat.factorial num_people

theorem pet_store_combinations : 
  ways_to_choose_pets * permutations_of_choices = 43200 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l59_5972


namespace NUMINAMATH_CALUDE_sqrt_sum_irrational_l59_5984

theorem sqrt_sum_irrational (a b : ℚ) 
  (ha : Irrational (Real.sqrt a)) 
  (hb : Irrational (Real.sqrt b)) : 
  Irrational (Real.sqrt a + Real.sqrt b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_irrational_l59_5984


namespace NUMINAMATH_CALUDE_distance_is_600_km_l59_5990

/-- The distance between the starting points of two persons traveling towards each other -/
def distance_between_starting_points (speed1 speed2 : ℝ) (travel_time : ℝ) : ℝ :=
  (speed1 + speed2) * travel_time

/-- Theorem stating that the distance between starting points is 600 km -/
theorem distance_is_600_km (speed1 speed2 travel_time : ℝ) 
  (h1 : speed1 = 70)
  (h2 : speed2 = 80)
  (h3 : travel_time = 4) :
  distance_between_starting_points speed1 speed2 travel_time = 600 := by
  sorry

#check distance_is_600_km

end NUMINAMATH_CALUDE_distance_is_600_km_l59_5990


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l59_5955

theorem greatest_value_quadratic_inequality :
  ∃ (x_max : ℝ), x_max = 7 ∧
  (∀ x : ℝ, x^2 - 12*x + 35 ≤ 0 → x ≤ x_max) ∧
  (x_max^2 - 12*x_max + 35 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l59_5955


namespace NUMINAMATH_CALUDE_ornament_shop_profit_maximization_l59_5908

/-- Ornament shop profit maximization problem -/
theorem ornament_shop_profit_maximization :
  ∀ (cost_A cost_B selling_price total_quantity : ℕ) 
    (min_B max_B_ratio discount_threshold discount_rate : ℕ),
  cost_A = 1400 →
  cost_B = 630 →
  cost_A = 2 * cost_B →
  selling_price = 15 →
  total_quantity = 600 →
  min_B = 390 →
  max_B_ratio = 4 →
  discount_threshold = 150 →
  discount_rate = 40 →
  ∃ (quantity_A quantity_B profit : ℕ),
    quantity_A + quantity_B = total_quantity ∧
    quantity_B ≥ min_B ∧
    quantity_B ≤ max_B_ratio * quantity_A ∧
    quantity_A = 210 ∧
    quantity_B = 390 ∧
    profit = 3630 ∧
    (∀ (other_quantity_A other_quantity_B other_profit : ℕ),
      other_quantity_A + other_quantity_B = total_quantity →
      other_quantity_B ≥ min_B →
      other_quantity_B ≤ max_B_ratio * other_quantity_A →
      other_profit ≤ profit) :=
by sorry

end NUMINAMATH_CALUDE_ornament_shop_profit_maximization_l59_5908


namespace NUMINAMATH_CALUDE_product_of_roots_cubic_equation_l59_5937

theorem product_of_roots_cubic_equation :
  let f : ℝ → ℝ := λ x => 3 * x^3 - x^2 - 20 * x + 27
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x, f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ r₁ * r₂ * r₃ = -9 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_cubic_equation_l59_5937


namespace NUMINAMATH_CALUDE_fraction_equality_l59_5920

theorem fraction_equality (w x y z : ℝ) (hw : w ≠ 0) 
  (h : (x + 6*y - 3*z) / (-3*x + 4*w) = (-2*y + z) / (x - w) ∧ 
       (-2*y + z) / (x - w) = 2/3) : 
  x / w = 2/3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l59_5920


namespace NUMINAMATH_CALUDE_simplify_expression_l59_5993

theorem simplify_expression (p q r : ℝ) 
  (hp : p ≠ 7) (hq : q ≠ 8) (hr : r ≠ 9) : 
  (p - 7) / (9 - r) * (q - 8) / (7 - p) * (r - 9) / (8 - q) = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l59_5993


namespace NUMINAMATH_CALUDE_equal_pairs_infinity_l59_5975

def infinite_sequence (a : ℤ → ℝ) : Prop :=
  ∀ n : ℤ, a n = (1/4) * (a (n-1) + a (n+1))

theorem equal_pairs_infinity (a : ℤ → ℝ) :
  infinite_sequence a →
  (∃ i j : ℤ, i ≠ j ∧ a i = a j) →
  ∃ f : ℕ → (ℤ × ℤ), (∀ n : ℕ, (f n).1 ≠ (f n).2 ∧ a (f n).1 = a (f n).2) ∧
                      (∀ m n : ℕ, m ≠ n → f m ≠ f n) :=
by sorry

end NUMINAMATH_CALUDE_equal_pairs_infinity_l59_5975


namespace NUMINAMATH_CALUDE_quadruple_solutions_l59_5969

theorem quadruple_solutions : 
  ∀ (a b c d : ℕ+), 
    (a * b + 2 * a - b = 58 ∧ 
     b * c + 4 * b + 2 * c = 300 ∧ 
     c * d - 6 * c + 4 * d = 101) → 
    ((a = 3 ∧ b = 26 ∧ c = 7 ∧ d = 13) ∨ 
     (a = 15 ∧ b = 2 ∧ c = 73 ∧ d = 7)) := by
  sorry

end NUMINAMATH_CALUDE_quadruple_solutions_l59_5969


namespace NUMINAMATH_CALUDE_gcf_72_108_l59_5919

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by
  sorry

end NUMINAMATH_CALUDE_gcf_72_108_l59_5919


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l59_5928

theorem purely_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 3*a + 2)
  (z.re = 0 ∧ z.im ≠ 0) → a = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l59_5928


namespace NUMINAMATH_CALUDE_shoe_pairs_count_l59_5945

theorem shoe_pairs_count (total_shoes : ℕ) (prob_same_color : ℚ) : 
  total_shoes = 16 → 
  prob_same_color = 1 / 15 → 
  (total_shoes / 2 : ℕ) = 8 := by
sorry

end NUMINAMATH_CALUDE_shoe_pairs_count_l59_5945


namespace NUMINAMATH_CALUDE_units_digit_of_n_l59_5914

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_n (m n : ℕ) :
  m * n = 23^5 →
  units_digit m = 4 →
  units_digit n = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l59_5914


namespace NUMINAMATH_CALUDE_inscription_satisfies_conditions_l59_5942

structure Box where
  color : String
  maker : String
  inscription : String

def is_bellini (b : Box) : Prop := b.maker = "Bellini"
def is_cellini (b : Box) : Prop := b.maker = "Cellini"

def inscription : String := "Either both caskets are made by Bellini, or at least one of them is made by a member of the Cellini family"

theorem inscription_satisfies_conditions (golden_box silver_box : Box) 
  (h1 : golden_box.inscription = silver_box.inscription)
  (h2 : golden_box.inscription = inscription)
  (h3 : silver_box.inscription = inscription) :
  (¬ (is_bellini golden_box ∨ is_bellini silver_box) → 
    (is_cellini golden_box ∨ is_cellini silver_box)) ∧
  (is_bellini golden_box ∧ is_bellini silver_box) :=
by sorry

#check inscription_satisfies_conditions

end NUMINAMATH_CALUDE_inscription_satisfies_conditions_l59_5942


namespace NUMINAMATH_CALUDE_people_reached_in_day_l59_5992

/-- The number of people reached after n hours of message spreading -/
def people_reached (n : ℕ) : ℕ :=
  2^(n+1) - 1

/-- Theorem stating the number of people reached in 24 hours -/
theorem people_reached_in_day : people_reached 24 = 2^24 - 1 := by
  sorry

#eval people_reached 24

end NUMINAMATH_CALUDE_people_reached_in_day_l59_5992


namespace NUMINAMATH_CALUDE_solution_mixture_proof_l59_5910

theorem solution_mixture_proof (x : ℝ) 
  (h1 : x + 20 = 100) -- First solution is x% carbonated water and 20% lemonade
  (h2 : 0.6799999999999997 * x + 0.32000000000000003 * 55 = 72) -- Mixture equation
  : x = 80 := by
  sorry

end NUMINAMATH_CALUDE_solution_mixture_proof_l59_5910


namespace NUMINAMATH_CALUDE_tennis_tournament_n_is_five_l59_5951

/-- Represents a tennis tournament with the given conditions --/
structure TennisTournament where
  n : ℕ
  total_players : ℕ := 5 * n
  total_matches : ℕ := (total_players * (total_players - 1)) / 2
  women_wins : ℕ
  men_wins : ℕ
  no_ties : women_wins + men_wins = total_matches
  win_ratio : women_wins * 2 = men_wins * 3

/-- The theorem stating that n must be 5 for the given conditions --/
theorem tennis_tournament_n_is_five :
  ∀ t : TennisTournament, t.n = 5 := by sorry

end NUMINAMATH_CALUDE_tennis_tournament_n_is_five_l59_5951


namespace NUMINAMATH_CALUDE_gas_volume_ranking_l59_5935

-- Define the regions
inductive Region
| West
| NonWest
| Russia

-- Define the gas volume per capita for each region
def gas_volume (r : Region) : ℝ :=
  match r with
  | Region.West => 21428
  | Region.NonWest => 26848.55
  | Region.Russia => 302790.13

-- Theorem to prove the ranking
theorem gas_volume_ranking :
  gas_volume Region.Russia > gas_volume Region.NonWest ∧
  gas_volume Region.NonWest > gas_volume Region.West :=
by sorry

end NUMINAMATH_CALUDE_gas_volume_ranking_l59_5935


namespace NUMINAMATH_CALUDE_solution_set_range_of_a_l59_5901

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 4| + |x + 1|

-- Theorem for the solution of f(x) ≤ 9
theorem solution_set (x : ℝ) : f x ≤ 9 ↔ x ∈ Set.Icc (-2) 4 := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Icc 0 2, f x = -x^2 + a) ↔ a ∈ Set.Icc (19/4) 7 := by sorry

end NUMINAMATH_CALUDE_solution_set_range_of_a_l59_5901


namespace NUMINAMATH_CALUDE_product_of_real_parts_quadratic_complex_l59_5999

theorem product_of_real_parts_quadratic_complex (x : ℂ) :
  x^2 + 3*x = -2 + 2*I →
  ∃ (s₁ s₂ : ℂ), (s₁^2 + 3*s₁ = -2 + 2*I) ∧ 
                 (s₂^2 + 3*s₂ = -2 + 2*I) ∧
                 (s₁.re * s₂.re = (5 - 2*Real.sqrt 5) / 4) :=
by sorry

end NUMINAMATH_CALUDE_product_of_real_parts_quadratic_complex_l59_5999


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l59_5906

theorem opposite_of_negative_three : -((-3) : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l59_5906


namespace NUMINAMATH_CALUDE_equation_solution_l59_5970

theorem equation_solution :
  ∃ x : ℝ, (3 / (x - 2) = 2 / (x - 1)) ∧ (x = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l59_5970


namespace NUMINAMATH_CALUDE_remaining_cooking_time_l59_5903

def total_potatoes : ℕ := 15
def cooked_potatoes : ℕ := 6
def cooking_time_per_potato : ℕ := 8

theorem remaining_cooking_time : 
  (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 72 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cooking_time_l59_5903


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_l59_5948

/-- Proves that the cost of each chocolate bar is $3 -/
theorem chocolate_bar_cost (initial_bars : ℕ) (unsold_bars : ℕ) (total_revenue : ℚ) : 
  initial_bars = 7 → unsold_bars = 4 → total_revenue = 9 → 
  (total_revenue / (initial_bars - unsold_bars : ℚ)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_l59_5948


namespace NUMINAMATH_CALUDE_ladybug_leaves_l59_5940

theorem ladybug_leaves (ladybugs_per_leaf : ℕ) (total_ladybugs : ℕ) (h1 : ladybugs_per_leaf = 139) (h2 : total_ladybugs = 11676) :
  total_ladybugs / ladybugs_per_leaf = 84 := by
sorry

end NUMINAMATH_CALUDE_ladybug_leaves_l59_5940


namespace NUMINAMATH_CALUDE_quadratic_equations_root_range_l59_5902

/-- The range of real numbers for a, such that at most two of the given three quadratic equations do not have real roots -/
theorem quadratic_equations_root_range : 
  {a : ℝ | (∃ x : ℝ, x^2 - a*x + 9 = 0) ∨ 
           (∃ x : ℝ, x^2 + a*x - 2*a = 0) ∨ 
           (∃ x : ℝ, x^2 + (a+1)*x + 9/4 = 0)} = 
  {a : ℝ | a ≤ -4 ∨ a ≥ 0} := by sorry

end NUMINAMATH_CALUDE_quadratic_equations_root_range_l59_5902


namespace NUMINAMATH_CALUDE_circle_center_satisfies_conditions_l59_5931

/-- The center of a circle satisfying given conditions -/
theorem circle_center_satisfies_conditions :
  let center : ℝ × ℝ := (-18, -11)
  let line1 : ℝ → ℝ → ℝ := λ x y => 3 * x - 4 * y - 20
  let line2 : ℝ → ℝ → ℝ := λ x y => 3 * x - 4 * y + 40
  let midline : ℝ → ℝ → ℝ := λ x y => 3 * x - 4 * y + 10
  let line3 : ℝ → ℝ → ℝ := λ x y => x - 3 * y - 15
  (midline center.1 center.2 = 0) ∧ (line3 center.1 center.2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_satisfies_conditions_l59_5931


namespace NUMINAMATH_CALUDE_sandy_marks_theorem_l59_5941

theorem sandy_marks_theorem (correct_marks : ℕ) (incorrect_marks : ℕ) 
  (total_sums : ℕ) (correct_sums : ℕ) 
  (h1 : correct_marks = 3) 
  (h2 : incorrect_marks = 2) 
  (h3 : total_sums = 30) 
  (h4 : correct_sums = 22) :
  correct_marks * correct_sums - incorrect_marks * (total_sums - correct_sums) = 50 := by
  sorry

end NUMINAMATH_CALUDE_sandy_marks_theorem_l59_5941


namespace NUMINAMATH_CALUDE_square_fraction_l59_5998

theorem square_fraction (a b : ℕ+) (h : (a.val * b.val + 1) ∣ (a.val^2 + b.val^2)) :
  ∃ (k : ℕ), (a.val^2 + b.val^2) / (a.val * b.val + 1) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_fraction_l59_5998


namespace NUMINAMATH_CALUDE_sticker_distribution_l59_5986

/-- The number of ways to distribute indistinguishable objects among distinct containers -/
def distribute (objects : ℕ) (containers : ℕ) : ℕ :=
  Nat.choose (objects + containers - 1) (containers - 1)

/-- Theorem: Distributing 10 indistinguishable stickers among 5 distinct sheets of paper -/
theorem sticker_distribution : distribute 10 5 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l59_5986


namespace NUMINAMATH_CALUDE_pencil_distribution_l59_5943

theorem pencil_distribution (total_pens : ℕ) (total_pencils : ℕ) (max_students : ℕ) :
  total_pens = 891 →
  max_students = 81 →
  total_pens % max_students = 0 →
  total_pencils % max_students = 0 :=
by sorry

end NUMINAMATH_CALUDE_pencil_distribution_l59_5943


namespace NUMINAMATH_CALUDE_continuity_at_two_l59_5983

noncomputable def f (x : ℝ) : ℝ := (x^3 - 8) / (x^2 - 4)

theorem continuity_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 2| ∧ |x - 2| < δ → |f x - 3| < ε :=
sorry

end NUMINAMATH_CALUDE_continuity_at_two_l59_5983


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l59_5905

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (12 * q) * Real.sqrt (8 * q^2) * Real.sqrt (9 * q^5) = 12 * q^4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l59_5905


namespace NUMINAMATH_CALUDE_unique_losses_l59_5977

/-- Represents a participant in the badminton tournament -/
structure Participant where
  id : Fin 16
  gamesWon : Nat
  gamesLost : Nat

/-- The set of all participants in the tournament -/
def Tournament := Fin 16 → Participant

theorem unique_losses (t : Tournament) : 
  (∀ i j : Fin 16, i ≠ j → (t i).gamesWon ≠ (t j).gamesWon) →
  (∀ i : Fin 16, (t i).gamesWon + (t i).gamesLost = 15) →
  (∀ i : Fin 16, (t i).gamesWon < 16) →
  (∀ i j : Fin 16, i ≠ j → (t i).gamesLost ≠ (t j).gamesLost) :=
by sorry

end NUMINAMATH_CALUDE_unique_losses_l59_5977


namespace NUMINAMATH_CALUDE_smallest_sum_of_bases_l59_5936

theorem smallest_sum_of_bases : ∃ (a b : ℕ), 
  (a > 6 ∧ b > 6) ∧ 
  (6 * a + 2 = 2 * b + 6) ∧ 
  (∀ (a' b' : ℕ), (a' > 6 ∧ b' > 6) → (6 * a' + 2 = 2 * b' + 6) → a + b ≤ a' + b') ∧
  a + b = 26 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_bases_l59_5936


namespace NUMINAMATH_CALUDE_hilt_bee_count_l59_5997

/-- The number of bees Mrs. Hilt saw on the first day -/
def first_day_bees : ℕ := 144

/-- The multiplier for the number of bees on the second day -/
def bee_multiplier : ℕ := 3

/-- The number of bees Mrs. Hilt saw on the second day -/
def second_day_bees : ℕ := first_day_bees * bee_multiplier

theorem hilt_bee_count : second_day_bees = 432 := by
  sorry

end NUMINAMATH_CALUDE_hilt_bee_count_l59_5997


namespace NUMINAMATH_CALUDE_a4_plus_b4_equals_228_l59_5932

theorem a4_plus_b4_equals_228 (a b : ℝ) 
  (h1 : (a^2 - b^2)^2 = 100) 
  (h2 : (a^3 * b^3) = 512) : 
  a^4 + b^4 = 228 := by
sorry

end NUMINAMATH_CALUDE_a4_plus_b4_equals_228_l59_5932


namespace NUMINAMATH_CALUDE_milk_packets_returned_l59_5917

/-- Given information about milk packets and their prices, prove the number of returned packets. -/
theorem milk_packets_returned (total : ℕ) (avg_price all_remaining returned : ℚ) :
  total = 5 ∧ 
  avg_price = 20 ∧ 
  all_remaining = 12 ∧ 
  returned = 32 →
  ∃ (x : ℕ), 
    x ≤ total ∧ 
    (total : ℚ) * avg_price = (total - x : ℚ) * all_remaining + (x : ℚ) * returned ∧
    x = 2 := by
  sorry

end NUMINAMATH_CALUDE_milk_packets_returned_l59_5917


namespace NUMINAMATH_CALUDE_sqrt_86400_simplified_l59_5934

theorem sqrt_86400_simplified : Real.sqrt 86400 = 120 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_86400_simplified_l59_5934


namespace NUMINAMATH_CALUDE_simplify_power_of_power_l59_5929

theorem simplify_power_of_power (x : ℝ) : (5 * x^2)^4 = 625 * x^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_of_power_l59_5929


namespace NUMINAMATH_CALUDE_completing_square_result_l59_5944

theorem completing_square_result (x : ℝ) :
  x^2 - 4*x - 1 = 0 → (x - 2)^2 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_completing_square_result_l59_5944


namespace NUMINAMATH_CALUDE_sum_lower_bound_l59_5930

theorem sum_lower_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * b = a + b + 3) :
  6 ≤ a + b := by
  sorry

end NUMINAMATH_CALUDE_sum_lower_bound_l59_5930


namespace NUMINAMATH_CALUDE_complex_cube_equality_l59_5947

theorem complex_cube_equality (c d : ℝ) (h : d > 0) :
  (c + d * Complex.I) ^ 3 = (c - d * Complex.I) ^ 3 ↔ d / c = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_complex_cube_equality_l59_5947


namespace NUMINAMATH_CALUDE_function_identity_l59_5991

theorem function_identity (f : ℝ → ℝ) 
  (h : ∀ a x : ℝ, a < x ∧ x < a + 100 → a ≤ f x ∧ f x ≤ a + 100) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l59_5991


namespace NUMINAMATH_CALUDE_right_triangle_point_distance_l59_5965

theorem right_triangle_point_distance (h d x : ℝ) : 
  h > 0 → d > 0 → x > 0 →
  x + Real.sqrt ((x + h)^2 + d^2) = h + d →
  x = h * d / (2 * h + d) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_point_distance_l59_5965


namespace NUMINAMATH_CALUDE_some_number_solution_l59_5971

theorem some_number_solution :
  ∃ x : ℝ, x * 13.26 + x * 9.43 + x * 77.31 = 470 ∧ x = 4.7 := by
  sorry

end NUMINAMATH_CALUDE_some_number_solution_l59_5971


namespace NUMINAMATH_CALUDE_tom_bonus_percentage_l59_5916

/-- Calculates the percentage of bonus points per customer served -/
def bonus_percentage (customers_per_hour : ℕ) (hours_worked : ℕ) (total_bonus_points : ℕ) : ℚ :=
  (total_bonus_points : ℚ) / ((customers_per_hour * hours_worked) : ℚ) * 100

/-- Proves that the bonus percentage for Tom is 20% -/
theorem tom_bonus_percentage :
  bonus_percentage 10 8 16 = 20 := by
  sorry

#eval bonus_percentage 10 8 16

end NUMINAMATH_CALUDE_tom_bonus_percentage_l59_5916


namespace NUMINAMATH_CALUDE_special_function_upper_bound_l59_5913

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) ∧ 
  (f 1 = 1) ∧
  (∀ x₁ x₂, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂)

/-- The main theorem -/
theorem special_function_upper_bound 
  (f : ℝ → ℝ) (h : SpecialFunction f) : 
  ∀ x ∈ Set.Icc 0 1, f x ≤ 2 * x :=
by sorry

end NUMINAMATH_CALUDE_special_function_upper_bound_l59_5913


namespace NUMINAMATH_CALUDE_inverse_proportion_through_point_l59_5956

/-- The inverse proportion function passing through (2, -1) has k = -2 --/
theorem inverse_proportion_through_point (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, x ≠ 0 → k / x = -1 / 2) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_through_point_l59_5956


namespace NUMINAMATH_CALUDE_triangle_area_with_given_conditions_l59_5988

noncomputable def triangle_area (r : ℝ) (R : ℝ) (A B C : ℝ) : ℝ :=
  4 * r * R * Real.sin A

theorem triangle_area_with_given_conditions (r R A B C : ℝ) 
  (h_inradius : r = 4)
  (h_circumradius : R = 9)
  (h_angle_condition : 2 * Real.cos A = Real.cos B + Real.cos C) :
  triangle_area r R A B C = 8 * Real.sqrt 181 := by
  sorry

#check triangle_area_with_given_conditions

end NUMINAMATH_CALUDE_triangle_area_with_given_conditions_l59_5988


namespace NUMINAMATH_CALUDE_problem_statement_l59_5921

theorem problem_statement : 65 * 1515 - 25 * 1515 + 1515 = 62115 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l59_5921


namespace NUMINAMATH_CALUDE_constant_function_inequality_l59_5987

theorem constant_function_inequality (f : ℝ → ℝ) :
  (∀ x y z : ℝ, f (x + y) + f (y + z) + f (z + x) ≥ 3 * f (x + 2*y + 3*z)) →
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
by sorry

end NUMINAMATH_CALUDE_constant_function_inequality_l59_5987


namespace NUMINAMATH_CALUDE_infinitely_many_m_with_1000_nonzero_bits_l59_5946

def count_nonzero_bits (m : ℕ) : ℕ :=
  (m.bits.filter (· ≠ 0)).length

theorem infinitely_many_m_with_1000_nonzero_bits :
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ count_nonzero_bits m = 1000 :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_m_with_1000_nonzero_bits_l59_5946


namespace NUMINAMATH_CALUDE_truck_gas_ratio_l59_5973

/-- Proves the ratio of gas in a truck's tank to its total capacity before filling -/
theorem truck_gas_ratio (truck_capacity car_capacity added_gas : ℚ) 
  (h1 : truck_capacity = 20)
  (h2 : car_capacity = 12)
  (h3 : added_gas = 18)
  (h4 : (1/3) * car_capacity + added_gas = truck_capacity + car_capacity) :
  (truck_capacity - ((1/3) * car_capacity + added_gas - car_capacity)) / truck_capacity = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_truck_gas_ratio_l59_5973


namespace NUMINAMATH_CALUDE_cookies_sold_l59_5996

/-- Proves the number of cookies sold given the problem conditions -/
theorem cookies_sold (original_cupcake_price original_cookie_price : ℚ)
  (price_reduction : ℚ) (cupcakes_sold : ℕ) (total_revenue : ℚ)
  (h1 : original_cupcake_price = 3)
  (h2 : original_cookie_price = 2)
  (h3 : price_reduction = 1/2)
  (h4 : cupcakes_sold = 16)
  (h5 : total_revenue = 32) :
  (total_revenue - cupcakes_sold * (original_cupcake_price * price_reduction)) / (original_cookie_price * price_reduction) = 8 := by
  sorry

end NUMINAMATH_CALUDE_cookies_sold_l59_5996


namespace NUMINAMATH_CALUDE_sally_forgot_seven_poems_l59_5979

/-- Represents the number of poems in different categories --/
structure PoemCounts where
  initial : ℕ
  correct : ℕ
  mixed : ℕ

/-- Calculates the number of completely forgotten poems --/
def forgotten_poems (counts : PoemCounts) : ℕ :=
  counts.initial - (counts.correct + counts.mixed)

/-- Theorem stating that Sally forgot 7 poems --/
theorem sally_forgot_seven_poems : 
  let sally_counts : PoemCounts := ⟨15, 5, 3⟩
  forgotten_poems sally_counts = 7 := by
  sorry

end NUMINAMATH_CALUDE_sally_forgot_seven_poems_l59_5979


namespace NUMINAMATH_CALUDE_tens_digit_of_smallest_divisible_l59_5980

-- Define the smallest positive integer divisible by 20, 16, and 2016
def smallest_divisible : ℕ := 10080

-- Define a function to get the tens digit of a natural number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Theorem statement
theorem tens_digit_of_smallest_divisible :
  tens_digit smallest_divisible = 8 ∧
  ∀ m : ℕ, m > 0 ∧ 20 ∣ m ∧ 16 ∣ m ∧ 2016 ∣ m → m ≥ smallest_divisible :=
by sorry

end NUMINAMATH_CALUDE_tens_digit_of_smallest_divisible_l59_5980


namespace NUMINAMATH_CALUDE_unique_partition_count_l59_5909

/-- The number of ways to partition n into three distinct positive integers -/
def partition_count (n : ℕ) : ℕ :=
  (n - 1) * (n - 2) / 2 - 3 * ((n / 2) - 2) - 1

/-- Theorem stating that 18 is the only positive integer satisfying the condition -/
theorem unique_partition_count :
  ∀ n : ℕ, n > 0 → (partition_count n = n + 1 ↔ n = 18) := by sorry

end NUMINAMATH_CALUDE_unique_partition_count_l59_5909


namespace NUMINAMATH_CALUDE_x_value_proof_l59_5968

theorem x_value_proof (x y : ℝ) : 
  x / (x + 1) = (y^2 + 3*y + 1) / (y^2 + 3*y + 2) → x = y^2 + 3*y + 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l59_5968


namespace NUMINAMATH_CALUDE_factorization_equality_l59_5985

theorem factorization_equality (x y : ℝ) : 
  3 * x^3 - 6 * x^2 * y + 3 * x * y^2 = 3 * x * (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l59_5985


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l59_5952

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  swimmer : ℝ  -- Speed of the swimmer in still water
  stream : ℝ   -- Speed of the stream

/-- Calculates the effective speed given the swimmer's speed and stream speed. -/
def effectiveSpeed (s : SwimmerSpeed) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Theorem: Given the conditions of the swimming problem, the swimmer's speed in still water is 5.5 km/h. -/
theorem swimmer_speed_in_still_water 
  (s : SwimmerSpeed) 
  (h1 : effectiveSpeed s true = 35 / 5)   -- Downstream condition
  (h2 : effectiveSpeed s false = 20 / 5)  -- Upstream condition
  : s.swimmer = 5.5 := by
  sorry

#eval 5.5  -- To check if the value 5.5 is recognized correctly

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l59_5952


namespace NUMINAMATH_CALUDE_dinner_bill_calculation_l59_5961

theorem dinner_bill_calculation 
  (appetizer_cost : ℝ) 
  (entree_cost : ℝ) 
  (dessert_cost : ℝ) 
  (tip_percentage : ℝ) 
  (h1 : appetizer_cost = 9)
  (h2 : entree_cost = 20)
  (h3 : dessert_cost = 11)
  (h4 : tip_percentage = 0.3) :
  appetizer_cost + 2 * entree_cost + dessert_cost + 
  (appetizer_cost + 2 * entree_cost + dessert_cost) * tip_percentage = 78 :=
by sorry

end NUMINAMATH_CALUDE_dinner_bill_calculation_l59_5961


namespace NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l59_5904

/-- Prove that the polar curve equation ρ = √2 cos(θ - π/4) is equivalent to the rectangular coordinate equation (x - 1/2)² + (y - 1/2)² = 1/2 -/
theorem polar_to_rectangular_equivalence (x y ρ θ : ℝ) :
  (ρ = Real.sqrt 2 * Real.cos (θ - π / 4)) ∧
  (x = ρ * Real.cos θ) ∧
  (y = ρ * Real.sin θ) →
  (x - 1 / 2) ^ 2 + (y - 1 / 2) ^ 2 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l59_5904


namespace NUMINAMATH_CALUDE_power_five_fifteen_div_power_twentyfive_six_l59_5918

theorem power_five_fifteen_div_power_twentyfive_six :
  5^15 / 25^6 = 125 := by
sorry

end NUMINAMATH_CALUDE_power_five_fifteen_div_power_twentyfive_six_l59_5918


namespace NUMINAMATH_CALUDE_calories_per_chip_l59_5926

/-- Represents the number of chips in a bag -/
def chips_per_bag : ℕ := 24

/-- Represents the cost of a bag in dollars -/
def cost_per_bag : ℚ := 2

/-- Represents the total calories Peter wants to consume -/
def total_calories : ℕ := 480

/-- Represents the total amount Peter needs to spend in dollars -/
def total_spent : ℚ := 4

/-- Theorem stating that each chip contains 10 calories -/
theorem calories_per_chip : 
  (total_calories : ℚ) / (total_spent / cost_per_bag * chips_per_bag) = 10 := by
  sorry

end NUMINAMATH_CALUDE_calories_per_chip_l59_5926


namespace NUMINAMATH_CALUDE_smallest_number_with_distinct_sums_ending_in_two_l59_5924

def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def digitSumSequence (x : Nat) : List Nat :=
  [x, sumOfDigits x, sumOfDigits (sumOfDigits x), sumOfDigits (sumOfDigits (sumOfDigits x))]

theorem smallest_number_with_distinct_sums_ending_in_two :
  ∀ y : Nat, y < 2999 →
    ¬(List.Pairwise (· ≠ ·) (digitSumSequence y) ∧
      (digitSumSequence y).getLast? = some 2) ∧
    (List.Pairwise (· ≠ ·) (digitSumSequence 2999) ∧
     (digitSumSequence 2999).getLast? = some 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_distinct_sums_ending_in_two_l59_5924


namespace NUMINAMATH_CALUDE_sin_750_degrees_l59_5923

theorem sin_750_degrees (h : ∀ x, Real.sin (x + 2 * Real.pi) = Real.sin x) : 
  Real.sin (750 * Real.pi / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_750_degrees_l59_5923


namespace NUMINAMATH_CALUDE_art_club_teams_l59_5925

theorem art_club_teams (n : ℕ) (h : n.choose 2 = 15) :
  n.choose 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_art_club_teams_l59_5925


namespace NUMINAMATH_CALUDE_max_value_expression_l59_5967

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hbc : b > c) (hca : c > a) (ha_neq_zero : a ≠ 0) : 
  ∃ (x : ℝ), x = ((2*a + b)^2 + (b - 2*c)^2 + (c - a)^2) / a^2 ∧ 
  x ≤ 44 ∧ 
  ∀ (y : ℝ), y = ((2*a + b)^2 + (b - 2*c)^2 + (c - a)^2) / a^2 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l59_5967


namespace NUMINAMATH_CALUDE_complex_number_location_l59_5907

theorem complex_number_location :
  let z : ℂ := (1 - Complex.I) / (1 + Complex.I)
  let w : ℂ := z / (1 + Complex.I)
  (w.re < 0) ∧ (w.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l59_5907


namespace NUMINAMATH_CALUDE_vector_parallel_coordinates_l59_5962

/-- Given two vectors a and b in ℝ², prove that if |a| = 2√5, b = (1,2), and a is parallel to b,
    then a = (2,4) or a = (-2,-4) -/
theorem vector_parallel_coordinates (a b : ℝ × ℝ) :
  (norm a = 2 * Real.sqrt 5) →
  (b = (1, 2)) →
  (∃ (k : ℝ), a = k • b) →
  (a = (2, 4) ∨ a = (-2, -4)) :=
sorry

end NUMINAMATH_CALUDE_vector_parallel_coordinates_l59_5962


namespace NUMINAMATH_CALUDE_apple_sales_proof_l59_5959

/-- The number of apples sold by Reginald --/
def apples_sold : ℕ := 20

/-- The price of each apple in dollars --/
def apple_price : ℚ := 1.25

/-- The cost of Reginald's bike in dollars --/
def bike_cost : ℚ := 80

/-- The fraction of the bike cost that the repairs cost --/
def repair_cost_fraction : ℚ := 1/4

/-- The fraction of earnings remaining after repairs --/
def remaining_earnings_fraction : ℚ := 1/5

theorem apple_sales_proof :
  apples_sold = 20 ∧
  apple_price = 1.25 ∧
  bike_cost = 80 ∧
  repair_cost_fraction = 1/4 ∧
  remaining_earnings_fraction = 1/5 ∧
  (apples_sold : ℚ) * apple_price - bike_cost * repair_cost_fraction = 
    remaining_earnings_fraction * ((apples_sold : ℚ) * apple_price) :=
by sorry

end NUMINAMATH_CALUDE_apple_sales_proof_l59_5959


namespace NUMINAMATH_CALUDE_min_surface_area_cubic_pile_l59_5966

/-- Represents the dimensions of a brick -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the surface area of a cube given its side length -/
def cubeSurfaceArea (sideLength : ℕ) : ℕ :=
  6 * sideLength * sideLength

/-- Theorem: The minimum surface area of a cubic pile of bricks -/
theorem min_surface_area_cubic_pile (brick : BrickDimensions)
  (h1 : brick.length = 25)
  (h2 : brick.width = 15)
  (h3 : brick.height = 5) :
  ∃ (sideLength : ℕ), cubeSurfaceArea sideLength = 33750 ∧
    ∀ (otherSideLength : ℕ), cubeSurfaceArea otherSideLength ≥ 33750 := by
  sorry

end NUMINAMATH_CALUDE_min_surface_area_cubic_pile_l59_5966


namespace NUMINAMATH_CALUDE_cube_volume_puzzle_l59_5900

theorem cube_volume_puzzle (a : ℝ) : 
  a > 0 → 
  (a + 2) * (a - 2) * a = a^3 - 8 → 
  a^3 = 8 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_puzzle_l59_5900


namespace NUMINAMATH_CALUDE_jack_remaining_gift_card_value_jack_gift_card_return_l59_5927

/-- Calculates the remaining value of gift cards Jack can return after sending some to a scammer. -/
theorem jack_remaining_gift_card_value 
  (bb_count : ℕ) (bb_value : ℕ) (wm_count : ℕ) (wm_value : ℕ) 
  (bb_sent : ℕ) (wm_sent : ℕ) : ℕ :=
  let total_bb := bb_count * bb_value
  let total_wm := wm_count * wm_value
  let sent_bb := bb_sent * bb_value
  let sent_wm := wm_sent * wm_value
  let remaining_bb := total_bb - sent_bb
  let remaining_wm := total_wm - sent_wm
  remaining_bb + remaining_wm

/-- Proves that Jack can return gift cards worth $3900. -/
theorem jack_gift_card_return : 
  jack_remaining_gift_card_value 6 500 9 200 1 2 = 3900 := by
  sorry

end NUMINAMATH_CALUDE_jack_remaining_gift_card_value_jack_gift_card_return_l59_5927


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l59_5960

/-- A line passing through (2, 3) with equal x and y intercepts -/
def EqualInterceptLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (a : ℝ), (p.1 / a + p.2 / a = 1 ∧ a ≠ 0) ∨ (p.1 = 2 ∧ p.2 = 3) ∨ (p.1 = 0 ∧ p.2 = 0)}

theorem equal_intercept_line_equation :
  EqualInterceptLine = {p : ℝ × ℝ | p.1 + p.2 - 5 = 0} ∪ {p : ℝ × ℝ | 3 * p.1 - 2 * p.2 = 0} :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l59_5960


namespace NUMINAMATH_CALUDE_odd_prime_fifth_power_difference_l59_5912

theorem odd_prime_fifth_power_difference (p : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) 
  (hx : ∃ (x y : ℤ), (x : ℝ)^5 - (y : ℝ)^5 = p) :
  ∃ (v : ℤ), Odd v ∧ Real.sqrt ((4 * p + 1 : ℝ) / 5) = ((v^2 : ℝ) + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_fifth_power_difference_l59_5912


namespace NUMINAMATH_CALUDE_jeans_price_markup_l59_5964

theorem jeans_price_markup (cost : ℝ) (h : cost > 0) :
  let retailer_price := cost * 1.4
  let customer_price := retailer_price * 1.1
  (customer_price - cost) / cost = 0.54 := by
sorry

end NUMINAMATH_CALUDE_jeans_price_markup_l59_5964


namespace NUMINAMATH_CALUDE_round_trip_speed_l59_5974

/-- Given a round trip where:
    - The outbound journey is at speed p km/h
    - The return journey is at 3 km/h
    - The average speed is (24/q) km/h
    - p = 4
    Then q = 7 -/
theorem round_trip_speed (p q : ℝ) (hp : p = 4) : 
  (2 / ((1/p) + (1/3)) = 24/q) → q = 7 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_speed_l59_5974


namespace NUMINAMATH_CALUDE_debby_initial_bottles_l59_5938

/-- The number of water bottles Debby bought initially -/
def initial_bottles : ℕ := sorry

/-- The number of bottles Debby drinks per day -/
def bottles_per_day : ℕ := 15

/-- The number of days Debby drank water -/
def days_drinking : ℕ := 11

/-- The number of bottles Debby has left -/
def bottles_left : ℕ := 99

/-- Theorem stating that Debby bought 264 water bottles initially -/
theorem debby_initial_bottles : initial_bottles = 264 := by
  sorry

end NUMINAMATH_CALUDE_debby_initial_bottles_l59_5938


namespace NUMINAMATH_CALUDE_no_cubic_linear_terms_implies_value_l59_5915

theorem no_cubic_linear_terms_implies_value (m n : ℝ) :
  (∀ x : ℝ, m * x^3 - 2 * x^2 + 3 * x - 4 * x^3 + 5 * x^2 - n * x = 3 * x^2) →
  m^2 - 2 * m * n + n^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_cubic_linear_terms_implies_value_l59_5915


namespace NUMINAMATH_CALUDE_birthday_800th_day_l59_5939

/-- Given a person born on a Tuesday, their 800th day of life will fall on a Thursday. -/
theorem birthday_800th_day (birth_day : Nat) (days_passed : Nat) : 
  birth_day = 2 → days_passed = 800 → (birth_day + days_passed) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_birthday_800th_day_l59_5939


namespace NUMINAMATH_CALUDE_factor_expression_l59_5982

theorem factor_expression (y : ℝ) : 5 * y * (y - 4) + 2 * (y - 4) = (5 * y + 2) * (y - 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l59_5982


namespace NUMINAMATH_CALUDE_fraction_simplification_l59_5957

theorem fraction_simplification :
  let x : ℚ := 1/3
  1 / (1 / x^1 + 1 / x^2 + 1 / x^3) = 1 / 39 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l59_5957


namespace NUMINAMATH_CALUDE_ExistFunctionsWithSpecifiedPeriods_l59_5994

-- Define the concept of a smallest positive period for a function
def SmallestPositivePeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  (∀ x, f (x + p) = f x) ∧ 
  (∀ q, 0 < q → q < p → ∃ x, f (x + q) ≠ f x)

-- State the theorem
theorem ExistFunctionsWithSpecifiedPeriods : 
  ∃ (f g : ℝ → ℝ), 
    SmallestPositivePeriod f 6 ∧ 
    SmallestPositivePeriod g 2 ∧ 
    SmallestPositivePeriod (λ x => f x + g x) 3 := by
  sorry


end NUMINAMATH_CALUDE_ExistFunctionsWithSpecifiedPeriods_l59_5994


namespace NUMINAMATH_CALUDE_correct_distribution_probability_l59_5953

/-- Represents the number of guests -/
def num_guests : ℕ := 4

/-- Represents the total number of rolls -/
def total_rolls : ℕ := 8

/-- Represents the number of cheese rolls -/
def cheese_rolls : ℕ := 4

/-- Represents the number of fruit rolls -/
def fruit_rolls : ℕ := 4

/-- Represents the number of rolls per guest -/
def rolls_per_guest : ℕ := 2

/-- The probability of each guest getting one cheese roll and one fruit roll -/
def probability_correct_distribution : ℚ := 1 / 35

theorem correct_distribution_probability :
  probability_correct_distribution = 
    (cheese_rolls.choose 1 * fruit_rolls.choose 1 / (total_rolls.choose 2)) *
    ((cheese_rolls - 1).choose 1 * (fruit_rolls - 1).choose 1 / ((total_rolls - 2).choose 2)) *
    ((cheese_rolls - 2).choose 1 * (fruit_rolls - 2).choose 1 / ((total_rolls - 4).choose 2)) *
    1 := by sorry

#check correct_distribution_probability

end NUMINAMATH_CALUDE_correct_distribution_probability_l59_5953


namespace NUMINAMATH_CALUDE_power_equation_l59_5949

theorem power_equation (a m n : ℝ) (h1 : a^m = 6) (h2 : a^n = 6) : a^(2*m - n) = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l59_5949


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l59_5978

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l59_5978


namespace NUMINAMATH_CALUDE_inequality_proof_l59_5958

theorem inequality_proof (m n : ℝ) (h1 : m < n) (h2 : n < 0) : n / m + m / n > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l59_5958


namespace NUMINAMATH_CALUDE_painted_cells_theorem_l59_5954

/-- Represents a rectangular grid with painted and unpainted cells. -/
structure PaintedRectangle where
  rows : Nat
  cols : Nat
  painted_cells : Nat

/-- Checks if the given PaintedRectangle satisfies the problem conditions. -/
def is_valid_painting (rect : PaintedRectangle) : Prop :=
  ∃ k l : Nat,
    rect.rows = 2 * k + 1 ∧
    rect.cols = 2 * l + 1 ∧
    k * l = 74 ∧
    rect.painted_cells = (2 * k + 1) * (2 * l + 1) - 74

/-- The main theorem stating the only possible numbers of painted cells. -/
theorem painted_cells_theorem :
  ∀ rect : PaintedRectangle,
    is_valid_painting rect →
    (rect.painted_cells = 373 ∨ rect.painted_cells = 301) :=
by sorry

end NUMINAMATH_CALUDE_painted_cells_theorem_l59_5954


namespace NUMINAMATH_CALUDE_min_sum_of_primes_for_99_consecutive_sum_l59_5963

/-- The sum of 99 consecutive natural numbers -/
def sum_99_consecutive (x : ℕ) : ℕ := 99 * x

/-- Predicate to check if a number is prime -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem min_sum_of_primes_for_99_consecutive_sum :
  ∃ (a b c d : ℕ), 
    is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧
    (∃ x : ℕ, sum_99_consecutive x = a * b * c * d) ∧
    (∀ a' b' c' d' : ℕ, 
      is_prime a' ∧ is_prime b' ∧ is_prime c' ∧ is_prime d' ∧
      (∃ x : ℕ, sum_99_consecutive x = a' * b' * c' * d') →
      a + b + c + d ≤ a' + b' + c' + d') ∧
    a + b + c + d = 70 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_primes_for_99_consecutive_sum_l59_5963


namespace NUMINAMATH_CALUDE_homework_time_difference_l59_5976

/-- Proves that the difference in time taken by Sarah and Samuel to finish their homework is 48 minutes -/
theorem homework_time_difference (samuel_time sarah_time_hours : ℝ) : 
  samuel_time = 30 → 
  sarah_time_hours = 1.3 → 
  sarah_time_hours * 60 - samuel_time = 48 := by
sorry

end NUMINAMATH_CALUDE_homework_time_difference_l59_5976


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l59_5995

-- Define the circle M
def circle_M (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*y - 3 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Theorem statement
theorem circle_and_line_properties :
  ∀ (E F : ℝ × ℝ),
  circle_M (-Real.sqrt 3) 0 ∧
  circle_M (Real.sqrt 3) 0 ∧
  circle_M 0 (-3) ∧
  (∃ (t : ℝ), line_l (E.1) (E.2) ∧ line_l (F.1) (F.2)) ∧
  (E.1 - F.1)^2 + (E.2 - F.2)^2 = 15 →
  (∀ (x y : ℝ), circle_M x y ↔ x^2 + y^2 + 2*y - 3 = 0) ∧
  (∀ (x y : ℝ), line_l x y ↔ (y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x)) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l59_5995


namespace NUMINAMATH_CALUDE_sum_angles_regular_star_5_l59_5922

/-- A regular 5-pointed star inscribed in a circle -/
structure RegularStar5 where
  /-- The angle at each tip of the star -/
  tip_angle : ℝ
  /-- The number of points in the star -/
  num_points : ℕ
  /-- The number of points is 5 -/
  h_num_points : num_points = 5

/-- The sum of angles at the tips of a regular 5-pointed star is 540° -/
theorem sum_angles_regular_star_5 (star : RegularStar5) : 
  star.num_points * star.tip_angle = 540 := by
  sorry

end NUMINAMATH_CALUDE_sum_angles_regular_star_5_l59_5922


namespace NUMINAMATH_CALUDE_danielle_age_l59_5989

/-- Given the ages of Anna, Ben, Carlos, and Danielle, prove that Danielle is 22 years old. -/
theorem danielle_age (anna ben carlos danielle : ℕ)
  (h1 : anna = ben - 4)
  (h2 : ben = carlos + 3)
  (h3 : danielle = carlos + 6)
  (h4 : anna = 15) :
  danielle = 22 := by
sorry

end NUMINAMATH_CALUDE_danielle_age_l59_5989


namespace NUMINAMATH_CALUDE_sales_growth_rate_l59_5981

def initial_sales : ℝ := 10000
def final_sales : ℝ := 12100
def months_between : ℕ := 2

theorem sales_growth_rate :
  ∃ (r : ℝ), r > 0 ∧ (1 + r) ^ months_between = final_sales / initial_sales ∧ r = 0.1 := by
sorry

end NUMINAMATH_CALUDE_sales_growth_rate_l59_5981


namespace NUMINAMATH_CALUDE_bookmark_position_l59_5950

/-- Represents a book with pages and a bookmark --/
structure Book where
  pages : ℕ
  coverThickness : ℕ
  bookmarkPosition : ℕ

/-- Calculates the total thickness of a book in page-equivalent units --/
def bookThickness (b : Book) : ℕ := b.pages + 2 * b.coverThickness

/-- The problem setup --/
def bookshelfProblem (book1 book2 : Book) : Prop :=
  book1.pages = 250 ∧
  book2.pages = 250 ∧
  book1.coverThickness = 10 ∧
  book2.coverThickness = 10 ∧
  book1.bookmarkPosition = 125 ∧
  (bookThickness book1 + bookThickness book2) / 3 = book1.bookmarkPosition + book1.coverThickness + book2.bookmarkPosition

theorem bookmark_position (book1 book2 : Book) :
  bookshelfProblem book1 book2 → book2.bookmarkPosition = 35 :=
by sorry

end NUMINAMATH_CALUDE_bookmark_position_l59_5950


namespace NUMINAMATH_CALUDE_difference_of_squares_l59_5933

theorem difference_of_squares (x y : ℝ) : (x + 2*y) * (-2*y + x) = x^2 - 4*y^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l59_5933
