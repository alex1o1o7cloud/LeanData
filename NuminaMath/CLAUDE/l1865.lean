import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_two_range_for_empty_solution_set_l1865_186582

def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |x - a|

theorem solution_set_for_a_equals_two :
  {x : ℝ | f 2 x > 2} = {x : ℝ | x > 3/2} := by sorry

theorem range_for_empty_solution_set :
  {a : ℝ | a > 0 ∧ ∀ x, f a x < 2*a} = {a : ℝ | a > 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_two_range_for_empty_solution_set_l1865_186582


namespace NUMINAMATH_CALUDE_will_earnings_l1865_186570

def hourly_wage : ℕ := 8
def monday_hours : ℕ := 8
def tuesday_hours : ℕ := 2

theorem will_earnings : 
  hourly_wage * monday_hours + hourly_wage * tuesday_hours = 80 := by
  sorry

end NUMINAMATH_CALUDE_will_earnings_l1865_186570


namespace NUMINAMATH_CALUDE_damaged_manuscript_multiplication_l1865_186534

theorem damaged_manuscript_multiplication : ∃ (x y : ℕ), 
  x > 0 ∧ y > 0 ∧
  10 ≤ x * 8 ∧ x * 8 < 100 ∧
  100 ≤ x * (y / 10) ∧ x * (y / 10) < 1000 ∧
  y % 10 = 8 ∧
  x * y = 1176 := by
sorry

end NUMINAMATH_CALUDE_damaged_manuscript_multiplication_l1865_186534


namespace NUMINAMATH_CALUDE_walmart_gift_card_value_l1865_186523

/-- Given information about gift cards and their usage, determine the value of each Walmart gift card -/
theorem walmart_gift_card_value 
  (best_buy_count : ℕ) 
  (best_buy_value : ℕ) 
  (walmart_count : ℕ) 
  (used_best_buy : ℕ) 
  (used_walmart : ℕ) 
  (total_remaining_value : ℕ) :
  best_buy_count = 6 →
  best_buy_value = 500 →
  walmart_count = 9 →
  used_best_buy = 1 →
  used_walmart = 2 →
  total_remaining_value = 3900 →
  (walmart_count - used_walmart) * 
    ((total_remaining_value - (best_buy_count - used_best_buy) * best_buy_value) / 
     (walmart_count - used_walmart)) = 
  (walmart_count - used_walmart) * 200 :=
by sorry

end NUMINAMATH_CALUDE_walmart_gift_card_value_l1865_186523


namespace NUMINAMATH_CALUDE_cube_decomposition_l1865_186551

/-- The smallest odd number in the decomposition of m³ -/
def smallest_odd (m : ℕ+) : ℕ := 2 * (m - 1) + 3

/-- The number of odd terms in the decomposition of m³ -/
def num_terms (m : ℕ+) : ℕ := (m + 2) * (m - 1) / 2

theorem cube_decomposition (m : ℕ+) :
  smallest_odd m = 91 → m = 10 := by sorry

end NUMINAMATH_CALUDE_cube_decomposition_l1865_186551


namespace NUMINAMATH_CALUDE_triangle_circumscribed_circle_diameter_l1865_186550

theorem triangle_circumscribed_circle_diameter 
  (a : ℝ) (A : ℝ) (D : ℝ) :
  a = 10 ∧ A = π/4 ∧ D = a / Real.sin A → D = 10 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_circumscribed_circle_diameter_l1865_186550


namespace NUMINAMATH_CALUDE_escalator_walking_rate_l1865_186537

/-- Calculates the walking rate of a person on an escalator -/
theorem escalator_walking_rate 
  (escalator_speed : ℝ) 
  (escalator_length : ℝ) 
  (time_taken : ℝ) 
  (h1 : escalator_speed = 15)
  (h2 : escalator_length = 180)
  (h3 : time_taken = 10) :
  (escalator_length / time_taken) - escalator_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_escalator_walking_rate_l1865_186537


namespace NUMINAMATH_CALUDE_elaine_jerry_ratio_l1865_186595

/-- Represents the time spent in the pool by each person --/
structure PoolTime where
  jerry : ℚ
  elaine : ℚ
  george : ℚ
  kramer : ℚ

/-- Conditions of the problem --/
def pool_conditions (t : PoolTime) : Prop :=
  t.jerry = 3 ∧
  t.george = t.elaine / 3 ∧
  t.kramer = 0 ∧
  t.jerry + t.elaine + t.george + t.kramer = 11

/-- The theorem to be proved --/
theorem elaine_jerry_ratio (t : PoolTime) :
  pool_conditions t → t.elaine / t.jerry = 2 := by
  sorry


end NUMINAMATH_CALUDE_elaine_jerry_ratio_l1865_186595


namespace NUMINAMATH_CALUDE_arithmetic_is_F_sequence_l1865_186531

def is_F_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n ≥ 3, ∃ i j : ℕ, i ≠ j ∧ i < n ∧ j < n ∧ a n = a i + a j

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a n = 2 * n

theorem arithmetic_is_F_sequence :
  ∀ a : ℕ → ℝ, arithmetic_sequence a → is_F_sequence a :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_is_F_sequence_l1865_186531


namespace NUMINAMATH_CALUDE_complete_square_property_l1865_186529

/-- A function to represent a quadratic expression of the form (p + qx)² + (r + sx)² -/
def quadraticExpression (p q r s : ℝ) (x : ℝ) : ℝ :=
  (p + q * x)^2 + (r + s * x)^2

/-- Predicate to check if a quadratic expression is a complete square -/
def isCompleteSquare (f : ℝ → ℝ) : Prop :=
  ∃ (k l : ℝ), ∀ x, f x = (k * x + l)^2

theorem complete_square_property 
  (a b c a' b' c' : ℝ) 
  (h1 : isCompleteSquare (quadraticExpression a b a' b'))
  (h2 : isCompleteSquare (quadraticExpression a c a' c')) :
  isCompleteSquare (quadraticExpression b c b' c') := by
  sorry

end NUMINAMATH_CALUDE_complete_square_property_l1865_186529


namespace NUMINAMATH_CALUDE_total_oil_needed_l1865_186587

/-- Represents the oil requirements for a bicycle -/
structure BikeOil where
  wheel : ℕ  -- Oil needed for one wheel
  chain : ℕ  -- Oil needed for the chain
  pedals : ℕ -- Oil needed for the pedals
  brakes : ℕ -- Oil needed for the brakes

/-- Calculates the total oil needed for a bicycle -/
def totalOilForBike (bike : BikeOil) : ℕ :=
  2 * bike.wheel + bike.chain + bike.pedals + bike.brakes

/-- The oil requirements for the first bicycle -/
def bike1 : BikeOil := 
  { wheel := 20, chain := 15, pedals := 8, brakes := 10 }

/-- The oil requirements for the second bicycle -/
def bike2 : BikeOil := 
  { wheel := 25, chain := 18, pedals := 10, brakes := 12 }

/-- The oil requirements for the third bicycle -/
def bike3 : BikeOil := 
  { wheel := 30, chain := 20, pedals := 12, brakes := 15 }

/-- Theorem stating the total oil needed for all three bicycles -/
theorem total_oil_needed : 
  totalOilForBike bike1 + totalOilForBike bike2 + totalOilForBike bike3 = 270 := by
  sorry

end NUMINAMATH_CALUDE_total_oil_needed_l1865_186587


namespace NUMINAMATH_CALUDE_set_intersection_equality_l1865_186575

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem set_intersection_equality : M ∩ N = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l1865_186575


namespace NUMINAMATH_CALUDE_incorrect_value_calculation_l1865_186514

theorem incorrect_value_calculation (n : ℕ) (initial_mean correct_mean correct_value : ℝ) 
  (h1 : n = 25)
  (h2 : initial_mean = 190)
  (h3 : correct_mean = 191.4)
  (h4 : correct_value = 165) :
  ∃ incorrect_value : ℝ,
    incorrect_value = n * correct_mean - (n - 1) * initial_mean - correct_value ∧
    incorrect_value = 200 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_value_calculation_l1865_186514


namespace NUMINAMATH_CALUDE_staff_discount_percentage_l1865_186545

theorem staff_discount_percentage (d : ℝ) : 
  d > 0 →  -- Assuming the original price is positive
  let discounted_price := 0.85 * d  -- Price after 15% discount
  let final_price := 0.765 * d      -- Price staff member pays
  let staff_discount_percent := (discounted_price - final_price) / discounted_price * 100
  staff_discount_percent = 10 := by
sorry

end NUMINAMATH_CALUDE_staff_discount_percentage_l1865_186545


namespace NUMINAMATH_CALUDE_described_relationship_is_correlation_l1865_186524

/-- Represents a variable in a statistical relationship -/
structure Variable where
  name : String
  is_independent : Bool

/-- Represents a relationship between two variables -/
structure Relationship where
  x : Variable
  y : Variable
  is_uncertain : Bool
  y_has_randomness : Bool

/-- Defines what a correlation is -/
def is_correlation (r : Relationship) : Prop :=
  r.x.is_independent ∧ 
  ¬r.y.is_independent ∧ 
  r.is_uncertain ∧ 
  r.y_has_randomness

/-- Theorem stating that the described relationship is a correlation -/
theorem described_relationship_is_correlation (x y : Variable) (r : Relationship) 
  (h1 : x.is_independent)
  (h2 : ¬y.is_independent)
  (h3 : r.x = x)
  (h4 : r.y = y)
  (h5 : r.is_uncertain)
  (h6 : r.y_has_randomness) :
  is_correlation r := by
  sorry


end NUMINAMATH_CALUDE_described_relationship_is_correlation_l1865_186524


namespace NUMINAMATH_CALUDE_javier_exercise_minutes_l1865_186572

/-- Proves that Javier exercised for 50 minutes each day given the conditions of the problem -/
theorem javier_exercise_minutes : ℕ → Prop :=
  fun x => 
    (∀ d : ℕ, d ≤ 7 → x > 0) →  -- Javier exercised some minutes every day for one week
    (3 * 90 + 7 * x = 620) →   -- Total exercise time for both Javier and Sanda
    x = 50                     -- Javier exercised for 50 minutes each day

/-- Proof of the theorem -/
lemma prove_javier_exercise_minutes : ∃ x : ℕ, javier_exercise_minutes x :=
  sorry

end NUMINAMATH_CALUDE_javier_exercise_minutes_l1865_186572


namespace NUMINAMATH_CALUDE_divisibility_after_subtraction_l1865_186532

theorem divisibility_after_subtraction :
  ∃ (n : ℕ), n = 15 ∧ (427398 - 3) % n = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_divisibility_after_subtraction_l1865_186532


namespace NUMINAMATH_CALUDE_water_formed_hcl_nahco3_l1865_186527

/-- Represents a chemical compound in a reaction -/
structure Compound where
  name : String
  moles : ℚ

/-- Represents a chemical reaction -/
structure Reaction where
  reactants : List Compound
  products : List Compound

/-- The balanced chemical equation for the reaction of HCl and NaHCO₃ -/
def hcl_nahco3_reaction : Reaction :=
  { reactants := [
      { name := "HCl", moles := 1 },
      { name := "NaHCO₃", moles := 1 }
    ],
    products := [
      { name := "NaCl", moles := 1 },
      { name := "CO₂", moles := 1 },
      { name := "H₂O", moles := 1 }
    ]
  }

/-- Calculate the amount of a specific product formed in a reaction -/
def amount_formed (reaction : Reaction) (product_name : String) (limiting_reagent_moles : ℚ) : ℚ :=
  let product := reaction.products.find? (fun c => c.name = product_name)
  match product with
  | some p => p.moles * limiting_reagent_moles
  | none => 0

/-- Theorem: The amount of water formed when 2 moles of HCl react with 2 moles of NaHCO₃ is 2 moles -/
theorem water_formed_hcl_nahco3 :
  amount_formed hcl_nahco3_reaction "H₂O" 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_formed_hcl_nahco3_l1865_186527


namespace NUMINAMATH_CALUDE_nell_total_cards_l1865_186569

def initial_cards : ℝ := 304.5
def received_cards : ℝ := 276.25

theorem nell_total_cards : initial_cards + received_cards = 580.75 := by
  sorry

end NUMINAMATH_CALUDE_nell_total_cards_l1865_186569


namespace NUMINAMATH_CALUDE_quadratic_equation_conversion_quadratic_coefficients_l1865_186547

theorem quadratic_equation_conversion (x : ℝ) : 
  (x^2 - 8*x = 10) ↔ (x^2 - 8*x - 10 = 0) :=
by sorry

theorem quadratic_coefficients :
  ∃ (a b c : ℝ), (∀ x, x^2 - 8*x - 10 = 0 ↔ a*x^2 + b*x + c = 0) ∧ 
  a = 1 ∧ b = -8 ∧ c = -10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_conversion_quadratic_coefficients_l1865_186547


namespace NUMINAMATH_CALUDE_dragon_unicorn_equivalence_l1865_186535

theorem dragon_unicorn_equivalence (R U : Prop) :
  (R → U) ↔ ((¬U → ¬R) ∧ (¬R ∨ U)) :=
sorry

end NUMINAMATH_CALUDE_dragon_unicorn_equivalence_l1865_186535


namespace NUMINAMATH_CALUDE_rachels_homework_l1865_186504

/-- Rachel's homework problem -/
theorem rachels_homework (reading_pages : ℕ) (math_pages : ℕ) : 
  reading_pages = 4 → reading_pages = math_pages + 1 → math_pages = 3 :=
by sorry

end NUMINAMATH_CALUDE_rachels_homework_l1865_186504


namespace NUMINAMATH_CALUDE_unique_solution_triple_sqrt_plus_four_l1865_186508

theorem unique_solution_triple_sqrt_plus_four :
  ∃! x : ℝ, x > 0 ∧ x = 3 * Real.sqrt x + 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_triple_sqrt_plus_four_l1865_186508


namespace NUMINAMATH_CALUDE_f_image_is_zero_to_eight_l1865_186565

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the domain
def D : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

-- Theorem statement
theorem f_image_is_zero_to_eight :
  Set.image f D = { y | 0 ≤ y ∧ y ≤ 8 } := by
  sorry

end NUMINAMATH_CALUDE_f_image_is_zero_to_eight_l1865_186565


namespace NUMINAMATH_CALUDE_cost_per_load_is_25_cents_l1865_186574

/-- Calculates the cost per load in cents when buying detergent on sale -/
def cost_per_load_cents (loads_per_bottle : ℕ) (sale_price_per_bottle : ℚ) : ℚ :=
  let total_cost := 2 * sale_price_per_bottle
  let total_loads := 2 * loads_per_bottle
  (total_cost / total_loads) * 100

/-- Theorem stating that the cost per load is 25 cents under given conditions -/
theorem cost_per_load_is_25_cents (loads_per_bottle : ℕ) (sale_price_per_bottle : ℚ) 
    (h1 : loads_per_bottle = 80)
    (h2 : sale_price_per_bottle = 20) :
  cost_per_load_cents loads_per_bottle sale_price_per_bottle = 25 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_load_is_25_cents_l1865_186574


namespace NUMINAMATH_CALUDE_problem_solution_l1865_186580

theorem problem_solution (x y z : ℝ) 
  (eq1 : 12 * x - 9 * y^2 = 7)
  (eq2 : 6 * y - 9 * z^2 = -2)
  (eq3 : 12 * z - 9 * x^2 = 4) :
  6 * x^2 + 9 * y^2 + 12 * z^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1865_186580


namespace NUMINAMATH_CALUDE_asymptote_sum_l1865_186583

theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x^3 + A*x^2 + B*x + C = (x + 1)*(x - 3)*(x - 4)) → 
  A + B + C = 11 := by
  sorry

end NUMINAMATH_CALUDE_asymptote_sum_l1865_186583


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l1865_186546

theorem mans_rowing_speed (river_speed : ℝ) (round_trip_time : ℝ) (total_distance : ℝ) (still_water_speed : ℝ) : 
  river_speed = 2 →
  round_trip_time = 1 →
  total_distance = 5.333333333333333 →
  still_water_speed = 7.333333333333333 →
  (total_distance / 2) / (round_trip_time / 2) = still_water_speed - river_speed ∧
  (total_distance / 2) / (round_trip_time / 2) = still_water_speed + river_speed :=
by sorry

end NUMINAMATH_CALUDE_mans_rowing_speed_l1865_186546


namespace NUMINAMATH_CALUDE_phd_time_calculation_l1865_186500

/-- Calculates the total time John spent on his PhD --/
def total_phd_time (acclimation_time : ℝ) (basics_time : ℝ) (research_multiplier : ℝ) 
  (sabbatical_time : ℝ) (dissertation_fraction : ℝ) (conference_time : ℝ) : ℝ :=
  let research_time := basics_time * (1 + research_multiplier) + sabbatical_time
  let dissertation_time := acclimation_time * dissertation_fraction + conference_time
  acclimation_time + basics_time + research_time + dissertation_time

theorem phd_time_calculation :
  total_phd_time 1 2 0.75 0.5 0.5 0.25 = 7.75 := by
  sorry

end NUMINAMATH_CALUDE_phd_time_calculation_l1865_186500


namespace NUMINAMATH_CALUDE_adult_tickets_correct_l1865_186564

/-- The number of adult tickets sold at the Rotary Club's Omelet Breakfast --/
def adult_tickets : ℕ :=
  let small_children_tickets : ℕ := 53
  let older_children_tickets : ℕ := 35
  let senior_tickets : ℕ := 37
  let small_children_omelet : ℚ := 1/2
  let older_children_omelet : ℕ := 1
  let adult_omelet : ℕ := 2
  let senior_omelet : ℚ := 3/2
  let extra_omelets : ℕ := 25
  let total_eggs : ℕ := 584
  let eggs_per_omelet : ℕ := 3
  26

theorem adult_tickets_correct : adult_tickets = 26 := by
  sorry

end NUMINAMATH_CALUDE_adult_tickets_correct_l1865_186564


namespace NUMINAMATH_CALUDE_max_children_in_class_l1865_186579

theorem max_children_in_class (x : ℕ) : 
  (∃ (chocolates_per_box : ℕ),
    -- Original plan with 6 boxes
    6 * chocolates_per_box = 10 * x + 40 ∧
    -- New plan with 4 boxes
    4 * chocolates_per_box ≥ 8 * (x - 1) + 4 ∧
    4 * chocolates_per_box < 8 * (x - 1) + 8) →
  x ≤ 23 :=
sorry

end NUMINAMATH_CALUDE_max_children_in_class_l1865_186579


namespace NUMINAMATH_CALUDE_snow_probability_l1865_186577

theorem snow_probability : 
  let p1 : ℚ := 1/5  -- probability of snow for each of the first 5 days
  let p2 : ℚ := 1/3  -- probability of snow for each of the next 5 days
  let days1 : ℕ := 5  -- number of days with probability p1
  let days2 : ℕ := 5  -- number of days with probability p2
  let prob_at_least_one_snow : ℚ := 1 - (1 - p1)^days1 * (1 - p2)^days2
  prob_at_least_one_snow = 726607/759375 := by
sorry

end NUMINAMATH_CALUDE_snow_probability_l1865_186577


namespace NUMINAMATH_CALUDE_arithmetic_log_implies_square_product_converse_not_always_true_l1865_186502

-- Define a predicate for arithmetic sequence of logarithms
def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  ∃ (a d : ℝ), (Real.log x = a) ∧ (Real.log y = a + d) ∧ (Real.log z = a + 2*d)

-- Define the theorem
theorem arithmetic_log_implies_square_product (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  is_arithmetic_sequence x y z → y^2 = x*z :=
by sorry

-- Define a counterexample to show the converse is not necessarily true
theorem converse_not_always_true :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ y^2 = x*z ∧ ¬(is_arithmetic_sequence x y z) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_log_implies_square_product_converse_not_always_true_l1865_186502


namespace NUMINAMATH_CALUDE_exponential_inequality_l1865_186576

theorem exponential_inequality (x : ℝ) : 3^x < (1:ℝ)/27 ↔ x < -3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l1865_186576


namespace NUMINAMATH_CALUDE_max_imag_part_is_sin_45_l1865_186507

-- Define the complex polynomial
def f (z : ℂ) : ℂ := z^6 - z^4 + z^2 - 1

-- Define the set of roots
def roots : Set ℂ := {z : ℂ | f z = 0}

-- Theorem statement
theorem max_imag_part_is_sin_45 :
  ∃ (z : ℂ), z ∈ roots ∧ 
    ∀ (w : ℂ), w ∈ roots → Complex.im w ≤ Complex.im z ∧ 
      Complex.im z = Real.sin (π/4) :=
sorry

end NUMINAMATH_CALUDE_max_imag_part_is_sin_45_l1865_186507


namespace NUMINAMATH_CALUDE_race_time_comparison_l1865_186525

theorem race_time_comparison (a V : ℝ) (h_a : a > 0) (h_V : V > 0) :
  let planned_time := a / V
  let first_half_time := a / (2 * 1.25 * V)
  let second_half_time := a / (2 * 0.8 * V)
  let actual_time := first_half_time + second_half_time
  actual_time > planned_time := by sorry

end NUMINAMATH_CALUDE_race_time_comparison_l1865_186525


namespace NUMINAMATH_CALUDE_smallest_multiples_sum_l1865_186589

theorem smallest_multiples_sum : ∃ c d : ℕ,
  (c ≥ 10 ∧ c < 100 ∧ c % 5 = 0 ∧ ∀ x : ℕ, x ≥ 10 ∧ x < 100 ∧ x % 5 = 0 → c ≤ x) ∧
  (d ≥ 100 ∧ d < 1000 ∧ d % 7 = 0 ∧ ∀ y : ℕ, y ≥ 100 ∧ y < 1000 ∧ y % 7 = 0 → d ≤ y) ∧
  c + d = 115 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiples_sum_l1865_186589


namespace NUMINAMATH_CALUDE_nathan_warmth_increase_l1865_186522

def blankets_in_closet : ℕ := 14
def warmth_per_blanket : ℕ := 3

def warmth_increase (blankets_used : ℕ) : ℕ :=
  blankets_used * warmth_per_blanket

theorem nathan_warmth_increase :
  warmth_increase (blankets_in_closet / 2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_nathan_warmth_increase_l1865_186522


namespace NUMINAMATH_CALUDE_max_value_of_trig_function_l1865_186510

theorem max_value_of_trig_function :
  let f : ℝ → ℝ := λ x ↦ Real.sin (x / 2) + Real.sqrt 3 * Real.cos (x / 2)
  ∀ x : ℝ, f x ≤ 2 ∧ ∃ x₀ : ℝ, f x₀ = 2 := by sorry

end NUMINAMATH_CALUDE_max_value_of_trig_function_l1865_186510


namespace NUMINAMATH_CALUDE_tangent_line_to_exponential_curve_l1865_186511

/-- A line is tangent to a curve if it intersects the curve at exactly one point and has the same slope as the curve at that point. -/
def is_tangent (f g : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f x₀ = g x₀ ∧ (deriv f) x₀ = (deriv g) x₀

/-- The problem statement -/
theorem tangent_line_to_exponential_curve (a : ℝ) :
  is_tangent (fun x => x - 3) (fun x => Real.exp (x + a)) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_exponential_curve_l1865_186511


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l1865_186528

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + (initial * percentage / 100) :=
by sorry

theorem increase_80_by_150_percent :
  80 * (1 + 150 / 100) = 200 :=
by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l1865_186528


namespace NUMINAMATH_CALUDE_floor_equality_l1865_186585

theorem floor_equality (m : ℝ) (h : m ≥ 3) :
  ⌊m * (m + 1) / (2 * (2 * m - 1))⌋ = ⌊(m + 1) / 4⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_equality_l1865_186585


namespace NUMINAMATH_CALUDE_green_face_probability_octahedral_die_l1865_186543

/-- An octahedral die with green and yellow faces -/
structure OctahedralDie where
  total_faces : Nat
  green_faces : Nat
  yellow_faces : Nat

/-- The probability of rolling a green face on an octahedral die -/
def green_face_probability (die : OctahedralDie) : Rat :=
  die.green_faces / die.total_faces

/-- Theorem: The probability of rolling a green face on an octahedral die
    with 5 green faces and 3 yellow faces is 5/8 -/
theorem green_face_probability_octahedral_die :
  let die : OctahedralDie := {
    total_faces := 8,
    green_faces := 5,
    yellow_faces := 3
  }
  green_face_probability die = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_green_face_probability_octahedral_die_l1865_186543


namespace NUMINAMATH_CALUDE_parallel_implies_m_eq_neg_one_perpendicular_implies_m_eq_one_plus_minus_two_sqrt_two_l1865_186578

-- Define the vectors
def OA : Fin 2 → ℝ := ![(-1 : ℝ), 3]
def OB : Fin 2 → ℝ := ![3, -1]
def OC (m : ℝ) : Fin 2 → ℝ := ![m, 1]

-- Define vector operations
def vector_sub (v w : Fin 2 → ℝ) : Fin 2 → ℝ := λ i => v i - w i
def parallel (v w : Fin 2 → ℝ) : Prop := ∃ k : ℝ, ∀ i, v i = k * w i
def perpendicular (v w : Fin 2 → ℝ) : Prop := v 0 * w 0 + v 1 * w 1 = 0

-- Define the theorems
theorem parallel_implies_m_eq_neg_one (m : ℝ) :
  parallel (vector_sub OB OA) (OC m) → m = -1 := by sorry

theorem perpendicular_implies_m_eq_one_plus_minus_two_sqrt_two (m : ℝ) :
  perpendicular (vector_sub (OC m) OA) (vector_sub (OC m) OB) →
  (m = 1 + 2 * Real.sqrt 2 ∨ m = 1 - 2 * Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_parallel_implies_m_eq_neg_one_perpendicular_implies_m_eq_one_plus_minus_two_sqrt_two_l1865_186578


namespace NUMINAMATH_CALUDE_banana_ratio_l1865_186567

/-- Theorem about the ratio of bananas in Raj's basket to bananas eaten -/
theorem banana_ratio (initial_bananas : ℕ) (bananas_left_on_tree : ℕ) (bananas_eaten : ℕ) :
  initial_bananas = 310 →
  bananas_left_on_tree = 100 →
  bananas_eaten = 70 →
  (initial_bananas - bananas_left_on_tree - bananas_eaten) / bananas_eaten = 2 := by
  sorry


end NUMINAMATH_CALUDE_banana_ratio_l1865_186567


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l1865_186521

theorem solution_set_equivalence (x : ℝ) :
  (3*x - 1) / (2 - x) ≥ 1 ↔ 3/4 ≤ x ∧ x < 2 := by
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l1865_186521


namespace NUMINAMATH_CALUDE_narcissistic_numbers_l1865_186590

theorem narcissistic_numbers : 
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ 
    n = (n / 100)^3 + ((n % 100) / 10)^3 + (n % 10)^3} = 
  {153, 370, 371, 407} := by
sorry

end NUMINAMATH_CALUDE_narcissistic_numbers_l1865_186590


namespace NUMINAMATH_CALUDE_parabola_equation_l1865_186513

/-- A parabola with vertex at the origin and focus on the y-axis. -/
structure Parabola where
  p : ℝ  -- The focal parameter of the parabola

/-- The line y = 2x + 1 -/
def line (x : ℝ) : ℝ := 2 * x + 1

/-- The chord length intercepted by the line y = 2x + 1 on the parabola -/
def chordLength (p : Parabola) : ℝ := sorry

theorem parabola_equation (p : Parabola) :
  chordLength p = Real.sqrt 15 →
  (∀ x y : ℝ, y = p.p * x^2 ∨ y = -3 * p.p * x^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1865_186513


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1865_186599

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right-angled triangle condition
  a^2 + b^2 + c^2 = 2500 →  -- sum of squares condition
  c = 25 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1865_186599


namespace NUMINAMATH_CALUDE_farm_animals_count_l1865_186530

theorem farm_animals_count : 
  ∀ (total_legs ducks dogs : ℕ),
  total_legs = 24 →
  ducks = 4 →
  total_legs = 2 * ducks + 4 * dogs →
  ducks + dogs = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_farm_animals_count_l1865_186530


namespace NUMINAMATH_CALUDE_student_calculation_l1865_186509

theorem student_calculation (x : ℤ) (h : x = 110) : 3 * x - 220 = 110 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_l1865_186509


namespace NUMINAMATH_CALUDE_number_added_problem_l1865_186597

theorem number_added_problem (x : ℝ) : 
  3 * (2 * 5 + x) = 57 → x = 9 := by
sorry

end NUMINAMATH_CALUDE_number_added_problem_l1865_186597


namespace NUMINAMATH_CALUDE_test_score_difference_l1865_186506

theorem test_score_difference (score_60 score_75 score_85 score_95 : ℝ)
  (percent_60 percent_75 percent_85 percent_95 : ℝ) :
  score_60 = 60 ∧ 
  score_75 = 75 ∧ 
  score_85 = 85 ∧ 
  score_95 = 95 ∧
  percent_60 = 0.2 ∧
  percent_75 = 0.4 ∧
  percent_85 = 0.25 ∧
  percent_95 = 0.15 ∧
  percent_60 + percent_75 + percent_85 + percent_95 = 1 →
  let mean := percent_60 * score_60 + percent_75 * score_75 + 
              percent_85 * score_85 + percent_95 * score_95
  let median := score_75
  abs (mean - median) = 2.5 := by
sorry

end NUMINAMATH_CALUDE_test_score_difference_l1865_186506


namespace NUMINAMATH_CALUDE_pages_left_after_eleven_days_l1865_186512

/-- Represents the number of pages left unread after reading for a given number of days -/
def pages_left (total_pages : ℕ) (pages_per_day : ℕ) (days : ℕ) : ℕ :=
  total_pages - pages_per_day * days

/-- Theorem stating that reading 15 pages a day for 11 days from a 250-page book leaves 85 pages unread -/
theorem pages_left_after_eleven_days :
  pages_left 250 15 11 = 85 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_after_eleven_days_l1865_186512


namespace NUMINAMATH_CALUDE_x_value_l1865_186557

theorem x_value (x : ℚ) (h : 1/3 - 1/4 = 4/x) : x = 48 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1865_186557


namespace NUMINAMATH_CALUDE_f_monotone_increasing_iff_a_range_l1865_186591

/-- A function f(x) = 2x^2 - ax + 5 that is monotonically increasing on [1, +∞) -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - a * x + 5

/-- The property of f being monotonically increasing on [1, +∞) -/
def monotone_increasing (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x < y → f a x < f a y

/-- The theorem stating the range of a for which f is monotonically increasing on [1, +∞) -/
theorem f_monotone_increasing_iff_a_range :
  ∀ a : ℝ, monotone_increasing a ↔ a ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_iff_a_range_l1865_186591


namespace NUMINAMATH_CALUDE_coupon_savings_difference_l1865_186598

/-- Represents the savings from Coupon A (15% off the listed price) -/
def savingsA (price : ℝ) : ℝ := 0.15 * price

/-- Represents the savings from Coupon B ($30 off the listed price) -/
def savingsB : ℝ := 30

/-- Represents the savings from Coupon C (25% off the amount exceeding $100) -/
def savingsC (price : ℝ) : ℝ := 0.25 * (price - 100)

/-- The theorem stating the difference between max and min prices where Coupon A is optimal -/
theorem coupon_savings_difference : 
  ∃ (x y : ℝ), 
    x > 100 ∧ y > 100 ∧
    (∀ p, p > 100 → savingsA p ≥ savingsB → savingsA p ≥ savingsC p → p ≥ x) ∧
    (∀ p, p > 100 → savingsA p ≥ savingsB → savingsA p ≥ savingsC p → p ≤ y) ∧
    y - x = 50 := by
  sorry

end NUMINAMATH_CALUDE_coupon_savings_difference_l1865_186598


namespace NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l1865_186556

/-- A dodecahedron is a 3-dimensional figure with 12 pentagonal faces and 20 vertices,
    where 3 faces meet at each vertex. -/
structure Dodecahedron where
  vertices : Nat
  faces : Nat
  faces_per_vertex : Nat
  vertices_eq : vertices = 20
  faces_eq : faces = 12
  faces_per_vertex_eq : faces_per_vertex = 3

/-- An interior diagonal of a dodecahedron is a segment connecting two vertices
    which do not lie on a common face. -/
def interior_diagonal (d : Dodecahedron) : Nat :=
  sorry

/-- The number of interior diagonals in a dodecahedron is 160. -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) :
  interior_diagonal d = 160 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l1865_186556


namespace NUMINAMATH_CALUDE_angle_properties_l1865_186593

theorem angle_properties (α : Real) 
  (h1 : 0 < α) (h2 : α < π) (h3 : Real.sin α + Real.cos α = 1/5) : 
  (Real.tan α = -4/3) ∧ 
  (Real.sin α ^ 2 - 3 * Real.sin α * Real.cos α - 4 * Real.cos α ^ 2 = 16/25) := by
  sorry

end NUMINAMATH_CALUDE_angle_properties_l1865_186593


namespace NUMINAMATH_CALUDE_total_books_count_l1865_186563

theorem total_books_count (sam_books joan_books tom_books alice_books : ℕ)
  (h1 : sam_books = 110)
  (h2 : joan_books = 102)
  (h3 : tom_books = 125)
  (h4 : alice_books = 97) :
  sam_books + joan_books + tom_books + alice_books = 434 := by
  sorry

end NUMINAMATH_CALUDE_total_books_count_l1865_186563


namespace NUMINAMATH_CALUDE_total_cost_calculation_l1865_186592

def tshirt_price : ℝ := 8
def sweater_price : ℝ := 18
def jacket_price : ℝ := 80
def jeans_price : ℝ := 35
def shoe_price : ℝ := 60

def jacket_discount : ℝ := 0.1
def shoe_discount : ℝ := 0.15

def clothing_tax_rate : ℝ := 0.05
def shoe_tax_rate : ℝ := 0.08

def tshirt_quantity : ℕ := 6
def sweater_quantity : ℕ := 4
def jacket_quantity : ℕ := 5
def jeans_quantity : ℕ := 3
def shoe_quantity : ℕ := 2

theorem total_cost_calculation :
  let tshirt_cost := tshirt_price * tshirt_quantity
  let sweater_cost := sweater_price * sweater_quantity
  let jacket_cost := jacket_price * jacket_quantity * (1 - jacket_discount)
  let jeans_cost := jeans_price * jeans_quantity
  let shoe_cost := shoe_price * shoe_quantity * (1 - shoe_discount)
  
  let clothing_subtotal := tshirt_cost + sweater_cost + jacket_cost + jeans_cost
  let shoe_subtotal := shoe_cost
  
  let clothing_tax := clothing_subtotal * clothing_tax_rate
  let shoe_tax := shoe_subtotal * shoe_tax_rate
  
  let total_cost := clothing_subtotal + shoe_subtotal + clothing_tax + shoe_tax
  
  total_cost = 724.41 := by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l1865_186592


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1865_186538

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^2 + y^2 + z^2 = 1) :
  x*y/z + y*z/x + z*x/y ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1865_186538


namespace NUMINAMATH_CALUDE_card_game_proof_l1865_186555

theorem card_game_proof (total_credits : ℕ) (red_cards : ℕ) (red_credit_value : ℕ) (blue_credit_value : ℕ)
  (h1 : total_credits = 84)
  (h2 : red_cards = 8)
  (h3 : red_credit_value = 3)
  (h4 : blue_credit_value = 5) :
  ∃ (blue_cards : ℕ), red_cards + blue_cards = 20 ∧ 
    red_cards * red_credit_value + blue_cards * blue_credit_value = total_credits :=
by
  sorry

end NUMINAMATH_CALUDE_card_game_proof_l1865_186555


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1865_186542

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  (1 / x + 1 / y) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1865_186542


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1865_186533

theorem min_value_of_expression (x : ℝ) :
  ∃ (m : ℝ), m = -784 ∧ ∀ (y : ℝ), (15 - y) * (13 - y) * (15 + y) * (13 + y) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1865_186533


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l1865_186554

/-- The perimeter of a trapezoid JKLM with given coordinates -/
theorem trapezoid_perimeter : 
  let J : ℝ × ℝ := (-2, -4)
  let K : ℝ × ℝ := (-2, 1)
  let L : ℝ × ℝ := (6, 7)
  let M : ℝ × ℝ := (6, -4)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist J K + dist K L + dist L M + dist M J = 34 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l1865_186554


namespace NUMINAMATH_CALUDE_symmetry_condition_l1865_186541

theorem symmetry_condition (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  (∀ x y : ℝ, y = (p * x + q) / (r * x + s) → -x = (p * (-y) + q) / (r * (-y) + s)) →
  p - s = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_condition_l1865_186541


namespace NUMINAMATH_CALUDE_quadrilateral_wx_length_l1865_186581

-- Define the quadrilateral WXYZ
structure Quadrilateral :=
  (W X Y Z : ℝ × ℝ)

-- Define the circle
def Circle := (ℝ × ℝ) → Prop

-- Define the inscribed property
def inscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

-- Define the diameter property
def is_diameter (W Z : ℝ × ℝ) (c : Circle) : Prop := sorry

-- Define the angle measure
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the length of a segment
def segment_length (A B : ℝ × ℝ) : ℝ := sorry

theorem quadrilateral_wx_length 
  (q : Quadrilateral) 
  (c : Circle) 
  (h1 : inscribed q c)
  (h2 : is_diameter q.W q.Z c)
  (h3 : segment_length q.W q.Z = 2)
  (h4 : segment_length q.X q.Z = segment_length q.Y q.W)
  (h5 : angle_measure q.W q.X q.Y = 72 * π / 180) :
  segment_length q.W q.X = Real.cos (18 * π / 180) * Real.sqrt (2 * (1 - Real.sin (18 * π / 180))) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_wx_length_l1865_186581


namespace NUMINAMATH_CALUDE_zachary_did_more_pushups_l1865_186549

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 51

/-- The number of push-ups David did -/
def david_pushups : ℕ := 44

/-- The difference in push-ups between Zachary and David -/
def pushup_difference : ℕ := zachary_pushups - david_pushups

theorem zachary_did_more_pushups : pushup_difference = 7 := by
  sorry

end NUMINAMATH_CALUDE_zachary_did_more_pushups_l1865_186549


namespace NUMINAMATH_CALUDE_age_difference_l1865_186559

theorem age_difference (a b : ℕ) : 
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 →  -- Ensure a and b are single digits
  (10 * a + b) + 10 = 3 * ((10 * b + a) + 10) → -- In 10 years, Mary's age is 3 times Jane's
  (10 * a + b) - (10 * b + a) = 54 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l1865_186559


namespace NUMINAMATH_CALUDE_xyz_square_equality_implies_zero_l1865_186501

theorem xyz_square_equality_implies_zero (x y z : ℤ) : 
  x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

-- Note: The second part of the original problem doesn't have a definitive answer,
-- so we'll omit it from the Lean statement.

end NUMINAMATH_CALUDE_xyz_square_equality_implies_zero_l1865_186501


namespace NUMINAMATH_CALUDE_impossible_to_divide_into_l_pieces_l1865_186503

/-- Represents a chessboard cell --/
inductive Cell
| Black
| White

/-- Represents an L-shaped piece --/
structure LPiece :=
(cells : Fin 4 → Cell)

/-- Represents a chessboard --/
def Chessboard := Fin 8 → Fin 8 → Cell

/-- Returns the color of a cell based on its coordinates --/
def cellColor (row col : Fin 8) : Cell :=
  if (row.val + col.val) % 2 = 0 then Cell.Black else Cell.White

/-- Checks if a cell is in the central 2x2 square --/
def isCentralSquare (row col : Fin 8) : Prop :=
  (row = 3 ∨ row = 4) ∧ (col = 3 ∨ col = 4)

/-- Represents the modified chessboard with central 2x2 square removed --/
def ModifiedChessboard : Type :=
  { cell : Fin 8 × Fin 8 // ¬isCentralSquare cell.1 cell.2 }

/-- The main theorem stating that it's impossible to divide the modified chessboard into L-shaped pieces --/
theorem impossible_to_divide_into_l_pieces :
  ¬∃ (pieces : List LPiece), 
    (pieces.length > 0) ∧ 
    (∀ (cell : ModifiedChessboard), ∃! (piece : LPiece) (i : Fin 4), 
      piece ∈ pieces ∧ piece.cells i = cellColor cell.val.1 cell.val.2) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_divide_into_l_pieces_l1865_186503


namespace NUMINAMATH_CALUDE_linear_diophantine_equation_solutions_l1865_186553

theorem linear_diophantine_equation_solutions
  (a b c x₀ y₀ : ℤ)
  (h_gcd : Int.gcd a b = 1)
  (h_solution : a * x₀ + b * y₀ = c) :
  ∀ x y : ℤ, a * x + b * y = c →
    ∃ k : ℤ, x = x₀ + k * b ∧ y = y₀ - k * a :=
sorry

end NUMINAMATH_CALUDE_linear_diophantine_equation_solutions_l1865_186553


namespace NUMINAMATH_CALUDE_line_relationship_sum_l1865_186517

/-- Represents a line in the form Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.A * l2.B = l1.B * l2.A

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.A * l2.A + l1.B * l2.B = 0

theorem line_relationship_sum (m n : ℝ) : 
  let l1 : Line := ⟨2, 2, -1⟩
  let l2 : Line := ⟨4, n, 3⟩
  let l3 : Line := ⟨m, 6, 1⟩
  parallel l1 l2 → perpendicular l1 l3 → m + n = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_relationship_sum_l1865_186517


namespace NUMINAMATH_CALUDE_simplified_fraction_ratio_l1865_186573

theorem simplified_fraction_ratio (m : ℝ) (c d : ℤ) :
  (∃ (k : ℝ), (5 * m + 15) / 5 = k ∧ k = c * m + d) →
  d / c = 3 :=
by sorry

end NUMINAMATH_CALUDE_simplified_fraction_ratio_l1865_186573


namespace NUMINAMATH_CALUDE_number_difference_l1865_186586

theorem number_difference (a b : ℕ) (h1 : a + b = 72) (h2 : a = 30) (h3 : b = 42) :
  b - a = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1865_186586


namespace NUMINAMATH_CALUDE_celebrity_baby_photo_match_probability_l1865_186548

/-- The number of celebrities and baby photos -/
def n : ℕ := 4

/-- The probability of correctly matching all celebrities with their baby photos -/
def correct_match_probability : ℚ := 1 / (n.factorial : ℚ)

/-- Theorem stating that the probability of correctly matching all celebrities
    with their baby photos when guessing at random is 1/24 -/
theorem celebrity_baby_photo_match_probability :
  correct_match_probability = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_celebrity_baby_photo_match_probability_l1865_186548


namespace NUMINAMATH_CALUDE_tv_price_increase_l1865_186515

theorem tv_price_increase (x : ℝ) 
  (h1 : (1 + x / 100) * 0.8 = 1 + 28 / 100) : x = 60 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_increase_l1865_186515


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_theorem_l1865_186584

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The length of the smaller base -/
  smallerBase : ℝ
  /-- The height of the trapezoid -/
  height : ℝ
  /-- The diagonal is perpendicular to the lateral side -/
  diagonalPerpendicular : Bool

/-- Properties of the isosceles trapezoid -/
def trapezoidProperties : IsoscelesTrapezoid :=
  { smallerBase := 3
  , height := 2
  , diagonalPerpendicular := true }

/-- The theorem stating the properties of the isosceles trapezoid -/
theorem isosceles_trapezoid_theorem (t : IsoscelesTrapezoid) 
  (h1 : t = trapezoidProperties) :
  ∃ (largerBase acuteAngle : ℝ),
    largerBase = 5 ∧ 
    acuteAngle = Real.arctan 2 :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_theorem_l1865_186584


namespace NUMINAMATH_CALUDE_basketball_games_lost_l1865_186520

/-- Calculates the number of games lost given the total games and games won -/
def games_lost (total_games : ℕ) (games_won : ℕ) : ℕ :=
  total_games - games_won

/-- Theorem: Given 12 total games and 8 games won, the number of games lost is 4 -/
theorem basketball_games_lost :
  let total_games : ℕ := 12
  let games_won : ℕ := 8
  games_lost total_games games_won = 4 := by
  sorry


end NUMINAMATH_CALUDE_basketball_games_lost_l1865_186520


namespace NUMINAMATH_CALUDE_planes_parallel_l1865_186544

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the geometric relations
variable (belongs_to : Point → Line → Prop)
variable (subset : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (noncoplanar : Line → Line → Prop)

-- State the theorem
theorem planes_parallel 
  (α β : Plane) (a b : Line) :
  noncoplanar a b →
  subset a α →
  subset b β →
  parallel_line_plane a β →
  parallel_line_plane b α →
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_l1865_186544


namespace NUMINAMATH_CALUDE_equation_solutions_l1865_186539

theorem equation_solutions : 
  ∃ (x₁ x₂ x₃ x₄ : ℝ), 
    (x₁ = (-1 + Real.sqrt 6) / 5 ∧ 
     x₂ = (-1 - Real.sqrt 6) / 5 ∧ 
     5 * x₁^2 + 2 * x₁ - 1 = 0 ∧ 
     5 * x₂^2 + 2 * x₂ - 1 = 0) ∧
    (x₃ = 3 ∧ 
     x₄ = -4 ∧ 
     x₃ * (x₃ - 3) - 4 * (3 - x₃) = 0 ∧ 
     x₄ * (x₄ - 3) - 4 * (3 - x₄) = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1865_186539


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l1865_186505

theorem sum_of_fourth_powers (a b : ℂ) 
  (h1 : (a + 1) * (b + 1) = 2)
  (h2 : (a^2 + 1) * (b^2 + 1) = 32) :
  ∃ x y : ℂ, 
    (x^4 + 1) * (y^4 + 1) + (a^4 + 1) * (b^4 + 1) = 1924 ∧
    ((x + 1) * (y + 1) = 2 ∧ (x^2 + 1) * (y^2 + 1) = 32) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l1865_186505


namespace NUMINAMATH_CALUDE_min_value_2a5_plus_a4_l1865_186552

/-- A geometric sequence with positive terms satisfying a specific condition -/
structure GeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  condition : 2 * a 4 + a 3 - 2 * a 2 - a 1 = 8

/-- The minimum value of 2a_5 + a_4 for the given geometric sequence -/
theorem min_value_2a5_plus_a4 (seq : GeometricSequence) :
  ∃ m : ℝ, m = 12 * Real.sqrt 3 ∧ ∀ x : ℝ, (2 * seq.a 5 + seq.a 4) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_2a5_plus_a4_l1865_186552


namespace NUMINAMATH_CALUDE_circumcircle_area_l1865_186588

theorem circumcircle_area (a b c : ℝ) (A B C : ℝ) (R : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  A < π / 2 ∧ B < π / 2 ∧ C < π / 2 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  c / Real.sin C = 2 * R ∧
  A = 5 * π / 12 ∧
  B = π / 4 ∧
  c = 4 →
  π * R^2 = 16 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_circumcircle_area_l1865_186588


namespace NUMINAMATH_CALUDE_m_zero_sufficient_not_necessary_l1865_186596

/-- Represents a quadratic equation in two variables -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- Checks if a quadratic equation represents a circle -/
def isCircle (eq : QuadraticEquation) : Prop :=
  eq.a = eq.b ∧ eq.a ≠ 0 ∧ eq.c^2 + eq.d^2 - 4 * eq.a * eq.f > 0

/-- The specific equation x^2 + y^2 - 4x + 2y + m = 0 -/
def specificEquation (m : ℝ) : QuadraticEquation :=
  { a := 1, b := 1, c := -4, d := 2, e := 0, f := m }

/-- Theorem stating that m = 0 is sufficient but not necessary for the equation to represent a circle -/
theorem m_zero_sufficient_not_necessary :
  (∀ m : ℝ, m = 0 → isCircle (specificEquation m)) ∧
  ¬(∀ m : ℝ, isCircle (specificEquation m) → m = 0) :=
sorry

end NUMINAMATH_CALUDE_m_zero_sufficient_not_necessary_l1865_186596


namespace NUMINAMATH_CALUDE_entertainment_committee_combinations_l1865_186561

theorem entertainment_committee_combinations (n : ℕ) : 
  (n.choose 3 = 20) → (n.choose 2 = 15) := by
  sorry

end NUMINAMATH_CALUDE_entertainment_committee_combinations_l1865_186561


namespace NUMINAMATH_CALUDE_power_product_equality_l1865_186562

theorem power_product_equality (x : ℝ) : 2 * (x^3 * x^2) = 2 * x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l1865_186562


namespace NUMINAMATH_CALUDE_triangle_area_with_perpendicular_medians_l1865_186540

-- Define the triangle XYZ
def Triangle (X Y Z : ℝ × ℝ) : Prop := sorry

-- Define a median of a triangle
def Median (A B C M : ℝ × ℝ) : Prop := sorry

-- Define perpendicular lines
def Perpendicular (A B C D : ℝ × ℝ) : Prop := sorry

-- Define the length of a line segment
def Length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the altitude of a triangle
def Altitude (A B C H : ℝ × ℝ) : Prop := sorry

-- Define the area of a triangle
def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

theorem triangle_area_with_perpendicular_medians 
  (X Y Z U V : ℝ × ℝ) 
  (h1 : Triangle X Y Z)
  (h2 : Median X Y Z U)
  (h3 : Median Y Z X V)
  (h4 : Perpendicular X U Y V)
  (h5 : Length X U = 10)
  (h6 : Length Y V = 24)
  (h7 : ∃ H, Altitude Z X Y H ∧ Length Z H = 16) :
  TriangleArea X Y Z = 160 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_perpendicular_medians_l1865_186540


namespace NUMINAMATH_CALUDE_project_hours_l1865_186571

theorem project_hours (total_hours : ℕ) (kate_hours : ℕ) : 
  total_hours = 135 → 
  2 * kate_hours + kate_hours + 6 * kate_hours = total_hours →
  6 * kate_hours - kate_hours = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_project_hours_l1865_186571


namespace NUMINAMATH_CALUDE_dice_roll_probability_l1865_186566

def is_valid_roll (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6

def meets_conditions (a b c : ℕ) : Prop :=
  a * b * c = 72 ∧ a + b + c = 13

def total_outcomes : ℕ := 6 * 6 * 6

def favorable_outcomes : ℕ := 6

theorem dice_roll_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 36 :=
sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l1865_186566


namespace NUMINAMATH_CALUDE_exists_common_divisor_l1865_186536

/-- A function from positive integers to integers greater than or equal to 2 -/
def PositiveIntegerFunction := ℕ+ → ℕ

/-- The property that f(m+n) divides f(m) + f(n) for all positive integers m and n -/
def HasDivisibilityProperty (f : PositiveIntegerFunction) : Prop :=
  ∀ m n : ℕ+, (f (m + n) : ℤ) ∣ (f m + f n : ℤ)

/-- The main theorem -/
theorem exists_common_divisor
  (f : PositiveIntegerFunction)
  (h1 : ∀ n : ℕ+, f n ≥ 2)
  (h2 : HasDivisibilityProperty f) :
  ∃ c : ℕ+, c > 1 ∧ ∀ n : ℕ+, (c : ℤ) ∣ (f n : ℤ) :=
sorry

end NUMINAMATH_CALUDE_exists_common_divisor_l1865_186536


namespace NUMINAMATH_CALUDE_workers_problem_l1865_186519

/-- Given a number of workers that can complete a job in 25 days, 
    and adding 10 workers reduces the time to 15 days, 
    prove that the original number of workers is 15. -/
theorem workers_problem (W : ℕ) : 
  W * 25 = (W + 10) * 15 → W = 15 := by sorry

end NUMINAMATH_CALUDE_workers_problem_l1865_186519


namespace NUMINAMATH_CALUDE_pen_cost_is_47_l1865_186558

/-- The cost of a pen in cents -/
def pen_cost : ℕ := 47

/-- The cost of a pencil in cents -/
def pencil_cost : ℕ := sorry

/-- Six pens and five pencils cost 380 cents -/
axiom condition1 : 6 * pen_cost + 5 * pencil_cost = 380

/-- Three pens and eight pencils cost 298 cents -/
axiom condition2 : 3 * pen_cost + 8 * pencil_cost = 298

/-- The cost of a pen is 47 cents -/
theorem pen_cost_is_47 : pen_cost = 47 := by sorry

end NUMINAMATH_CALUDE_pen_cost_is_47_l1865_186558


namespace NUMINAMATH_CALUDE_larrys_to_keiths_score_ratio_l1865_186518

/-- Given that Keith scored 3 points, Danny scored 5 more marks than Larry,
    and the total amount of marks scored by the three students is 26,
    prove that the ratio of Larry's score to Keith's score is 3:1 -/
theorem larrys_to_keiths_score_ratio (keith_score larry_score danny_score : ℕ) : 
  keith_score = 3 →
  danny_score = larry_score + 5 →
  keith_score + larry_score + danny_score = 26 →
  larry_score / keith_score = 3 / 1 := by
sorry

end NUMINAMATH_CALUDE_larrys_to_keiths_score_ratio_l1865_186518


namespace NUMINAMATH_CALUDE_length_to_breadth_ratio_l1865_186516

/-- Represents a rectangular plot -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  area : ℝ
  length_multiple_of_breadth : ∃ (k : ℝ), length = k * breadth
  area_eq : area = length * breadth

/-- Theorem: The ratio of length to breadth is 3:1 for a rectangular plot with area 2028 and breadth 26 -/
theorem length_to_breadth_ratio (plot : RectangularPlot) 
  (h_area : plot.area = 2028)
  (h_breadth : plot.breadth = 26) :
  plot.length / plot.breadth = 3 := by
sorry

end NUMINAMATH_CALUDE_length_to_breadth_ratio_l1865_186516


namespace NUMINAMATH_CALUDE_train_length_is_60_l1865_186594

/-- Two trains with equal length on parallel tracks -/
structure TrainSystem where
  train_length : ℝ
  fast_speed : ℝ
  slow_speed : ℝ
  passing_time : ℝ

/-- The train system satisfies the given conditions -/
def valid_train_system (ts : TrainSystem) : Prop :=
  ts.fast_speed = 72 * (5/18) ∧  -- 72 km/h in m/s
  ts.slow_speed = 54 * (5/18) ∧  -- 54 km/h in m/s
  ts.passing_time = 24

/-- Theorem stating that the length of each train is 60 meters -/
theorem train_length_is_60 (ts : TrainSystem) 
  (h : valid_train_system ts) : ts.train_length = 60 := by
  sorry

#check train_length_is_60

end NUMINAMATH_CALUDE_train_length_is_60_l1865_186594


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1865_186560

theorem polynomial_expansion (t : ℝ) : 
  (3 * t^2 - 2 * t + 3) * (-2 * t^2 + 3 * t - 4) = 
  -6 * t^4 + 13 * t^3 - 24 * t^2 + 17 * t - 12 := by sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1865_186560


namespace NUMINAMATH_CALUDE_prob_rain_at_least_one_day_l1865_186526

def prob_rain_friday : ℝ := 0.6
def prob_rain_saturday : ℝ := 0.7
def prob_rain_sunday : ℝ := 0.4

theorem prob_rain_at_least_one_day :
  let prob_no_rain_friday := 1 - prob_rain_friday
  let prob_no_rain_saturday := 1 - prob_rain_saturday
  let prob_no_rain_sunday := 1 - prob_rain_sunday
  let prob_no_rain_all_days := prob_no_rain_friday * prob_no_rain_saturday * prob_no_rain_sunday
  let prob_rain_at_least_one_day := 1 - prob_no_rain_all_days
  prob_rain_at_least_one_day = 0.928 := by
sorry

end NUMINAMATH_CALUDE_prob_rain_at_least_one_day_l1865_186526


namespace NUMINAMATH_CALUDE_sqrt_product_difference_l1865_186568

theorem sqrt_product_difference (x y z w : ℝ) : 
  x = Real.sqrt 108 → 
  y = Real.sqrt 128 → 
  z = Real.sqrt 6 → 
  w = Real.sqrt 18 → 
  x * y * z - w = 288 - 3 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_difference_l1865_186568
