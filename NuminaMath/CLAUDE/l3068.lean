import Mathlib

namespace NUMINAMATH_CALUDE_dolly_initial_tickets_l3068_306807

/- Define the number of rides for each attraction -/
def ferris_wheel_rides : Nat := 2
def roller_coaster_rides : Nat := 3
def log_ride_rides : Nat := 7

/- Define the ticket cost for each attraction -/
def ferris_wheel_cost : Nat := 2
def roller_coaster_cost : Nat := 5
def log_ride_cost : Nat := 1

/- Define the additional tickets needed -/
def additional_tickets : Nat := 6

/- Theorem to prove -/
theorem dolly_initial_tickets : 
  (ferris_wheel_rides * ferris_wheel_cost + 
   roller_coaster_rides * roller_coaster_cost + 
   log_ride_rides * log_ride_cost) - 
  additional_tickets = 20 := by
  sorry

end NUMINAMATH_CALUDE_dolly_initial_tickets_l3068_306807


namespace NUMINAMATH_CALUDE_infinitely_many_unreachable_integers_l3068_306875

/-- Sum of digits in base b -/
def sum_of_digits (b : ℕ) (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem infinitely_many_unreachable_integers (b : ℕ) (h : b ≥ 2) :
  ∀ M : ℕ, ∃ S : Finset ℕ, (Finset.card S = M) ∧ 
  (∀ k ∈ S, ∀ n : ℕ, n + sum_of_digits b n ≠ k) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_unreachable_integers_l3068_306875


namespace NUMINAMATH_CALUDE_optimal_route_unchanged_for_given_network_l3068_306846

/-- Represents the transportation network of a country -/
structure TransportNetwork where
  num_cities : Nat
  capital_travel_time : Real
  city_connection_time : Real
  initial_transfer_time : Real
  reduced_transfer_time : Real

/-- Calculates the travel time via the capital -/
def time_via_capital (network : TransportNetwork) (transfer_time : Real) : Real :=
  2 * network.capital_travel_time + transfer_time

/-- Calculates the maximum travel time via cyclic connections -/
def time_via_cycle (network : TransportNetwork) (transfer_time : Real) : Real :=
  5 * network.city_connection_time + 4 * transfer_time

/-- Determines if the optimal route remains unchanged after reducing transfer time -/
def optimal_route_unchanged (network : TransportNetwork) : Prop :=
  let initial_time_via_capital := time_via_capital network network.initial_transfer_time
  let initial_time_via_cycle := time_via_cycle network network.initial_transfer_time
  let reduced_time_via_capital := time_via_capital network network.reduced_transfer_time
  let reduced_time_via_cycle := time_via_cycle network network.reduced_transfer_time
  (initial_time_via_capital ≤ initial_time_via_cycle) ∧
  (reduced_time_via_capital ≤ reduced_time_via_cycle)

theorem optimal_route_unchanged_for_given_network :
  optimal_route_unchanged
    { num_cities := 11
    , capital_travel_time := 7
    , city_connection_time := 3
    , initial_transfer_time := 2
    , reduced_transfer_time := 1.5 } := by
  sorry

end NUMINAMATH_CALUDE_optimal_route_unchanged_for_given_network_l3068_306846


namespace NUMINAMATH_CALUDE_complement_A_in_S_l3068_306825

def S : Set ℕ := {x | 0 ≤ x ∧ x ≤ 5}
def A : Set ℕ := {x | 1 < x ∧ x < 5}

theorem complement_A_in_S : 
  (S \ A) = {0, 1, 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_S_l3068_306825


namespace NUMINAMATH_CALUDE_average_mark_first_class_l3068_306886

theorem average_mark_first_class 
  (n1 : ℕ) (n2 : ℕ) (avg2 : ℝ) (avg_total : ℝ)
  (h1 : n1 = 30)
  (h2 : n2 = 50)
  (h3 : avg2 = 80)
  (h4 : avg_total = 65) :
  (n1 + n2) * avg_total = n1 * ((n1 + n2) * avg_total - n2 * avg2) / n1 + n2 * avg2 :=
by sorry

end NUMINAMATH_CALUDE_average_mark_first_class_l3068_306886


namespace NUMINAMATH_CALUDE_three_greater_than_negative_five_l3068_306873

theorem three_greater_than_negative_five : 3 > -5 := by
  sorry

end NUMINAMATH_CALUDE_three_greater_than_negative_five_l3068_306873


namespace NUMINAMATH_CALUDE_distinct_primes_in_product_l3068_306879

theorem distinct_primes_in_product : 
  let n := 12 * 13 * 14 * 15
  Finset.card (Nat.factors n).toFinset = 5 := by
sorry

end NUMINAMATH_CALUDE_distinct_primes_in_product_l3068_306879


namespace NUMINAMATH_CALUDE_x_x_minus_3_is_quadratic_l3068_306860

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation_in_one_variable (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x(x-3) = 0 -/
def f (x : ℝ) : ℝ := x * (x - 3)

/-- Theorem: x(x-3) = 0 is a quadratic equation in one variable -/
theorem x_x_minus_3_is_quadratic : is_quadratic_equation_in_one_variable f := by
  sorry


end NUMINAMATH_CALUDE_x_x_minus_3_is_quadratic_l3068_306860


namespace NUMINAMATH_CALUDE_points_per_treasure_l3068_306876

theorem points_per_treasure (total_treasures : ℕ) (total_score : ℕ) (points_per_treasure : ℕ) : 
  total_treasures = 7 → total_score = 63 → points_per_treasure * total_treasures = total_score → points_per_treasure = 9 := by
  sorry

end NUMINAMATH_CALUDE_points_per_treasure_l3068_306876


namespace NUMINAMATH_CALUDE_march_first_is_wednesday_l3068_306885

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a date in March -/
structure MarchDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Function to get the day of the week for a given number of days before a reference day -/
def daysBefore (referenceDay : DayOfWeek) (days : Nat) : DayOfWeek :=
  sorry

theorem march_first_is_wednesday (march13 : MarchDate) 
  (h : march13.day = 13 ∧ march13.dayOfWeek = DayOfWeek.Monday) :
  ∃ (march1 : MarchDate), march1.day = 1 ∧ march1.dayOfWeek = DayOfWeek.Wednesday :=
  sorry

end NUMINAMATH_CALUDE_march_first_is_wednesday_l3068_306885


namespace NUMINAMATH_CALUDE_concert_ticket_price_l3068_306870

/-- The price of each ticket in dollars -/
def ticket_price : ℚ := 4

/-- The total number of tickets bought -/
def total_tickets : ℕ := 8

/-- The total amount spent in dollars -/
def total_spent : ℚ := 32

theorem concert_ticket_price : 
  ticket_price * total_tickets = total_spent :=
sorry

end NUMINAMATH_CALUDE_concert_ticket_price_l3068_306870


namespace NUMINAMATH_CALUDE_exam_probability_l3068_306819

/-- The probability of passing the exam -/
def prob_pass : ℚ := 4/7

/-- The probability of not passing the exam -/
def prob_not_pass : ℚ := 1 - prob_pass

theorem exam_probability : prob_not_pass = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_exam_probability_l3068_306819


namespace NUMINAMATH_CALUDE_cubic_root_identity_l3068_306853

theorem cubic_root_identity (a b : ℝ) (h1 : a ≠ b) (h2 : (Real.rpow a (1/3) + Real.rpow b (1/3))^3 = a^2 * b^2) : 
  (3*a + 1)*(3*b + 1) - 3*a^2*b^2 = 1 := by sorry

end NUMINAMATH_CALUDE_cubic_root_identity_l3068_306853


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_a_value_l3068_306882

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / 4 - y^2 / a = 1

-- Define the asymptotes of the hyperbola
def asymptote (a : ℝ) (x y : ℝ) : Prop := y = (Real.sqrt a / 2) * x ∨ y = -(Real.sqrt a / 2) * x

-- Theorem statement
theorem hyperbola_asymptote_a_value :
  ∀ a : ℝ, a > 1 →
  asymptote a 2 (Real.sqrt 3) →
  hyperbola a 2 (Real.sqrt 3) →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_a_value_l3068_306882


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3068_306839

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 8 →                  -- One leg measures 8 meters
  (1/2) * a * b = 48 →     -- Area is 48 square meters
  a^2 + b^2 = c^2 →        -- Pythagorean theorem for right triangle
  c = 4 * Real.sqrt 13 :=  -- Hypotenuse length is 4√13 meters
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3068_306839


namespace NUMINAMATH_CALUDE_people_speaking_both_languages_l3068_306893

/-- Given a group of people with specified language abilities, calculate the number who speak both languages. -/
theorem people_speaking_both_languages 
  (total : ℕ) 
  (latin : ℕ) 
  (french : ℕ) 
  (neither : ℕ) 
  (h_total : total = 25)
  (h_latin : latin = 13)
  (h_french : french = 15)
  (h_neither : neither = 6) :
  latin + french - (total - neither) = 9 := by
sorry

end NUMINAMATH_CALUDE_people_speaking_both_languages_l3068_306893


namespace NUMINAMATH_CALUDE_no_solution_exists_l3068_306845

theorem no_solution_exists : ¬∃ (a b : ℕ+), a^2 - 23 = b^11 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3068_306845


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3068_306842

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.sin y = 2010)
  (h2 : x + 2010 * Real.cos y = 2009)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2009 + Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3068_306842


namespace NUMINAMATH_CALUDE_inequality_proof_l3068_306878

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hsum : a + b + c = 1) :
  3/16 ≤ (a/(1+a))^2 + (b/(1+b))^2 + (c/(1+c))^2 ∧ (a/(1+a))^2 + (b/(1+b))^2 + (c/(1+c))^2 ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3068_306878


namespace NUMINAMATH_CALUDE_pedal_triangle_largest_angle_l3068_306830

/-- Represents an acute triangle with vertices A, B, C and corresponding angles α, β, γ. -/
structure AcuteTriangle where
  α : Real
  β : Real
  γ : Real
  acute_angles : α ≤ β ∧ β ≤ γ ∧ γ < Real.pi / 2
  angle_sum : α + β + γ = Real.pi

/-- Represents the pedal triangle of an acute triangle. -/
def PedalTriangle (t : AcuteTriangle) : Prop :=
  ∃ (largest_pedal_angle : Real),
    largest_pedal_angle = Real.pi - 2 * t.α ∧
    largest_pedal_angle ≥ t.γ

/-- 
Theorem: The largest angle in the pedal triangle of an acute triangle is at least 
as large as the largest angle in the original triangle. Equality holds when the 
original triangle is isosceles with the equal angles at least 60°.
-/
theorem pedal_triangle_largest_angle (t : AcuteTriangle) : 
  PedalTriangle t ∧ 
  (Real.pi - 2 * t.α = t.γ ↔ t.α = t.β ∧ t.γ ≥ Real.pi / 3) := by
  sorry


end NUMINAMATH_CALUDE_pedal_triangle_largest_angle_l3068_306830


namespace NUMINAMATH_CALUDE_machine_production_l3068_306828

/-- Given that 4 machines produce x units in 6 days at a constant rate,
    prove that 16 machines will produce 2x units in 3 days at the same rate. -/
theorem machine_production (x : ℝ) : 
  (∃ (rate : ℝ), rate > 0 ∧ 4 * rate * 6 = x) →
  (∃ (output : ℝ), 16 * (x / (4 * 6)) * 3 = output ∧ output = 2 * x) :=
by sorry

end NUMINAMATH_CALUDE_machine_production_l3068_306828


namespace NUMINAMATH_CALUDE_existence_of_subset_l3068_306801

theorem existence_of_subset (n : ℕ+) (t : ℝ) (a : Fin (2*n.val-1) → ℝ) (ht : t ≠ 0) :
  ∃ (s : Finset (Fin (2*n.val-1))), s.card = n.val ∧
    ∀ (i j : Fin (2*n.val-1)), i ∈ s → j ∈ s → i ≠ j → a i - a j ≠ t :=
sorry

end NUMINAMATH_CALUDE_existence_of_subset_l3068_306801


namespace NUMINAMATH_CALUDE_six_legs_is_insect_l3068_306834

/-- Represents an animal with a certain number of legs -/
structure Animal where
  legs : ℕ

/-- Definition of an insect based on number of legs -/
def is_insect (a : Animal) : Prop := a.legs = 6

/-- Theorem stating that an animal with 6 legs satisfies the definition of an insect -/
theorem six_legs_is_insect (a : Animal) (h : a.legs = 6) : is_insect a := by
  sorry

end NUMINAMATH_CALUDE_six_legs_is_insect_l3068_306834


namespace NUMINAMATH_CALUDE_range_of_m_l3068_306863

theorem range_of_m (x : ℝ) (m : ℝ) : 
  (∃ x ∈ Set.Ioo (π/2) π, 2 * Real.sin x ^ 2 - Real.sqrt 3 * Real.sin (2 * x) + m - 1 = 0) →
  m ∈ Set.Ioo (-2) (-1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3068_306863


namespace NUMINAMATH_CALUDE_exponent_division_l3068_306835

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^6 / a^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3068_306835


namespace NUMINAMATH_CALUDE_divisors_of_2018_or_2019_l3068_306877

theorem divisors_of_2018_or_2019 (h1 : Nat.Prime 673) (h2 : Nat.Prime 1009) :
  (Finset.filter (fun n => n ∣ 2018 ∨ n ∣ 2019) (Finset.range 2020)).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_2018_or_2019_l3068_306877


namespace NUMINAMATH_CALUDE_boxes_left_l3068_306894

/-- The number of boxes Jerry started with -/
def initial_boxes : ℕ := 10

/-- The number of boxes Jerry sold -/
def sold_boxes : ℕ := 5

/-- Theorem: Jerry has 5 boxes left after selling -/
theorem boxes_left : initial_boxes - sold_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_boxes_left_l3068_306894


namespace NUMINAMATH_CALUDE_no_integer_coordinate_equilateral_triangle_l3068_306868

theorem no_integer_coordinate_equilateral_triangle :
  ¬ ∃ (A B C : ℤ × ℤ), 
    (A.1 ≠ B.1 ∨ A.2 ≠ B.2) ∧ 
    (B.1 ≠ C.1 ∨ B.2 ≠ C.2) ∧ 
    (C.1 ≠ A.1 ∨ C.2 ≠ A.2) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 := by
  sorry


end NUMINAMATH_CALUDE_no_integer_coordinate_equilateral_triangle_l3068_306868


namespace NUMINAMATH_CALUDE_sixth_root_equation_solution_l3068_306871

theorem sixth_root_equation_solution (x : ℝ) :
  (x^2 * (x^4)^(1/3))^(1/6) = 4 ↔ x = 4^(18/5) := by sorry

end NUMINAMATH_CALUDE_sixth_root_equation_solution_l3068_306871


namespace NUMINAMATH_CALUDE_triangle_third_side_l3068_306832

theorem triangle_third_side (a b m : ℝ) (ha : a = 11) (hb : b = 23) (hm : m = 10) :
  ∃ c : ℝ, c = 30 ∧ m^2 = (2*a^2 + 2*b^2 - c^2) / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_l3068_306832


namespace NUMINAMATH_CALUDE_sqrt_x6_plus_x4_l3068_306889

theorem sqrt_x6_plus_x4 (x : ℝ) : Real.sqrt (x^6 + x^4) = |x|^2 * Real.sqrt (x^2 + 1) := by sorry

end NUMINAMATH_CALUDE_sqrt_x6_plus_x4_l3068_306889


namespace NUMINAMATH_CALUDE_max_product_ab_l3068_306831

theorem max_product_ab (a b : ℝ) 
  (h : ∀ x : ℝ, Real.exp x ≥ a * (x - 1) + b) : 
  a * b ≤ (1/2) * Real.exp 3 := by
sorry

end NUMINAMATH_CALUDE_max_product_ab_l3068_306831


namespace NUMINAMATH_CALUDE_altered_detergent_amount_is_180_l3068_306826

/-- Represents the ratio of components in a cleaning solution -/
structure CleaningSolution :=
  (bleach : ℚ)
  (detergent : ℚ)
  (fabricSoftener : ℚ)
  (water : ℚ)

/-- Calculates the amount of detergent in the altered solution -/
def alteredDetergentAmount (original : CleaningSolution) (alteredWaterAmount : ℚ) : ℚ :=
  let bleachToDetergentRatio := original.bleach / original.detergent * 3
  let fabricSoftenerToDetergentRatio := (original.fabricSoftener / original.detergent) / 2
  let detergentToWaterRatio := (original.detergent / original.water) * (2/3)
  
  let newDetergentToWaterRatio := detergentToWaterRatio * alteredWaterAmount
  
  newDetergentToWaterRatio

/-- Theorem stating that the altered solution contains 180 liters of detergent -/
theorem altered_detergent_amount_is_180 :
  let original := CleaningSolution.mk 4 40 60 100
  let alteredWaterAmount := 300
  alteredDetergentAmount original alteredWaterAmount = 180 := by
  sorry

end NUMINAMATH_CALUDE_altered_detergent_amount_is_180_l3068_306826


namespace NUMINAMATH_CALUDE_beths_crayon_packs_l3068_306800

/-- The number of crayon packs Beth has after distribution and finding more -/
def beths_total_packs (initial_packs : ℚ) (total_friends : ℕ) (new_packs : ℚ) : ℚ :=
  (initial_packs / total_friends) + new_packs

/-- Theorem stating Beth's total packs under the given conditions -/
theorem beths_crayon_packs : 
  beths_total_packs 4 10 6 = 6.4 := by sorry

end NUMINAMATH_CALUDE_beths_crayon_packs_l3068_306800


namespace NUMINAMATH_CALUDE_BF_length_is_four_l3068_306861

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D E F : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_right_angled_at_A_and_C (q : Quadrilateral) : Prop := sorry
def E_and_F_on_AC (q : Quadrilateral) : Prop := sorry
def DE_perpendicular_to_AC (q : Quadrilateral) : Prop := sorry
def BF_perpendicular_to_AC (q : Quadrilateral) : Prop := sorry

-- Define the given lengths
def AE_length (q : Quadrilateral) : ℝ := 4
def DE_length (q : Quadrilateral) : ℝ := 6
def CE_length (q : Quadrilateral) : ℝ := 6

-- Define the length of BF
def BF_length (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem BF_length_is_four (q : Quadrilateral) 
  (h1 : is_right_angled_at_A_and_C q)
  (h2 : E_and_F_on_AC q)
  (h3 : DE_perpendicular_to_AC q)
  (h4 : BF_perpendicular_to_AC q) :
  BF_length q = 4 := by sorry

end NUMINAMATH_CALUDE_BF_length_is_four_l3068_306861


namespace NUMINAMATH_CALUDE_second_rewind_time_l3068_306824

theorem second_rewind_time (total_time first_segment first_rewind second_segment third_segment : ℕ) : 
  total_time = 120 ∧ 
  first_segment = 35 ∧ 
  first_rewind = 5 ∧ 
  second_segment = 45 ∧ 
  third_segment = 20 → 
  total_time - (first_segment + first_rewind + second_segment + third_segment) = 15 := by
  sorry

end NUMINAMATH_CALUDE_second_rewind_time_l3068_306824


namespace NUMINAMATH_CALUDE_remaining_bulbs_correct_l3068_306841

def calculate_remaining_bulbs (initial_led : ℕ) (initial_incandescent : ℕ)
  (used_led : ℕ) (used_incandescent : ℕ)
  (alex_percent : ℚ) (bob_percent : ℚ) (charlie_led_percent : ℚ) (charlie_incandescent_percent : ℚ)
  : (ℕ × ℕ) :=
  sorry

theorem remaining_bulbs_correct :
  let initial_led := 24
  let initial_incandescent := 16
  let used_led := 10
  let used_incandescent := 6
  let alex_percent := 1/2
  let bob_percent := 1/4
  let charlie_led_percent := 1/5
  let charlie_incandescent_percent := 3/10
  calculate_remaining_bulbs initial_led initial_incandescent used_led used_incandescent
    alex_percent bob_percent charlie_led_percent charlie_incandescent_percent = (6, 6) :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_bulbs_correct_l3068_306841


namespace NUMINAMATH_CALUDE_proportion_third_number_l3068_306862

theorem proportion_third_number (y : ℝ) : 
  (0.75 : ℝ) / 1.05 = y / 7 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_proportion_third_number_l3068_306862


namespace NUMINAMATH_CALUDE_amount_from_cars_and_buses_is_309_l3068_306849

/-- Calculates the amount raised from cars and buses given the total amount raised and the amounts from other vehicle types. -/
def amount_from_cars_and_buses (total_raised : ℕ) (suv_charge truck_charge motorcycle_charge : ℕ) (num_suvs num_trucks num_motorcycles : ℕ) : ℕ :=
  total_raised - (suv_charge * num_suvs + truck_charge * num_trucks + motorcycle_charge * num_motorcycles)

/-- Theorem stating that the amount raised from cars and buses is $309. -/
theorem amount_from_cars_and_buses_is_309 :
  amount_from_cars_and_buses 500 12 10 15 3 8 5 = 309 := by
  sorry

end NUMINAMATH_CALUDE_amount_from_cars_and_buses_is_309_l3068_306849


namespace NUMINAMATH_CALUDE_expansion_temperature_difference_l3068_306856

-- Define the initial conditions and coefficients
def initial_length : ℝ := 2
def initial_temp : ℝ := 80
def alpha_iron : ℝ := 0.0000118
def alpha_zinc : ℝ := 0.000031
def length_difference : ℝ := 0.0015

-- Define the function for the length of a rod at temperature x
def rod_length (alpha : ℝ) (x : ℝ) : ℝ :=
  initial_length * (1 + alpha * (x - initial_temp))

-- Define the theorem to prove
theorem expansion_temperature_difference :
  ∃ (x₁ x₂ : ℝ),
    x₁ ≠ x₂ ∧
    (rod_length alpha_zinc x₁ - rod_length alpha_iron x₁ = length_difference ∨
     rod_length alpha_iron x₁ - rod_length alpha_zinc x₁ = length_difference) ∧
    (rod_length alpha_zinc x₂ - rod_length alpha_iron x₂ = length_difference ∨
     rod_length alpha_iron x₂ - rod_length alpha_zinc x₂ = length_difference) ∧
    ((x₁ = 119 ∧ x₂ = 41) ∨ (x₁ = 41 ∧ x₂ = 119)) :=
sorry

end NUMINAMATH_CALUDE_expansion_temperature_difference_l3068_306856


namespace NUMINAMATH_CALUDE_tg_plus_ctg_values_l3068_306895

-- Define the trigonometric functions
noncomputable def sec (x : ℝ) : ℝ := 1 / Real.cos x
noncomputable def cosec (x : ℝ) : ℝ := 1 / Real.sin x
noncomputable def tg (x : ℝ) : ℝ := Real.tan x
noncomputable def ctg (x : ℝ) : ℝ := 1 / Real.tan x

-- State the theorem
theorem tg_plus_ctg_values (x : ℝ) :
  sec x - cosec x = 4 * Real.sqrt 3 →
  (tg x + ctg x = -6 ∨ tg x + ctg x = 8) :=
by sorry

end NUMINAMATH_CALUDE_tg_plus_ctg_values_l3068_306895


namespace NUMINAMATH_CALUDE_boat_trip_distance_l3068_306857

/-- Proves that given a boat with speed 8 kmph in standing water, a stream with speed 6 kmph,
    and a round trip time of 120 hours, the distance to the destination is 210 km. -/
theorem boat_trip_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : boat_speed = 8) 
  (h2 : stream_speed = 6) 
  (h3 : total_time = 120) : 
  (boat_speed + stream_speed) * (boat_speed - stream_speed) * total_time / 
  (2 * (boat_speed + stream_speed) * (boat_speed - stream_speed)) = 210 := by
  sorry

end NUMINAMATH_CALUDE_boat_trip_distance_l3068_306857


namespace NUMINAMATH_CALUDE_paving_cost_calculation_l3068_306808

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a rectangular floor with length 5.5 m and width 3.75 m at a rate of Rs. 400 per square metre is Rs. 8250 -/
theorem paving_cost_calculation :
  paving_cost 5.5 3.75 400 = 8250 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_calculation_l3068_306808


namespace NUMINAMATH_CALUDE_slope_equation_l3068_306884

theorem slope_equation (m : ℝ) (h1 : m > 0) 
  (h2 : (m - 3) / (1 - m) = m) : m = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_slope_equation_l3068_306884


namespace NUMINAMATH_CALUDE_complex_sum_representation_l3068_306847

theorem complex_sum_representation : ∃ (r θ : ℝ), 
  15 * Complex.exp (Complex.I * (π / 7)) + 15 * Complex.exp (Complex.I * (9 * π / 14)) = r * Complex.exp (Complex.I * θ) ∧ 
  r = 15 * Real.sqrt 2 ∧ 
  θ = 11 * π / 28 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_representation_l3068_306847


namespace NUMINAMATH_CALUDE_number_equation_l3068_306854

theorem number_equation (x : ℝ) : 0.833 * x = -60 → x = -72 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l3068_306854


namespace NUMINAMATH_CALUDE_rooks_diagonal_move_l3068_306859

/-- Represents a position on an 8x8 chessboard -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents a configuration of 8 rooks on an 8x8 chessboard -/
structure RookConfiguration :=
  (positions : Fin 8 → Position)
  (no_attacks : ∀ i j, i ≠ j → 
    (positions i).row ≠ (positions j).row ∧ 
    (positions i).col ≠ (positions j).col)

/-- Checks if a position is adjacent diagonally to another position -/
def is_adjacent_diagonal (p1 p2 : Position) : Prop :=
  (p1.row.val + 1 = p2.row.val ∧ p1.col.val + 1 = p2.col.val) ∨
  (p1.row.val + 1 = p2.row.val ∧ p1.col.val = p2.col.val + 1) ∨
  (p1.row.val = p2.row.val + 1 ∧ p1.col.val + 1 = p2.col.val) ∨
  (p1.row.val = p2.row.val + 1 ∧ p1.col.val = p2.col.val + 1)

/-- The main theorem to be proved -/
theorem rooks_diagonal_move (initial : RookConfiguration) :
  ∃ (final : RookConfiguration),
    ∀ i, is_adjacent_diagonal (initial.positions i) (final.positions i) :=
sorry

end NUMINAMATH_CALUDE_rooks_diagonal_move_l3068_306859


namespace NUMINAMATH_CALUDE_sector_circumradius_l3068_306883

theorem sector_circumradius (r : ℝ) (θ : ℝ) (h1 : r = 8) (h2 : θ = 2 * π / 3) :
  let R := r / (2 * Real.sin (θ / 2))
  R = 8 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_sector_circumradius_l3068_306883


namespace NUMINAMATH_CALUDE_sequence_periodicity_l3068_306806

theorem sequence_periodicity 
  (a b : ℕ → ℤ) 
  (h : ∀ n ≥ 3, (a n - a (n-1)) * (a n - a (n-2)) + (b n - b (n-1)) * (b n - b (n-2)) = 0) :
  ∃ k : ℕ+, a k = a (k + 2008) :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l3068_306806


namespace NUMINAMATH_CALUDE_can_mark_any_rational_ratio_l3068_306814

/-- Represents the ability to mark points on a segment -/
structure SegmentMarker where
  /-- Mark a point that divides a segment in half -/
  mark_half : ∀ (a b : ℝ), ∃ (c : ℝ), c = (a + b) / 2
  /-- Mark a point that divides a segment in the ratio n:(n+1) -/
  mark_ratio : ∀ (a b : ℝ) (n : ℕ), ∃ (c : ℝ), (c - a) / (b - c) = n / (n + 1)

/-- Theorem stating that with given marking abilities, any rational ratio can be achieved -/
theorem can_mark_any_rational_ratio (marker : SegmentMarker) :
  ∀ (a b : ℝ) (p q : ℕ), ∃ (c : ℝ), (c - a) / (b - c) = p / q :=
sorry

end NUMINAMATH_CALUDE_can_mark_any_rational_ratio_l3068_306814


namespace NUMINAMATH_CALUDE_only_q_is_true_l3068_306815

theorem only_q_is_true (p q m : Prop) 
  (h1 : (p ∨ q ∨ m) ∧ (¬(p ∧ q) ∧ ¬(p ∧ m) ∧ ¬(q ∧ m)))  -- Only one of p, q, and m is true
  (h2 : (p ∨ ¬(p ∨ q) ∨ m) ∧ (¬(p ∧ ¬(p ∨ q)) ∧ ¬(p ∧ m) ∧ ¬(¬(p ∨ q) ∧ m)))  -- Only one judgment is incorrect
  : q := by
sorry


end NUMINAMATH_CALUDE_only_q_is_true_l3068_306815


namespace NUMINAMATH_CALUDE_pizzeria_sales_l3068_306833

theorem pizzeria_sales (small_price large_price total_sales small_count : ℕ)
  (h1 : small_price = 2)
  (h2 : large_price = 8)
  (h3 : total_sales = 40)
  (h4 : small_count = 8) :
  ∃ large_count : ℕ,
    large_count * large_price + small_count * small_price = total_sales ∧
    large_count = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_pizzeria_sales_l3068_306833


namespace NUMINAMATH_CALUDE_paper_cutting_equations_l3068_306837

/-- Represents the paper cutting scenario in a seventh-grade class. -/
theorem paper_cutting_equations (x y : ℕ) : 
  (x + y = 12 ∧ 6 * x = 3 * (4 * y)) ↔ 
  (x = number_of_sheets_for_stars ∧ 
   y = number_of_sheets_for_flowers ∧ 
   total_sheets_used = 12 ∧ 
   stars_per_sheet = 6 ∧ 
   flowers_per_sheet = 4 ∧ 
   total_stars = 3 * total_flowers) :=
sorry

end NUMINAMATH_CALUDE_paper_cutting_equations_l3068_306837


namespace NUMINAMATH_CALUDE_office_clerks_count_l3068_306848

/-- Calculates the number of clerks in an office given specific salary information. -/
theorem office_clerks_count (total_avg : ℚ) (officer_avg : ℚ) (clerk_avg : ℚ) (officer_count : ℕ) :
  total_avg = 90 →
  officer_avg = 600 →
  clerk_avg = 84 →
  officer_count = 2 →
  ∃ (clerk_count : ℕ), 
    (officer_count * officer_avg + clerk_count * clerk_avg) / (officer_count + clerk_count) = total_avg ∧
    clerk_count = 170 :=
by sorry

end NUMINAMATH_CALUDE_office_clerks_count_l3068_306848


namespace NUMINAMATH_CALUDE_photo_arrangements_count_l3068_306855

/-- The number of people in the group --/
def group_size : ℕ := 5

/-- The number of arrangements where two specific people are adjacent --/
def adjacent_arrangements : ℕ := 2 * (group_size - 1).factorial

/-- The number of arrangements where three specific people are adjacent --/
def triple_adjacent_arrangements : ℕ := 2 * (group_size - 2).factorial

/-- The number of valid arrangements --/
def valid_arrangements : ℕ := adjacent_arrangements - triple_adjacent_arrangements

theorem photo_arrangements_count : valid_arrangements = 36 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangements_count_l3068_306855


namespace NUMINAMATH_CALUDE_a_range_for_increasing_f_l3068_306892

/-- A cubic function f(x) that is increasing on the entire real line. -/
noncomputable def f (a : ℝ) : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + 7*a*x

/-- The property that f is increasing on the entire real line. -/
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- The theorem stating the range of a for which f is increasing. -/
theorem a_range_for_increasing_f :
  ∀ a : ℝ, is_increasing (f a) ↔ 0 ≤ a ∧ a ≤ 21 :=
sorry

end NUMINAMATH_CALUDE_a_range_for_increasing_f_l3068_306892


namespace NUMINAMATH_CALUDE_sin_sum_identity_l3068_306850

theorem sin_sum_identity : 
  Real.sin (13 * π / 180) * Real.sin (58 * π / 180) + 
  Real.sin (77 * π / 180) * Real.sin (32 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_identity_l3068_306850


namespace NUMINAMATH_CALUDE_new_students_weight_l3068_306858

/-- Given a class of 8 students, prove that when two students weighing 70kg and 80kg 
    are replaced and the average weight decreases by 2 kg, 
    the combined weight of the two new students is 134 kg. -/
theorem new_students_weight (total_weight : ℝ) : 
  (total_weight - 150 + 134) / 8 = total_weight / 8 - 2 := by
  sorry

end NUMINAMATH_CALUDE_new_students_weight_l3068_306858


namespace NUMINAMATH_CALUDE_number_of_boys_l3068_306867

theorem number_of_boys (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 300 →
  boys + girls = total →
  girls = boys * total / 100 →
  boys = 75 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l3068_306867


namespace NUMINAMATH_CALUDE_union_condition_intersection_empty_l3068_306891

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 7}
def B (t : ℝ) : Set ℝ := {x : ℝ | t + 1 ≤ x ∧ x ≤ 2*t - 2}

-- Statement 1: A ∪ B = A if and only if t ∈ (-∞, 9/2]
theorem union_condition (t : ℝ) : A ∪ B t = A ↔ t ≤ 9/2 := by sorry

-- Statement 2: A ∩ B = ∅ if and only if t ∈ (-∞, 3) ∪ (6, +∞)
theorem intersection_empty (t : ℝ) : A ∩ B t = ∅ ↔ t < 3 ∨ t > 6 := by sorry

end NUMINAMATH_CALUDE_union_condition_intersection_empty_l3068_306891


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3068_306864

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_union_theorem :
  (U \ A) ∪ (U \ B) = {0, 1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3068_306864


namespace NUMINAMATH_CALUDE_marys_max_earnings_l3068_306865

/-- Represents Mary's work schedule and pay structure --/
structure WorkSchedule where
  maxHours : Nat
  regularRate : ℕ
  overtimeRate1 : ℕ
  overtimeRate2 : ℕ
  weekendBonus : ℕ
  milestoneBonus : ℕ

/-- Calculates the maximum weekly earnings based on the given work schedule --/
def maxWeeklyEarnings (schedule : WorkSchedule) : ℕ :=
  let regularPay := schedule.regularRate * 40
  let overtimePay1 := schedule.overtimeRate1 * 10
  let overtimePay2 := schedule.overtimeRate2 * 10
  let weekendBonus := schedule.weekendBonus * 2
  regularPay + overtimePay1 + overtimePay2 + weekendBonus + schedule.milestoneBonus

/-- Mary's work schedule --/
def marysSchedule : WorkSchedule := {
  maxHours := 60
  regularRate := 10
  overtimeRate1 := 12
  overtimeRate2 := 15
  weekendBonus := 50
  milestoneBonus := 100
}

/-- Theorem stating that Mary's maximum weekly earnings are $875 --/
theorem marys_max_earnings :
  maxWeeklyEarnings marysSchedule = 875 := by
  sorry


end NUMINAMATH_CALUDE_marys_max_earnings_l3068_306865


namespace NUMINAMATH_CALUDE_line_property_l3068_306811

/-- Given a line passing through points (2, -1) and (-1, 6), prove that 3m - 2b = -19 where m is the slope and b is the y-intercept -/
theorem line_property (m b : ℚ) : 
  (∀ (x y : ℚ), (x = 2 ∧ y = -1) ∨ (x = -1 ∧ y = 6) → y = m * x + b) →
  3 * m - 2 * b = -19 := by
  sorry

end NUMINAMATH_CALUDE_line_property_l3068_306811


namespace NUMINAMATH_CALUDE_factorial_ratio_l3068_306813

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 48 = 2450 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3068_306813


namespace NUMINAMATH_CALUDE_first_year_after_2020_with_digit_sum_18_l3068_306817

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isValidYear (year : ℕ) : Prop :=
  year > 2020 ∧ sumOfDigits year = 18

theorem first_year_after_2020_with_digit_sum_18 :
  ∀ year : ℕ, isValidYear year → year ≥ 2799 :=
sorry

end NUMINAMATH_CALUDE_first_year_after_2020_with_digit_sum_18_l3068_306817


namespace NUMINAMATH_CALUDE_luke_trivia_score_l3068_306844

/-- Luke's trivia game score calculation -/
theorem luke_trivia_score (points_per_round : ℕ) (num_rounds : ℕ) :
  points_per_round = 146 →
  num_rounds = 157 →
  points_per_round * num_rounds = 22822 := by
  sorry

end NUMINAMATH_CALUDE_luke_trivia_score_l3068_306844


namespace NUMINAMATH_CALUDE_turnip_bag_weights_l3068_306829

def bag_weights : List Nat := [13, 15, 16, 17, 21, 24]

def total_weight : Nat := bag_weights.sum

structure BagDistribution where
  turnip_weight : Nat
  onion_weights : List Nat
  carrot_weights : List Nat

def is_valid_distribution (d : BagDistribution) : Prop :=
  d.turnip_weight ∈ bag_weights ∧
  (d.onion_weights ++ d.carrot_weights).sum = total_weight - d.turnip_weight ∧
  d.carrot_weights.sum = 2 * d.onion_weights.sum ∧
  (d.onion_weights ++ d.carrot_weights).toFinset ⊆ bag_weights.toFinset.erase d.turnip_weight

theorem turnip_bag_weights :
  ∀ d : BagDistribution, is_valid_distribution d → d.turnip_weight = 13 ∨ d.turnip_weight = 16 := by
  sorry

end NUMINAMATH_CALUDE_turnip_bag_weights_l3068_306829


namespace NUMINAMATH_CALUDE_cube_edge_length_l3068_306898

/-- Given a cube with surface area 18 dm², prove that the length of its edge is √3 dm. -/
theorem cube_edge_length (S : ℝ) (edge : ℝ) (h1 : S = 18) (h2 : S = 6 * edge ^ 2) : 
  edge = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cube_edge_length_l3068_306898


namespace NUMINAMATH_CALUDE_inequality_proof_l3068_306810

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  a * b ≤ 1/4 ∧ Real.sqrt a + Real.sqrt b ≤ Real.sqrt 2 ∧ a^2 + b^2 ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3068_306810


namespace NUMINAMATH_CALUDE_short_students_fraction_l3068_306843

/-- Given a class with the following properties:
  * There are 400 total students
  * There are 90 tall students
  * There are 150 students with average height
  Prove that the fraction of short students to the total number of students is 2/5 -/
theorem short_students_fraction (total : ℕ) (tall : ℕ) (average : ℕ) 
  (h_total : total = 400)
  (h_tall : tall = 90)
  (h_average : average = 150) :
  (total - tall - average : ℚ) / total = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_short_students_fraction_l3068_306843


namespace NUMINAMATH_CALUDE_scientific_notation_170000_l3068_306804

theorem scientific_notation_170000 :
  170000 = 1.7 * (10 : ℝ)^5 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_170000_l3068_306804


namespace NUMINAMATH_CALUDE_sqrt_c_value_l3068_306888

theorem sqrt_c_value (a b c : ℝ) :
  (a^2 + 2020 * a + c = 0) →
  (b^2 + 2020 * b + c = 0) →
  (a / b + b / a = 98) →
  Real.sqrt c = 202 := by
sorry

end NUMINAMATH_CALUDE_sqrt_c_value_l3068_306888


namespace NUMINAMATH_CALUDE_zeros_order_l3068_306821

noncomputable def f (x : ℝ) := Real.exp x + x
noncomputable def g (x : ℝ) := Real.log x + x
noncomputable def h (x : ℝ) := Real.log x - 1

theorem zeros_order (a b c : ℝ) 
  (ha : f a = 0) 
  (hb : g b = 0) 
  (hc : h c = 0) : 
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_zeros_order_l3068_306821


namespace NUMINAMATH_CALUDE_polygon_sides_l3068_306836

theorem polygon_sides (S : ℝ) (h : S = 1080) :
  ∃ n : ℕ, n > 2 ∧ (n - 2) * 180 = S ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3068_306836


namespace NUMINAMATH_CALUDE_gcd_seven_digit_special_l3068_306827

def seven_digit_special (n : ℕ) : ℕ := 1001000 * n + n / 100

theorem gcd_seven_digit_special :
  ∃ (k : ℕ), k > 0 ∧ ∀ (n : ℕ), 100 ≤ n ∧ n < 1000 → 
    (k ∣ seven_digit_special n) ∧
    ∀ (m : ℕ), m > k ∧ (∀ (p : ℕ), 100 ≤ p ∧ p < 1000 → m ∣ seven_digit_special p) → False :=
by sorry

end NUMINAMATH_CALUDE_gcd_seven_digit_special_l3068_306827


namespace NUMINAMATH_CALUDE_prime_divisibility_l3068_306816

theorem prime_divisibility (p a b : ℕ) : 
  Prime p → 
  p ≠ 3 → 
  a > 0 → 
  b > 0 → 
  p ∣ (a + b) → 
  p^2 ∣ (a^3 + b^3) → 
  p^2 ∣ (a + b) ∨ p^3 ∣ (a^3 + b^3) := by
sorry

end NUMINAMATH_CALUDE_prime_divisibility_l3068_306816


namespace NUMINAMATH_CALUDE_intersection_points_correct_l3068_306820

/-- Parallelogram with given dimensions divided into three equal areas -/
structure EqualAreaParallelogram where
  AB : ℝ
  AD : ℝ
  BE : ℝ
  h_AB : AB = 153
  h_AD : AD = 180
  h_BE : BE = 135

/-- The points where perpendicular lines intersect AD -/
def intersection_points (p : EqualAreaParallelogram) : ℝ × ℝ :=
  (96, 156)

/-- Theorem stating that the intersection points are correct -/
theorem intersection_points_correct (p : EqualAreaParallelogram) :
  intersection_points p = (96, 156) :=
sorry

end NUMINAMATH_CALUDE_intersection_points_correct_l3068_306820


namespace NUMINAMATH_CALUDE_constant_remainder_iff_b_eq_neg_four_thirds_l3068_306838

-- Define the polynomials
def f (b : ℚ) (x : ℚ) : ℚ := 12 * x^3 - 9 * x^2 + b * x + 8
def g (x : ℚ) : ℚ := 3 * x^2 - 4 * x + 2

-- Define the remainder function
def remainder (b : ℚ) (x : ℚ) : ℚ := f b x - g x * ((4 * x) + (b + 7) / 3)

-- Theorem statement
theorem constant_remainder_iff_b_eq_neg_four_thirds :
  (∃ (c : ℚ), ∀ (x : ℚ), remainder b x = c) ↔ b = -4/3 :=
sorry

end NUMINAMATH_CALUDE_constant_remainder_iff_b_eq_neg_four_thirds_l3068_306838


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l3068_306840

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 - b * x + c

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h_solution_set : ∀ x : ℝ, f a b c x > 0 ↔ -1 < x ∧ x < 2) :
  (a + b + c = 0) ∧ (a < 0) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_inequality_properties_l3068_306840


namespace NUMINAMATH_CALUDE_total_employee_purchase_price_l3068_306822

/-- Represents an item in the store -/
structure Item where
  name : String
  wholesale_cost : ℝ
  markup : ℝ
  employee_discount : ℝ

/-- Calculates the final price for an employee -/
def employee_price (item : Item) : ℝ :=
  item.wholesale_cost * (1 + item.markup) * (1 - item.employee_discount)

/-- The three items in the store -/
def video_recorder : Item :=
  { name := "Video Recorder", wholesale_cost := 200, markup := 0.20, employee_discount := 0.30 }

def digital_camera : Item :=
  { name := "Digital Camera", wholesale_cost := 150, markup := 0.25, employee_discount := 0.20 }

def smart_tv : Item :=
  { name := "Smart TV", wholesale_cost := 800, markup := 0.15, employee_discount := 0.25 }

/-- Theorem: The total amount paid by an employee for all three items is $1008 -/
theorem total_employee_purchase_price :
  employee_price video_recorder + employee_price digital_camera + employee_price smart_tv = 1008 := by
  sorry

end NUMINAMATH_CALUDE_total_employee_purchase_price_l3068_306822


namespace NUMINAMATH_CALUDE_derivative_parity_l3068_306802

-- Define even and odd functions
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem derivative_parity (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (IsEven f → IsOdd f') ∧ (IsOdd f → IsEven f') := by sorry

end NUMINAMATH_CALUDE_derivative_parity_l3068_306802


namespace NUMINAMATH_CALUDE_parallelogram_diagonals_l3068_306899

/-- Represents a parallelogram with given side length and diagonal lengths -/
structure Parallelogram where
  side : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- Check if three lengths can form a triangle -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The main theorem to prove -/
theorem parallelogram_diagonals (p : Parallelogram) : 
  p.side = 10 →
  (p.diagonal1 = 20 ∧ p.diagonal2 = 30) ↔
  (canFormTriangle (p.side) (p.diagonal1 / 2) (p.diagonal2 / 2) ∧
   ¬(canFormTriangle p.side 2 3) ∧
   ¬(canFormTriangle p.side 3 4) ∧
   ¬(canFormTriangle p.side 4 6)) :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_diagonals_l3068_306899


namespace NUMINAMATH_CALUDE_expression_equality_l3068_306823

theorem expression_equality : 
  (Real.sqrt (4/3) + Real.sqrt 3) * Real.sqrt 6 - (Real.sqrt 20 - Real.sqrt 5) / Real.sqrt 5 = 5 * Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3068_306823


namespace NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l3068_306866

/-- Proves that the ratio of time taken to row upstream to downstream is 2:1 --/
theorem upstream_downstream_time_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : boat_speed = 51)
  (h2 : stream_speed = 17) : 
  (boat_speed + stream_speed) / (boat_speed - stream_speed) = 2 := by
  sorry

#check upstream_downstream_time_ratio

end NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l3068_306866


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l3068_306852

def f (x : ℝ) : ℝ := -x^2 + 4*x + 5

theorem max_min_values_of_f :
  let a : ℝ := 1
  let b : ℝ := 4
  (∀ x ∈ Set.Icc a b, f x ≤ 9) ∧
  (∃ x ∈ Set.Icc a b, f x = 9) ∧
  (∀ x ∈ Set.Icc a b, f x ≥ 5) ∧
  (∃ x ∈ Set.Icc a b, f x = 5) :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l3068_306852


namespace NUMINAMATH_CALUDE_largest_constant_divisor_l3068_306874

theorem largest_constant_divisor (n : ℤ) : 
  let x : ℤ := 4 * n - 1
  ∃ (k : ℤ), (12 * x + 2) * (8 * x + 6) * (6 * x + 3) = 60 * k ∧ 
  ∀ (m : ℤ), m > 60 → 
    ∃ (l : ℤ), (12 * x + 2) * (8 * x + 6) * (6 * x + 3) ≠ m * l :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_divisor_l3068_306874


namespace NUMINAMATH_CALUDE_xy_and_x3y_plus_x2_l3068_306803

theorem xy_and_x3y_plus_x2 (x y : ℝ) 
  (hx : x = 2 + Real.sqrt 3) 
  (hy : y = 2 - Real.sqrt 3) : 
  x * y = 1 ∧ x^3 * y + x^2 = 14 + 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_xy_and_x3y_plus_x2_l3068_306803


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l3068_306809

/-- Given an ellipse C and a line l, if l intersects C at two points with a specific distance, then the y-intercept of l is 0. -/
theorem ellipse_line_intersection (x y m : ℝ) : 
  (4 * x^2 + y^2 = 1) →  -- Ellipse equation
  (y = x + m) →          -- Line equation
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
    (4 * A.1^2 + A.2^2 = 1) ∧ 
    (4 * B.1^2 + B.2^2 = 1) ∧ 
    (A.2 = A.1 + m) ∧ 
    (B.2 = B.1 + m) ∧ 
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = (2 * Real.sqrt 10 / 5)^2)) →
  m = 0 := by
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l3068_306809


namespace NUMINAMATH_CALUDE_general_term_formula_l3068_306869

/-- Given a sequence {a_n} where S_n is the sum of its first n terms -/
def S (n : ℕ) : ℝ := 3^n - 1

/-- The general term of the sequence -/
def a (n : ℕ) : ℝ := 2 * 3^(n - 1)

/-- Theorem stating that the given general term formula is correct -/
theorem general_term_formula (n : ℕ) (h : n ≥ 1) : 
  a n = S n - S (n - 1) := by sorry

end NUMINAMATH_CALUDE_general_term_formula_l3068_306869


namespace NUMINAMATH_CALUDE_line_slope_equals_k_l3068_306851

/-- Given a line passing through points (-1, -4) and (5, k), 
    if the slope of the line is equal to k, then k = 4/5 -/
theorem line_slope_equals_k (k : ℚ) : 
  (k - (-4)) / (5 - (-1)) = k → k = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_equals_k_l3068_306851


namespace NUMINAMATH_CALUDE_second_item_cost_price_l3068_306890

/-- Given two items sold together for 432 yuan, where one item is sold at a 20% loss
    and the combined sale results in a 20% profit, prove that the cost price of the second item is 90 yuan. -/
theorem second_item_cost_price (total_selling_price : ℝ) (loss_percentage : ℝ) (profit_percentage : ℝ) 
  (h1 : total_selling_price = 432)
  (h2 : loss_percentage = 0.20)
  (h3 : profit_percentage = 0.20) :
  ∃ (cost_price_1 cost_price_2 : ℝ),
    cost_price_1 * (1 - loss_percentage) = total_selling_price / 2 ∧
    total_selling_price = (cost_price_1 + cost_price_2) * (1 + profit_percentage) ∧
    cost_price_2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_second_item_cost_price_l3068_306890


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3068_306880

/-- Calculates the area of a rectangular field given specific fencing conditions -/
theorem rectangular_field_area 
  (uncovered_side : ℝ) 
  (total_fencing : ℝ) 
  (h1 : uncovered_side = 20) 
  (h2 : total_fencing = 88) : 
  uncovered_side * ((total_fencing - uncovered_side) / 2) = 680 :=
by
  sorry

#check rectangular_field_area

end NUMINAMATH_CALUDE_rectangular_field_area_l3068_306880


namespace NUMINAMATH_CALUDE_product_357_sum_28_l3068_306897

theorem product_357_sum_28 (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
  a * b * c * d = 357 → 
  a + b + c + d = 28 := by
sorry

end NUMINAMATH_CALUDE_product_357_sum_28_l3068_306897


namespace NUMINAMATH_CALUDE_multiplication_problem_l3068_306812

theorem multiplication_problem : 8 * (1 / 15) * 30 * 3 = 48 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_l3068_306812


namespace NUMINAMATH_CALUDE_kaleb_total_score_l3068_306896

/-- Kaleb's score in the first half of the game -/
def first_half_score : ℕ := 43

/-- Kaleb's score in the second half of the game -/
def second_half_score : ℕ := 23

/-- Kaleb's total score in the game -/
def total_score : ℕ := first_half_score + second_half_score

/-- Theorem stating that Kaleb's total score is 66 points -/
theorem kaleb_total_score : total_score = 66 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_total_score_l3068_306896


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l3068_306881

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (heq : x + 2*y + 2*x*y = 8) :
  ∀ z, z = x + 2*y → z ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l3068_306881


namespace NUMINAMATH_CALUDE_min_distance_squared_l3068_306805

def is_geometric_progression (x y z : ℝ) : Prop :=
  ∃ r : ℝ, y = x * r ∧ z = y * r

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem min_distance_squared (x y z : ℝ) :
  is_geometric_progression x y z →
  is_arithmetic_progression (x * y) (y * z) (x * z) →
  z ≥ 1 →
  x ≠ y →
  y ≠ z →
  x ≠ z →
  (∀ x' y' z' : ℝ, 
    is_geometric_progression x' y' z' →
    is_arithmetic_progression (x' * y') (y' * z') (x' * z') →
    z' ≥ 1 →
    x' ≠ y' →
    y' ≠ z' →
    x' ≠ z' →
    (x - 1)^2 + (y - 1)^2 + (z - 1)^2 ≤ (x' - 1)^2 + (y' - 1)^2 + (z' - 1)^2) →
  (x - 1)^2 + (y - 1)^2 + (z - 1)^2 = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_squared_l3068_306805


namespace NUMINAMATH_CALUDE_switch_circuit_probability_l3068_306818

theorem switch_circuit_probability (P_A P_AB : ℝ) 
  (h1 : P_A = 1/2) 
  (h2 : P_AB = 1/5) : 
  P_AB / P_A = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_switch_circuit_probability_l3068_306818


namespace NUMINAMATH_CALUDE_journey_distance_l3068_306872

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h1 : total_time = 40)
  (h2 : speed1 = 20)
  (h3 : speed2 = 30)
  (h4 : total_time = (distance / 2) / speed1 + (distance / 2) / speed2) :
  distance = 960 :=
by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l3068_306872


namespace NUMINAMATH_CALUDE_dhoni_doll_expenditure_l3068_306887

theorem dhoni_doll_expenditure :
  ∀ (total_spent : ℕ) (large_price small_price : ℕ),
    large_price = 6 →
    small_price = large_price - 2 →
    (total_spent / small_price) - (total_spent / large_price) = 25 →
    total_spent = 300 := by
  sorry

end NUMINAMATH_CALUDE_dhoni_doll_expenditure_l3068_306887
